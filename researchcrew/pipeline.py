"""The research loop: Plan -> Retrieve -> Research -> Critique -> (re-research gaps)* -> Write.

Design notes
- Retrieval is deterministic Python (not LLM tool-calling): cheaper, reproducible, and removes the
  most common failure mode of small open models (malformed tool calls).
- Every passage gets an Evidence ID. Agents may only cite those IDs; hallucinated IDs are stripped
  and the final bibliography is built from real metadata (see citations.py).
- The critic is hybrid: an LLM reviewer plus deterministic guardrails (e.g. "no evidence at all").
- The loop is bounded (max_iterations) and stops early when the critic is satisfied or when a
  re-search finds nothing new.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Callable

from . import prompts
from .agents import AgentError, AgentRunner
from .citations import finalize, sanitize
from .config import Settings
from .evidence import EvidenceStore
from .models import CriticReview, Finding, Gap, ResearchResult, SubQuestion
from .utils import clip, extract_json

log = logging.getLogger(__name__)
EventCallback = Callable[[str, str], None]


class ResearchPipeline:
    def __init__(self, runner: AgentRunner, retriever, settings: Settings,
                 on_event: EventCallback | None = None) -> None:
        self.runner, self.retriever, self.s = runner, retriever, settings
        self._on_event = on_event or (lambda stage, msg: None)

    def _emit(self, stage: str, msg: str) -> None:
        log.info("[%s] %s", stage, msg)
        self._on_event(stage, msg)

    # ------------------------------------------------------------------ planning
    def plan(self, question: str) -> list[SubQuestion]:
        raw = self.runner.run("planner", prompts.planner(question, self.s.max_subquestions),
                              prompts.PLANNER_EXPECTED)
        subs: list[SubQuestion] = []
        try:
            for i, item in enumerate(extract_json(raw).get("sub_questions", []), start=1):
                q = str(item.get("question", "")).strip()
                if not q:
                    continue
                queries = [str(x).strip() for x in item.get("search_queries", []) if str(x).strip()]
                subs.append(SubQuestion(id=i, question=q, search_queries=queries[:2] or [q]))
        except (ValueError, AttributeError) as exc:
            log.warning("planner output unparseable (%s); falling back to single question", exc)
        if not subs:
            subs = [SubQuestion(id=1, question=question, search_queries=[question])]
        return subs[: self.s.max_subquestions]

    # ----------------------------------------------------------------- retrieval
    def _collect(self, sub: SubQuestion, queries: list[str], store: EvidenceStore) -> list[str]:
        """Retrieve for `queries`, attach to `sub`, return IDs that are NEW for this sub-question."""
        new: list[str] = []
        for q in queries:
            for ev in self.retriever.retrieve(q):
                ev = store.add(ev)
                if ev.id not in sub.evidence_ids:
                    sub.evidence_ids.append(ev.id)
                    new.append(ev.id)
        # keep the best N (new evidence first so follow-ups are never crowded out)
        cap = self.s.max_evidence_per_sub
        if len(sub.evidence_ids) > cap:
            old = [i for i in sub.evidence_ids if i not in new]
            old.sort(key=lambda i: store.get(i).score, reverse=True)
            keep_new = sorted(new, key=lambda i: store.get(i).score, reverse=True)[:cap]
            sub.evidence_ids = keep_new + old[: cap - len(keep_new)]
        return [i for i in new if i in sub.evidence_ids]

    # ------------------------------------------------------------------ research
    def _research(self, question: str, sub: SubQuestion, store: EvidenceStore,
                  previous: Finding | None = None, critique: str = "") -> Finding:
        evidence = store.many(sub.evidence_ids)
        if not evidence:
            return Finding(sub.id, sub.question, "No evidence could be retrieved for this sub-question.",
                           "low", "all", [], (previous.revision + 1) if previous else 0)
        block = "\n\n".join(e.for_prompt() for e in evidence)
        raw = self.runner.run("researcher",
                              prompts.researcher(question, sub.question, block,
                                                 previous.answer if previous else "", critique),
                              prompts.RESEARCHER_EXPECTED)
        finding = self._parse_finding(raw, sub)
        finding.answer, _ = sanitize(finding.answer, set(sub.evidence_ids))
        finding.revision = (previous.revision + 1) if previous else 0
        return finding

    @staticmethod
    def _parse_finding(raw: str, sub: SubQuestion) -> Finding:
        m = re.search(r"ANSWER:\s*(.*?)\s*CONFIDENCE:\s*(\w+)\s*(?:UNRESOLVED:\s*(.*))?$",
                      raw, re.DOTALL | re.IGNORECASE)
        if m:
            answer, conf, unresolved = m.group(1).strip(), m.group(2).lower(), (m.group(3) or "").strip()
        else:
            answer, conf, unresolved = raw.strip(), "low", ""
        if conf not in {"high", "medium", "low"}:
            conf = "medium"
        return Finding(sub.id, sub.question, answer, conf, unresolved, list(sub.evidence_ids))

    def critique(self, question: str, subs: list[SubQuestion], findings: dict[int, Finding],
                 store: EvidenceStore, iteration: int) -> CriticReview:
        parts = []
        for sub in subs:
            f = findings[sub.id]
            evs = store.many(sub.evidence_ids)
            web = sum(e.source_type == "web" for e in evs)
            parts.append(f"Sub-question {sub.id}: {sub.question}\n"
                         f"Confidence: {f.confidence} | sources: {len(evs)} ({web} web, {len(evs) - web} pdf)\n"
                         f"Answer: {clip(f.answer, 900)}\nUnresolved: {f.unresolved or 'none'}")
        block = "\n\n".join(parts)
        score, verdict, summary, gaps, ok = 0.0, "sufficient", "", [], True
        prompt = prompts.critic(question, block, self.s.sufficiency_score)
        data, error = None, None
        for attempt in range(2):          # one retry if the model answers with something that is not JSON
            try:
                hint = "" if attempt == 0 else "\n\nYour previous reply was not valid JSON. Reply with ONLY the JSON object."
                data = extract_json(self.runner.run("critic", prompt + hint, prompts.CRITIC_EXPECTED))
                break
            except ValueError as exc:
                error = exc
            except AgentError as exc:     # runner already retried with backoff; do not multiply waits
                error = exc
                break
        if data is not None:
            try:
                score = float(data.get("score", 5))
                verdict = str(data.get("verdict", "insufficient")).lower()
                summary = str(data.get("summary", ""))
                valid = {s.id for s in subs}
                for g in data.get("gaps", [])[:3]:
                    sid = int(g.get("sub_question_id", -1))
                    if sid in valid:
                        qs = [str(x) for x in g.get("follow_up_queries", []) if str(x).strip()]
                        gaps.append(Gap(sid, str(g.get("issue", "")), qs[:2]))
            except (ValueError, TypeError) as exc:
                data, error = None, exc
        if data is None:
            ok = False
            log.warning("critic failed (%s); proceeding with deterministic checks only", error)
            summary = f"Critic unavailable ({str(error)[:160]}); only automatic checks applied."

        # deterministic guardrails: a sub-question with zero evidence is always a gap
        flagged = {g.sub_id for g in gaps}
        for sub in subs:
            if not sub.evidence_ids and sub.id not in flagged:
                gaps.append(Gap(sub.id, "No evidence was retrieved.",
                                [sub.question, f"{sub.question} overview"], auto=True))
        if ok and score >= self.s.sufficiency_score:
            verdict = "sufficient"
        if any(g.auto for g in gaps):          # missing evidence overrides an optimistic critic
            verdict = "insufficient"
        return CriticReview(iteration, score, verdict, summary, gaps[: self.s.max_gaps], ok)

    def _fill_gaps(self, question: str, review: CriticReview, subs: list[SubQuestion],
                   findings: dict[int, Finding], store: EvidenceStore) -> bool:
        by_id = {s.id: s for s in subs}
        progress = False
        for gap in review.gaps:
            sub = by_id[gap.sub_id]
            queries = gap.follow_up_queries or [clip(f"{sub.question} {gap.issue}", 180)]
            self._emit("retrieve", f"Gap in Q{sub.id}: {gap.issue} -> searching {queries}")
            new_ids = self._collect(sub, queries, store)
            if not new_ids:
                self._emit("retrieve", f"Q{sub.id}: no new evidence found")
                continue
            progress = True
            self._emit("research", f"Q{sub.id}: re-researching with {len(new_ids)} new sources")
            findings[sub.id] = self._research(question, sub, store, findings[sub.id], gap.issue)
        return progress

    # ------------------------------------------------------------------- writer
    def write(self, question: str, subs: list[SubQuestion], findings: dict[int, Finding],
              review: CriticReview, store: EvidenceStore) -> str:
        used = sorted({i for f in findings.values() for i in f.evidence_ids}, key=lambda x: int(x[1:]))
        block = "\n\n".join(f"### Finding {s.id}: {s.question}\n(confidence: {findings[s.id].confidence})\n"
                            f"{findings[s.id].answer}" for s in subs)
        limitations = " ".join(filter(None, [review.summary] + [g.issue for g in review.gaps]))
        try:
            text = self.runner.run("writer", prompts.writer(question, block, used, limitations),
                                   prompts.WRITER_EXPECTED)
            text, _ = sanitize(text, set(used))
            return text
        except AgentError as exc:
            log.warning("writer failed (%s); using deterministic fallback report", exc)
            return self._fallback_report(question, subs, findings)

    @staticmethod
    def _fallback_report(question: str, subs: list[SubQuestion], findings: dict[int, Finding]) -> str:
        lines = [f"# Research Report: {question}", "", "## Key Findings", ""]
        for s in subs:
            f = findings[s.id]
            lines += [f"### {s.question}", f.answer, f"*Confidence: {f.confidence}*", ""]
        return "\n".join(lines)

    # --------------------------------------------------------------------- main
    def run(self, question: str, max_iterations: int | None = None) -> ResearchResult:
        t0 = time.time()
        max_iter = self.s.max_iterations if max_iterations is None else max_iterations
        store = EvidenceStore()

        self._emit("plan", "Planner is decomposing the question")
        subs = self.plan(question)
        self._emit("plan", f"{len(subs)} sub-questions: " + " | ".join(s.question for s in subs))

        findings: dict[int, Finding] = {}
        for sub in subs:
            self._emit("retrieve", f"Q{sub.id}: searching {sub.search_queries}")
            self._collect(sub, sub.search_queries, store)
            self._emit("research", f"Q{sub.id}: researching with {len(sub.evidence_ids)} sources")
            findings[sub.id] = self._research(question, sub, store)

        reviews: list[CriticReview] = []
        rounds = 0
        while True:
            self._emit("critique", f"Critic review #{rounds + 1}")
            review = self.critique(question, subs, findings, store, rounds + 1)
            reviews.append(review)
            self._emit("critique", (f"score {review.score:g}/10, verdict: {review.verdict}, "
                                    f"{len(review.gaps)} gap(s)") if review.ok else
                                   "critic unavailable - see terminal log; continuing without review")
            if review.verdict == "sufficient" or not review.gaps or rounds >= max_iter:
                break
            rounds += 1
            if not self._fill_gaps(question, review, subs, findings, store):
                self._emit("critique", "No new evidence available; stopping the loop")
                break

        self._emit("write", "Writer is composing the report")
        ordered = [findings[s.id] for s in subs]
        draft = self.write(question, subs, findings, reviews[-1], store)
        final = finalize(draft, store)
        stats = {**final.stats, "sub_questions": len(subs), "iterations": rounds,
                 "evidence_retrieved": len(store), "final_score": reviews[-1].score}
        self._emit("done", f"{stats['unique_sources']} sources cited, {stats['citations']} citations")
        return ResearchResult(question, subs, ordered, reviews, final.text, final.references,
                              store.all(), stats, round(time.time() - t0, 1))
