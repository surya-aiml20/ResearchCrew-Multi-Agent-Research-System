"""All prompt templates in one place so behaviour is easy to audit and tune."""
from __future__ import annotations

PLANNER_EXPECTED = "A single JSON object with a 'sub_questions' list."


def planner(question: str, n: int) -> str:
    return f"""Decompose the research question into 3 to {n} focused, non-overlapping sub-questions
that together fully answer it. For each, write 1-2 concise web-search queries (keywords, not sentences).

Research question: {question}

Return ONLY valid JSON, no commentary:
{{"sub_questions": [{{"id": 1, "question": "...", "search_queries": ["...", "..."]}}]}}"""


RESEARCHER_EXPECTED = "Three labelled sections: ANSWER, CONFIDENCE, UNRESOLVED."


def researcher(question: str, sub_question: str, evidence_block: str,
               previous: str = "", critique: str = "") -> str:
    revision = ""
    if previous:
        revision = f"""
This is a REVISION. Your previous draft was:
{previous}

The reviewer flagged this issue: {critique}
Use the new evidence to fix it. Keep claims that are still supported.
"""
    return f"""You are answering one sub-question of a larger research task, using ONLY the evidence below.

Overall question: {question}
Sub-question: {sub_question}
{revision}
EVIDENCE (each item has an ID like [E3]):
{evidence_block}

Rules:
- Every factual claim MUST end with the ID(s) of the evidence supporting it, e.g. "... grew 40% [E2]" or "[E1, E4]".
- Use only IDs that appear above. Never invent facts, numbers, or IDs.
- If sources disagree, say so and cite both. If evidence is thin, say so honestly.
- 120-220 words, plain prose.

Reply in EXACTLY this format:
ANSWER: <your cited answer>
CONFIDENCE: <high|medium|low>
UNRESOLVED: <what the evidence could not answer, or "none">"""


CRITIC_EXPECTED = "A single JSON object with score, verdict, summary and gaps."


def critic(question: str, findings_block: str, threshold: float) -> str:
    return f"""You are a rigorous research reviewer. Assess whether the findings below adequately answer the
research question, using these criteria: coverage of the question, strength and diversity of evidence,
internal contradictions, unsupported or uncited claims, and stated unresolved items.

Research question: {question}

FINDINGS:
{findings_block}

Score 1-10 (>= {threshold:g} means good enough to write up). Only report a gap if additional searching
could plausibly fix it. Report at most 3 gaps, most important first. For each gap give 1-2 NEW, specific
search queries that differ from generic restatements.

Return ONLY valid JSON:
{{"score": 7, "verdict": "sufficient|insufficient", "summary": "one or two sentences",
  "gaps": [{{"sub_question_id": 1, "issue": "what is missing or weak", "follow_up_queries": ["...", "..."]}}]}}"""


WRITER_EXPECTED = "A well-structured Markdown research report with inline [E#] citations."


def writer(question: str, findings_block: str, allowed_ids: list[str], limitations: str) -> str:
    return f"""Write the final research report in Markdown from the verified findings below.

Research question: {question}

FINDINGS (already cited with evidence IDs):
{findings_block}

Reviewer notes on remaining limitations: {limitations or "none"}

Structure:
# <Specific, informative title>
## Executive Summary  (3-5 sentences)
## Key Findings  (one subsection per finding, synthesised - do not just copy)
## Limitations & Open Questions

Rules:
- Cite with the existing IDs only (allowed: {", ".join(allowed_ids)}), e.g. [E2] or [E1, E3].
- Do not add facts that are not in the findings. Do not write a Sources/References section; it is generated automatically.
- Be honest about low-confidence areas. Professional, concise tone."""
