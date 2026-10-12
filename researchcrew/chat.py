"""Conversation logic for the chat UI.

Per message the router picks one of three modes:
  pdf       - the question is about the PDF(s) uploaded IN THIS CHAT  -> answer only from the PDF
  research  - needs new information                                    -> full crew (web + this chat's PDFs)
  followup  - answerable from the earlier conversation                 -> no retrieval

PDF isolation is structural: each chat has its own vector collection, so a new chat has no PDFs.
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict

from .agents import AgentError, AgentRunner
from .citations import finalize
from .evidence import EvidenceStore
from .utils import clip, extract_json

log = logging.getLogger(__name__)

# message schema (JSON-serialisable):
#   {"role": "user", "content": str}
#   {"role": "assistant", "type": "research", "content": <report md>, "trace": dict, ...}
#   {"role": "assistant", "type": "pdf",      "content": <cited answer>, "passages": [dict]}
#   {"role": "assistant", "type": "chat",     "content": str}
#   {"role": "assistant", "type": "notice",   "content": "📎 Indexed ..."}   (not sent to LLMs)

PDF_REF_RE = re.compile(
    r"\b(pdf|pdfs|document|documents|file|paper|uploaded|upload|attachment|attached|this doc|the doc)\b", re.I)
OVERVIEW_RE = re.compile(
    r"(summar\w*|overview|tell me about|what('s| is) (it|this|the \w+) about|key (points|takeaways)|"
    r"main (points|ideas|topics)|table of contents|outline|contents?\b|explain (the|this) )", re.I)


def mentions_pdf(message: str) -> bool:
    return bool(PDF_REF_RE.search(message))


def is_overview(message: str) -> bool:
    return bool(OVERVIEW_RE.search(message))


def history_digest(messages: list[dict], max_messages: int = 6, report_chars: int = 3000) -> str:
    lines = []
    for m in [m for m in messages if m.get("type") != "notice"][-max_messages:]:
        if m["role"] == "user":
            lines.append(f"USER: {clip(m['content'], 600)}")
        else:
            limit = report_chars if m.get("type") in {"research", "pdf"} else 1200
            lines.append(f"ASSISTANT ({m.get('type', 'chat')}): {clip(m['content'], limit)}")
    return "\n\n".join(lines)


def _router_prompt(message: str, digest: str, pdf_names: list[str], pdf_score: float | None) -> str:
    if pdf_names:
        pdf_info = (f"PDFs uploaded in this chat: {', '.join(pdf_names)}.\n"
                    f"Best similarity between the message and the PDF content: "
                    f"{pdf_score:.2f} (0.45+ usually means the PDF contains relevant text).")
    else:
        pdf_info = "No PDFs have been uploaded in this chat."
    return f"""Conversation so far:
{digest or '(empty)'}

{pdf_info}

New user message: {message}

Choose how to handle it:
- "pdf": it refers to the uploaded document(s) OR is plausibly answered by their content
  (summaries, "what does it say about X", specific facts, definitions found in the document).
  Only valid if PDFs exist.
- "research": it needs NEW information from the web or a new topic unrelated to the PDFs.
- "followup": it can be answered from the earlier conversation (shorten, reformat, clarify, compare).
Also rewrite the message as a fully standalone question (resolve "it", "that", "the second one").

Return ONLY JSON: {{"mode": "pdf|research|followup", "standalone_question": "..."}}"""


def route(runner: AgentRunner, message: str, messages: list[dict],
          pdf_names: list[str] | None = None, pdf_score: float | None = None) -> tuple[str, str]:
    """Returns (mode, standalone_question)."""
    pdf_names = pdf_names or []
    if pdf_names and mentions_pdf(message):          # explicit reference: no LLM needed
        return "pdf", message
    if not messages and not pdf_names:
        return "research", message
    try:
        raw = runner.run("router", _router_prompt(message, history_digest(messages), pdf_names, pdf_score),
                         "A JSON object with mode and standalone_question.")
        data = extract_json(raw)
        mode = str(data.get("mode", "research")).lower()
        if mode not in {"pdf", "research", "followup"} or (mode == "pdf" and not pdf_names):
            mode = "research"
        if mode == "followup" and not messages:
            mode = "research"
        return mode, (str(data.get("standalone_question") or message).strip() or message)
    except (AgentError, ValueError) as exc:
        log.warning("router failed (%s); falling back", exc)
        return ("pdf" if pdf_names and (pdf_score or 0) >= 0.45 else "research"), message


def answer_followup(runner: AgentRunner, message: str, messages: list[dict]) -> str:
    digest = history_digest(messages, max_messages=6, report_chars=6000)
    prompt = f"""You are continuing a research conversation. Answer the user's new message using ONLY the
earlier conversation below. Keep any [n] citation markers that appear in the text you rely on. If the
earlier research does not contain the answer, say so plainly and suggest a research question to run next.

CONVERSATION:
{digest}

NEW MESSAGE: {message}"""
    return runner.run("chat", prompt, "A concise, accurate markdown answer.")


def answer_from_pdf(runner: AgentRunner, store, message: str, standalone: str | None = None,
                    k: int = 8, max_passages: int = 14) -> tuple[str, list[dict]]:
    """RAG over the PDFs of the current chat only. Returns (cited_markdown, passages)."""
    query = standalone or message
    ev = EvidenceStore()
    hits = store.search(query, k=k, min_score=0.15)
    if is_overview(message) or is_overview(query) or len(hits) < 2:
        hits = hits + store.overview(10)          # broad coverage for "tell me about the PDF"
    for h in hits[:max_passages]:
        ev.add(h)
    if not len(ev):
        return "I couldn't extract any text from the PDF(s) in this chat.", []

    block = "\n\n".join(e.for_prompt(900) for e in ev.all())
    prompt = f"""You answer questions about the PDF document(s) the user uploaded in THIS chat, using ONLY the
passages below. Each passage has an ID like [E3] plus its file and page.

QUESTION: {query}

PASSAGES:
{block}

Rules:
- Cite every claim with the passage ID(s), e.g. "... [E2]" or "[E1, E4]". Use only IDs shown above.
- For summary / "what is this about" requests, synthesise the main themes across passages (use short
  markdown bullets) and make clear what the document covers.
- If the passages do not contain the answer, say "The uploaded PDF doesn't appear to cover this" and
  suggest what could be searched on the web instead. Never use outside knowledge or invent details.
- Do not write a Sources section; it is added automatically."""
    raw = runner.run("researcher", prompt, "A cited markdown answer grounded only in the passages.")
    final = finalize(raw, ev)
    return final.text, [asdict(e) for e in ev.all()]