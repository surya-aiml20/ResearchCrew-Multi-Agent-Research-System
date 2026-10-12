"""Citation hygiene: the LLM never gets to invent a source.

The writer cites evidence as [E3] / [E1, E4]. After generation we:
  1. drop any ID that does not exist in the evidence store (hallucinated citations),
  2. renumber the survivors [1], [2], ... by order of first appearance,
  3. build the "Sources" section programmatically from real retrieved metadata.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .evidence import EvidenceStore

CITE_RE = re.compile(r"\[\s*(E\d+(?:\s*[,;]\s*E\d+)*)\s*\]")
_TRAILING_SOURCES = re.compile(r"\n#{1,4}\s*(sources|references|bibliography)\b.*",
                               re.IGNORECASE | re.DOTALL)


def _ids(group: str) -> list[str]:
    return re.findall(r"E\d+", group)


def sanitize(text: str, allowed: set[str]) -> tuple[str, int]:
    """Remove citations whose IDs are not in `allowed`. Returns (text, removed_count)."""
    removed = 0

    def repl(m: re.Match) -> str:
        nonlocal removed
        ids = _ids(m.group(1))
        keep = [i for i in ids if i in allowed]
        removed += len(ids) - len(keep)
        return f"[{', '.join(keep)}]" if keep else ""

    return CITE_RE.sub(repl, text), removed


@dataclass
class FinalizedReport:
    text: str
    references: list[dict[str, Any]]
    stats: dict[str, Any]


def finalize(report: str, store: EvidenceStore) -> FinalizedReport:
    report = _TRAILING_SOURCES.sub("", report).rstrip()
    order: dict[str, int] = {}
    invalid = 0
    total = 0

    def repl(m: re.Match) -> str:
        nonlocal invalid, total
        nums: list[int] = []
        for ev_id in _ids(m.group(1)):
            if store.get(ev_id) is None:
                invalid += 1
                continue
            order.setdefault(ev_id, len(order) + 1)
            nums.append(order[ev_id])
            total += 1
        nums = sorted(set(nums))
        return f"[{', '.join(map(str, nums))}]" if nums else ""

    body = CITE_RE.sub(repl, report)
    body = re.sub(r"[ \t]+([.,;:])", r"\1", body)  # tidy spaces left by removed markers

    refs: list[dict[str, Any]] = []
    for ev_id, n in sorted(order.items(), key=lambda kv: kv[1]):
        ev = store.get(ev_id)
        assert ev is not None
        refs.append({"n": n, "evidence_id": ev_id, "type": ev.source_type,
                     "title": ev.title, "location": ev.location})

    if refs:
        lines = ["", "## Sources", ""]
        for r in refs:
            if r["type"] == "web":
                lines.append(f"{r['n']}. {r['title']} - <{r['location']}>")
            else:
                lines.append(f"{r['n']}. {r['location']} (PDF)")
        body = body + "\n" + "\n".join(lines) + "\n"

    stats = {
        "citations": total,
        "unique_sources": len(refs),
        "web_sources": sum(1 for r in refs if r["type"] == "web"),
        "pdf_sources": sum(1 for r in refs if r["type"] == "pdf"),
        "invalid_citations_removed": invalid,
    }
    return FinalizedReport(body, refs, stats)
