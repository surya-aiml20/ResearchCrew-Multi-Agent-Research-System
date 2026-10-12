"""Typed domain objects shared across the pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .utils import clip


@dataclass
class Evidence:
    source_type: str          # "web" | "pdf"
    title: str
    location: str             # URL for web, "file.pdf, p.3" for PDFs
    text: str
    score: float = 0.0
    id: str = ""              # assigned by EvidenceStore, e.g. "E7"

    def for_prompt(self, max_chars: int = 700) -> str:
        return (f"[{self.id}] ({self.source_type.upper()}) {self.title} - {self.location}\n"
                f"{clip(self.text, max_chars)}")


@dataclass
class SubQuestion:
    id: int
    question: str
    search_queries: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class Finding:
    sub_id: int
    question: str
    answer: str
    confidence: str = "medium"
    unresolved: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    revision: int = 0


@dataclass
class Gap:
    sub_id: int
    issue: str
    follow_up_queries: list[str] = field(default_factory=list)
    auto: bool = False        # True when raised by deterministic guardrails, not the LLM critic


@dataclass
class CriticReview:
    iteration: int
    score: float
    verdict: str              # "sufficient" | "insufficient"
    summary: str
    gaps: list[Gap] = field(default_factory=list)
    ok: bool = True 


@dataclass
class ResearchResult:
    question: str
    plan: list[SubQuestion]
    findings: list[Finding]
    reviews: list[CriticReview]
    report: str
    references: list[dict[str, Any]]
    evidence: list[Evidence]
    stats: dict[str, Any]
    elapsed_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
