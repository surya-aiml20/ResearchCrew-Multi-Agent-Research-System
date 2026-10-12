"""Hybrid retrieval: live web + local PDF knowledge base, with graceful degradation."""
from __future__ import annotations

import logging

from ..models import Evidence
from .vector_store import VectorStore
from .web_search import SerperClient

log = logging.getLogger(__name__)


class Retriever:
    def __init__(self, store: VectorStore | None = None, web: SerperClient | None = None,
                 k_web: int = 4, k_pdf: int = 4, pdf_min_score: float = 0.30) -> None:
        self.store, self.web = store, web
        self.k_web, self.k_pdf, self.pdf_min_score = k_web, k_pdf, pdf_min_score
        self.warnings: list[str] = []

    @property
    def has_sources(self) -> bool:
        return bool(self.web) or bool(self.store and self.store.count())

    def retrieve(self, query: str) -> list[Evidence]:
        results: list[Evidence] = []
        if self.store is not None:
            try:
                results += self.store.search(query, self.k_pdf, self.pdf_min_score)
            except Exception as exc:  # noqa: BLE001
                self._warn(f"PDF retrieval failed: {exc}")
        if self.web is not None:
            try:
                results += self.web.search(query, self.k_web)
            except Exception as exc:  # noqa: BLE001
                self._warn(f"Web search failed for '{query[:60]}': {exc}")
        return results

    def _warn(self, msg: str) -> None:
        log.warning(msg)
        if msg not in self.warnings:
            self.warnings.append(msg)
