"""Evidence registry: gives every retrieved passage a stable ID (E1, E2, ...) for citation."""
from __future__ import annotations

import hashlib

from .models import Evidence


class EvidenceStore:
    def __init__(self) -> None:
        self._items: dict[str, Evidence] = {}
        self._keys: dict[str, str] = {}

    @staticmethod
    def _key(ev: Evidence) -> str:
        body = " ".join(ev.text.split())[:300]
        return hashlib.sha1(f"{ev.source_type}|{ev.location}|{body}".encode()).hexdigest()

    def add(self, ev: Evidence) -> Evidence:
        """Register evidence; identical passages are de-duplicated and keep their original ID."""
        key = self._key(ev)
        if key in self._keys:
            existing = self._items[self._keys[key]]
            existing.score = max(existing.score, ev.score)
            return existing
        ev.id = f"E{len(self._items) + 1}"
        self._items[ev.id] = ev
        self._keys[key] = ev.id
        return ev

    def get(self, ev_id: str) -> Evidence | None:
        return self._items.get(ev_id)

    def many(self, ids: list[str]) -> list[Evidence]:
        return [self._items[i] for i in ids if i in self._items]

    def all(self) -> list[Evidence]:
        return list(self._items.values())

    def __len__(self) -> int:
        return len(self._items)
