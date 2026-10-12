"""Multi-chat persistence. One JSON file per chat in CHATS_DIR.

Each chat also owns a private ChromaDB collection named ``chat_<id>`` (see app.py), so PDFs
uploaded in one chat are never visible in another.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path


class ChatStore:
    def __init__(self, directory: str) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, chat_id: str) -> Path:
        return self.dir / f"{chat_id}.json"

    def create(self) -> dict:
        now = time.time()
        chat = {"id": uuid.uuid4().hex[:10], "title": "New chat", "created": now,
                "updated": now, "messages": []}
        self.save(chat)
        return chat

    def load(self, chat_id: str) -> dict | None:
        try:
            data = json.loads(self._path(chat_id).read_text(encoding="utf-8"))
            return data if isinstance(data, dict) and "messages" in data else None
        except (OSError, ValueError):
            return None

    def save(self, chat: dict) -> None:
        if chat.get("title") == "New chat":
            first = next((m["content"] for m in chat["messages"] if m["role"] == "user"), "")
            if first:
                chat["title"] = " ".join(first.split())[:48]
        chat["updated"] = time.time()
        tmp = self._path(chat["id"]).with_suffix(".tmp")
        tmp.write_text(json.dumps(chat, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._path(chat["id"]))

    def list(self) -> list[dict]:
        out = []
        for f in self.dir.glob("*.json"):
            c = self.load(f.stem)
            if c:
                out.append({"id": c["id"], "title": c.get("title", "Chat"),
                            "updated": c.get("updated", 0), "n": len(c["messages"])})
        return sorted(out, key=lambda c: c["updated"], reverse=True)

    def delete(self, chat_id: str) -> None:
        try:
            self._path(chat_id).unlink()
        except OSError:
            pass