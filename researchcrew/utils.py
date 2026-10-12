"""Small, dependency-free helpers."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of an LLM response (tolerates fences / chatter)."""
    candidates = [text] + _FENCE.findall(text)
    decoder = json.JSONDecoder()
    for chunk in candidates:
        for match in re.finditer(r"\{", chunk):
            try:
                obj, _ = decoder.raw_decode(chunk[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    raise ValueError("No JSON object found in model output")


def clip(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"
