"""Central configuration. Everything is overridable via environment variables / .env."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()
# CrewAI telemetry off by default (keeps console clean, no outbound calls).
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

log = logging.getLogger(__name__)

# Groq shut these down on 16 Aug 2026; map old IDs (e.g. from a stale .env) to the official replacements.
RETIRED_MODELS = {
    "groq/llama-3.3-70b-versatile": "groq/openai/gpt-oss-120b",
    "groq/llama-3.1-8b-instant": "groq/openai/gpt-oss-20b",
    "groq/qwen/qwen3-32b": "groq/openai/gpt-oss-120b",
    "groq/meta-llama/llama-4-scout-17b-16e-instruct": "groq/openai/gpt-oss-120b",
}


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).lower() in {"1", "true", "yes", "on"}


def _model(name: str, default: str) -> str:
    value = _env(name, default)
    if not value.startswith("groq/") and "/" in value and value.split("/")[0] in {"openai", "llama", "qwen"}:
        value = "groq/" + value          # allow bare Groq IDs like "openai/gpt-oss-120b"
    if value in RETIRED_MODELS:
        log.warning("Model %s is retired on Groq; using %s instead", value, RETIRED_MODELS[value])
        value = RETIRED_MODELS[value]
    return value


class ConfigError(RuntimeError):
    """Raised when required configuration (e.g. an API key) is missing."""


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = ""
    serper_api_key: str = ""
    model_smart: str = "groq/openai/gpt-oss-120b"
    model_fast: str = "groq/openai/gpt-oss-20b"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chroma_dir: str = "./data/chroma"
    chats_dir: str = "./data/chats"
    k_web: int = 4
    k_pdf: int = 4
    pdf_min_score: float = 0.30
    fetch_pages: bool = True
    max_iterations: int = 2
    max_subquestions: int = 4
    sufficiency_score: float = 8.0
    max_evidence_per_sub: int = 8
    max_gaps: int = 3
    chunk_size: int = 900
    chunk_overlap: int = 150

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            groq_api_key=_env("GROQ_API_KEY", ""),
            serper_api_key=_env("SERPER_API_KEY", ""),
            model_smart=_model("MODEL_SMART", cls.model_smart),
            model_fast=_model("MODEL_FAST", cls.model_fast),
            embedding_model=_env("EMBEDDING_MODEL", cls.embedding_model),
            chroma_dir=_env("CHROMA_DIR", cls.chroma_dir),
            chats_dir=_env("CHATS_DIR", cls.chats_dir),
            k_web=int(_env("K_WEB", str(cls.k_web))),
            k_pdf=int(_env("K_PDF", str(cls.k_pdf))),
            pdf_min_score=float(_env("PDF_MIN_SCORE", str(cls.pdf_min_score))),
            fetch_pages=_bool("FETCH_PAGES", cls.fetch_pages),
            max_iterations=int(_env("MAX_ITERATIONS", str(cls.max_iterations))),
            max_subquestions=int(_env("MAX_SUBQUESTIONS", str(cls.max_subquestions))),
            sufficiency_score=float(_env("SUFFICIENCY_SCORE", str(cls.sufficiency_score))),
        )