"""Command-line interface.

    python cli.py "How do RAG and fine-tuning compare for domain QA?" --pdf paper.pdf --out report.md
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from researchcrew.agents import CrewAgentRunner
from researchcrew.config import ConfigError, Settings
from researchcrew.pipeline import ResearchPipeline
from researchcrew.retrieval import (BGEEmbedder, Retriever, SerperClient, VectorStore,
                                    chunk_pages, load_pdf)
from researchcrew.utils import setup_logging


def main() -> int:
    ap = argparse.ArgumentParser(description="ResearchCrew - multi-agent research with a critic loop")
    ap.add_argument("question")
    ap.add_argument("--pdf", nargs="*", default=[], help="PDF files to index for RAG")
    ap.add_argument("--max-iters", type=int, help="max critic-driven re-research rounds")
    ap.add_argument("--no-web", action="store_true", help="disable live web search")
    ap.add_argument("--out", default="report.md", help="where to write the Markdown report")
    ap.add_argument("--json", help="optionally dump the full trace (plan, critic loop, evidence)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    import logging
    setup_logging(logging.INFO if args.verbose else logging.WARNING)
    s = Settings.from_env()
    if args.max_iters is not None:
        s = replace(s, max_iterations=args.max_iters)

    store = None
    if args.pdf or Path(s.chroma_dir).exists():
        store = VectorStore(BGEEmbedder(s.embedding_model), s.chroma_dir)
        for p in args.pdf:
            doc_id, name, pages = load_pdf(p)
            n = store.add_chunks(chunk_pages(doc_id, name, pages, s.chunk_size, s.chunk_overlap))
            print(f"indexed {name}: {n} chunks", file=sys.stderr)

    web = None if args.no_web or not s.serper_api_key else SerperClient(s.serper_api_key, s.fetch_pages)
    retriever = Retriever(store, web, s.k_web, s.k_pdf, s.pdf_min_score)
    if not retriever.has_sources:
        print("No sources available: set SERPER_API_KEY and/or pass --pdf.", file=sys.stderr)
        return 2

    try:
        runner = CrewAgentRunner(s)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    pipe = ResearchPipeline(runner, retriever, s, on_event=lambda st, m: print(f"[{st:>8}] {m}", file=sys.stderr))
    result = pipe.run(args.question)
    Path(args.out).write_text(result.report, encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(f"\nReport written to {args.out} ({result.elapsed_s}s) - stats: {result.stats}", file=sys.stderr)
    for w in retriever.warnings:
        print(f"warning: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
