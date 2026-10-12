"""Live web search via the Serper.dev API, enriched with page-text excerpts."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import requests

from ..models import Evidence

log = logging.getLogger(__name__)
SERPER_URL = "https://google.serper.dev/search"
_UA = {"User-Agent": "Mozilla/5.0 (compatible; ResearchCrew/1.0; +research-bot)"}


def fetch_excerpt(url: str, max_chars: int = 1500, timeout: float = 8.0) -> str:
    """Best-effort readable text from a web page. Never raises."""
    try:
        if url.lower().endswith(".pdf"):
            return ""
        r = requests.get(url, headers=_UA, timeout=timeout)
        if r.status_code != 200 or "html" not in r.headers.get("content-type", "").lower():
            return ""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
            tag.decompose()
        paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join(p for p in paras if len(p) > 60)
        return text[:max_chars]
    except Exception as exc:  # noqa: BLE001 - scraping is inherently flaky
        log.debug("excerpt fetch failed for %s: %s", url, exc)
        return ""


class SerperClient:
    def __init__(self, api_key: str, fetch_pages: bool = True, fetch_top_n: int = 2,
                 timeout: float = 15.0) -> None:
        if not api_key:
            raise ValueError("SERPER_API_KEY is required for web search")
        self.api_key = api_key
        self.fetch_pages = fetch_pages
        self.fetch_top_n = fetch_top_n
        self.timeout = timeout

    def search(self, query: str, k: int = 4) -> list[Evidence]:
        resp = requests.post(SERPER_URL, timeout=self.timeout,
                             headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
                             json={"q": query, "num": max(k, 3)})
        resp.raise_for_status()
        organic = resp.json().get("organic", [])[:k]

        excerpts: dict[int, str] = {}
        if self.fetch_pages and organic:
            targets = list(enumerate(organic))[: self.fetch_top_n]
            with ThreadPoolExecutor(max_workers=len(targets)) as pool:
                for (i, _), text in zip(targets, pool.map(lambda t: fetch_excerpt(t[1]["link"]), targets)):
                    excerpts[i] = text

        out: list[Evidence] = []
        for rank, item in enumerate(organic):
            body = item.get("snippet", "")
            if excerpts.get(rank):
                body = f"{body}\n\nPage excerpt: {excerpts[rank]}"
            if not body.strip():
                continue
            out.append(Evidence(source_type="web", title=item.get("title", item["link"]),
                                location=item["link"], text=body,
                                score=round(1.0 - 0.1 * rank, 3)))
        return out
