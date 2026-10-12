"""ChromaDB persistent vector store. Embeddings are computed by us (BGE) and passed in explicitly,
which keeps behaviour identical across ChromaDB versions.

One *collection per chat* gives hard isolation of uploaded PDFs between chats."""
from __future__ import annotations

from .embeddings import Embedder
from .pdf_loader import Chunk
from ..models import Evidence


def _order_key(chunk_id: str, meta: dict) -> tuple:
    try:
        j = int(chunk_id.rsplit(":", 1)[1])
    except (IndexError, ValueError):
        j = 0
    return (meta.get("source", ""), int(meta.get("page", 0)), j)


class VectorStore:
    def __init__(self, embedder: Embedder, persist_dir: str = "./data/chroma",
                 collection: str = "researchcrew_docs") -> None:
        import chromadb

        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = collection
        self.col = self.client.get_or_create_collection(name=collection)

    def add_chunks(self, chunks: list[Chunk], batch: int = 128) -> int:
        for i in range(0, len(chunks), batch):
            part = chunks[i:i + batch]
            self.col.upsert(
                ids=[c.id for c in part],
                documents=[c.text for c in part],
                embeddings=self.embedder.embed_documents([c.text for c in part]),
                metadatas=[{"source": c.source, "page": c.page, "doc_id": c.doc_id} for c in part],
            )
        return len(chunks)

    def search(self, query: str, k: int = 4, min_score: float = 0.0) -> list[Evidence]:
        n = self.col.count()
        if n == 0 or not query.strip():
            return []
        res = self.col.query(query_embeddings=[self.embedder.embed_query(query)],
                             n_results=min(k, n), include=["documents", "metadatas", "distances"])
        out: list[Evidence] = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            score = 1.0 - dist / 2.0   # Chroma returns squared L2; vectors are unit-norm => cosine
            if score < min_score:
                continue
            out.append(Evidence(source_type="pdf", title=meta["source"],
                                location=f"{meta['source']}, p.{meta['page']}",
                                text=doc, score=round(score, 4)))
        return out

    def overview(self, n: int = 10) -> list[Evidence]:
        """Evenly sampled chunks across every document, in reading order.
        Used for 'summarise / what is this PDF about' questions, where similarity search alone
        would only return a few look-alike passages."""
        data = self.col.get(include=["documents", "metadatas"])
        ids, docs, metas = data["ids"], data["documents"], data["metadatas"]
        if not ids:
            return []
        by_source: dict[str, list[tuple]] = {}
        for cid, doc, meta in zip(ids, docs, metas):
            by_source.setdefault(meta["source"], []).append((_order_key(cid, meta), doc, meta))
        budget = max(3, n // len(by_source))
        out: list[Evidence] = []
        for items in by_source.values():
            items.sort(key=lambda t: t[0])
            take = min(budget, len(items))
            for i in range(take):
                _, doc, meta = items[int(i * len(items) / take)]
                out.append(Evidence(source_type="pdf", title=meta["source"],
                                    location=f"{meta['source']}, p.{meta['page']}",
                                    text=doc, score=1.0))
        return out

    def count(self) -> int:
        return self.col.count()

    def sources(self) -> dict[str, int]:
        data = self.col.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for m in data["metadatas"] or []:
            counts[m["source"]] = counts.get(m["source"], 0) + 1
        return counts

    def clear(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.col = self.client.get_or_create_collection(name=self.collection_name)

    def drop(self) -> None:
        """Permanently delete this collection (used when a chat is deleted)."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:  # noqa: BLE001
            pass