import fitz

from researchcrew.retrieval import HashEmbedder, VectorStore, chunk_pages, load_pdf


def make_pdf(pages: list[str]) -> bytes:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 550, 780), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


PAGES = [
    "Retrieval augmented generation combines a retriever with a language model. "
    "The retriever fetches relevant passages from a vector database such as ChromaDB. " * 4,
    "Fine tuning updates model weights on domain data. It is expensive but can change style and behaviour. " * 4,
]


def test_pdf_loading_and_chunking_preserves_pages():
    doc_id, name, pages = load_pdf(make_pdf(PAGES), "demo.pdf")
    assert name == "demo.pdf" and [p for p, _ in pages] == [1, 2]
    chunks = chunk_pages(doc_id, name, pages, chunk_size=200, overlap=40)
    assert len(chunks) > 2
    assert {c.page for c in chunks} == {1, 2}
    assert all(len(c.text) <= 200 for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_chunker_terminates_on_unbroken_text():
    chunks = chunk_pages("d", "f.pdf", [(1, "x" * 5000)], chunk_size=300, overlap=50)
    assert len(chunks) >= 16


def test_vector_store_retrieves_relevant_page(tmp_path):
    doc_id, name, pages = load_pdf(make_pdf(PAGES), "demo.pdf")
    store = VectorStore(HashEmbedder(), str(tmp_path / "chroma"), collection="test_col")
    store.add_chunks(chunk_pages(doc_id, name, pages, 300, 50))
    assert store.count() > 0 and store.sources()["demo.pdf"] == store.count()
    hits = store.search("fine tuning model weights domain data", k=2)
    assert hits and hits[0].source_type == "pdf" and hits[0].location.endswith("p.2")
    # re-indexing the same file is idempotent (upsert on stable IDs)
    n = store.count()
    store.add_chunks(chunk_pages(doc_id, name, pages, 300, 50))
    assert store.count() == n
    store.clear()
    assert store.count() == 0 and store.search("anything") == []
