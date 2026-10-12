from researchcrew.citations import finalize, sanitize
from researchcrew.evidence import EvidenceStore
from researchcrew.models import Evidence


def _store():
    s = EvidenceStore()
    s.add(Evidence("web", "Alpha", "https://a.com", "alpha text"))
    s.add(Evidence("pdf", "doc.pdf", "doc.pdf, p.3", "pdf text"))
    s.add(Evidence("web", "Gamma", "https://g.com", "gamma text"))
    return s


def test_sanitize_removes_unknown_ids():
    text, removed = sanitize("Fact [E1] and [E9] and [E2, E8].", {"E1", "E2"})
    assert removed == 2
    assert "[E9]" not in text and "E8" not in text
    assert "[E1]" in text and "[E2]" in text


def test_finalize_renumbers_by_first_appearance_and_builds_sources():
    s = _store()
    out = finalize("# T\nB first [E3]. Then A [E1, E2]. Again [E3].\n\n## Sources\n1. made up", s)
    assert "[1]" in out.text and "[2, 3]" in out.text
    assert "made up" not in out.text
    assert [r["evidence_id"] for r in out.references] == ["E3", "E1", "E2"]
    assert out.stats["unique_sources"] == 3 and out.stats["pdf_sources"] == 1
    assert "## Sources" in out.text and "https://a.com" in out.text and "doc.pdf, p.3" in out.text


def test_finalize_drops_hallucinated_ids_and_counts_them():
    out = finalize("Claim [E1] and bogus [E42].", _store())
    assert out.stats["invalid_citations_removed"] == 1
    assert "E42" not in out.text and out.stats["citations"] == 1


def test_evidence_store_dedupes():
    s = EvidenceStore()
    a = s.add(Evidence("web", "T", "u", "same text", 0.2))
    b = s.add(Evidence("web", "T", "u", "same text", 0.9))
    assert a.id == b.id and len(s) == 1 and a.score == 0.9
