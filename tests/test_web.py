from unittest.mock import MagicMock, patch

from researchcrew.retrieval.web_search import SerperClient


def test_serper_parses_results_without_network():
    payload = {"organic": [{"title": "A", "link": "https://a.com", "snippet": "alpha snippet"},
                           {"title": "B", "link": "https://b.com", "snippet": ""},
                           {"title": "C", "link": "https://c.com", "snippet": "gamma snippet"}]}
    resp = MagicMock(); resp.json.return_value = payload; resp.raise_for_status = lambda: None
    with patch("researchcrew.retrieval.web_search.requests.post", return_value=resp):
        out = SerperClient("key", fetch_pages=False).search("q", k=3)
    assert [e.location for e in out] == ["https://a.com", "https://c.com"]   # empty snippet skipped
    assert out[0].score > out[1].score and out[0].source_type == "web"
