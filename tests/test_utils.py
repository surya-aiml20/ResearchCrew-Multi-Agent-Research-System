import pytest

from researchcrew.utils import extract_json


def test_extract_json_plain_and_fenced_and_chatty():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure!\n```json\n{"a": {"b": [1, 2]}}\n```\nDone.') == {"a": {"b": [1, 2]}}
    assert extract_json('blah { not json } then {"ok": true} tail') == {"ok": True}


def test_extract_json_raises_when_absent():
    with pytest.raises(ValueError):
        extract_json("no json here")
