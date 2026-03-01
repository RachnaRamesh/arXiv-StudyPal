import json
from rag.ingest import filter_records, record_to_doc, AI_CATEGORIES


def test_filter_records_positive():
    rec = {"categories": "cs.AI cs.LG"}
    assert filter_records(rec)


def test_filter_records_negative():
    rec = {"categories": "math.PR"}
    assert not filter_records(rec)


def test_record_to_doc():
    rec = {"title": "Test", "abstract": "Abstract", "authors": "A", "categories": "cs.AI", "update_date": "2020", "id": "1234"}
    doc = record_to_doc(rec)
    assert "Test" in doc["text"]
    assert doc["metadata"]["arxiv_id"] == "1234"
