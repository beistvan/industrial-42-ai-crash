"""Tests for the NGramBaseline."""
from __future__ import annotations

import pytest

from src.ml import NGramBaseline


@pytest.mark.unit
def test_ngram_fits_and_predicts_on_toy_data():
    seqs = {
        "mosfet": {
            "s1": ["A", "B", "C", "D"],
            "s2": ["A", "B", "C", "E"],
            "s3": ["A", "B", "X"],
        },
        "igbt": {},
        "ic": {},
    }
    m = NGramBaseline(max_order=3).fit(seqs)
    # After 'A B', the top-1 should be 'C' (seen twice) and then 'X' (once).
    top = m.predict_topk("mosfet", ["A", "B"], k=3)
    assert top[0] == "C"
    assert "X" in top


@pytest.mark.unit
def test_ngram_backoff_when_suffix_unknown():
    seqs = {"mosfet": {"s1": ["A", "B", "C", "D", "E"]}, "igbt": {}, "ic": {}}
    m = NGramBaseline(max_order=4).fit(seqs)
    # Unseen suffix should still produce predictions via backoff.
    top = m.predict_topk("mosfet", ["NEVER_SEEN_TOKEN"], k=3)
    assert isinstance(top, list)
    assert len(top) > 0


@pytest.mark.unit
def test_ngram_complete_stops_on_stop_token():
    seqs = {"mosfet": {"s1": ["A", "B", "SHIP LOT"]}, "igbt": {}, "ic": {}}
    m = NGramBaseline(max_order=2).fit(seqs)
    out = m.complete("mosfet", ["A"], max_steps=10, stop_token="SHIP LOT")
    assert out[-1] == "SHIP LOT"
    assert len(out) <= 11


@pytest.mark.unit
def test_ngram_save_load_roundtrip(tmp_path):
    seqs = {"mosfet": {"s1": ["A", "B", "C", "B", "C"]}, "igbt": {}, "ic": {}}
    m = NGramBaseline(max_order=3).fit(seqs)
    p = tmp_path / "ngram.pkl"
    m.save(p)
    loaded = NGramBaseline.load(p)
    assert loaded.max_order == m.max_order
    assert loaded.predict_topk("mosfet", ["A", "B"], k=2) == \
           m.predict_topk("mosfet", ["A", "B"], k=2)


@pytest.mark.unit
def test_ngram_stats_shape():
    seqs = {"mosfet": {"s1": ["A", "B"]}, "igbt": {}, "ic": {}}
    m = NGramBaseline(max_order=4).fit(seqs)
    stats = m.stats()
    assert stats["model"] == "ngram_suffix_backoff"
    assert stats["max_order"] == 4
    assert "mosfet" in stats["indexed_suffixes_per_family"]
