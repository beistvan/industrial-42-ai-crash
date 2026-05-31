"""Tests for Wave 5 parameter enrichment."""
from __future__ import annotations

from src.data.param_enrichment import (
    enrich_step_sequence,
    is_param_token,
    param_token_for,
    prefix_end_for_steps,
)
from src.data.step_metadata import describe_step


def test_step_metadata_loads_parameters():
    meta = describe_step("mosfet", "RECEIVE WAFER LOT")
    assert meta["parameters"], "expected fab parameters for canonical MOSFET step"


def test_enrich_inserts_param_tokens_after_steps():
    steps = ["RECEIVE WAFER LOT", "LOT IDENTIFICATION"]
    enriched = enrich_step_sequence("mosfet", steps, mode="after_step")
    assert enriched[0] == steps[0]
    assert is_param_token(enriched[1])
    assert enriched[2] == steps[1]


def test_param_token_is_deterministic():
    a = param_token_for("mosfet", "GATE OXIDE GROWTH")
    b = param_token_for("mosfet", "GATE OXIDE GROWTH")
    assert a == b
    assert a.startswith("<p:")


def test_prefix_end_includes_param_after_last_kept_step():
    stream = ["S0", param_token_for("ic", "S0"), "S1", param_token_for("ic", "S1")]
    assert prefix_end_for_steps(stream, 1) == 2
