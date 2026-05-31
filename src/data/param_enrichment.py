"""Optional fab-parameter enrichment for training sequences.

Reference CSVs map each canonical STEP to fab-level parameters. Wave 5 inserts
deterministic auxiliary tokens *after* each step during training only. Judge
eval and submission outputs remain STEP tokens only.

Training stream (example):
    <bos> <fam:mosfet> STEP_A <p:abc…> STEP_B <p:def…> … <eos>

Inference / dev eval:
    <bos> <fam:mosfet> STEP_A STEP_B …   (no param tokens — matches eval inputs)
"""
from __future__ import annotations

import hashlib
from typing import Literal

from src.data.infineon_loader import FAMILIES, Family, Sequences
from src.data.step_metadata import load_step_metadata

ParamEnrichment = Literal["none", "after_step"]

PARAM_TOKEN_PREFIX = "<p:"


def is_param_token(token: str) -> bool:
    return token.startswith(PARAM_TOKEN_PREFIX)


def param_token_for(family: Family, step: str) -> str:
    """Stable auxiliary token for one (family, step) pair."""
    digest = hashlib.sha1(f"{family}|{step}".encode("utf-8")).hexdigest()[:10]
    return f"{PARAM_TOKEN_PREFIX}{digest}>"


def enrich_step_sequence(
    family: Family,
    steps: list[str],
    *,
    mode: ParamEnrichment = "after_step",
) -> list[str]:
    """Return a token stream with optional param aux tokens after each step."""
    if mode == "none":
        return list(steps)
    meta = load_step_metadata()
    out: list[str] = []
    for step in steps:
        out.append(step)
        params = meta.get(family, {}).get(step, {}).get("parameters", "")
        if params:
            out.append(param_token_for(family, step))
    return out


def enrich_sequences(
    sequences: dict[Family, Sequences],
    *,
    mode: ParamEnrichment = "after_step",
) -> dict[Family, Sequences]:
    """Deep-copy family maps with enriched step streams."""
    if mode == "none":
        return {fam: dict(seqs) for fam, seqs in sequences.items()}
    enriched: dict[Family, Sequences] = {}
    for fam in FAMILIES:
        enriched[fam] = {
            sid: enrich_step_sequence(fam, steps, mode=mode)
            for sid, steps in sequences.get(fam, {}).items()
        }
    return enriched


def prefix_end_for_steps(token_stream: list[str], keep_steps: int) -> int:
    """Exclusive end index in `token_stream` after `keep_steps` process steps.

    Includes any param aux token immediately following the last kept step.
    """
    kept = 0
    for i, token in enumerate(token_stream):
        if is_param_token(token):
            continue
        kept += 1
        if kept == keep_steps:
            end = i + 1
            if end < len(token_stream) and is_param_token(token_stream[end]):
                end += 1
            return end
    return len(token_stream)


def collect_param_tokens(sequences: dict[Family, Sequences]) -> set[str]:
    """All param aux tokens appearing in enriched streams."""
    found: set[str] = set()
    for seqs in sequences.values():
        for stream in seqs.values():
            found.update(t for t in stream if is_param_token(t))
    return found
