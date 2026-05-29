"""Tests for the Infineon loader and Vocabulary."""
from __future__ import annotations

import pytest

from src.data import FAMILIES, load_all_families, load_family
from src.data.infineon_loader import (
    BOS_TOKEN,
    EOS_TOKEN,
    FAMILY_TOKENS,
    PAD_TOKEN,
    SPECIAL_TOKENS,
    UNK_TOKEN,
    Vocabulary,
)


@pytest.mark.unit
def test_load_one_family_mosfet():
    seqs = load_family("mosfet")
    assert len(seqs) == 1000, f"expected 1000 MOSFET sequences, got {len(seqs)}"
    one = next(iter(seqs.values()))
    assert one[0] == "RECEIVE WAFER LOT"
    assert one[-1] == "SHIP LOT"
    assert 100 < len(one) < 200


@pytest.mark.unit
def test_load_all_families_shapes():
    all_seqs = load_all_families()
    assert set(all_seqs.keys()) == set(FAMILIES)
    for fam in FAMILIES:
        assert len(all_seqs[fam]) == 1000
        # Spot-check one sequence per family.
        steps = next(iter(all_seqs[fam].values()))
        assert steps[0] == "RECEIVE WAFER LOT"
        assert steps[-1] == "SHIP LOT"


@pytest.mark.unit
def test_vocabulary_specials_reserved():
    vocab = Vocabulary.from_step_tokens(["RECEIVE WAFER LOT", "SHIP LOT", "ETCH"])
    assert vocab.pad_id == 0
    assert vocab.bos_id == 1
    assert vocab.eos_id == 2
    assert vocab.unk_id == 3
    for token in SPECIAL_TOKENS:
        assert token in vocab.token_to_id
    # Step tokens are sorted, so 'ETCH' < 'RECEIVE WAFER LOT' < 'SHIP LOT'.
    step_ids = [vocab.token_to_id[t] for t in ("ETCH", "RECEIVE WAFER LOT", "SHIP LOT")]
    assert step_ids == sorted(step_ids)


@pytest.mark.unit
def test_vocabulary_encode_decode_round_trip():
    vocab = Vocabulary.from_step_tokens(["A", "B", "C"])
    ids = vocab.encode(["B", "A", "C", "MISSING"])
    assert ids[:3] == [vocab.token_to_id[t] for t in ("B", "A", "C")]
    assert ids[3] == vocab.unk_id
    decoded = vocab.decode(ids)
    assert decoded[:3] == ["B", "A", "C"]
    assert decoded[3] == UNK_TOKEN


@pytest.mark.unit
def test_vocabulary_from_sequences_covers_real_data():
    all_seqs = load_all_families()
    vocab = Vocabulary.from_sequences(all_seqs)
    # Every real step token should be present (no UNKs on training data).
    seen_unk = 0
    for fam_seqs in all_seqs.values():
        for steps in fam_seqs.values():
            for tok in steps:
                if vocab.token_to_id.get(tok, vocab.unk_id) == vocab.unk_id:
                    seen_unk += 1
    assert seen_unk == 0
    # Sanity: vocab size within the documented ~120 step + specials range.
    n_steps = len(vocab) - len(SPECIAL_TOKENS)
    assert 100 <= n_steps <= 200, f"unexpected step-token count: {n_steps}"


@pytest.mark.unit
def test_family_tokens_distinct():
    ids = {Vocabulary.from_step_tokens([]).token_to_id[FAMILY_TOKENS[f]] for f in FAMILIES}
    assert len(ids) == 3
