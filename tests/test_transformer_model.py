"""Unit tests for the compact Transformer sequence model.

These tests intentionally do not skip when PyTorch is missing. A skip would make
`make smoke` look green while the step-5 Transformer path was never verified.
Install CPU PyTorch with `make setup-cpu` before running the strict test suite.
"""
from __future__ import annotations

import pytest

from src.data import Vocabulary
from src.data.infineon_loader import SPECIAL_TOKENS
from src.ml.transformer_model import require_torch

try:
    import torch  # noqa: F401
except Exception:  # pragma: no cover - environment-specific
    torch = None  # type: ignore[assignment]


def _require_torch_for_test() -> None:
    if torch is None:
        pytest.fail(
            "PyTorch is required for Transformer tests. Run `make setup-cpu` "
            "or `python3 -m pip install --upgrade --index-url "
            "https://download.pytorch.org/whl/cpu torch`.",
            pytrace=False,
        )


@pytest.fixture()
def toy_sequences():
    return {
        "mosfet": {
            "s1": ["A", "B", "SHIP LOT"],
            "s2": ["A", "C", "SHIP LOT"],
        },
        "igbt": {"s3": ["X", "Y", "SHIP LOT"]},
        "ic": {},
    }


@pytest.mark.unit
def test_transformer_dataset_and_collate_shapes(toy_sequences):
    _require_torch_for_test()
    from src.ml.transformer_model import SequenceTrainingDataset, collate_lm_batch

    vocab = Vocabulary.from_sequences(toy_sequences)
    dataset = SequenceTrainingDataset(toy_sequences, vocab, max_len=16)
    batch = collate_lm_batch([dataset[0], dataset[1]], pad_id=vocab.pad_id)

    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["input_ids"].shape[0] == 2


@pytest.mark.unit
def test_transformer_forward_and_predict_api(toy_sequences):
    _require_torch_for_test()
    from src.ml.transformer_model import (
        ProcessTransformerNet,
        TransformerConfig,
        TransformerProcessModel,
    )

    vocab = Vocabulary.from_sequences(toy_sequences)
    config = TransformerConfig(d_model=16, n_layers=1, n_heads=2, dropout=0.0,
                               max_len=16, dim_feedforward=32)
    net = ProcessTransformerNet(len(vocab), config, vocab.pad_id)
    wrapper = TransformerProcessModel(net, vocab, config, device="cpu")

    preds = wrapper.predict_topk("mosfet", ["A"], k=3)
    assert len(preds) > 0
    assert all(token not in SPECIAL_TOKENS for token in preds)

    completed = wrapper.complete("mosfet", ["A"], max_steps=2)
    assert completed[:1] == ["A"]

    lp = wrapper.sequence_log_prob("mosfet", ["A", "B", "SHIP LOT"])
    assert lp <= 0.0


@pytest.mark.unit
def test_transformer_save_load_roundtrip(tmp_path, toy_sequences):
    _require_torch_for_test()
    from src.ml.transformer_model import (
        ProcessTransformerNet,
        TransformerConfig,
        TransformerProcessModel,
    )

    vocab = Vocabulary.from_sequences(toy_sequences)
    config = TransformerConfig(d_model=16, n_layers=1, n_heads=2, dropout=0.0,
                               max_len=16, dim_feedforward=32)
    net = ProcessTransformerNet(len(vocab), config, vocab.pad_id)
    model = TransformerProcessModel(net, vocab, config, device="cpu",
                                    metadata={"test": True})
    path = tmp_path / "toy_transformer.pt"
    model.save(path)

    loaded = TransformerProcessModel.load(path, device="cpu")
    assert loaded.config.d_model == 16
    assert loaded.vocab.id_to_token == model.vocab.id_to_token
    assert loaded.metadata["test"] is True


@pytest.mark.unit
def test_require_torch_does_not_raise_when_installed():
    _require_torch_for_test()
    require_torch()
