"""Small decoder-only Transformer for Infineon process sequences.

The model is deliberately lightweight: the process vocabulary is only about
150 tokens, so a family-conditioned causal language model is enough for the
hackathon tasks without pulling in a large LLM dependency.

Input convention:
    <bos>, <fam:mosfet|igbt|ic>, STEP_1, STEP_2, ..., STEP_N

Training objective:
    next-token prediction over the same token stream, with <eos> appended to
    completed training sequences. At inference time the public API mirrors the
    n-gram baseline: `predict_topk()` for Task 1 and `complete()` for Task 2.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from src.data.infineon_loader import (
    BOS_TOKEN,
    EOS_TOKEN,
    FAMILY_TOKENS,
    PAD_TOKEN,
    SPECIAL_TOKENS,
    Family,
    Sequences,
    Vocabulary,
)

try:  # Keep the repo importable in CPU-only/minimal environments.
    import torch
    from torch import Tensor, nn
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - exercised only when torch is absent.
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = Any  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment]


@dataclass(frozen=True)
class TransformerConfig:
    """Hyperparameters for the compact process-sequence Transformer."""

    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1
    max_len: int = 180
    dim_feedforward: int | None = None

    def resolved_dim_feedforward(self) -> int:
        return self.dim_feedforward or 4 * self.d_model

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TransformerConfig":
        known = {field: payload[field] for field in cls.__dataclass_fields__  # type: ignore[attr-defined]
                 if field in payload}
        return cls(**known)


def require_torch() -> None:
    """Raise a clear, actionable error when PyTorch is not installed."""
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for the compact Transformer model, even for CPU-only "
            "local smoke runs. Fix locally with `make setup-cpu` or manually with "
            "`python3 -m pip install --upgrade --index-url "
            "https://download.pytorch.org/whl/cpu torch`. On Leonardo/GPU, install "
            "the CUDA-enabled PyTorch build from the HPC environment docs."
        )


if torch is not None:

    class ProcessTransformerNet(nn.Module):
        """Causal decoder-style Transformer implemented with TransformerEncoder.

        `nn.TransformerEncoder` plus an upper-triangular causal mask is a compact
        way to implement decoder-only language modelling without cross-attention.
        """

        def __init__(self, vocab_size: int, config: TransformerConfig, pad_id: int) -> None:
            super().__init__()
            if config.d_model % config.n_heads != 0:
                raise ValueError("d_model must be divisible by n_heads")
            self.config = config
            self.pad_id = pad_id
            self.token_embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=pad_id)
            self.position_embedding = nn.Embedding(config.max_len, config.d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.resolved_dim_feedforward(),
                dropout=config.dropout,
                batch_first=True,
                norm_first=False,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=config.n_layers)
            self.final_norm = nn.LayerNorm(config.d_model)
            self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)
            self.lm_head.weight = self.token_embedding.weight

        def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
            batch_size, seq_len = input_ids.shape
            if seq_len > self.config.max_len:
                raise ValueError(
                    f"input length {seq_len} exceeds max_len={self.config.max_len}"
                )
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            x = self.token_embedding(input_ids) + self.position_embedding(positions)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device),
                diagonal=1,
            )
            padding_mask = None
            if attention_mask is not None:
                padding_mask = attention_mask == 0
            x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
            x = self.final_norm(x)
            return self.lm_head(x)

else:  # pragma: no cover - import guard only.

    class ProcessTransformerNet:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


class SequenceTrainingDataset(Dataset):
    """Tokenized next-token examples built from family-indexed sequences."""

    def __init__(
        self,
        sequences_per_family: dict[Family, Sequences],
        vocab: Vocabulary,
        *,
        max_len: int = 180,
    ) -> None:
        require_torch()
        self.vocab = vocab
        self.max_len = max_len
        self.examples: list[list[int]] = []
        for family, sequences in sequences_per_family.items():
            for steps in sequences.values():
                self.examples.extend(_encode_sequence_examples(family, steps, vocab, max_len))
        if not self.examples:
            raise ValueError("No training examples were produced")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> list[int]:
        return self.examples[index]


def _encode_sequence_examples(
    family: Family,
    steps: list[str],
    vocab: Vocabulary,
    max_len: int,
) -> list[list[int]]:
    """Encode one full route into one or more LM examples.

    For the current official data, `max_len=180` is enough for all families.
    The chunking branch is still included so the same model can train on longer
    generated data later without failing.
    """
    if max_len < 8:
        raise ValueError("max_len must be at least 8")
    token_budget = max_len + 1  # ids are later shifted into x/y of length max_len.
    full_tokens = [BOS_TOKEN, FAMILY_TOKENS[family], *steps, EOS_TOKEN]
    if len(full_tokens) <= token_budget:
        return [vocab.encode(full_tokens)]

    examples: list[list[int]] = []
    step_capacity = max_len - 2
    for start in range(0, len(steps), step_capacity):
        chunk = steps[start:start + step_capacity]
        tokens = [BOS_TOKEN, FAMILY_TOKENS[family], *chunk]
        if start + len(chunk) >= len(steps):
            tokens.append(EOS_TOKEN)
        if len(tokens) >= 3:
            examples.append(vocab.encode(tokens[:token_budget]))
    return examples


def collate_lm_batch(examples: list[list[int]], pad_id: int) -> dict[str, Tensor]:
    """Pad variable-length token streams for next-token prediction."""
    require_torch()
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    max_len = max(len(ids) - 1 for ids in examples)
    input_ids = torch.full((len(examples), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long)
    for row, ids in enumerate(examples):
        x = ids[:-1]
        y = ids[1:]
        input_ids[row, :len(x)] = torch.tensor(x, dtype=torch.long)
        labels[row, :len(y)] = torch.tensor(y, dtype=torch.long)
        attention_mask[row, :len(x)] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class TransformerProcessModel:
    """Inference wrapper exposing the same API as `NGramBaseline`."""

    def __init__(
        self,
        net: ProcessTransformerNet,
        vocab: Vocabulary,
        config: TransformerConfig,
        *,
        device: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        require_torch()
        self.vocab = vocab
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = net.to(self.device)
        self.net.eval()
        self.metadata = metadata or {}
        self._blocked_nextstep_ids = self._special_ids(include_eos=True)
        self._blocked_completion_ids = self._special_ids(include_eos=False)

    def _special_ids(self, *, include_eos: bool) -> set[int]:
        blocked = set()
        for token in SPECIAL_TOKENS:
            if token == EOS_TOKEN and not include_eos:
                continue
            blocked.add(self.vocab.token_to_id[token])
        return blocked

    def _prefix_ids(self, family: Family, prefix: list[str]) -> list[int]:
        context_budget = self.config.max_len - 2
        trimmed_prefix = prefix[-context_budget:] if context_budget > 0 else []
        tokens = [BOS_TOKEN, FAMILY_TOKENS[family], *trimmed_prefix]
        return self.vocab.encode(tokens)

    @torch.no_grad() if torch is not None else (lambda fn: fn)
    def _next_logits(self, family: Family, prefix: list[str]) -> Tensor:
        ids = self._prefix_ids(family, prefix)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        logits = self.net(input_ids, attention_mask=attention_mask)
        return logits[0, -1].detach().clone()

    def _ranked_token_ids(
        self,
        family: Family,
        prefix: list[str],
        *,
        k: int,
        allow_eos: bool,
    ) -> list[int]:
        logits = self._next_logits(family, prefix)
        blocked = self._blocked_completion_ids if allow_eos else self._blocked_nextstep_ids
        for token_id in blocked:
            logits[token_id] = -torch.inf
        ranked = torch.argsort(logits, descending=True).tolist()
        out: list[int] = []
        for token_id in ranked:
            token = self.vocab.id_to_token[token_id]
            if token == PAD_TOKEN or token == BOS_TOKEN:
                continue
            out.append(token_id)
            if len(out) >= k:
                break
        return out

    def predict_topk(self, family: Family, prefix: list[str], k: int = 5) -> list[str]:
        """Return top-k process-step predictions for Task 1."""
        token_ids = self._ranked_token_ids(family, prefix, k=k, allow_eos=False)
        return self.vocab.decode(token_ids)

    def complete(
        self,
        family: Family,
        prefix: list[str],
        *,
        max_steps: int = 200,
        stop_token: str = "SHIP LOT",
    ) -> list[str]:
        """Greedy completion for Task 2."""
        out = list(prefix)
        for _ in range(max_steps):
            token_id = self._ranked_token_ids(family, out, k=1, allow_eos=True)[0]
            token = self.vocab.id_to_token[token_id]
            if token == EOS_TOKEN:
                break
            out.append(token)
            if token == stop_token:
                break
        return out

    def save(self, path: Path, *, metadata: dict[str, Any] | None = None) -> None:
        """Persist model weights plus vocab/config metadata."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": "transformer_decoder_only",
            "config": asdict(self.config),
            "vocab_tokens": list(self.vocab.id_to_token),
            "state_dict": self.net.cpu().state_dict(),
            "metadata": metadata if metadata is not None else self.metadata,
        }
        torch.save(payload, path)
        self.net.to(self.device)

    @classmethod
    def load(cls, path: Path, *, device: str | None = None) -> "TransformerProcessModel":
        """Load a saved TransformerProcessModel from `train_transformer.py`."""
        require_torch()
        payload = torch.load(path, map_location=device or "cpu")
        config = TransformerConfig.from_dict(payload["config"])
        tokens = tuple(payload["vocab_tokens"])
        vocab = Vocabulary(id_to_token=tokens, token_to_id={t: i for i, t in enumerate(tokens)})
        net = ProcessTransformerNet(len(vocab), config, vocab.pad_id)
        net.load_state_dict(payload["state_dict"])
        return cls(net, vocab, config, device=device, metadata=payload.get("metadata") or {})

    def stats(self) -> dict[str, Any]:
        return {
            "model": "transformer_decoder_only",
            "config": asdict(self.config),
            "vocab_size": len(self.vocab),
            "metadata": self.metadata,
        }


def build_transformer_model(
    vocab: Vocabulary,
    config: TransformerConfig,
    *,
    device: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> TransformerProcessModel:
    """Create an untrained TransformerProcessModel wrapper."""
    require_torch()
    net = ProcessTransformerNet(len(vocab), config, vocab.pad_id)
    return TransformerProcessModel(net, vocab, config, device=device, metadata=metadata)


def iter_step_tokens(sequences_per_family: dict[Family, Sequences]) -> Iterable[str]:
    """Yield raw process-step strings from a family-indexed sequence mapping."""
    for sequences in sequences_per_family.values():
        for steps in sequences.values():
            yield from steps
