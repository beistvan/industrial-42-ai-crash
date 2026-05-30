#!/usr/bin/env python3
"""Train the compact decoder-only Transformer for the Infineon sequence tasks.

The model uses the same public inference API as the n-gram baseline:
`predict_topk(family, prefix, k)` and `complete(family, prefix)`. This means the
existing local evaluator, submission writer, and Streamlit app can swap between
baseline and Transformer checkpoints without changing task logic.

Quick smoke run:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/train_transformer.py \
        --limit-train-sequences 2 --epochs 1 --skip-eval \
        --d-model 16 --n-layers 1 --n-heads 2 --dim-feedforward 32 --batch-size 2

Small production run:
    python scripts/train_transformer.py --config configs/transformer_small.yaml

Medium Leonardo/GPU run:
    python scripts/train_transformer.py --config configs/transformer_medium.yaml --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import (  # noqa: E402
    FAMILIES,
    Vocabulary,
    load_all_families,
    load_extra_families,
    merge_sequence_maps,
)
from src.eval.run_eval import evaluate_all  # noqa: E402
from src.ml.transformer_model import (  # noqa: E402
    ProcessTransformerNet,
    SequenceTrainingDataset,
    TransformerConfig,
    TransformerProcessModel,
    collate_lm_batch,
    require_torch,
)

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
except Exception:  # pragma: no cover - handled in main via require_torch().
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]


def _read_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return payload


def _load_split(
    split_dir: Path,
    *,
    limit_per_family: int | None = None,
    limit_dev_per_family: int | None = None,
    extra_data_dir: Path | None = None,
    limit_extra_per_family: int | None = None,
):
    train_ids = json.loads((split_dir / "train_ids.json").read_text(encoding="utf-8"))
    dev_ids_path = split_dir / "dev_ids.json"
    dev_ids = json.loads(dev_ids_path.read_text(encoding="utf-8")) if dev_ids_path.exists() else {}
    all_seqs = load_all_families()
    train = {}
    dev = {}
    for fam in FAMILIES:
        ids = list(train_ids.get(fam, []))
        if limit_per_family is not None:
            ids = ids[:limit_per_family]
        train[fam] = {sid: all_seqs[fam][sid] for sid in ids}
        dev_source_ids = list(dev_ids.get(fam, []))
        if limit_dev_per_family is not None:
            dev_source_ids = dev_source_ids[:limit_dev_per_family]
        dev[fam] = {sid: all_seqs[fam][sid] for sid in dev_source_ids}
    extras = load_extra_families(extra_data_dir)
    if limit_extra_per_family is not None:
        extras = {
            fam: dict(list(extras.get(fam, {}).items())[:limit_extra_per_family])
            for fam in FAMILIES
        }
    train = merge_sequence_maps(train, extras)
    extra_counts = {fam: len(extras.get(fam, {})) for fam in FAMILIES}
    return train, dev, extra_counts


def _make_optimizer(model: nn.Module, *, lr: float, weight_decay: float):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("bias") or "norm" in name.lower() or "embedding" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def _batch_loss(model: nn.Module, batch: dict, *, pad_id: int) -> torch.Tensor:
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]
    logits = model(input_ids, attention_mask=attention_mask)
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=pad_id,
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer,
    device: torch.device,
    pad_id: int,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        loss = _batch_loss(model, batch, pad_id=pad_id)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_batches += 1
    return total_loss / max(total_batches, 1)


@torch.no_grad() if torch is not None else (lambda fn: fn)
def _eval_lm_loss(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    pad_id: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = _batch_loss(model, batch, pad_id=pad_id)
        total_loss += float(loss.detach().cpu())
        total_batches += 1
    return total_loss / max(total_batches, 1)


def _resolve_args_config(args) -> tuple[TransformerConfig, dict[str, Any]]:
    yaml_cfg = _read_yaml(args.config)
    model_cfg = dict(yaml_cfg.get("model", {}))
    train_cfg = dict(yaml_cfg.get("training", {}))

    for key in ["d_model", "n_layers", "n_heads", "dropout", "max_len", "dim_feedforward"]:
        value = getattr(args, key, None)
        if value is not None:
            model_cfg[key] = value

    config = TransformerConfig.from_dict(model_cfg)
    defaults = {
        "epochs": 8,
        "batch_size": 128,
        "lr": 5e-4,
        "weight_decay": 1e-2,
        "grad_clip": 1.0,
        "num_workers": 0,
    }
    defaults.update(train_cfg)
    if args.epochs is not None:
        defaults["epochs"] = args.epochs
    if args.batch_size is not None:
        defaults["batch_size"] = args.batch_size
    if args.lr is not None:
        defaults["lr"] = args.lr
    return config, defaults


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "configs" / "transformer_small.yaml")
    parser.add_argument("--split-dir", type=Path,
                        default=REPO_ROOT / "data" / "processed" / "splits")
    parser.add_argument("--eval-dir", type=Path,
                        default=REPO_ROOT / "data" / "processed" / "dev_eval")
    parser.add_argument("--model-path", type=Path,
                        default=REPO_ROOT / "models" / "transformer_small.pt")
    parser.add_argument("--metrics-path", type=Path,
                        default=REPO_ROOT / "artifacts" / "transformer_metrics.json")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train-sequences", type=int,
                        help="Debug only: cap training sequences per family.")
    parser.add_argument("--limit-dev-sequences", type=int,
                        help="Debug only: cap dev sequences per family for LM loss.")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip task-level dev evaluation after training.")
    parser.add_argument("--extra-data-dir", type=Path, default=None,
                        help="Optional directory of generated extra CSVs to augment training.")
    parser.add_argument("--limit-extra-sequences", type=int,
                        help="Debug only: cap generated extra sequences per family.")

    # Config overrides. Leave unset by default so YAML remains the source of truth.
    parser.add_argument("--d-model", dest="d_model", type=int)
    parser.add_argument("--n-layers", dest="n_layers", type=int)
    parser.add_argument("--n-heads", dest="n_heads", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--max-len", dest="max_len", type=int)
    parser.add_argument("--dim-feedforward", dest="dim_feedforward", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", dest="batch_size", type=int)
    parser.add_argument("--lr", type=float)
    args = parser.parse_args()

    require_torch()
    if not (args.split_dir / "train_ids.json").exists():
        raise SystemExit(
            f"Missing {args.split_dir/'train_ids.json'}. "
            "Run: python scripts/make_dev_split.py --force"
        )

    config, train_cfg = _resolve_args_config(args)
    requested_device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if requested_device == "auto":
        requested_device = "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but torch.cuda.is_available() is false")
    device = torch.device(requested_device)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    print(f"[1/5] Loading split from {args.split_dir}")
    train, dev, extra_counts = _load_split(
        args.split_dir,
        limit_per_family=args.limit_train_sequences,
        limit_dev_per_family=args.limit_dev_sequences,
        extra_data_dir=args.extra_data_dir,
        limit_extra_per_family=args.limit_extra_sequences,
    )
    print("      train: " + ", ".join(f"{f}={len(s)}" for f, s in train.items()))
    print("      dev  : " + ", ".join(f"{f}={len(s)}" for f, s in dev.items()))
    if args.extra_data_dir:
        print("      extras: " + ", ".join(f"{f}={extra_counts[f]}" for f in FAMILIES))

    print("[2/5] Building vocabulary and tokenized datasets")
    vocab_source = {fam: {**train.get(fam, {}), **dev.get(fam, {})} for fam in FAMILIES}
    vocab = Vocabulary.from_sequences(vocab_source)
    train_ds = SequenceTrainingDataset(train, vocab, max_len=config.max_len)
    dev_ds = SequenceTrainingDataset(dev, vocab, max_len=config.max_len) if any(dev.values()) else None
    collate = partial(collate_lm_batch, pad_id=vocab.pad_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate,
    )
    dev_loader = None
    if dev_ds is not None:
        dev_loader = DataLoader(
            dev_ds,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(train_cfg.get("num_workers", 0)),
            collate_fn=collate,
        )
    print(f"      vocab_size={len(vocab)}, train_examples={len(train_ds)}")
    print(f"      config={asdict(config)}")

    print(f"[3/5] Training on {device}")
    net = ProcessTransformerNet(len(vocab), config, vocab.pad_id).to(device)
    optimizer = _make_optimizer(
        net,
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    history = []
    t0 = time.time()
    epochs = int(train_cfg["epochs"])
    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(
            net,
            train_loader,
            optimizer=optimizer,
            device=device,
            pad_id=vocab.pad_id,
            grad_clip=float(train_cfg["grad_clip"]),
        )
        dev_loss = None
        if dev_loader is not None:
            dev_loss = _eval_lm_loss(net, dev_loader, device=device, pad_id=vocab.pad_id)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ppl": math.exp(min(train_loss, 20.0)),
            "dev_loss": dev_loss,
            "dev_ppl": math.exp(min(dev_loss, 20.0)) if dev_loss is not None else None,
        }
        history.append(row)
        dev_text = f", dev_loss={dev_loss:.4f}" if dev_loss is not None else ""
        print(f"      epoch {epoch:02d}/{epochs}: train_loss={train_loss:.4f}{dev_text}")
    train_seconds = time.time() - t0

    print(f"[4/5] Saving checkpoint -> {args.model_path}")
    metadata = {
        "seed": args.seed,
        "config_path": str(args.config),
        "train_cfg": train_cfg,
        "train_counts": {f: len(s) for f, s in train.items()},
        "dev_counts": {f: len(s) for f, s in dev.items()},
        "extra_data_dir": str(args.extra_data_dir) if args.extra_data_dir else None,
        "extra_counts": extra_counts,
        "history": history,
        "train_seconds": round(train_seconds, 2),
    }
    wrapper = TransformerProcessModel(net, vocab, config, device=str(device), metadata=metadata)
    wrapper.save(args.model_path, metadata=metadata)

    payload: dict[str, Any] = {
        "model": wrapper.stats(),
        "train_seconds": round(train_seconds, 2),
        "history": history,
        "extra_data_dir": str(args.extra_data_dir) if args.extra_data_dir else None,
        "extra_counts": extra_counts,
    }

    if args.skip_eval:
        print("[5/5] Skipping task-level dev evaluation (--skip-eval)")
    else:
        print(f"[5/5] Evaluating task metrics on {args.eval_dir}")
        t1 = time.time()
        metrics = evaluate_all(wrapper, args.eval_dir)
        eval_seconds = time.time() - t1
        payload["eval_seconds"] = round(eval_seconds, 2)
        payload["metrics"] = metrics.to_dict()
        print(json.dumps(metrics.to_dict(), indent=2))

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote metrics: {args.metrics_path}")


if __name__ == "__main__":
    main()
