"""Deterministic train/dev splits per product family."""
from __future__ import annotations

import random
from dataclasses import dataclass

from src.data.infineon_loader import Family, FAMILIES, Sequences


@dataclass(frozen=True)
class SplitResult:
    """Train and dev sequences per family. Sequence ids are preserved."""
    train: dict[Family, Sequences]
    dev: dict[Family, Sequences]

    def train_lengths(self) -> dict[Family, int]:
        return {f: len(s) for f, s in self.train.items()}

    def dev_lengths(self) -> dict[Family, int]:
        return {f: len(s) for f, s in self.dev.items()}


def deterministic_split(
    sequences_per_family: dict[Family, Sequences],
    *,
    dev_per_family: int = 100,
    seed: int = 7,
) -> SplitResult:
    """Hold out `dev_per_family` sequences per family for local evaluation.

    Splits are deterministic across Python processes and machines: sequence ids
    are sorted, then shuffled with a family-specific Random instance derived
    from `seed` and the stable FAMILIES index. Do **not** use Python's built-in
    `hash()` here because it is intentionally randomized between processes.
    """
    train: dict[Family, Sequences] = {}
    dev: dict[Family, Sequences] = {}
    for family_index, family in enumerate(FAMILIES):
        if family not in sequences_per_family:
            continue
        seqs = sequences_per_family[family]
        if dev_per_family <= 0 or dev_per_family >= len(seqs):
            raise ValueError(
                f"dev_per_family={dev_per_family} must be between 1 and "
                f"{len(seqs) - 1} for {family}"
            )
        ids = sorted(seqs.keys())
        rng = random.Random(seed * 10_000 + family_index)
        rng.shuffle(ids)
        dev_ids = set(ids[:dev_per_family])
        train[family] = {sid: seqs[sid] for sid in ids if sid not in dev_ids}
        dev[family] = {sid: seqs[sid] for sid in ids if sid in dev_ids}
    return SplitResult(train=train, dev=dev)
