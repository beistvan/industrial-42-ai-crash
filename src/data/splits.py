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

    Splits are deterministic for a given seed: sequence ids are sorted, then
    shuffled with the provided seed. The same seed always produces the same
    dev set.
    """
    train: dict[Family, Sequences] = {}
    dev: dict[Family, Sequences] = {}
    for family in FAMILIES:
        if family not in sequences_per_family:
            continue
        seqs = sequences_per_family[family]
        if dev_per_family >= len(seqs):
            raise ValueError(
                f"dev_per_family={dev_per_family} >= total sequences "
                f"({len(seqs)}) for {family}"
            )
        ids = sorted(seqs.keys())
        rng = random.Random(seed + hash(family) % 10_000)
        rng.shuffle(ids)
        dev_ids = set(ids[:dev_per_family])
        train[family] = {sid: seqs[sid] for sid in ids if sid not in dev_ids}
        dev[family] = {sid: seqs[sid] for sid in ids if sid in dev_ids}
    return SplitResult(train=train, dev=dev)
