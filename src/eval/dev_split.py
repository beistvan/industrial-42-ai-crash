"""Build local dev eval inputs mirroring the official eval format.

Produces:
  - eval_input_valid_dev.csv      (Tasks 1 & 2): each dev sequence truncated
                                  at 60% and 80% completion, long format
                                  SEQUENCE_ID, STEP.
  - eval_input_valid_dev_gold.csv (gold full sequences for scoring).
  - eval_input_anomaly_dev.csv    (Task 3): valid dev sequences mixed with
                                  rule-violating sequences produced by
                                  swapping / dropping critical steps.
  - eval_input_anomaly_dev_gold.csv (per-sequence labels + injected rule).

These are *local* dev inputs — when the organizers ship the official eval CSVs
they go to data/raw/infineon/eval/ and replace these.
"""
from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

from src.data.infineon_loader import Family, Sequences
from src.eval.rule_validator import is_valid_sequence, violation_rules

TRUNCATION_FRACTIONS: tuple[float, ...] = (0.6, 0.8)


@dataclass(frozen=True)
class DevEvalPaths:
    valid_input: Path
    valid_gold: Path
    anomaly_input: Path
    anomaly_gold: Path


def truncate(steps: list[str], fraction: float) -> list[str]:
    n = max(1, int(math.floor(len(steps) * fraction)))
    return steps[:n]


def _seq_id(family: Family, sid: str, fraction: float) -> str:
    pct = int(round(fraction * 100))
    return f"dev_{family}_{sid}_t{pct}"


def write_valid_dev(
    dev_per_family: dict[Family, Sequences],
    out_input: Path,
    out_gold: Path,
) -> None:
    out_input.parent.mkdir(parents=True, exist_ok=True)
    out_gold.parent.mkdir(parents=True, exist_ok=True)
    with out_input.open("w", newline="", encoding="utf-8") as fi, \
         out_gold.open("w", newline="", encoding="utf-8") as fg:
        wi, wg = csv.writer(fi), csv.writer(fg)
        wi.writerow(["SEQUENCE_ID", "FAMILY", "STEP"])
        wg.writerow(["SEQUENCE_ID", "FAMILY", "STEP"])
        for family, seqs in dev_per_family.items():
            for sid, steps in seqs.items():
                for frac in TRUNCATION_FRACTIONS:
                    eval_sid = _seq_id(family, sid, frac)
                    for step in truncate(steps, frac):
                        wi.writerow([eval_sid, family, step])
                    for step in steps:
                        wg.writerow([eval_sid, family, step])


# ---- Anomaly injection -----------------------------------------------------

def _inject_violation(steps: list[str], rng: random.Random) -> tuple[list[str], str] | None:
    """Try a few mutations to produce a rule-violating sequence.

    Returns (new_steps, injected_rule_label) or None if we couldn't trigger
    any rule violation in this sequence. We re-validate to confirm.
    """
    strategies = [
        ("RULE_DEP_NO_CLEAN_drop_clean", _drop_clean_steps),
        ("RULE_ETCH_NO_MASK_drop_develop", _drop_develops),
        ("RULE_SHIP_BEFORE_TEST_move_ship", _move_ship_early),
        ("shuffle_middle", _shuffle_middle),
    ]
    rng.shuffle(strategies)
    for label, fn in strategies:
        mutated = fn(steps, rng)
        if mutated is not None and mutated != steps and not is_valid_sequence(mutated):
            return mutated, label
    return None


def _drop_clean_steps(steps: list[str], rng: random.Random) -> list[str] | None:
    targets = {"PRE CLEAN WAFER", "RCA CLEAN 1", "RCA CLEAN 2", "HF DIP",
               "PRE DEPOSITION CLEAN", "PRE METAL CLEAN", "RCA CLEAN"}
    new = [s for s in steps if s not in targets]
    return new if len(new) != len(steps) else None


def _drop_develops(steps: list[str], rng: random.Random) -> list[str] | None:
    targets = {"DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"}
    new = [s for s in steps if s not in targets]
    return new if len(new) != len(steps) else None


def _move_ship_early(steps: list[str], rng: random.Random) -> list[str] | None:
    if "SHIP LOT" not in steps or "WAFER SORT TEST" not in steps:
        return None
    new = [s for s in steps if s != "SHIP LOT"]
    # Insert SHIP LOT before the sort test.
    try:
        idx = new.index("WAFER SORT TEST")
    except ValueError:
        return None
    if idx == 0:
        return None
    new.insert(rng.randint(0, idx - 1) if idx > 1 else 0, "SHIP LOT")
    return new


def _shuffle_middle(steps: list[str], rng: random.Random) -> list[str] | None:
    if len(steps) < 20:
        return None
    new = list(steps)
    middle = new[5:-5]
    rng.shuffle(middle)
    new[5:-5] = middle
    return new


def write_anomaly_dev(
    dev_per_family: dict[Family, Sequences],
    out_input: Path,
    out_gold: Path,
    *,
    seed: int = 17,
) -> dict[str, int]:
    rng = random.Random(seed)
    out_input.parent.mkdir(parents=True, exist_ok=True)
    out_gold.parent.mkdir(parents=True, exist_ok=True)
    valid_rows: list[tuple[str, Family, list[str]]] = []
    invalid_rows: list[tuple[str, Family, list[str], str]] = []
    for family, seqs in dev_per_family.items():
        for sid, steps in seqs.items():
            eval_sid_valid = f"dev_anom_{family}_{sid}_valid"
            valid_rows.append((eval_sid_valid, family, steps))
            res = _inject_violation(list(steps), rng)
            if res is not None:
                mutated, label = res
                eval_sid_inv = f"dev_anom_{family}_{sid}_invalid"
                invalid_rows.append((eval_sid_inv, family, mutated, label))
    combined: list[tuple[str, Family, list[str], bool, str]] = []
    for sid, fam, steps in valid_rows:
        combined.append((sid, fam, steps, True, ""))
    for sid, fam, steps, label in invalid_rows:
        combined.append((sid, fam, steps, False, label))
    rng.shuffle(combined)
    with out_input.open("w", newline="", encoding="utf-8") as fi, \
         out_gold.open("w", newline="", encoding="utf-8") as fg:
        wi, wg = csv.writer(fi), csv.writer(fg)
        wi.writerow(["SEQUENCE_ID", "FAMILY", "STEP"])
        wg.writerow(["SEQUENCE_ID", "FAMILY", "VALID", "INJECTED_LABEL", "DETECTED_RULES"])
        for sid, fam, steps, is_valid, label in combined:
            for step in steps:
                wi.writerow([sid, fam, step])
            detected = ";".join(violation_rules(steps))
            wg.writerow([sid, fam, int(is_valid), label, detected])
    return {
        "valid": len(valid_rows),
        "invalid": len(invalid_rows),
        "total": len(combined),
    }


def build_dev_eval(
    dev_per_family: dict[Family, Sequences],
    out_dir: Path,
    *,
    anomaly_seed: int = 17,
) -> tuple[DevEvalPaths, dict]:
    paths = DevEvalPaths(
        valid_input=out_dir / "eval_input_valid_dev.csv",
        valid_gold=out_dir / "eval_input_valid_dev_gold.csv",
        anomaly_input=out_dir / "eval_input_anomaly_dev.csv",
        anomaly_gold=out_dir / "eval_input_anomaly_dev_gold.csv",
    )
    write_valid_dev(dev_per_family, paths.valid_input, paths.valid_gold)
    counts = write_anomaly_dev(
        dev_per_family, paths.anomaly_input, paths.anomaly_gold, seed=anomaly_seed
    )
    return paths, counts
