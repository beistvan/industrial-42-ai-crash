#!/usr/bin/env python3
"""Generate optional extra valid Infineon process routes for local augmentation.

This is the local, GPU-free part of step 6. It uses the official Infineon
`generate_sequences.py` implementation already vendored under
`data/raw/infineon/training_data/`, writes generated CSVs under
`data/generated/infineon/`, and records a manifest for reproducibility.

Examples:
    python scripts/generate_extra_sequences.py --count-per-family 250 --seed 101
    python scripts/train_ngram.py --extra-data-dir data/generated/infineon
    python scripts/train_transformer.py --extra-data-dir data/generated/infineon --epochs 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.infineon_loader import FAMILIES, _load_official_module  # noqa: E402

FAMILY_FILE_STEMS = {
    "mosfet": "MOSFET",
    "igbt": "IGBT",
    "ic": "IC",
}


def _row_count(path: Path) -> int:
    with path.open(encoding="utf-8-sig") as f:
        return max(sum(1 for _ in f) - 1, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count-per-family", type=int, default=250,
                        help="How many generated sequences per family.")
    parser.add_argument("--seed", type=int, default=101,
                        help="Base seed; family index is added for reproducibility.")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "data" / "generated" / "infineon")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing generated CSVs.")
    args = parser.parse_args()

    if args.count_per_family <= 0:
        raise SystemExit("--count-per-family must be positive")

    gen = _load_official_module()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "count_per_family": args.count_per_family,
        "base_seed": args.seed,
        "files": [],
    }

    for idx, family in enumerate(FAMILIES):
        seed = args.seed + idx
        stem = FAMILY_FILE_STEMS[family]
        out_path = args.out_dir / f"{stem}_extra_{args.count_per_family}_seed{seed}.csv"
        if out_path.exists() and not args.force:
            raise FileExistsError(
                f"Refusing to overwrite {out_path}. Pass --force to regenerate."
            )
        print(f"Generating {family} count={args.count_per_family} seed={seed} -> {out_path}")
        sequences = gen.generate_dataset(
            family,
            args.count_per_family,
            seed=seed,
            validate=True,
        )
        gen.write_csv(out_path, sequences)
        # Validate once more through the official validator for a clear log.
        invalid = sum(1 for steps in sequences if gen.validate_sequence(steps))
        if invalid:
            raise RuntimeError(f"Generated {invalid} invalid {family} sequences")
        path_text = str(out_path)
        try:
            path_text = str(out_path.relative_to(REPO_ROOT))
        except ValueError:
            pass
        manifest["files"].append({
            "family": family,
            "path": path_text,
            "seed": seed,
            "sequences": len(sequences),
            "rows": _row_count(out_path),
        })

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote manifest: {manifest_path}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
