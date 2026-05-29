"""Load Infineon industrial sequence data and build a shared vocabulary.

Wraps the official `read_csv_sequences` from
`data/raw/infineon/training_data/generate_sequences.py` so we never diverge
from how the scoring script parses CSVs.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

Family = Literal["mosfet", "igbt", "ic"]
FAMILIES: tuple[Family, ...] = ("mosfet", "igbt", "ic")
Sequences = dict[str, list[str]]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "infineon"
_OFFICIAL_SCRIPT = DEFAULT_DATA_ROOT / "training_data" / "generate_sequences.py"

# Special tokens reserved by the modeling layer. They never collide with
# real step strings (which are uppercase English phrases).
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
FAMILY_TOKENS: dict[Family, str] = {
    "mosfet": "<fam:mosfet>",
    "igbt": "<fam:igbt>",
    "ic": "<fam:ic>",
}
SPECIAL_TOKENS: tuple[str, ...] = (
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
    *FAMILY_TOKENS.values(),
)


def _load_official_module():
    """Import `generate_sequences.py` by file path (it lives under data/raw/)."""
    if not _OFFICIAL_SCRIPT.exists():
        raise FileNotFoundError(
            f"Official Infineon generator not found at {_OFFICIAL_SCRIPT}. "
            f"Run `git pull` or refer to data/raw/infineon/SOURCE.md."
        )
    mod_name = "_infineon_generate_sequences"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, _OFFICIAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {_OFFICIAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _variants_path(family: Family, data_root: Path) -> Path:
    name = {"mosfet": "MOSFET_variants.csv",
            "igbt": "IGBT_variants.csv",
            "ic": "IC_variants.csv"}[family]
    return data_root / "training_data" / name


def load_family(
    family: Family,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> Sequences:
    """Read all sequences for one family. Returns dict[seq_id -> list[step]]."""
    if family not in FAMILIES:
        raise ValueError(f"Unknown family {family!r}; expected one of {FAMILIES}")
    path = _variants_path(family, data_root)
    if not path.exists():
        raise FileNotFoundError(f"Missing variants CSV for {family}: {path}")
    mod = _load_official_module()
    return mod.read_csv_sequences(path)


def load_all_families(
    data_root: Path = DEFAULT_DATA_ROOT,
) -> dict[Family, Sequences]:
    """Read sequences for all three families. Keys are kept in FAMILIES order."""
    return {fam: load_family(fam, data_root) for fam in FAMILIES}


def load_sequence_csv(path: Path) -> Sequences:
    """Read any long-format Infineon sequence CSV via the official parser.

    The CSV must have the same layout as the official training data: at least
    `SEQUENCE_ID` and `STEP` columns. Extra columns such as `FAMILY` are
    ignored by the official parser.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing sequence CSV: {path}")
    mod = _load_official_module()
    return mod.read_csv_sequences(path)


def _infer_family_from_path(path: Path) -> Family | None:
    """Best-effort family inference for generated CSV filenames."""
    low = path.name.lower()
    for family in FAMILIES:
        if family in low:
            return family
    return None


def load_extra_families(extra_data_dir: Path | None) -> dict[Family, Sequences]:
    """Load optional generated training sequences from a directory.

    The helper scans `*.csv` files under `extra_data_dir` and assigns each file
    to a family by filename (`mosfet`, `igbt`, or `ic`). Sequence ids are
    prefixed with the source filename stem, so generated ids cannot collide with
    official training ids. Missing directories simply return empty family maps.
    """
    out: dict[Family, Sequences] = {fam: {} for fam in FAMILIES}
    if extra_data_dir is None or not extra_data_dir.exists():
        return out
    for csv_path in sorted(extra_data_dir.glob("*.csv")):
        family = _infer_family_from_path(csv_path)
        if family is None:
            continue
        for sid, steps in load_sequence_csv(csv_path).items():
            out[family][f"{csv_path.stem}::{sid}"] = steps
    return out


def merge_sequence_maps(
    base: dict[Family, Sequences],
    extra: dict[Family, Sequences] | None,
) -> dict[Family, Sequences]:
    """Return a family-indexed copy of `base` augmented with `extra`."""
    merged: dict[Family, Sequences] = {fam: dict(base.get(fam, {})) for fam in FAMILIES}
    if not extra:
        return merged
    for family in FAMILIES:
        merged[family].update(extra.get(family, {}))
    return merged


@dataclass(frozen=True)
class Vocabulary:
    """Deterministic token -> int mapping shared across all three families.

    Special tokens are assigned the lowest ids (see SPECIAL_TOKENS order):
    pad=0, bos=1, eos=2, unk=3, fam:mosfet=4, fam:igbt=5, fam:ic=6, then
    process step tokens sorted alphabetically for reproducibility.
    """
    id_to_token: tuple[str, ...]
    token_to_id: dict[str, int] = field(hash=False)

    def __len__(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_id(self) -> int: return self.token_to_id[PAD_TOKEN]
    @property
    def bos_id(self) -> int: return self.token_to_id[BOS_TOKEN]
    @property
    def eos_id(self) -> int: return self.token_to_id[EOS_TOKEN]
    @property
    def unk_id(self) -> int: return self.token_to_id[UNK_TOKEN]

    def family_id(self, family: Family) -> int:
        return self.token_to_id[FAMILY_TOKENS[family]]

    def encode(self, tokens: Iterable[str]) -> list[int]:
        return [self.token_to_id.get(t, self.unk_id) for t in tokens]

    def decode(self, ids: Iterable[int]) -> list[str]:
        return [self.id_to_token[i] for i in ids]

    @classmethod
    def from_step_tokens(cls, step_tokens: Iterable[str]) -> "Vocabulary":
        steps_sorted = sorted({s for s in step_tokens if s not in SPECIAL_TOKENS})
        ordered: list[str] = list(SPECIAL_TOKENS) + steps_sorted
        token_to_id = {tok: i for i, tok in enumerate(ordered)}
        return cls(id_to_token=tuple(ordered), token_to_id=token_to_id)

    @classmethod
    def from_sequences(
        cls,
        sequences_per_family: dict[Family, Sequences],
    ) -> "Vocabulary":
        seen: set[str] = set()
        for seqs in sequences_per_family.values():
            for steps in seqs.values():
                seen.update(steps)
        return cls.from_step_tokens(seen)
