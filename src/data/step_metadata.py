"""Lookup STEP → DESCRIPTION / fab parameters from official reference CSVs.

Training sequences (`*_variants.csv`) contain STEP tokens only. The reference
files map each canonical step name to human-readable text and optional fab-level
parameters. Use this for demos, reports, and (future) auxiliary training signals.

Judge submission outputs remain STEP tokens only.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

from src.data.infineon_loader import FAMILIES, Family, PROJECT_ROOT

REFERENCE_DIR = PROJECT_ROOT / "data" / "raw" / "infineon" / "reference"

_FAMILY_FILES: dict[Family, tuple[str, str | None]] = {
    "mosfet": ("MOSFET_Longdescr.csv", "MOSFET_longdescription_parameters.csv"),
    "igbt": ("IGBT_Longdescr.csv", "IGBT_longdescription_parameters.csv"),
    "ic": ("IC_Longdescr.csv", "IC_longdescription_parameters.csv"),
}


def _normalize_header(name: str) -> str:
    return name.strip().lstrip("\ufeff").strip('"')


def _normalize_row(row: dict[str, str]) -> dict[str, str]:
    return {_normalize_header(k): v for k, v in row.items()}


def _read_two_col(path: Path, value_col: str) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        value_key = _normalize_header(value_col)
        for raw in reader:
            row = _normalize_row(raw)
            step = (row.get("STEP") or "").strip()
            val = (row.get(value_key) or "").strip()
            if step and val:
                out[step] = val
    return out


@lru_cache(maxsize=1)
def load_step_metadata() -> dict[Family, dict[str, dict[str, str]]]:
    """Return {family: {step: {description, parameters}}}."""
    meta: dict[Family, dict[str, dict[str, str]]] = {}
    for fam in FAMILIES:
        descr_file, param_file = _FAMILY_FILES[fam]
        descriptions = _read_two_col(REFERENCE_DIR / descr_file, "DESCRIPTION")
        parameters: dict[str, str] = {}
        if param_file:
            param_path = REFERENCE_DIR / param_file
            if param_path.exists():
                with param_path.open(encoding="utf-8-sig", newline="") as fh:
                    reader = csv.DictReader(fh)
                    fields = [
                        _normalize_header(c)
                        for c in (reader.fieldnames or [])
                        if _normalize_header(c) != "STEP"
                    ]
                    for raw in reader:
                        row = _normalize_row(raw)
                        step = (row.get("STEP") or "").strip()
                        if not step:
                            continue
                        parts = [
                            f"{col}: {row[col].strip()}"
                            for col in fields
                            if row.get(col, "").strip()
                        ]
                        if parts:
                            parameters[step] = "; ".join(parts)
        steps = set(descriptions) | set(parameters)
        meta[fam] = {
            s: {
                "description": descriptions.get(s, ""),
                "parameters": parameters.get(s, ""),
            }
            for s in steps
        }
    return meta


def describe_step(family: Family, step: str) -> dict[str, str]:
    """Return description/parameters dict for one step (may be empty strings)."""
    return load_step_metadata().get(family, {}).get(step, {"description": "", "parameters": ""})
