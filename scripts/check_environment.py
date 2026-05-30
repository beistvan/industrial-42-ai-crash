#!/usr/bin/env python3
"""Check local runtime dependencies for the Infineon hackathon repo.

The n-gram baseline is pure Python/scikit-learn. The small Transformer in step 5
is deliberately lightweight, but it still needs PyTorch even for CPU-only local
smoke training. This script makes missing dependencies fail early with an
actionable message instead of showing hidden pytest skips or a late stack trace.
"""
from __future__ import annotations

import argparse
import importlib.util
import platform
import sys
from textwrap import dedent


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _torch_status() -> tuple[bool, str]:
    if not _has_module("torch"):
        return False, "not installed"
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-specific
        return False, f"installed but failed to import: {exc}"
    cuda = getattr(torch, "cuda", None)
    cuda_text = "cuda=unavailable"
    if cuda is not None:
        try:
            cuda_text = f"cuda={'available' if cuda.is_available() else 'unavailable'}"
        except Exception:  # pragma: no cover
            cuda_text = "cuda=unknown"
    return True, f"torch={torch.__version__}, {cuda_text}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-torch", action="store_true",
                        help="Fail if PyTorch is missing. Required for step 5 Transformer targets.")
    parser.add_argument("--require-streamlit", action="store_true",
                        help="Fail if Streamlit is missing. Required for run-demo/run-dashboard.")
    args = parser.parse_args()

    print(f"Python: {sys.version.split()[0]} ({platform.system()} {platform.machine()})")

    missing_base = []
    for name in ["numpy", "pandas", "sklearn", "yaml", "pytest"]:
        if not _has_module(name):
            missing_base.append(name)
    if args.require_streamlit and not _has_module("streamlit"):
        missing_base.append("streamlit")
    if missing_base:
        print("Missing dependencies: " + ", ".join(missing_base), file=sys.stderr)
        print("Run: make setup", file=sys.stderr)
        raise SystemExit(1)

    ok_torch, torch_msg = _torch_status()
    print(f"PyTorch: {torch_msg}")
    if args.require_torch and not ok_torch:
        print(
            dedent(
                """
                PyTorch is required for the compact Transformer model, even on CPU.

                Fix locally with one of these commands:
                  make setup-cpu

                or, manually:
                  python3 -m pip install --upgrade --index-url https://download.pytorch.org/whl/cpu torch

                On Leonardo/GPU, install the CUDA-enabled PyTorch build according to the
                HPC environment docs, then re-run:
                  make train-transformer-smoke
                """
            ).strip(),
            file=sys.stderr,
        )
        raise SystemExit(1)

    print("Environment check passed.")


if __name__ == "__main__":
    main()
