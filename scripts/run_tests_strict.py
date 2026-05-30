#!/usr/bin/env python3
"""Run pytest and fail if any test is skipped.

Use this when preparing the final hackathon repo: skipped Transformer tests often
mean PyTorch was not installed locally, so the step-5 Transformer path was not
really verified.
"""
from __future__ import annotations

import os
import sys

import pytest


class _SkipCounter:
    def __init__(self) -> None:
        self.skipped = 0

    def pytest_runtest_logreport(self, report) -> None:  # pragma: no cover - pytest hook
        if report.skipped:
            self.skipped += 1


def main() -> None:
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    counter = _SkipCounter()
    exit_code = pytest.main(["-q", "-rs"], plugins=[counter])
    if exit_code != 0:
        raise SystemExit(exit_code)
    if counter.skipped:
        print(
            f"Strict test check FAILED: pytest reported {counter.skipped} skipped test(s). "
            "Run `make setup-cpu` and then `make test-strict` again.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print("Strict test check passed: no skipped tests.")


if __name__ == "__main__":
    main()
