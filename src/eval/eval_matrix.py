"""Four-arm evaluation matrix for pitch-ready before/after comparisons.

Arms on the dev holdout:
  A — n-gram baseline (Level 1)
  B — T1 Transformer specialist
  C — T2 specialist + rule-constrained beam
  D — submission hybrid (T1 next-step/anomaly + T2 completion)

Missing checkpoints are marked ``status: unavailable`` — the matrix still
completes without silent degradation.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from src.eval.run_eval import (
    TaskMetrics,
    evaluate_all,
    evaluate_anomaly,
    evaluate_completion,
    evaluate_next_step,
)
from src.ml import load_sequence_model

REPO_ROOT = Path(__file__).resolve().parents[2]

ArmStatus = Literal["ok", "unavailable", "partial"]


@dataclass
class ArmSpec:
    arm_id: str
    name: str
    description: str
    model_paths: dict[str, Path]
    rule_constrained: bool = True
    candidate_pool: int = 5


@dataclass
class ArmResult:
    arm_id: str
    name: str
    description: str
    status: ArmStatus
    model_paths: dict[str, str]
    missing: list[str] = field(default_factory=list)
    metrics: dict[str, Any] | None = None
    runtime_s: float = 0.0
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_arm_specs(repo_root: Path | None = None) -> list[ArmSpec]:
    root = repo_root or REPO_ROOT
    ngram = root / "models" / "ngram_baseline.pkl"
    t1 = root / "models" / "sweeps" / "h_mod_nosched_mrr.pt.best"
    t2 = root / "models" / "sweeps" / "g_drop15_nosched_t2.pt.best"
    return [
        ArmSpec(
            arm_id="A",
            name="baseline",
            description="N-gram suffix-backoff (order 12) — Level 1 floor",
            model_paths={"all": ngram},
        ),
        ArmSpec(
            arm_id="B",
            name="t1_transformer",
            description="Wave 3 T1 specialist (h_mod_nosched_mrr) — all tasks",
            model_paths={"all": t1},
        ),
        ArmSpec(
            arm_id="C",
            name="t2_specialist",
            description="Wave 2 T2 specialist + rule-constrained beam — all tasks",
            model_paths={"all": t2},
        ),
        ArmSpec(
            arm_id="D",
            name="submission_hybrid",
            description="Submission: T1 next-step/anomaly + T2 completion",
            model_paths={"task1": t1, "task3": t1, "task2": t2},
        ),
    ]


def _resolve_models(
    spec: ArmSpec,
    *,
    device: str | None,
) -> tuple[dict[str, Any], list[str]]:
    loaded: dict[str, Any] = {}
    missing: list[str] = []
    for role, path in spec.model_paths.items():
        key = str(path)
        if not path.exists():
            missing.append(key)
            continue
        if key not in loaded:
            try:
                loaded[key] = load_sequence_model(path, device=device)
            except Exception:
                missing.append(key)
    return loaded, missing


def _model_for_role(spec: ArmSpec, role: str, loaded: dict[str, Any]) -> Any | None:
    if "all" in spec.model_paths:
        path = spec.model_paths["all"]
        return loaded.get(str(path))
    path = spec.model_paths.get(role)
    if path is None:
        return None
    return loaded.get(str(path))


def run_arm(
    spec: ArmSpec,
    eval_dir: Path,
    *,
    device: str | None = None,
) -> ArmResult:
    t0 = time.perf_counter()
    loaded, missing = _resolve_models(spec, device=device)

    expected_paths = {role: str(p) for role, p in spec.model_paths.items()}
    unique_expected = {str(p) for p in spec.model_paths.values()}
    if missing and set(missing) >= unique_expected:
        return ArmResult(
            arm_id=spec.arm_id,
            name=spec.name,
            description=spec.description,
            status="unavailable",
            model_paths=expected_paths,
            missing=missing,
            metrics=None,
            runtime_s=time.perf_counter() - t0,
            reason="all required checkpoints missing",
        )

    valid_in = eval_dir / "eval_input_valid_dev.csv"
    valid_gold = eval_dir / "eval_input_valid_dev_gold.csv"
    anomaly_in = eval_dir / "eval_input_anomaly_dev.csv"
    anomaly_gold = eval_dir / "eval_input_anomaly_dev_gold.csv"

    if spec.arm_id == "D":
        t1 = _model_for_role(spec, "task1", loaded)
        t2 = _model_for_role(spec, "task2", loaded)
        if t1 is None or t2 is None:
            return ArmResult(
                arm_id=spec.arm_id,
                name=spec.name,
                description=spec.description,
                status="unavailable",
                model_paths=expected_paths,
                missing=missing,
                metrics=None,
                runtime_s=time.perf_counter() - t0,
                reason="hybrid arm missing T1 and/or T2 checkpoint",
            )
        metrics = TaskMetrics(
            task1=evaluate_next_step(t1, valid_in, valid_gold),
            task2=evaluate_completion(
                t2,
                valid_in,
                valid_gold,
                rule_constrained=spec.rule_constrained,
                candidate_pool=spec.candidate_pool,
            ),
            task3=evaluate_anomaly(anomaly_in, anomaly_gold),
        ).to_dict()
        status: ArmStatus = "ok"
    else:
        model = _model_for_role(spec, "all", loaded)
        if model is None:
            return ArmResult(
                arm_id=spec.arm_id,
                name=spec.name,
                description=spec.description,
                status="unavailable",
                model_paths=expected_paths,
                missing=missing,
                metrics=None,
                runtime_s=time.perf_counter() - t0,
                reason="checkpoint missing or failed to load",
            )
        metrics = evaluate_all(
            model,
            eval_dir,
            rule_constrained=spec.rule_constrained,
            candidate_pool=spec.candidate_pool,
        ).to_dict()
        status = "partial" if missing else "ok"

    return ArmResult(
        arm_id=spec.arm_id,
        name=spec.name,
        description=spec.description,
        status=status,
        model_paths=expected_paths,
        missing=missing,
        metrics=metrics,
        runtime_s=round(time.perf_counter() - t0, 2),
    )


def run_eval_matrix(
    eval_dir: Path,
    *,
    arms: list[ArmSpec] | None = None,
    device: str | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    specs = arms or default_arm_specs(repo_root)
    results = [run_arm(spec, eval_dir, device=device) for spec in specs]
    available = [r for r in results if r.status != "unavailable" and r.metrics]
    summary: dict[str, Any] = {
        "arms_run": len(results),
        "arms_ok": sum(1 for r in results if r.status == "ok"),
        "arms_unavailable": sum(1 for r in results if r.status == "unavailable"),
    }
    if available:
        best_t1 = max(
            available,
            key=lambda r: (r.metrics or {}).get("task1_next_step", {}).get("overall", {}).get("mrr", 0),
        )
        best_t2 = min(
            available,
            key=lambda r: (r.metrics or {})
            .get("task2_completion", {})
            .get("overall", {})
            .get("normalized_edit_distance", 1.0),
        )
        summary["best_task1_mrr_arm"] = best_t1.arm_id
        summary["best_task2_ned_arm"] = best_t2.arm_id

    return {
        "eval_dir": str(eval_dir),
        "summary": summary,
        "arms": [r.to_dict() for r in results],
    }


def write_eval_matrix(
    eval_dir: Path,
    output_path: Path,
    *,
    device: str | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    payload = run_eval_matrix(eval_dir, device=device, repo_root=repo_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def matrix_comparison_table(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten matrix JSON into dashboard-friendly rows."""
    rows: list[dict[str, Any]] = []
    for arm in payload.get("arms", []):
        metrics = arm.get("metrics") or {}
        t1 = (metrics.get("task1_next_step") or {}).get("overall", {})
        t2 = (metrics.get("task2_completion") or {}).get("overall", {})
        t3 = metrics.get("task3_anomaly") or {}
        rows.append({
            "arm": arm.get("arm_id"),
            "name": arm.get("name"),
            "status": arm.get("status"),
            "task1_mrr": t1.get("mrr"),
            "task1_top1": t1.get("top1"),
            "task2_tok_acc": t2.get("token_accuracy"),
            "task2_ned": t2.get("normalized_edit_distance"),
            "task3_f1": t3.get("f1_invalid"),
            "task3_rule_attr": t3.get("rule_attribution_accuracy"),
            "runtime_s": arm.get("runtime_s"),
        })
    return rows
