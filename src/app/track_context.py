"""Shared Track 1 (Zero One) context for Streamlit dashboards."""

TRACK_DOC_URL = "https://docs.zero-one.lumos-consulting.at/tracks/track-1/"

BASELINE = {
    "task1_mrr": 0.807,
    "task1_top1": 0.687,
    "task2_token_acc": 0.421,
    "task2_ned": 0.227,
    "task3_f1": 1.0,
    "task3_rule_attr": 0.66,
}

SUBMISSION = {
    "task1_run": "h_mod_nosched_mrr",
    "task1_mrr": 0.8735,
    "task1_top1": 0.75,
    "task2_run": "g_drop15_nosched_t2",
    "task2_token_acc": 0.4545,
    "task2_ned": 0.223,
    "task3_f1": 1.0,
    "task3_rule_attr": 0.69,
}

# Four-arm eval matrix (pitch / dashboard). Arm D = shipped hybrid.
EVAL_ARMS = [
    {
        "arm_id": "A",
        "name": "baseline",
        "label": "N-gram (Level 1 floor)",
        "model_hint": "models/ngram_baseline.pkl",
    },
    {
        "arm_id": "B",
        "name": "t1_transformer",
        "label": "T1 specialist (Wave 3 modern)",
        "model_hint": "models/sweeps/h_mod_nosched_mrr.pt.best",
    },
    {
        "arm_id": "C",
        "name": "t2_specialist",
        "label": "T2 specialist + rule beam",
        "model_hint": "models/sweeps/g_drop15_nosched_t2.pt.best",
    },
    {
        "arm_id": "D",
        "name": "submission_hybrid",
        "label": "Submission hybrid (T1 + T2)",
        "model_hint": "T1 next-step/anomaly · T2 completion",
    },
]

# Published floor / ceiling references for honest benchmarking.
FLOOR_BASELINES = {
    "level1_ngram": {
        "source": "This repo — n-gram suffix-backoff order 12 on dev holdout",
        "task1_mrr": BASELINE["task1_mrr"],
        "task2_ned": BASELINE["task2_ned"],
    },
    "task3_rule_validator": {
        "source": "Organizer validate_sequence() grammar — upper bound for rules-only T3",
        "task3_f1": 1.0,
        "note": "Dev anomalies are injected rule violations; F1=1.0 expected with validator.",
    },
    "task3_lm_score": {
        "source": "Teacher-forced LM log-prob on T1 checkpoint (not a separate finetune)",
        "note": "SCORE column in anomaly.csv; detection still rule-based.",
    },
}

LEVELS = [
    {
        "level": "Level 1",
        "status": "done",
        "metric": f"MRR {BASELINE['task1_mrr']:.3f} · NED {BASELINE['task2_ned']:.3f}",
    },
    {
        "level": "Level 2",
        "status": "done",
        "metric": f"MRR {SUBMISSION['task1_mrr']:.3f} · NED {SUBMISSION['task2_ned']:.3f}",
    },
    {
        "level": "Level 3",
        "status": "done",
        "metric": f"MRR {SUBMISSION['task1_mrr']:.3f} · NED {SUBMISSION['task2_ned']:.3f}",
    },
]

TRACK_ALIGNMENT = """
**Track 1 — Learning and Benchmarking Process Logic** ([brief](%s))

| Track requirement | Our approach |
|---|---|
| Level 1: data + baseline | 3k official + 750 synthetic; n-gram baseline |
| Level 2: train + tune + visible benchmark | Transformer Waves 1–2; dev LEADERBOARD; hybrid submission |
| Level 3: scaling / architecture | Extras scaling study; Wave 3 modern arch (+0.05pp MRR); Wave 4 T2 experiment |
| 3 submission tasks | `nextstep.csv` + `completion.csv` + `anomaly.csv` |
| Baseline vs trained demo | Unified dashboard → Overview / Training tabs |
""" % TRACK_DOC_URL

PIPELINE_WAVES = [
    ("Wave 1", "Vanilla Transformer finalists", "done", "f_drop15_100_mrr · f_extras_1x_100_t2"),
    ("Wave 2", "Hyperparam fine grid", "done", "g_drop15_nosched_t2 best T2 (0.455)"),
    ("Wave 3", "Modern architecture", "done", "h_mod_nosched_mrr best T1 (0.874)"),
    ("Wave 4", "Task-2 prefix training", "done", "no gain vs Wave 2 T2 specialist"),
    ("Wave 5", "Parameter enrichment", "skipped", "not needed — Wave 3 beat T1 bar"),
    ("Wave 6", "Large model + seed sweep", "skipped", "optional post-submission experiment"),
]
