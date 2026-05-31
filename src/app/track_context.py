"""Shared Track 1 (Zero One) context for Streamlit dashboards."""

TRACK_DOC_URL = "https://docs.zero-one.lumos-consulting.at/tracks/track-1/"

BASELINE = {
    "task1_mrr": 0.807,
    "task2_token_acc": 0.421,
    "task3_rule_attr": 0.66,
}

SUBMISSION = {
    "task1_run": "h_mod_nosched_mrr",
    "task1_mrr": 0.8735,
    "task2_run": "g_drop15_nosched_t2",
    "task2_token_acc": 0.4545,
    "task3_rule_attr": 0.69,
}

LEVELS = [
    {
        "level": "Level 1",
        "status": "done",
        "metric": f"MRR {BASELINE['task1_mrr']:.3f} · tok {BASELINE['task2_token_acc']:.3f}",
    },
    {
        "level": "Level 2",
        "status": "done",
        "metric": f"MRR {SUBMISSION['task1_mrr']:.3f} · tok {SUBMISSION['task2_token_acc']:.3f}",
    },
    {
        "level": "Level 3",
        "status": "done",
        "metric": f"MRR {SUBMISSION['task1_mrr']:.3f} · tok {SUBMISSION['task2_token_acc']:.3f}",
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
