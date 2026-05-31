"""Shared Track 1 (Zero One) context for Streamlit dashboards."""

TRACK_DOC_URL = "https://docs.zero-one.lumos-consulting.at/tracks/track-1/"

# N-gram baseline (Level 1) — dev holdout
BASELINE = {
    "task1_top1": 0.687,
    "task1_mrr": 0.807,
    "task2_token_acc": 0.421,
    "task3_f1": 1.00,
    "task3_rule_attr": 0.66,
}

# Final hybrid submission (dev holdout). Live CSV picks come from
# LEADERBOARD_FINAL.csv via scripts/regenerate_submission.sh (HANDOFF.md).
SUBMISSION = {
    "task1_run": "h_mod_nosched_mrr",
    "task1_mrr": 0.8735,
    "task1_top1": 0.75,
    "task2_run": "g_drop15_nosched_t2",
    "task2_token_acc": 0.4545,
    "task3_f1": 1.00,
    "task3_rule_attr": 0.69,
}

LEVELS = [
    {
        "level": "Level 1",
        "title": "Data + baseline",
        "status": "done",
        "items": [
            "3k official sequences (IC / IGBT / MOSFET) + grammar docs",
            "+750 synthetic extras (250/family, seed=101)",
            "N-gram suffix-backoff baseline (order 12)",
        ],
        "metric": f"MRR {BASELINE['task1_mrr']:.3f} · tok {BASELINE['task2_token_acc']:.3f}",
    },
    {
        "level": "Level 2",
        "title": "Train → tune → benchmark",
        "status": "done",
        "items": [
            "Wave 1 Transformer finalists (vanilla ~4M params)",
            "Wave 2 hyperparam fine grid + hybrid Task-1/T2 specialists",
            "Dev leaderboard + unified Streamlit dashboard (`make run-dashboard`)",
        ],
        "metric": f"MRR {SUBMISSION['task1_mrr']:.3f} · tok {SUBMISSION['task2_token_acc']:.3f}",
    },
    {
        "level": "Level 3",
        "title": "Scaling & architecture stretch",
        "status": "done",
        "items": [
            "Data volume: 1× extras helps; 2× / 500-per-family hurts",
            "Wave 3 modern stack (RoPE + RMSNorm + SwiGLU) — T1 leader h_mod_nosched_mrr",
            "Wave 4 Task-2 prefix training — evaluated; did not beat g_drop15_nosched_t2",
        ],
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

SCALING_ROWS = [
    ("Real only", "m_real_only", "0", "~3 min"),
    ("+250/family (1×)", "m_real_extras_1x", "750", "~3 min"),
    ("+250/family (2× dup)", "m_real_extras_2x", "1500", "~3 min"),
    ("+500/family", "f_extras_500_100_mrr", "1500", "~35 min"),
]
