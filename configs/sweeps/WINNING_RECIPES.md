# Winning training recipes

Reproduce the **current hybrid submission** checkpoints with these sweep rows.
All YAMLs live in this directory; base architectures in `configs/transformer_*.yaml`.

| Task | Run | Sweep YAML | Row (0-based) | Base config | Checkpoint |
|---|---|---|---:|---|---|
| **T1 + T3** | `h_mod_nosched_mrr` | `leonardo_modern.yaml` | **4** | `transformer_modern.yaml` | `models/sweeps/h_mod_nosched_mrr.pt.best` |
| **T2** | `g_drop15_nosched_t2` | `leonardo_fine.yaml` | **4** | `transformer_medium.yaml` | `models/sweeps/g_drop15_nosched_t2.pt.best` |

## Task 1 — `h_mod_nosched_mrr` (Wave 3)

From `leonardo_modern.yaml` defaults + row overrides:

```yaml
base_config: configs/transformer_modern.yaml
arch: modern
extra_data_dir: data/generated/infineon
dropout: 0.15
epochs: 100
scheduler: none
warmup_steps: 0
save_best_by: dev_mrr
eval_task1_every: 1
eval_task2_every: 5
amp: true
label_smoothing: 0.1
```

Train one row:

```bash
python scripts/sweep_transformer.py \
  --sweep configs/sweeps/leonardo_modern.yaml --stage finalists --row 4
```

## Task 2 — `g_drop15_nosched_t2` (Wave 2)

From `leonardo_fine.yaml` row:

```yaml
base_config: configs/transformer_medium.yaml
extra_data_dir: data/generated/infineon
dropout: 0.15
epochs: 120
scheduler: none
warmup_steps: 0
save_best_by: dev_token_acc
eval_task1_every: 1
eval_task2_every: 5
amp: true
label_smoothing: 0.1
```

Train one row:

```bash
python scripts/sweep_transformer.py \
  --sweep configs/sweeps/leonardo_fine.yaml --stage finalists --row 4
```

## Related recipes (not current submission picks)

| Run | YAML | Row | Notes |
|---|---|---:|---|
| `f_drop15_100_mrr` | `leonardo_final.yaml` | 2 | Wave 1 T1 leader (superseded by `h_mod_nosched_mrr`) |
| `h_mod_nosched_t2` | `leonardo_lite.yaml` | 0 | Modern stack + T2 prefix training (experimental) |
| `h_mod_nosched_s7_mrr` | `leonardo_lite.yaml` | 1 | T1 recipe, seed 7 — dev MRR **0.867** (below `h_mod_nosched_mrr` 0.874) |
| `q_mod_large_nosched_mrr` | `leonardo_quick_arch.yaml` | 0 | Modern large (d384×8L), 60 ep — capacity probe, not submission pick |
| `q_mod_large_nosched_t2` | `leonardo_quick_arch.yaml` | 1 | Same stack + T2 prefix training (60 ep) |

Full sweep history: `leonardo_v1.yaml`, `leonardo_final.yaml`, `leonardo_fine.yaml`, `leonardo_modern.yaml`, `leonardo_task2.yaml`, `leonardo_params.yaml`, `leonardo_scale.yaml`, `leonardo_lite.yaml`, `leonardo_modern_tune.yaml`, `leonardo_quick_arch.yaml`.
