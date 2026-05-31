# Leonardo GPU runbook — Infineon industrial track

## Decision status

Submission baseline is locked in `result/submission/`. Use this for
demo and jury-facing outputs:

```bash
make smoke          # dev split + n-gram + tests
make run-dashboard  # unified Track 1 UI (metrics, leaderboard, training curves, live demo)
```

Reproduce or refresh submission CSVs on GPU — see [`HANDOFF.md`](../HANDOFF.md)
and [`README.md`](../README.md).

## Login and project transfer

The hackathon slides list these login nodes:

```bash
ssh <USER>@login01-ext.leonardo.cineca.it
ssh <USER>@login02-ext.leonardo.cineca.it
ssh <USER>@login05-ext.leonardo.cineca.it
ssh <USER>@login07-ext.leonardo.cineca.it
```

Preferred transfer because the repo is already pushed:

```bash
ssh <USER>@login01-ext.leonardo.cineca.it
cd $SCRATCH
# or another hackathon directory provided by mentors
git clone <YOUR_GIT_REPO_URL> industrial-infineon
cd industrial-infineon
```

Use `scp` only if git clone/pull is unavailable:

```bash
# local laptop
bash scripts/leonardo/pack_for_upload.sh industrial-infineon-leonardo-upload.tgz
scp industrial-infineon-leonardo-upload.tgz <USER>@login01-ext.leonardo.cineca.it:$SCRATCH/

# Leonardo
cd $SCRATCH
mkdir -p industrial-infineon && cd industrial-infineon
tar -xzf ../industrial-infineon-leonardo-upload.tgz
```

## Environment setup on Leonardo login node

```bash
cd $SCRATCH/industrial-infineon
bash scripts/leonardo/setup_env.sh
```

This creates `.venv`, installs `requirements.txt`, verifies PyTorch, creates the dev split, and generates a small extra-data set.

Manual equivalent:

```bash
module purge || true
module load python/3.11.7 || true
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt
python scripts/check_environment.py --require-torch
python scripts/make_dev_split.py --force
python scripts/generate_extra_sequences.py --count-per-family 250 --seed 101 --force
```

Do not run real training on a login node. Use Slurm.

## Slurm GPU sequence

Submit in this order:

```bash
sbatch scripts/leonardo/01_gpu_smoke.slurm
squeue --me
```

Inspect logs:

```bash
ls -ltr logs/
tail -f logs/inf-gpu-smoke-<JOBID>.out
cat logs/inf-gpu-smoke-<JOBID>.err
```

When smoke is green, either use the default 250 extra sequences per family or generate the larger 10k/family set first:

```bash
# optional but useful for the scaling-table story
sbatch scripts/leonardo/00_generate_extra_10k.slurm
```

Then train:

```bash
sbatch scripts/leonardo/02_train_small.slurm
sbatch scripts/leonardo/03_train_small_extra.slurm
```

If time remains:

```bash
sbatch scripts/leonardo/04_train_medium_extra.slurm
```

Prediction with the best Transformer checkpoint:

```bash
MODEL=models/transformer_small_extra.pt sbatch scripts/leonardo/05_predict_transformer.slurm
```

For a best checkpoint saved by `dev_mrr`:

```bash
MODEL=models/transformer_small_extra.pt.best sbatch scripts/leonardo/05_predict_transformer.slurm
```

## Monitoring

```bash
squeue --me
tail -f logs/<job-name>-<jobid>.out
scancel <JOBID>
```

The GPU scripts write:

```text
models/sweeps/*.pt.best
artifacts/sweeps/*.json
result/submission/{nextstep,completion,anomaly}.csv
```

## Back-copy results

From the local machine:

```bash
scripts/leonardo/fetch_results.sh \
  <USER>@login01-ext.leonardo.cineca.it:$SCRATCH/industrial-infineon \
  leonardo_results
```

Or directly:

```bash
scp -r <USER>@login01-ext.leonardo.cineca.it:$SCRATCH/industrial-infineon/models/sweeps ./models_sweeps_leonardo
scp -r <USER>@login01-ext.leonardo.cineca.it:$SCRATCH/industrial-infineon/artifacts/sweeps ./artifacts_sweeps_leonardo
scp -r <USER>@login01-ext.leonardo.cineca.it:$SCRATCH/industrial-infineon/result/submission ./results_submission_leonardo
```

## Final hybrid submission (dev holdout)

| Model | Task 1 Top-1 | Task 1 MRR | Task 2 token-acc | Role |
|---|---:|---:|---:|---|
| N-gram baseline | 0.687 | 0.807 | 0.421 | L1 |
| **`h_mod_nosched_mrr`** | **0.75** | **0.874** | 0.437 | T1 + anomaly |
| **`g_drop15_nosched_t2`** | 0.738 | 0.867 | **0.455** | T2 completion |

See [`artifacts/sweeps/LEADERBOARD_FINAL.md`](../artifacts/sweeps/LEADERBOARD_FINAL.md)
for the full 27-run table.

Regenerate submission CSVs after leaderboard updates (Leonardo login node submits
GPU predict jobs via Slurm — do not load transformers locally):

```bash
make leonardo-leaderboard-final
make regenerate-submission
```

Optional reservation: `SLURM_RESERVATION=s_tra_ncc make regenerate-submission`

Historical sweep scripts live under `scripts/leonardo/` (submit_sweep.sh, etc.).

| Wave | YAML | Rows |
|---|---|---:|
| 3 modern | `leonardo_modern.yaml` | 8 |
| 4 Task-2 | `leonardo_task2.yaml` | 8 |
| 5 params | `leonardo_params.yaml` | 6 |
| 6 scale | `leonardo_scale.yaml` | 8 |

## If a job fails

- `CUDA is not visible`: you are not on a GPU compute node or the Slurm request failed. Run only through `sbatch` on `boost_usr_prod` with `--gpus-per-task=1`.
- `reservation invalid`: ask mentors for the current hackathon reservation; replace `#SBATCH --reservation=s_tra_ncc` in the Slurm scripts or remove it.
- `module load python/3.11.7` fails: run `module avail python` and replace the version.
- `pip install` too slow or fails on compute node: install on login node before `sbatch`. Compute nodes may have limited/no internet.
- final Task-2 eval is slow: expected. Train scripts skip full Task-2 eval; use prediction on the selected checkpoint only.
