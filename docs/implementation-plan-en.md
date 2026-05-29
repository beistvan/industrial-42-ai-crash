# Implementation Plan (English translation of `implementation-plan-hu.md`)

> Faithful translation of the Hungarian original. Code blocks, tables, file
> paths, and external links are preserved verbatim.

Yes — **here is concretely what you need to do right now**: drop / shelve the
old mock-RandomForest direction in the industrial repo and rebuild around the
official Infineon case as a **hybrid sequence-learning + rule-validator
system**.

The winning strategy is *not* "throw a big LLM at it." The point of the case
is: **does the model actually learn real semiconductor process logic, or is
it just copying patterns?** So the strongest solution is:

> **Grammar-aware sequence model**
> baseline + trained token-sequence model + explicit process-rule validator +
> demo + measurable scaling experiment.

Only use Leonardo once the local pipeline already runs. Per the CINECA docs,
Leonardo is reached via SSH on host `login.leonardo.cineca.it`, 2FA is
mandatory, and the Booster partition is built from NVIDIA A100-64GB GPU nodes
with 4 GPUs per node. ([docs.hpc.cineca.it][1]) The AI:AT onboarding is also
explicitly structured around Leonardo login, Slurm, filesystem, and software
setup steps. ([AI Factory Austria AI:AT][2])

## Short diagnosis of the situation

The reality across the three repos right now:

1. **`zero_one_hack_01-main`** = the official hackathon repo.
   Treat this as the source of truth. It contains the Infineon track, the
   official training data, `generate_sequences.py`, the validator, the
   grammar, and the submission format.

2. **`industrial-models-that-learn-how-processes-unfold`** = your earlier
   scaffold. Right now it contains a mock telemetry / RandomForest anomaly
   demo. This no longer matches the case precisely, but it is a good repo
   skeleton: README, Streamlit app, tests, artifacts, Makefile.

3. **`zero-one-hack-ai-hackaton-main`** = a prep / orchestration pack.
   Only the submission / pitch / checklist parts of it are still useful.
   Don't take algorithmic direction from here.

## What to do now, in order

### 1. Repo consolidation: one repo should be THE repo

The main work lives in the industrial repo. Copy the official Infineon
material into it:

```bash
cd industrial-models-that-learn-how-processes-unfold

mkdir -p data/raw/infineon
cp -r ../zero_one_hack_01-main/tracks/industrial-infineon/training_data/* data/raw/infineon/

mkdir -p docs/official
cp ../zero_one_hack_01-main/tracks/industrial-infineon/Track_industrial_en.md docs/official/
cp ../zero_one_hack_01-main/tracks/industrial-infineon/training_data/generation_rules.md docs/official/
```

Then commit:

```bash
git checkout -b infineon-official-case
git add data/raw/infineon docs/official
git commit -m "Add official Infineon industrial sequence case data"
```

Very important: **do not commit the Leonardo link / token credentials, do not
put them in the README, do not share them on Discord**. Treat them as
secrets.

---

### 2. First milestone: official baseline, no GPU

In the next 1–2 hours, do not train a large model. First, have a local
baseline that can produce a submission file for all three tasks.

The three tasks:

| Task   | What to do                  | Best first solution                             |
| ------ | --------------------------- | ----------------------------------------------- |
| Task 1 | next-step top-5 prediction  | family-conditioned n-gram / suffix backoff      |
| Task 2 | sequence completion         | greedy / beam completion from the same model    |
| Task 3 | anomaly detection           | official `validate_sequence()` rule validator   |

**Task 3 should almost certainly be carried by a rule-based validator**,
because the 10 forbidden patterns are explicitly inside
`generate_sequences.py`. This is also strong from the judging angle: "not
just a black-box model; we expose process-rule attribution."

Initial file structure:

```text
src/
  data/
    infineon_loader.py
  ml/
    ngram_baseline.py
    transformer_model.py
  eval/
    local_eval.py
  inference/
    predict.py
scripts/
  make_dev_split.py
  train_ngram.py
  train_transformer.py
  predict_submission.py
extras/results/
  nextstep.csv
  completion.csv
  anomaly.csv
```

---

### 3. Build your own dev-eval set

Since the official eval input may come later / separately, build a local
validation set in exactly the same format.

The training-data statistics I saw:

| Family | Sequences |    Rows | Mean length | Unique steps |
| ------ | --------: | ------: | ----------: | -----------: |
| MOSFET |      1000 | 125,260 |      125.26 |          137 |
| IGBT   |      1000 | 147,959 |      147.96 |          147 |
| IC     |      1000 | 115,075 |      115.08 |          130 |

Your split could be e.g.:

```text
train: 900 sequence / family
dev:   100 sequence / family
```

From the dev set, generate:

```text
eval_input_valid_dev.csv
- 100 sequence × 3 family × 2 cut point = 600 rows
- cut point: 60% and 80%

eval_input_anomaly_dev.csv
- valid sequences + injected invalid sequences
```

This matters because you can already measure now:

```text
Top-1 / Top-3 / Top-5 / MRR
Completion token accuracy / edit distance
Anomaly F1 / rule attribution
```

---

### 4. Baseline model: n-gram + suffix backoff

For the first round, this:

```text
P(next_step | family, last_1_token)
P(next_step | family, last_2_tokens)
...
P(next_step | family, last_12_tokens)
```

Inference:

1. Look at the last 12 tokens of the partial sequence.
2. Search for an exact suffix match in the train sequences.
3. If none, back off to 11, 10, 9 … 1 tokens.
4. Return the top-5 next steps.
5. For completion, repeat the same greedy/beam-style until `SHIP LOT`.

This is fast, a strong baseline, and immediately comparable against a
trained transformer.

---

### 5. Improved model: small decoder-only transformer

No LLM needed. The vocabulary is ~150 tokens. That's ideal for a small
PyTorch model.

Input tokens:

```text
<BOS>, <FAMILY_MOSFET>, step_1, step_2, ..., step_n
```

Training objective:

```text
next-token prediction
```

Initial model configurations:

```yaml
small:
  d_model: 128
  n_layers: 4
  n_heads: 4
  dropout: 0.1
  max_len: 180
  batch_size: 128

medium:
  d_model: 256
  n_layers: 6
  n_heads: 8
  dropout: 0.1
  max_len: 180
  batch_size: 256
```

In the pitch you can then say:

> "We compare a memorization-prone n-gram baseline against a trained
> family-conditioned transformer, then constrain anomaly judgment with
> explicit process rules. This lets us separate the learned sequence
> distribution from hard manufacturing invariants."

That is much stronger than a plain "AI app."

---

### 6. Generate more data, but controlled

Because of the official generator you can produce additional valid
sequences:

```bash
cd data/raw/infineon

python generate_sequences.py --family mosfet --count 10000 --output MOSFET_extra_10k.csv --seed 101
python generate_sequences.py --family igbt   --count 10000 --output IGBT_extra_10k.csv   --seed 102
python generate_sequences.py --family ic     --count 10000 --output IC_extra_10k.csv     --seed 103
```

Then build a scaling table:

| Run         |    Data | Model              | Top-1 | Top-5 | Completion token acc | Notes          |
| ----------- | ------: | ------------------ | ----: | ----: | -------------------: | -------------- |
| baseline    |  3k seq | n-gram             |   ... |   ... |                  ... | no GPU         |
| small       |  3k seq | transformer-small  |   ... |   ... |                  ... | local/Leonardo |
| small+data  | 30k seq | transformer-small  |   ... |   ... |                  ... | generated      |
| medium+data | 30k seq | transformer-medium |   ... |   ... |                  ... | Leonardo       |

This is perfect material for the "European AI sovereignty / real
infrastructure / reproducible benchmark" story.

---

## Using Leonardo: only in phase 2

Don't start by fighting CUDA. Local first:

```bash
python scripts/make_dev_split.py
python scripts/train_ngram.py
python scripts/predict_submission.py --eval-valid data/eval_input_valid_dev.csv --eval-anomaly data/eval_input_anomaly_dev.csv
python src/eval/local_eval.py
```

Once that works, Leonardo:

```bash
ssh <your_user>@login.leonardo.cineca.it
```

Then:

```bash
git clone <your_repo_url>
cd industrial-models-that-learn-how-processes-unfold

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# torch install / module setup: per the Leonardo environment
```

Minimal Slurm script, with placeholders:

```bash
#!/bin/bash
#SBATCH --job-name=infineon-seq
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=runs/%x-%j.out
#SBATCH --error=runs/%x-%j.err

source .venv/bin/activate

python scripts/train_transformer.py \
  --config configs/transformer_small.yaml \
  --train-data data/processed/train_sequences.jsonl \
  --dev-data data/processed/dev_sequences.jsonl \
  --out runs/transformer_small
```

The exact account/QOS field will be specified by the hackathon Leonardo page
or by the infra mentor. Per the CINECA docs, the Booster side has both prod
and debug-style usage, with debug having short walltime; therefore start with
a debug / small job, then move to the prod / final training run.
([docs.hpc.cineca.it][1])

---

## Demo: what to show the jury

The demo should not be a generic dashboard. It should show exactly this:

1. **Partial process sequence input**

   ```text
   RECEIVE WAFER LOT → LOT IDENTIFICATION → INITIAL WAFER INSPECTION → ...
   ```

2. **Baseline top-5 vs transformer top-5**

   ```text
   Baseline: [HF DIP, ETCH, ...]
   Transformer: [MEASURE THICKNESS, MEASURE SURFACE PARTICLES, ...]
   ```

3. **Completion preview**

   * show the next 10–20 predicted steps;
   * mark whether it is valid per the grammar.

4. **Anomaly example**

   * feed in a sequence where e.g. there is no clean before deposition;
   * output:

     ```text
     Invalid
     Predicted rule: RULE_DEP_NO_CLEAN
     Explanation: deposition requires a clean step within the prior 12 steps.
     ```

5. **Metrics table**

   * n-gram baseline vs transformer;
   * anomaly validator F1 / rule attribution;
   * scaling run.

Demo narrative:

> "Our system learns the process-route distribution, but it also respects
> explicit semiconductor manufacturing invariants. This makes the benchmark
> interpretable: when the neural model predicts the next process step, the
> rule layer can validate whether the full route remains physically /
> process-logically plausible."

---

## Team split

**Person A — ML / backend**

* loader
* dev split
* n-gram baseline
* transformer
* metrics
* Leonardo run

**Person B — product / demo / submission**

* Streamlit rework
* REPORT.md
* slides
* demo script
* result tables
* repo cleanup
* no-secret check

Don't both do the same thing. The biggest mistake right now would be for
both of you to "try to train a model" while there is no submission pipeline.

---

## Concrete priority order for today

**Right now:**

```text
1. The industrial repo is THE main repo.
2. Copy in the official Infineon training_data.
3. Shelve the mock RandomForest baseline.
4. Loader + dev split.
5. Wire in the rule-based anomaly validator.
6. N-gram next-step / completion baseline.
7. Submission writer: nextstep.csv, completion.csv, anomaly.csv.
```

**After that:**

```text
8. Transformer train on a local small sample.
9. Leonardo smoke job.
10. Leonardo full small / medium training.
11. Streamlit demo: baseline vs trained.
12. REPORT.md + slides + 2-minute video.
```

## What NOT to do

Don't start building a RAG / LLM chatbot.
Don't tune the old mock-telemetry RandomForest.
Don't go to Leonardo before there is a locally working baseline + eval.
Don't commit credentials, token links, Discord invites, or secrets files.
Don't ship just a dashboard; ship measurable model output and an official
submission file.

## One-sentence strategic decision

**Build an Infineon-specific "process route learner": n-gram baseline + small
transformer + explicit grammar validator + reproducible eval + Streamlit
side-by-side demo.**

This is the path most likely to work, fastest to implement, and best
aligned with the jury.

[1]: https://docs.hpc.cineca.it/hpc/leonardo.html "Leonardo — CINECA HPC Documentation 1.0 documentation"
[2]: https://ai-at.eu/hpc-onboarding/ "HPC Onboarding | AI Factory Austria AI:AT"
