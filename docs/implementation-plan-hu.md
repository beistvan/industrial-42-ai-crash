Igen, itt **nagyon konkrétan most ezt kell csinálnotok**: az industrial repo régi mock-RandomForest irányát el kell dobni/félretenni, és az official Infineon case-re át kell építeni egy **hybrid sequence-learning + rule-validator rendszert**.

A nyerő stratégia nem az, hogy “ráengedünk egy nagy LLM-et”. A case lényege: **tanul-e a modell valódi félvezető processzlogikát, vagy csak mintázatot másol?** Ezért a legerősebb megoldás:

> **Grammar-aware sequence model**
> baseline + trainelt token-szekvencia modell + explicit process-rule validator + demo + mérhető scaling experiment.

A Leonardo-t csak akkor használjátok, amikor a lokális pipeline már fut. A CINECA doksi szerint Leonardo SSH-n a `login.leonardo.cineca.it` hoston érhető el, és 2FA kötelező; a Booster partíció NVIDIA A100-64GB GPU-s node-okból áll, node-onként 4 GPU-val. ([docs.hpc.cineca.it][1]) Az AI:AT onboarding is kifejezetten Leonardo-login, Slurm, fájlrendszer és software setup lépésekre van felépítve. ([AI Factory Austria AI:AT][2])

## A helyzet rövid diagnózisa

A három repóból most ez a valóság:

1. **`zero_one_hack_01-main`** = official hackathon repo.
   Ezt kell tekinteni truth source-nak. Benne van az Infineon track, official training data, `generate_sequences.py`, validator, grammar, submission format.

2. **`industrial-models-that-learn-how-processes-unfold`** = a ti korábbi scaffoldotok.
   Jelenleg mock telemetry / RandomForest anomaly demo van benne. Ez már nem passzol pontosan a case-hez, de jó repo-váz: README, Streamlit app, tests, artifacts, Makefile.

3. **`zero-one-hack-ai-hackaton-main`** = előkészítő/orchestration pack.
   Ebből már csak a submission/pitch/checklist részek hasznosak. Ne innen induljatok algoritmikailag.

## Mit csináljatok most, sorrendben

### 1. Repo-konszolidáció: egy repo legyen a fő repo

A fő munka az industrial repóban legyen. Oda másoljátok be az official Infineon anyagot:

```bash
cd industrial-models-that-learn-how-processes-unfold

mkdir -p data/raw/infineon
cp -r ../zero_one_hack_01-main/tracks/industrial-infineon/training_data/* data/raw/infineon/

mkdir -p docs/official
cp ../zero_one_hack_01-main/tracks/industrial-infineon/Track_industrial_en.md docs/official/
cp ../zero_one_hack_01-main/tracks/industrial-infineon/training_data/generation_rules.md docs/official/
```

Majd commit:

```bash
git checkout -b infineon-official-case
git add data/raw/infineon docs/official
git commit -m "Add official Infineon industrial sequence case data"
```

Nagyon fontos: a Leonardo linket/tokenes credentialt **ne commitoljátok, ne tegyétek README-be, ne osszátok meg Discordon**. Kezeljétek secretként.

---

### 2. Az első milestone: official baseline, nem GPU

A következő 1–2 órában ne traineljetek nagy modellt. Először legyen egy lokális baseline, ami mindhárom taskra tud submission fájlt írni.

A három task:

| Task   | Mit kell csinálni          | Legjobb első megoldás                         |
| ------ | -------------------------- | --------------------------------------------- |
| Task 1 | next-step top-5 prediction | family-conditioned n-gram / suffix backoff    |
| Task 2 | sequence completion        | greedy / beam completion ugyanebből           |
| Task 3 | anomaly detection          | official `validate_sequence()` rule validator |

A **Task 3-at szinte biztosan rule-based validatorral kell vinni**, mert a 10 forbidden pattern explicit benne van a `generate_sequences.py`-ban. Ez judge szempontból is erős: “not just a black-box model; we expose process-rule attribution.”

Első fájlstruktúra:

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

### 3. Csináljatok saját dev-eval setet

Mivel az official eval input lehet később/külön jön, csináljatok lokális validációt ugyanabban a formátumban.

A training data statisztikája, amit láttam:

| Family | Sequences |    Rows | Mean length | Unique steps |
| ------ | --------: | ------: | ----------: | -----------: |
| MOSFET |      1000 | 125,260 |      125.26 |          137 |
| IGBT   |      1000 | 147,959 |      147.96 |          147 |
| IC     |      1000 | 115,075 |      115.08 |          130 |

A saját split legyen például:

```text
train: 900 sequence / family
dev:   100 sequence / family
```

A dev-ből generáltok:

```text
eval_input_valid_dev.csv
- 100 sequence × 3 family × 2 cut point = 600 rows
- cut point: 60% és 80%

eval_input_anomaly_dev.csv
- valid sequences + injected invalid sequences
```

Ez azért fontos, mert így már most tudtok mérni:

```text
Top-1 / Top-3 / Top-5 / MRR
Completion token accuracy / edit distance
Anomaly F1 / rule attribution
```

---

### 4. Baseline modell: n-gram + suffix backoff

Első körben ez legyen:

```text
P(next_step | family, last_1_token)
P(next_step | family, last_2_tokens)
...
P(next_step | family, last_12_tokens)
```

Inference:

1. Nézd a partial sequence utolsó 12 tokenjét.
2. Keress exact suffix match-et a train sequences-ben.
3. Ha nincs, menj vissza 11, 10, 9 … 1 tokenre.
4. Adj top-5 következő lépést.
5. Completionnél ugyanezt greedy/beam módon ismételd `SHIP LOT`-ig.

Ez gyors, erős baseline, és rögtön összehasonlítható a trainelt transformerrel.

---

### 5. Improved model: kis decoder-only transformer

Nem kell LLM. A vocab kb. 150 token. Ez ideális kis PyTorch modellhez.

Input tokenek:

```text
<BOS>, <FAMILY_MOSFET>, step_1, step_2, ..., step_n
```

Training objective:

```text
next-token prediction
```

Model konfiguráció elsőre:

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

A pitchben ezt tudjátok mondani:

> “We compare a memorization-prone n-gram baseline against a trained family-conditioned transformer, then constrain anomaly judgment with explicit process rules. This lets us separate learned sequence distribution from hard manufacturing invariants.”

Ez sokkal erősebb, mint egy sima “AI app”.

---

### 6. Generáljatok több adatot, de kontrolláltan

Az official generator miatt tudtok plusz valid sequence-eket csinálni:

```bash
cd data/raw/infineon

python generate_sequences.py --family mosfet --count 10000 --output MOSFET_extra_10k.csv --seed 101
python generate_sequences.py --family igbt   --count 10000 --output IGBT_extra_10k.csv   --seed 102
python generate_sequences.py --family ic     --count 10000 --output IC_extra_10k.csv     --seed 103
```

Majd csináljatok scaling táblát:

| Run         |    Data | Model              | Top-1 | Top-5 | Completion token acc | Notes          |
| ----------- | ------: | ------------------ | ----: | ----: | -------------------: | -------------- |
| baseline    |  3k seq | n-gram             |   ... |   ... |                  ... | no GPU         |
| small       |  3k seq | transformer-small  |   ... |   ... |                  ... | local/Leonardo |
| small+data  | 30k seq | transformer-small  |   ... |   ... |                  ... | generated      |
| medium+data | 30k seq | transformer-medium |   ... |   ... |                  ... | Leonardo       |

Ez a “European AI sovereignty / real infrastructure / reproducible benchmark” sztorihoz tökéletes.

---

## Leonardo használata: csak a 2. fázisban

Most ne azzal kezdjétek, hogy CUDA-val küzdötök. Először local:

```bash
python scripts/make_dev_split.py
python scripts/train_ngram.py
python scripts/predict_submission.py --eval-valid data/eval_input_valid_dev.csv --eval-anomaly data/eval_input_anomaly_dev.csv
python src/eval/local_eval.py
```

Ha ez megy, akkor Leonardo:

```bash
ssh <your_user>@login.leonardo.cineca.it
```

Utána:

```bash
git clone <your_repo_url>
cd industrial-models-that-learn-how-processes-unfold

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# torch install / module setup: Leonardo environment szerint
```

Minimal Slurm script, placeholderrel:

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

A pontos account/QOS mezőt a hackathon Leonardo oldal vagy az infra mentor mondja meg. A CINECA doksi alapján a Booster oldalon van prod és debug jellegű használat, debug rövid walltime-mal; ezért először debug/small jobbal próbáljátok, aztán prod/final train. ([docs.hpc.cineca.it][1])

---

## Demo: mit mutassatok a zsűrinek

A demo ne általános dashboard legyen. Pont ezt mutassa:

1. **Partial process sequence bemenet**

   ```text
   RECEIVE WAFER LOT → LOT IDENTIFICATION → INITIAL WAFER INSPECTION → ...
   ```

2. **Baseline top-5 vs transformer top-5**

   ```text
   Baseline: [HF DIP, ETCH, ...]
   Transformer: [MEASURE THICKNESS, MEASURE SURFACE PARTICLES, ...]
   ```

3. **Completion preview**

   * mutassa a hátralévő 10–20 predicted stepet;
   * jelölje, hogy valid-e a grammar szerint.

4. **Anomaly example**

   * beadni egy sequence-et, ahol pl. deposition előtt nincs clean;
   * output:

     ```text
     Invalid
     Predicted rule: RULE_DEP_NO_CLEAN
     Explanation: deposition requires clean step within prior 12 steps.
     ```

5. **Metrics table**

   * n-gram baseline vs transformer;
   * anomaly validator F1/rule attribution;
   * scaling run.

A demo szövege legyen:

> “Our system learns the process route distribution, but it also respects explicit semiconductor manufacturing invariants. This makes the benchmark interpretable: when the neural model predicts the next process step, the rule layer can validate whether the full route remains physically/process-logically plausible.”

---

## Csapatfelosztás

**Person A — ML / backend**

* loader
* dev split
* n-gram baseline
* transformer
* metrics
* Leonardo run

**Person B — product / demo / submission**

* Streamlit átalakítás
* REPORT.md
* slides
* demo script
* result tables
* repo cleanup
* no-secret check

Ne mindketten ugyanazt csináljátok. Most a legnagyobb hiba az lenne, ha mindketten “modellt próbáltok trainelni”, miközben nincs submission pipeline.

---

## Konkrét mai prioritási sorrend

**Most azonnal:**

```text
1. Industrial repo legyen a fő repo.
2. Official Infineon training_data bemásolása.
3. Mock RandomForest baseline félretétele.
4. Loader + dev split.
5. Rule-based anomaly validator bekötése.
6. N-gram next-step/completion baseline.
7. Submission writer: nextstep.csv, completion.csv, anomaly.csv.
```

**Utána:**

```text
8. Transformer train local small sample.
9. Leonardo smoke job.
10. Leonardo full small/medium training.
11. Streamlit demo: baseline vs trained.
12. REPORT.md + slides + 2 perces video.
```

## Amit ne csináljatok

Ne kezdjetek RAG/LLM chatbotot építeni.
Ne tuningoljátok a régi mock telemetry RandomForestöt.
Ne menjetek Leonardo-ra addig, amíg nincs lokálisan működő baseline + eval.
Ne commitoljatok credentialt, token linket, Discord invite-ot, secrets fájlt.
Ne legyen csak dashboard; legyen mérhető model output és official submission fájl.

## Egy mondatos stratégiai döntés

**Építsetek egy Infineon-specific “process route learner”-t: n-gram baseline + small transformer + explicit grammar validator + reproducible eval + Streamlit side-by-side demo.**

Ez a legnagyobb eséllyel működő, gyorsan implementálható és zsűri-kompatibilis út.

[1]: https://docs.hpc.cineca.it/hpc/leonardo.html "Leonardo — CINECA HPC Documentation 1.0 documentation"
[2]: https://ai-at.eu/hpc-onboarding/ "HPC Onboarding | AI Factory Austria AI:AT"
