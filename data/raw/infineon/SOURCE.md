# Infineon Industrial AI Track — vendored data

Source: https://github.com/Lumos-Data/zero_one_hack_01/tree/main/tracks/industrial-infineon/training_data
Pulled: 2026-05-29 (commit on `main` at time of fetch).
License / use: hackathon participants only, per track briefing.

## Layout

```
data/raw/infineon/
├── SOURCE.md                              ← this file
├── reference/                             ← canonical single-sequence references
│   ├── synthetic_mosfet.csv               126-step canonical MOSFET recipe
│   ├── syntheticIGBT.csv                  151-step canonical IGBT recipe
│   ├── syntheticIC.csv                    107-step canonical IC recipe
│   ├── MOSFET_Longdescr.csv               step + plain-text description
│   ├── IGBT_Longdescr.csv
│   ├── IC_Longdescr.csv
│   ├── MOSFET_longdescription_parameters.csv   + realistic fab-level parameters
│   ├── IGBT_longdescription_parameters.csv
│   └── IC_longdescription_parameters.csv
└── training_data/                         ← bulk sequences + tooling
    ├── MOSFET_variants.csv                1,000 sequences (long format)
    ├── IGBT_variants.csv                  1,000 sequences (long format)
    ├── IC_variants.csv                    1,000 sequences (long format)
    ├── generate_sequences.py              CLI: generate / validate / estimate
    ├── generation_rules.md                full grammar + 10 forbidden patterns + eval protocol
    └── README.md                          upstream data quickstart
```

## Data contract

- `_variants.csv` files: long format, two columns — `SEQUENCE_ID, STEP` (one row per step).
- `reference/synthetic*.csv` files: single column `STEP`, one canonical sequence per family.
- Vocabulary: ~120 distinct step tokens across all three families.
- Every sequence starts with `RECEIVE WAFER LOT` and ends with `SHIP LOT`.

## Refresh

To re-pull (e.g. if upstream updates during the hackathon):

```bash
BASE="https://raw.githubusercontent.com/Lumos-Data/zero_one_hack_01/main/tracks/industrial-infineon/training_data"
cd data/raw/infineon
for f in synthetic_mosfet.csv syntheticIGBT.csv syntheticIC.csv \
         MOSFET_Longdescr.csv IGBT_Longdescr.csv IC_Longdescr.csv \
         MOSFET_longdescription_parameters.csv \
         IGBT_longdescription_parameters.csv \
         IC_longdescription_parameters.csv; do
  curl -fsSL "$BASE/$f" -o "reference/$f"
done
for f in MOSFET_variants.csv IGBT_variants.csv IC_variants.csv \
         generate_sequences.py generation_rules.md README.md; do
  curl -fsSL "$BASE/$f" -o "training_data/$f"
done
```

## Eval inputs (not included)

The two eval input files — `eval_input_valid.csv` (Tasks 1 & 2) and
`eval_input_anomaly.csv` (Task 3) — are distributed by the organizers at
hackathon kickoff and are **not** in this checkout. Drop them into
`data/raw/infineon/eval/` when received.
