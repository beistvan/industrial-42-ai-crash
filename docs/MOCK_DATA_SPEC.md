# Mock Data Spec — Industrial Sequence Model

This repo includes synthetic pre-hack data so the pipeline can run before sponsor data arrives.

File: `data/raw/mock_process_sequences.csv`

Columns:
- `run_id`, `recipe`, `step_index`, `step_name`, `timestamp`
- `temperature`, `pressure`, `vibration`, `tool_wear`, `duration_sec`
- `next_step`, `anomaly`, `quality_label`

Use it for sequence-model API shape, anomaly baseline, UI wiring, and evaluation rehearsals. Replace it with real process logs/sensor traces when provided.
