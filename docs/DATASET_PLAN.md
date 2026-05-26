# Dataset Plan — Industrial: Models that learn how processes unfold

## Preferred source
Event-provided dataset or case material.

## Fallback source
NASA turbofan, SECOM manufacturing, semiconductor process logs if provided, synthetic event logs

## Required data contract
Create `src/data/schema.py` with:
- input columns;
- target column;
- timestamp or sequence id if relevant;
- train/validation/test split strategy;
- leakage risks.

## Synthetic fallback
Synthetic data is allowed only to prove the pipeline before the event or as a clearly labeled fallback. Do not pretend it is real partner data.

## Minimum viable dataset size
- Demo mode: 100–1,000 rows/sequences.
- Real evaluation mode: as much as event data allows.
- GPU training mode: scale only after baseline and split are correct.
