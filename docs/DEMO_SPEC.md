# Demo Spec — Industrial: Models that learn how processes unfold

## Frontend
Streamlit process replay: timeline, predicted next state, anomaly score, bottleneck root-cause candidates

## Demo must show
1. User chooses or enters a scenario.
2. App shows model/agent output.
3. App shows confidence or uncertainty.
4. App shows why the output matters.
5. App shows evaluation evidence.

## Backup plan
If live app breaks, show:
- local screenshots;
- metrics table;
- saved sample predictions;
- 90-second screen recording.

## Do not show
- raw notebooks as the main demo;
- huge logs;
- claims without metrics;
- hidden manual intervention.

---

# Demo spec — Industrial: Models that learn how processes unfold

The demo is a Streamlit app run by:

```bash
make run-demo
```

## Demo purpose

Show that the track has a working end-to-end path:

1. data loaded;
2. baseline metric evidence shown;
3. scenario input accepted;
4. recommendation/decision generated;
5. confidence or explanation displayed;
6. data preview visible.

## Current demo flow

process scenario input -> normal/watch/anomaly decision -> expected next step -> confidence/explanation -> metric evidence -> data preview.

## Required sections in the page

- Title and one-sentence case framing.
- Baseline evidence from `artifacts/metrics.json`.
- Scenario controls with safe defaults.
- Recommendation/decision output.
- Confidence or human-review/uncertainty signal.
- Explanation/policy trace.
- Data sample table.
- Note that mock data will be replaced by event data after reveal.

## Tomorrow adaptation checklist

- [ ] Rename labels to match the official case.
- [ ] Replace synthetic CSV with real data source.
- [ ] Keep the same page sections even if the model changes.
- [ ] Show baseline vs improved metrics if an improved model is added.
- [ ] Keep a simple fallback rule so the demo never goes blank.

## Demo script, 45 seconds

1. “We start with a tested scaffold: loader, baseline, metrics, and demo.”
2. “Here is the real/scenario input.”
3. “The system returns this decision and confidence/explanation.”
4. “The metric card shows the baseline evidence.”
5. “The next improvement is targeted from the error analysis.”
