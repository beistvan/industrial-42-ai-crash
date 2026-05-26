from pathlib import Path
import json

required = [
    Path('README.md'),
    Path('src/app/main.py'),
    Path('docs/MODEL_PLAN.md'),
    Path('docs/EVALUATION_PLAN.md'),
    Path('docs/MOCK_DATA_SPEC.md'),
    Path('artifacts/metrics.json'),
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(f'Missing required files: {missing}')
metrics = json.loads(Path('artifacts/metrics.json').read_text(encoding='utf-8'))
if 'rows' not in metrics or metrics['rows'] <= 0:
    raise SystemExit('metrics.json does not contain positive row count')
print('Smoke test passed: scaffold, mock data, baseline metric artifact all exist.')
