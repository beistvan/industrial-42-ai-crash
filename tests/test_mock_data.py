from pathlib import Path
import json
import pandas as pd


def test_mock_data_and_metrics_exist():
    raw_files = list(Path('data/raw').glob('mock_*'))
    assert raw_files, 'Expected at least one generated mock data file in data/raw/. Run make generate-mock-data.'
    metrics_path = Path('artifacts/metrics.json')
    assert metrics_path.exists(), 'Expected artifacts/metrics.json. Run make train-baseline.'
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    assert metrics.get('rows', 0) > 0
    csv_files = [p for p in raw_files if p.suffix == '.csv']
    assert csv_files, 'Expected at least one CSV mock data file.'
    df = pd.read_csv(csv_files[0])
    assert len(df) > 10
