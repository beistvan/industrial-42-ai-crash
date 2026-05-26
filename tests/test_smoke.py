from pathlib import Path


def test_required_files_exist():
    assert Path('README.md').exists()
    assert Path('src/app/main.py').exists()
    assert Path('docs/MODEL_PLAN.md').exists()
