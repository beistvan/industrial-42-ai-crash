from pathlib import Path


def test_required_files_exist():
    assert Path("README.md").exists()
    assert Path("REPORT.md").exists()
    assert Path("src/app/dashboard.py").exists()
    assert Path("docs/SUBMISSION.md").exists()
    assert Path("HANDOFF.md").exists()
