from pathlib import Path


def test_required_workflow_directories_exist():
    root = Path(__file__).resolve().parents[1]
    required = ["scripts", "experiments", "submissions"]
    assert all((root / path).is_dir() for path in required)


def test_quality_script_contains_all_required_gates():
    root = Path(__file__).resolve().parents[1]
    check_script = (root / "scripts" / "check.py").read_text()
    assert all(gate in check_script for gate in ["pytest", "ruff", "mypy", "compileall"])
