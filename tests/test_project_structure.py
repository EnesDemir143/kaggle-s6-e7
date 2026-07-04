from pathlib import Path
import subprocess


def test_required_workflow_directories_exist():
    root = Path(__file__).resolve().parents[1]
    required = ["scripts", "outputs/experiments", "submissions"]
    assert all((root / path).is_dir() for path in required)


def test_quality_script_contains_all_required_gates():
    root = Path(__file__).resolve().parents[1]
    check_script = (root / "scripts" / "check.py").read_text()
    assert all(
        gate in check_script for gate in ["pytest", "ruff", "mypy", "compileall"]
    )


def test_pipeline_shell_scripts_have_valid_bash_syntax():
    root = Path(__file__).resolve().parents[1]
    scripts = [
        root / "scripts" / "dry_run_pipeline.sh",
        root / "scripts" / "experiment_runner.sh",
    ]
    subprocess.run(["bash", "-n", *map(str, scripts)], check=True)
