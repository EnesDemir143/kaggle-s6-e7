"""Project-wide paths and analysis constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "playground-series-s6e7"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
TARGET_COL = "health_condition"
ID_COL = "id"
EXPECTED_TARGETS = {"at-risk", "fit", "unhealthy"}
RANDOM_STATE = 42
PLOT_SAMPLE_SIZE = 100_000


def ensure_report_dirs() -> None:
    """Create report output directories for persisted notebook artifacts."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
