"""Project paths are resolved from this package, never from the current directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_WEEKLY_FILE = DATA_DIR / "test_45_factors_weekly.csv"
