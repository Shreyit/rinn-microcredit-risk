"""Project-wide paths and constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

POSITIVE_LABEL = 1  # default = missed repayment within 12 months
