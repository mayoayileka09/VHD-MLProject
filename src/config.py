from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True) 
class Paths:
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
    DATA_DIR: Path = PROJECT_ROOT / "data"
    REPORTS_DIR: Path = PROJECT_ROOT / "reports"
    FIGURES_DIR: Path = REPORTS_DIR / "figures"
    METRICS_DIR: Path = REPORTS_DIR / "metrics"
    MODELS_DIR: Path = PROJECT_ROOT / "models"

PATHS = Paths()