from pathlib import Path
from typing import Final

BASE_DIR: Final[Path] = Path(__file__).parent.parent.parent

DATA_URL: Final[str] = (
    "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
)

# Paths
PLOTS_DIR: Final[Path] = BASE_DIR / "plots"
MODELS_DIR: Final[Path] = BASE_DIR / "models"

PLOT_QUALITY_PATH: Final[Path] = PLOTS_DIR / "quality_distribution.png"
PLOT_CORR_PATH: Final[Path] = PLOTS_DIR / "correlation_heatmap.png"
PLOT_FEATURES_PATH: Final[Path] = PLOTS_DIR / "feature_importances.png"

MODEL_PATH: Final[Path] = MODELS_DIR / "wine_rf.pkl"

# ML defaults
TEST_SIZE: Final[float] = 0.2
RANDOM_STATE: Final[int] = 42
N_ESTIMATORS: Final[int] = 200
MAX_DEPTH: Final[int | None] = None
