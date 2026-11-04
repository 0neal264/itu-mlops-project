
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"

IMAGE_DATA_DIR = DATA_DIR / "images" / "raw"
METADATA_PATH = DATA_DIR / "images" / "metadata.csv"

TEST_DATA_DIR = DATA_DIR / "test"
TEST_METADATA_PATH = TEST_DATA_DIR / "metadata.csv"

MODELS_DIR = PROJ_ROOT / "models"
ARTIFACTS_DIR = PROJ_ROOT / "artifacts"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MLFLOW_STATE_DIR = PROJ_ROOT / "mlflow-state"

try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

except ModuleNotFoundError:
    pass
