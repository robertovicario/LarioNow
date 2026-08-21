# =========================
# Dependencies
# =========================

from pathlib import Path
import sys

ROOT_PATH = Path(__file__).resolve().parent
if ROOT_PATH.name in ["jobs", "notebook"]:
    ROOT_PATH = ROOT_PATH.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from lib import config as the_config
from lib import utils as the_utils

from train_model import train_model
from train_load import load_artifacts

# =========================
# Configurations
# =========================

# Paths
for path in the_config.PATHS:
    the_utils.ensure_path(path)

# =========================
# Pipeline
# =========================

def run_pipeline() -> None:

    the_config.refresh_logging()
    train_model()
    load_artifacts()

if __name__ == "__main__":
    run_pipeline()

# -------------------------
