# =========================
# Dependencies
# =========================

from config import config as the_config

from train_model import train_model
from train_load import load_artifacts

# =========================
# Configurations
# =========================

# Paths
for path in the_config.PATHS:
    the_config.ensure_path(path)

# =========================
# Methods
# =========================

def run_pipeline() -> None:

    the_config.refresh_logging()
    train_model()
    load_artifacts()

# =========================
# Entry Point
# =========================

if __name__ == "__main__":
    run_pipeline()

# -------------------------
