# =========================
# Dependencies
# =========================

from config import config as the_config

from data_extract import extract_data
from data_transform import transform_data
from data_load import load_data

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
    stations = extract_data()
    new_data = transform_data(stations, verbose=False)
    load_data(new_data)

# =========================
# Entry Point
# =========================

if __name__ == "__main__":
    run_pipeline()

# -------------------------
