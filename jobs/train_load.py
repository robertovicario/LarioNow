# =========================
# Dependencies
# =========================

from google.cloud import storage
from loguru import logger
from pathlib import Path
import sys

ROOT_PATH = Path(__file__).resolve().parent
if ROOT_PATH.name in ["jobs", "notebook"]:
    ROOT_PATH = ROOT_PATH.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from lib import config as the_config

# =========================
# Configurations
# =========================

# GCP
GCS_CLIENT = storage.Client()
GCS_BUCKET = GCS_CLIENT.bucket(the_config.GCS_BUCKET)

# =========================
# Methods
# =========================

def load_artifacts() -> None:

    logger.info(
f"""\n
# =========================
# (2) LOADING
# =========================
"""
    )

    # -------------------------
    # GCP
    # -------------------------

    blobs = GCS_BUCKET.list_blobs(prefix=the_config.GCS_PREFIX_LATEST)
    for blob in blobs:
        blob.delete()

    for file_path in (the_config.MODELS_LATEST_PATH).iterdir():
        if file_path.is_file():

            blob = GCS_BUCKET.blob(f"{the_config.GCS_PREFIX_LATEST}{file_path.name}")
            blob.upload_from_filename(str(file_path))

# -------------------------
