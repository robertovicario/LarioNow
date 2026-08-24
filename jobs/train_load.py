# =========================
# Dependencies
# =========================

from google.cloud import storage
from loguru import logger

from config import config as the_config

# =========================
# Configurations
# =========================

# GCP
GCS_CLIENT = storage.Client(project=the_config.GCP_PROJECT)
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
