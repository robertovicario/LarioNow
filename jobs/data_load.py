# =========================
# Dependencies
# =========================

from google.api_core.exceptions import NotFound
from google.cloud import bigquery
from loguru import logger
from pathlib import Path
from typing import Any
import shutil
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent
if PROJECT_ROOT.name == "jobs":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib import config as the_config

# =========================
# Configurations
# =========================

# GCP
BQ_CLIENT = bigquery.Client(project=the_config.GCP_PROJECT)

# =========================
# Methods
# =========================

def load_data(
    new_data: list[dict[str, Any]]
) -> None:

    logger.info(
f"""\n
# =========================
# (3) LOADING
# =========================
"""
    )

    dataset_ref = bigquery.DatasetReference(
        the_config.GCP_PROJECT,
        the_config.BQ_DATASET
    )
    try:

        # GET OPS (1)
        BQ_CLIENT.get_dataset(dataset_ref)

    except NotFound:

        # CREATE OPS (1)
        logger.warning(f"{the_config.LOG_TIMESTAMP} Dataset not found: {the_config.BQ_DATASET}")
        dataset_ref.location = "EU"
        BQ_CLIENT.create_dataset(dataset_ref)
        logger.success(f"{the_config.LOG_TIMESTAMP} CREATED: {the_config.BQ_DATASET}")

    try:

        # GET OPS (2)
        BQ_CLIENT.get_table(the_config.BQ_TABLE)

    except NotFound:

        # CREATE OPS (2)
        logger.warning(f"{the_config.LOG_TIMESTAMP} Table not found: {the_config.BQ_TABLE}")
        table = bigquery.Table(
            the_config.BQ_TABLE,
            schema=[
                bigquery.SchemaField("date", "DATE"),
                bigquery.SchemaField("year", "INTEGER"),
                bigquery.SchemaField("month", "INTEGER"),
                bigquery.SchemaField("day", "INTEGER"),
                bigquery.SchemaField("hour", "INTEGER"),
                bigquery.SchemaField("minute", "INTEGER"),
                bigquery.SchemaField("quarter", "INTEGER"),
                bigquery.SchemaField("week_of_year", "INTEGER"),
                bigquery.SchemaField("day_of_year", "INTEGER"),
                bigquery.SchemaField("day_of_week", "INTEGER"),
                bigquery.SchemaField("station", "STRING"),
                bigquery.SchemaField("city", "STRING"),
                bigquery.SchemaField("province", "STRING"),
                bigquery.SchemaField("latitude", "FLOAT"),
                bigquery.SchemaField("longitude", "FLOAT"),
                bigquery.SchemaField("temperature_c", "FLOAT"),
                bigquery.SchemaField("conf_temperature_c", "FLOAT"),
                bigquery.SchemaField("humidity_pct", "INTEGER"),
                bigquery.SchemaField("conf_humidity_pct", "FLOAT"),
                bigquery.SchemaField("dew_point_c", "FLOAT"),
                bigquery.SchemaField("conf_dew_point_c", "FLOAT"),
                bigquery.SchemaField("wind_speed_kmh", "FLOAT"),
                bigquery.SchemaField("conf_wind_speed_kmh", "FLOAT"),
                bigquery.SchemaField("wind_dir", "STRING"),
                bigquery.SchemaField("conf_wind_dir", "FLOAT"),
                bigquery.SchemaField("pressure_hpa", "FLOAT"),
                bigquery.SchemaField("conf_pressure_hpa", "FLOAT"),
                bigquery.SchemaField("rain_mm", "FLOAT"),
                bigquery.SchemaField("conf_rain_mm", "FLOAT"),
                bigquery.SchemaField("rain_mmh", "FLOAT"),
                bigquery.SchemaField("conf_rain_mmh", "FLOAT"),
                bigquery.SchemaField("conf_overall_gmean", "FLOAT")
            ],
        )
        BQ_CLIENT.create_table(table)
        for _ in range(the_config.N_RETRY):
            try:
                BQ_CLIENT.get_table(table.reference)
                break
            except NotFound:
                time.sleep(1)
        logger.success(f"{the_config.LOG_TIMESTAMP} CREATED: {the_config.BQ_TABLE}")

    if not new_data:
        logger.warning(f"{the_config.LOG_TIMESTAMP} No new data")
        return

    # -------------------------

    try:

        # INSERT OPS
        BQ_CLIENT.get_table(the_config.BQ_TABLE)
        errors = BQ_CLIENT.insert_rows_json(
            the_config.BQ_TABLE,
            new_data
        )

        if errors:
            logger.error(f"{the_config.LOG_TIMESTAMP} {str(errors)}")
            raise RuntimeError(str(errors))

    except Exception as e:
        logger.error(f"{the_config.LOG_TIMESTAMP} {str(e)}")
        raise

    # -------------------------

    # TMP Images (2)
    shutil.rmtree(the_config.TMP_IMG_PATH)
    logger.debug(f"{the_config.LOG_TIMESTAMP} ROWS: {len(new_data)}")
    logger.success(f"{the_config.LOG_TIMESTAMP} LOADED: {the_config.BQ_TABLE}")

# -------------------------
