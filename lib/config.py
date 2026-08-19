# =========================
# Dependencies
# =========================

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import os

# =========================
# Configurations
# =========================

# Web Scraping
N_RETRY = 10
BASE_URL = "http://www.centrometeolombardo.com/"
STATIONS_URL = f"{BASE_URL}/content.asp?CatId=273&ContentType=Stazioni"
IMG_URL = "http://rete.centrometeolombardo.com/{province}/{tag}/immagini/v.png"

# -------------------------

# Paths
ROOT_PATH = Path(__file__).resolve().parents[1]
MODELS_PATH = ROOT_PATH / "models"
MODELS_NB_PATH = MODELS_PATH / "notebook"
MODELS_LATEST_PATH = MODELS_PATH / "latest"
NB_OUT_PATH = ROOT_PATH / "notebook/out"
TMP_IMG_PATH = ROOT_PATH / "tmp/img"
LOCATIONS_JSON = ROOT_PATH / "config/locations.json"
STATIONS_JSON = ROOT_PATH / "config/stations.json"
PATHS = [
    MODELS_PATH,
    MODELS_NB_PATH,
    MODELS_LATEST_PATH,
    NB_OUT_PATH,
    TMP_IMG_PATH,
    LOCATIONS_JSON,
    STATIONS_JSON
]

# -------------------------

# Artifacts
LOCATIONS = {}
with open(LOCATIONS_JSON, "r") as f:
    LOCATIONS = json.load(f)

STATIONS = {}
with open(STATIONS_JSON, "r") as f:
    STATIONS = json.load(f)

# -------------------------

# Logging
LOCAL_TIMEZONE = ZoneInfo("Europe/Rome")
def refresh_logging():

    global NOW, ACTUAL_TIME, YEAR, MONTH, DAY, HOUR, MINUTE, LOG_TIMESTAMP
    NOW = datetime.now(LOCAL_TIMEZONE)

    rounded_min = ((NOW.minute + 2) // 5) * 5
    if rounded_min == 60:
        rounded_min = 0
        NOW = NOW + timedelta(hours=1)

    YEAR = NOW.strftime("%Y")
    MONTH = NOW.strftime("%m")
    DAY = NOW.strftime("%d")
    HOUR = NOW.strftime("%H")
    MINUTE = f"{rounded_min:02d}"
    ACTUAL_TIME = f"{YEAR}{MONTH}{DAY}-{HOUR}{MINUTE}"
    LOG_TIMESTAMP = f"[{YEAR}-{MONTH}-{DAY} @ {HOUR}:{MINUTE}]"
refresh_logging()

# -------------------------

# Google Cloud Platform (GCP)
GCP_PROJECT = os.getenv("GCP_PROJECT", "uninsubria-data-science")
BQ_DATASET = os.getenv("BQ_DATASET", "larionow_dataset")
BQ_TABLE_NAME = os.getenv("BQ_TABLE_NAME", "measurements")
BQ_TABLE = f"{GCP_PROJECT}.{BQ_DATASET}.{BQ_TABLE_NAME}"
GCS_BUCKET = os.getenv("GCS_BUCKET", "uninsubria-data-science-models")
GCS_PREFIX_MODELS = os.getenv("GCS_PREFIX_MODELS", "models/")
if not GCS_PREFIX_MODELS.endswith("/"):
    GCS_PREFIX_MODELS = f"{GCS_PREFIX_MODELS}/"
GCS_PREFIX_LATEST = f"{GCS_PREFIX_MODELS}latest/"

# -------------------------

# Feature Engineering
WIND_DIR_MAP = {
    "N": 0,
    "NNE": 22.5,
    "NE": 45,
    "ENE": 67.5,
    "E": 90,
    "ESE": 112.5,
    "SE": 135,
    "SSE": 157.5,
    "S": 180,
    "SSW": 202.5,
    "SW": 225,
    "WSW": 247.5,
    "W": 270,
    "WNW": 292.5,
    "NW": 315,
    "NNW": 337.5
}
LAGS = [1, 2, 3, 6, 12, 18, 24]
ROLLING_WINDOWS = [6, 12, 24]
FORECASTS = [30, 60, 90, 120]
HOLDOUT_MIN = 120
SAMPLING_MIN = 5
HOLDOUT_STEPS = HOLDOUT_MIN // SAMPLING_MIN

# Machine Learning
N_COLLECTION = 5472
N_SUBSAMPLING = 0  # 0 if no subsampling
HYPER_SPACE = {
	"sampling": N_COLLECTION * N_SUBSAMPLING,
    "automl_clf": {
		"time_budget": 60,
		"metric": "roc_auc"
	},
    "automl_reg": {
        "time_budget": 10,
		"metric": "rmse"
	}
}
CLASSIFICATION = {
    "targets": [
        "rain_flag"
	],
    "test_size": 0.2,
    "shuffle": True,
    "seed": 42
}
REGRESSION = {
    "targets" : [
		"temperature_c",
		"humidity_pct",
		"dew_point_c",
		"pressure_hpa",
		"wind_x",
		"wind_y"
	],
	"seed": 42
}

# Inference
N_STATIONS = (
    sum(len(stations)
    for stations in STATIONS.values())
)
INF_ROWS = N_STATIONS * HOLDOUT_STEPS

# Computer Vision
DEVICE = "cpu"
OCR_MODEL = "PP-OCRv5_server_rec"
OCR_FIELDS = [
    "temperature_c",
    "humidity_pct",
    "dew_point_c",
    "wind_speed_kmh",
    "wind_dir",
    "pressure_hpa",
    "rain_mm",
    "rain_mmh"
]

# -------------------------
