# =========================
# Dependencies
# =========================

from glob import glob
from loguru import logger
import numpy as np
from pathlib import Path
from paddleocr import TextRecognition
from typing import Any
import cv2
import os
import paddle
import sys

ROOT_PATH = Path(__file__).resolve().parent
if ROOT_PATH.name in ["jobs", "notebook"]:
    ROOT_PATH = ROOT_PATH.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from lib import config as the_config
from lib import utils as the_utils

# -------------------------

# Settings
paddle.disable_signal_handler()

# =========================
# Configurations
# =========================

# Models
MODEL_OCR = TextRecognition(
    model_name=the_config.CV_OCR_MODEL,
    device=the_config.CV_DEVICE
)

# =========================
# Methods
# =========================

def transform_data(
    stations: dict[str, dict[str, Any]],
    verbose: bool = False
) -> list[dict[str, Any]]:

    logger.info(
f"""\n
# =========================
# (2) TRANSFORMATION
# =========================
"""
    )

    # TMP Images (1)
    images = sorted(glob(os.path.join(the_config.TMP_IMG_PATH, "*.png")))
    total_img = len(images)

    # -------------------------

    new_data = []
    for i, img_file in enumerate(images, start=1):

        # Filename
        filename = os.path.basename(img_file)
        station, time_features = the_utils.parse_filename(filename)
        metadata = stations[station]
        city = metadata["city"]
        province = metadata["province"]
        latitude = metadata["latitude"]
        longitude = metadata["longitude"]
        logger.debug(f"{the_config.LOG_TIMESTAMP} [{i:02d}/{total_img:02d}]")
        logger.debug(f"{the_config.LOG_TIMESTAMP} STATION: {city}")

        # -------------------------

        try:

            # OpenCV Image
            img_cv = cv2.imread(img_file)
            if img_cv is None:
                logger.error(f"{the_config.LOG_TIMESTAMP} Unable to read img_cv: {filename}\n")
                continue

            img_cv = cv2.resize(
                img_cv,
                None,
                fx=2,
                fy=2,
                interpolation=cv2.INTER_LINEAR
            )

        except Exception as e:
            logger.error(f"{the_config.LOG_TIMESTAMP} {str(e)}\n")
            continue

        # -------------------------

        try:

            # ROIs Extraction (1)
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(
                hsv,
                np.array([35, 50, 50]),
                np.array([85, 255, 255]),
            )
            blue_mask = cv2.inRange(
                hsv,
                np.array([90, 50, 50]),
                np.array([130, 255, 255]),
            )

            # -------------------------

            # ROIs Extraction (2)
            green_rois = the_utils.extract_rois(img_cv, green_mask, "green")
            blue_rois  = the_utils.extract_rois(img_cv, blue_mask, "blue")
            all_rois = sorted(green_rois + blue_rois, key=lambda r: r["bbox"][0])

            # -------------------------

            # OCR Prediction
            pieces = the_utils.get_pieces(img_cv, green_mask, all_rois)
            values = the_utils.ocr_predict(
                MODEL_OCR,
                the_config.CV_OCR_FIELDS,
                pieces,
                verbose=verbose
            )

        except Exception as e:
            logger.error(f"{the_config.LOG_TIMESTAMP} {str(e)}\n")
            continue

        # -------------------------

        try:

            # Data Appending
            records = {
                "date": time_features["date"],
                "year": time_features["year"],
                "month": time_features["month"],
                "day": time_features["day"],
                "hour": time_features["hour"],
                "minute": time_features["minute"],
                "quarter": time_features["quarter"],
                "week_of_year": time_features["week_of_year"],
                "day_of_year": time_features["day_of_year"],
                "day_of_week": time_features["day_of_week"],
                "station": station,
                "city": city,
                "province": province,
                "latitude": latitude,
                "longitude": longitude
            }
            fields = {
                "temperature_c": the_utils.parse_float,
                "humidity_pct": the_utils.parse_int,
                "dew_point_c": the_utils.parse_float,
                "wind_speed_kmh": the_utils.parse_float,
                "wind_dir": the_utils.normalize_wind_dir,
                "pressure_hpa": the_utils.parse_float,
                "rain_mm": the_utils.parse_float,
                "rain_mmh": the_utils.parse_float
            }
            conf_fields = {
                "temperature_c": "conf_temperature_c",
                "humidity_pct": "conf_humidity_pct",
                "dew_point_c": "conf_dew_point_c",
                "wind_speed_kmh": "conf_wind_speed_kmh",
                "wind_dir": "conf_wind_dir",
                "pressure_hpa": "conf_pressure_hpa",
                "rain_mm": "conf_rain_mm",
                "rain_mmh": "conf_rain_mmh"
            }

            for field, parser in fields.items():

                try:
                    pred = values[field]
                    records[field] = parser(pred["rec_text"])
                    records[conf_fields[field]] = float(pred["rec_score"])
                except Exception as e:
                    records[field] = None
                    records[conf_fields[field]] = None
                    logger.warning(f"{the_config.LOG_TIMESTAMP} {str(e)}")
            new_data.append(records)

            rec_scores = [
                records[conf]
                for conf in conf_fields.values()
                if records[conf] is not None
            ]
            records["conf_overall_gmean"] = (
                round(np.exp(np.mean(np.log(rec_scores))), 3)
                if rec_scores else None
            )
        except Exception as e:
            logger.error(f"{the_config.LOG_TIMESTAMP} {str(e)}\n")
            continue
        logger.success(f"{the_config.LOG_TIMESTAMP} TRANSFORMED: {filename}\n")

    # -------------------------

    return new_data

# -------------------------
