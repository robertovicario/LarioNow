# =========================
# Dependencies
# =========================

from loguru import logger
from pathlib import Path
from urllib.parse import quote
import requests
import sys
import time

ROOT_PATH = Path(__file__).resolve().parent
if ROOT_PATH.name in ["jobs", "notebook"]:
    ROOT_PATH = ROOT_PATH.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from lib import config as the_config

# =========================
# Methods
# =========================

def extract_data(
    n: int = 0
) -> dict[str, dict]:

    logger.info(
f"""\n
# =========================
# (1) EXTRACTION
# =========================
"""
    )

    session = requests.Session()
    links = [
        (province, tag, metadata)
        for province, province_stations in the_config.STATIONS.items()
        for tag, metadata in province_stations.items()
    ]

    total_img = len(links)
    if n <= 0 or n > total_img:
        n = total_img

    # -------------------------

    stations = {}
    for i, (province, tag, metadata) in enumerate(links[:n], start=1):

        try:

            # Image Extraction (1)
            city = metadata["city"]
            stations[tag] = {
                **metadata,
                "province": province,
            }
            logger.debug(f"{the_config.LOG_TIMESTAMP} [{i:02d}/{total_img:02d}]")
            logger.debug(f"{the_config.LOG_TIMESTAMP} STATION: {city}")

            img_url = the_config.IMG_URL.format(
                province=quote(province),
                tag=quote(tag)
            )
            img_file = (
                the_config.TMP_IMG_PATH
                / f"{tag.lower()}-{the_config.ACTUAL_TIME}.png"
            )

            # -------------------------

            # Image Extraction (2)
            r = None
            for _ in range(the_config.N_RETRY):
                try:
                    r = session.get(img_url, timeout=30)
                    break
                except requests.exceptions.Timeout:
                    if _ == (the_config.N_RETRY - 1):
                        raise
                    time.sleep(1)

            r.raise_for_status()
            with open(img_file, "wb") as f:
                f.write(r.content)
            logger.success(f"{the_config.LOG_TIMESTAMP} EXTRACTED: {img_file.name}\n")

        except requests.exceptions.RequestException as e:
            logger.error(f"{the_config.LOG_TIMESTAMP} {str(e)}\n")
            continue

        except Exception as e:
            logger.error(f"{the_config.LOG_TIMESTAMP} {str(e)}\n")
            continue

    # -------------------------

    return stations

# -------------------------
