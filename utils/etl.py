# =========================
# Dependencies
# =========================

from datetime import datetime
from difflib import SequenceMatcher
from loguru import logger
import cv2
import numpy as np
import os
import re

from config import config as the_config

# =========================
# Methods
# =========================

def extract_rois(image, mask, roi_type):

    KERNEL_SIZE = (15, 3)
    PAD = 3

    # -------------------------

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_SIZE)
    mask = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # -------------------------

    rois = []
    for cnt in contours:

        x, y, w, h = cv2.boundingRect(cnt)
        if w < 15 or h < 15:
            continue

        x1 = max(0, x - PAD)
        y1 = max(0, y - PAD)
        x2 = min(image.shape[1], x + w + PAD)
        y2 = min(image.shape[0], y + h + PAD)
        rois.append({
            "type": roi_type,
            "bbox": (x1, y1, x2, y2),
            "image": image[y1:y2, x1:x2]
        })

    # -------------------------

    return rois

def get_pieces(image, mask, all_rois):

    def create_piece(group):
        x1 = min(r["bbox"][0] for r in group)
        y1 = min(r["bbox"][1] for r in group)
        x2 = max(r["bbox"][2] for r in group)
        y2 = max(r["bbox"][3] for r in group)

        return {
            "bbox": (x1, y1, x2, y2),
            "image": image[y1:y2, x1:x2].copy(),
            "green_mask": mask[y1:y2, x1:x2].copy(),
        }

    # -------------------------

    pieces = []
    current_green = []
    block_index = 0

    for roi in all_rois:

        if roi["type"] == "green":
            current_green.append(roi)
            continue

        if not current_green:
            continue

        block_index += 1
        if block_index == 5 and len(current_green) >= 2:
            gaps = [
                current_green[i + 1]["bbox"][0] - current_green[i]["bbox"][2]
                for i in range(len(current_green) - 1)
            ]

            split_idx = gaps.index(max(gaps)) + 1
            groups = (
                current_green[:split_idx],
                current_green[split_idx:],
            )
        else:
            groups = (current_green,)

        pieces.extend(create_piece(group) for group in groups)
        current_green = []

    # -------------------------

    if current_green:

        block_index += 1
        if block_index == 5 and len(current_green) >= 2:
            gaps = [
                current_green[i + 1]["bbox"][0] - current_green[i]["bbox"][2]
                for i in range(len(current_green) - 1)
            ]

            split_idx = gaps.index(max(gaps)) + 1
            groups = (
                current_green[:split_idx],
                current_green[split_idx:],
            )
        else:
            groups = (current_green,)
        pieces.extend(create_piece(group) for group in groups)

    # -------------------------

    return pieces

def ocr_predict(ocr_model, ocr_fields, pieces, verbose=False):

    values = {}
    for field, piece in zip(ocr_fields, pieces):

        img = piece["image"]
        img = cv2.resize(
            img,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_LINEAR
        )
        img = isolate_text(
            img,
            (35, 40, 40),
            (90, 255, 255),
            1
        )

        ocr_pred = ocr_model.predict(img)
        rec_text = str(ocr_pred[0]["rec_text"])
        rec_score = float(ocr_pred[0]["rec_score"])
        values[field] = {
            "rec_text": rec_text,
            "rec_score": rec_score
        }

        if verbose:
            label = f"{field}:".rjust(15)
            logger.debug(f"{label}{rec_text:>7} (confidence={rec_score:.3f})")

    # -------------------------

    return values

def isolate_text(img, hsv_lower, hsv_upper, dilate_iterations=0):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array(hsv_lower, dtype=np.uint8),
        np.array(hsv_upper, dtype=np.uint8)
    )

    if dilate_iterations > 0:
        kernel = np.ones((2, 2), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilate_iterations)

    isolated = np.full_like(img, 255)
    isolated[mask > 0] = (0, 0, 0)

    # -------------------------

    return isolated

def parse_filename(filename):

    station, date_str, hm = os.path.splitext(filename)[0].rsplit("-", 2)
    time_features = {
        "date": datetime.strptime(date_str, "%Y%m%d").date().isoformat(),
        "year": int(date_str[:4]),
        "month": int(date_str[4:6]),
        "day": int(date_str[6:]),
        "hour": int(hm[:2]),
        "minute": int(hm[2:]),
        "quarter": (int(hm[:2]) // 3) + 1,
        "week_of_year": datetime.strptime(date_str, "%Y%m%d").isocalendar()[1],
        "day_of_year": datetime.strptime(date_str, "%Y%m%d").timetuple().tm_yday,
        "day_of_week": datetime.strptime(date_str, "%Y%m%d").isocalendar()[2],
    }

    # -------------------------

    return station, time_features

def parse_int(text):

    text = str(text).replace("O", "0").replace("o", "0")
    match = re.search(r"\d+", text)

    if not match:
        return None

    # -------------------------

    return int(match.group())

def parse_float(text):

    text = str(text).replace("O", "0").replace("o", "0")
    text = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", text)
    text = text.replace(",", ".").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text)

    if not match:
        return None

    # -------------------------

    return float(match.group())

def normalize_wind_dir(text):

    if isinstance(text, dict):
        text = text.get("rec_text")

    if text is None:
        return None

    value = str(text).upper()
    value = re.sub(r"[^A-Z]", "", value)

    if not value:
        return None

    if value in the_config.FENG_WIND_DIR_MAP:
        return value

    best = max(
        the_config.FENG_WIND_DIR_MAP,
        key=lambda choice: SequenceMatcher(None, value, choice).ratio()
    )
    score = SequenceMatcher(None, value, best).ratio()

    # -------------------------

    return best if score >= 0.6 else None

# -------------------------
