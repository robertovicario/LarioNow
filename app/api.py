# =========================
# Dependencies
# =========================

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from google.cloud import bigquery, storage

import io
import joblib
import json
import pandas as pd

from config import config as the_config
from utils import pipeline as the_pipeline

# =========================
# Methods
# =========================

def load_data():
    return (
        bigquery
        .Client(project=the_config.GCP_PROJECT)
        .query(
            f"""
                SELECT * FROM `{the_config.BQ_TABLE}`
                ORDER BY TIMESTAMP(
                    DATETIME(year, month, day, hour, minute, 0)
                ) DESC
                LIMIT {the_config.INF_ROWS * 2}
            """
        ).to_dataframe()
    )

def load_models():

    # GCP
    GCS_BUCKET = (
        storage
        .Client(project=the_config.GCP_PROJECT)
        .bucket(the_config.GCS_BUCKET)
    )

    # Model -- Classification
    clf_blob = next(
        blob for blob in GCS_BUCKET.list_blobs(
            prefix=the_config.GCS_PREFIX_LATEST
        )
        if blob.name.endswith("_classifier.joblib")
    )
    clf = joblib.load(
        io.BytesIO(clf_blob.download_as_bytes())
    )

    # Model -- Regression
    reg_blob = next(
        blob for blob in GCS_BUCKET.list_blobs(
            prefix=the_config.GCS_PREFIX_LATEST
        )
        if blob.name.endswith("_regressor.joblib")
    )
    reg = joblib.load(
        io.BytesIO(reg_blob.download_as_bytes())
    )

    # -------------------------

    return clf, reg

def build_results_df():

    inf_df = load_data()
    clf, reg = load_models()

    clf_df = inf_df.copy()
    clf_df, _ = the_pipeline.feature_engineering_clf(
        inf_df, inference=True
    )
    clf_df = clf_df.assign(
        timestamp=lambda x: pd.to_datetime(
            x[["year", "month", "day", "hour", "minute"]]
        )
    )

    reg_df = inf_df.copy()
    reg_df, _ = the_pipeline.feature_engineering_reg(
        inf_df, inference=True
    )
    reg_df = reg_df.assign(
        timestamp=lambda x: pd.to_datetime(
            x[["year", "month", "day", "hour", "minute"]]
        )
    )

    # -------------------------

    return the_pipeline.exec_inference(clf, reg, clf_df, reg_df)

# =========================
# FastAPI
# =========================

# App
app = FastAPI(
    title="LarioNow API",
    version="1.0.0"
)

# =========================
# Endpoints
# =========================

@app.get("/")
def health():
    return {"status": "ok"}

@app.get("/results")
def results():
    return JSONResponse(
        content=json.loads(
            build_results_df().to_json(
                orient="records",
                date_format="iso",
                indent=4
            )
        )
    )

# -------------------------
