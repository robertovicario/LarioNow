# =========================
# Dependencies
# =========================

from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from google.cloud import bigquery, storage
from io import BytesIO
from zipfile import ZipFile

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

    # Caching
    inf_df = load_data()
    clf, reg = load_models()

    # Feature Engineering -- Classification
    clf_df = inf_df.copy()
    clf_df, _ = the_pipeline.feature_engineering_clf(
        inf_df, inference=True
    )
    clf_df = clf_df.assign(
        timestamp=lambda x: pd.to_datetime(
            x[["year", "month", "day", "hour", "minute"]]
        )
    )

    # Feature Engineering -- Regression
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
    version="1.0.0",
    docs_url="/docs"
)

# =========================
# Endpoints
# =========================

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/api/settings/cities")
def get_cities():

    df = build_results_df()
    return JSONResponse(
        content=(
            df["city"]
            .unique()
            .tolist()
        )
    )

@app.get("/api/predictions")
def get_predictions(city: str = None):

    df = build_results_df()
    if city is not None:
        df = df[df["city"].str.casefold() == city.casefold()]

    # -------------------------

    return JSONResponse(
        content=json.loads(
            df.to_json(
                orient="records",
                date_format="iso",
                indent=4
            )
        )
    )

@app.get("/api/predictions/extremes")
def get_extremes_from_cities():

    # Results
    df = build_results_df()
    content = {}

    for target in [
        "temperature_c",
        "humidity_pct",
        "dew_point_c",
        "pressure_hpa",
        "wind_speed_kmh",
        "rain_proba"
    ]:

        series = df[target].dropna()
        min_idx = series.idxmin()
        max_idx = series.idxmax()
        content[target] = {
            "min": {
                "value": df.loc[min_idx, target],
                "city": df.loc[min_idx, "city"]
            },
            "max": {
                "value": df.loc[max_idx, target],
                "city": df.loc[max_idx, "city"]
            }
        }

    # -------------------------

    return JSONResponse(content=content)

@app.get("/api/predictions/changes")
def get_changes_for_city(city: str):

    # Filter by city
    df = build_results_df()
    df = (
        df[
            df["city"].str.casefold() == city.casefold()
        ]
        .sort_values("timestamp")
    )

    # Results
    content = {}
    for target in [
        "temperature_c",
        "humidity_pct",
        "dew_point_c",
        "pressure_hpa",
        "wind_speed_kmh",
        "rain_proba"
    ]:

        changes = df[target].diff()
        content[target] = {
            f"lead_{int(previous_lead)}_{int(current_lead)}": float(change)
            for previous_lead, current_lead, change in zip(
                df["lead"].iloc[:-1],
                df["lead"].iloc[1:],
                changes.iloc[1:]
            )
            if pd.notna(change)
        }

    # -------------------------

    return JSONResponse(content=content)

@app.get("/api/model/metrics")
def get_model_metrics():

    # GCP
    GCS_BUCKET = (
        storage
        .Client(project=the_config.GCP_PROJECT)
        .bucket(the_config.GCS_BUCKET)
    )

    # Results
    content = {}
    for blob in GCS_BUCKET.list_blobs(
            prefix=the_config.GCS_PREFIX_LATEST
        ):

        filename = blob.name.rsplit("/", 1)[-1]
        if filename.endswith(".json"):

            artifact_name = filename.removesuffix(".json")
            content[artifact_name] = json.loads(
                blob.download_as_bytes()
            )

    # -------------------------

    return JSONResponse(content=content)

@app.get("/api/model/artifacts")
def download_model_artifacts():

    # GCP
    GCS_BUCKET = (
        storage
        .Client(project=the_config.GCP_PROJECT)
        .bucket(the_config.GCS_BUCKET)
    )

    # Results
    content = {}
    for blob in GCS_BUCKET.list_blobs(
        prefix=the_config.GCS_PREFIX_LATEST
    ):

        filename = blob.name.rsplit("/", 1)[-1]
        if not filename.endswith(".json"):
            content[filename] = blob.download_as_bytes()

    # -------------------------

    # Download
    zip_buffer = BytesIO()
    with ZipFile(zip_buffer, "w") as zip_file:
        for filename, data in content.items():
            zip_file.writestr(filename, data)
    zip_buffer.seek(0)

    # -------------------------

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                "attachment; filename=artifacts.zip"
            )
        }
    )

# -------------------------
