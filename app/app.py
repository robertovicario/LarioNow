# =========================
# Dependencies
# =========================

from datetime import datetime
from pathlib import Path
from google.cloud import bigquery, storage
import base64
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import sys

# Paths
ROOT_PATH = Path(__file__).resolve().parent
if ROOT_PATH.name in ["app", "jobs", "notebook"]:
    ROOT_PATH = ROOT_PATH.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from lib import config as the_config
from lib import utils as the_utils

# =========================
# Configurations
# =========================

RES_COLS = [
    "timestamp",
    "station", "city", "latitude", "longitude",
    "lead",
    "temperature_c",
    "humidity_pct",
    "dew_point_c",
    "pressure_hpa",
    "wind_x", "wind_y",
    "rain_proba"
]

# =========================
# Code
# =========================

@st.cache_data(ttl=300)
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

@st.cache_resource
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
    # display(clf)

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
    # display(reg)

    # -------------------------

    return clf, reg

def add_timestamp(df):

    return df.assign(
        timestamp=lambda x: pd.to_datetime(
            x[["year", "month", "day", "hour", "minute"]]
        )
    )

def drop_feature_columns(df, columns):

    return df.drop(columns=[c for c in columns if c in df.columns])

def build_results_df():

    inf_df = load_data()
    clf, reg = load_models()

    clf_df, _ = the_utils.feature_engineering_clf(
        inf_df.copy(), inference=True
    )
    clf_df = add_timestamp(clf_df)

    reg_df, _ = the_utils.feature_engineering_reg(
        inf_df.copy(), inference=True
    )
    reg_df = add_timestamp(reg_df)

    latest = (
        reg_df
        .sort_values(["station", "timestamp"])
        .groupby("station")
        .tail(1)
    )

    reg_drop = [
        "date",
        "province", "city", "station",
        "wind_speed_kmh", "wind_dir",
        "rain_mm", "rain_mmh",
        "timestamp",
    ]
    reg_drop += [c for c in reg_df.columns if c.lower().startswith("conf_")]
    X_latest_reg = drop_feature_columns(latest, reg_drop)

    reg_features = getattr(next(iter(reg.model_.values())), "feature_names_in_", None)
    if reg_features is not None:
        X_latest_reg = X_latest_reg[list(reg_features)]

    y_pred_reg = reg.predict(X_latest_reg)
    results_df = (
        y_pred_reg
        .rename_axis("row_index")
        .reset_index()
        .melt(id_vars="row_index", var_name="variable", value_name="value")
        .assign(
            lead=lambda x: x["variable"].str.extract(r"_lead_(\d+)$")[0].astype(int),
            variable=lambda x: x["variable"].str.replace(
                r"_lead_\d+$", "", regex=True
            )
        )
        .pivot(
            index=["row_index", "lead"],
            columns="variable",
            values="value"
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    station = latest[
        ["station", "city", "latitude", "longitude", "timestamp"]
    ].rename_axis("row_index").reset_index()
    results_df = results_df.merge(station, on="row_index")
    results_df["timestamp"] += pd.to_timedelta(results_df["lead"], unit="m")

    clf_features = getattr(clf.model_, "feature_names_in_", None)
    if clf_features is None:
        clf_drop = [
            "date", "year", "month", "day", "hour", "minute",
            "quarter", "week_of_year", "day_of_year", "day_of_week",
            "province", "city", "station",
            "wind_speed_kmh", "wind_dir",
            "rain_mm", "rain_mmh",
            "timestamp",
        ]
        clf_drop += [c for c in clf_df.columns if c.lower().startswith("conf_")]
        clf_features = drop_feature_columns(clf_df, clf_drop).columns.tolist()

    results_df["rain_proba"] = clf.predict_proba(results_df[list(clf_features)])[:, 1]
    results_df = results_df[RES_COLS]

    actual_df = (
        latest[
            [
                "timestamp",
                "station",
                "city",
                "latitude",
                "longitude",
                "temperature_c",
                "humidity_pct",
                "dew_point_c",
                "pressure_hpa",
                "wind_x",
                "wind_y",
                "rain_mm",
            ]
        ]
        .assign(
            lead=0,
            rain_proba=lambda x: x["rain_mm"].gt(0).astype(int),
        )
        .drop(columns=["rain_mm"])
    )

    return (
        pd.concat([actual_df, results_df], ignore_index=True)
        [RES_COLS]
        .sort_values(["city", "station", "lead"])
        .reset_index(drop=True)
    )

def build_stations_df():

    stations = []
    for _, tags in the_config.STATIONS.items():
        for _, info in tags.items():
            stations.append(
                {
                    "city": info["city"],
                    "province": info["province"],
                    "latitude": info["latitude"],
                    "longitude": info["longitude"],
                }
            )
    return pd.DataFrame(stations)

def main():

    # =========================
    # Settings
    # =========================

    logo_path = ROOT_PATH / "docs" / "theme" / "logo.svg"
    st.set_page_config(
        page_title="LarioNow",
        page_icon=logo_path,
        layout="wide"
    )

    # =========================
    # Configurations
    # =========================

    results_df = build_results_df()
    stations_df = build_stations_df()

    import base64
    logo_b64 = base64.b64encode(logo_path.read_bytes()).decode()

    st.markdown(
        f"""
        <div style="margin-bottom: 24px;display: flex;justify-content: center; align-items: center;">
            <div style="display: flex;align-items: center;gap: 24px;margin-bottom: 24px;">
                <img src="data:image/svg+xml;base64,{logo_b64}" style="width: 64px; height: 64px;">
                <h1 style="margin: 0;font-size: 32px;line-height: 1;">LarioNow</h1>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================
    # Map
    # =========================

    with st.container(border=True):

        st.subheader("🌦️ Lake Como Area — Meteorological Stations")
        col1, col2 = st.columns([1, 2])

        with col1:

            tmp_station = st.selectbox(
                f"Select station:",
                options=stations_df["city"].tolist(),
                index=0,
            )
            station = stations_df[stations_df["city"] == tmp_station].iloc[0]

            st.metric("Province", station["province"])
            st.metric("City", station["city"])
            st.metric("Latitude", f"{station['latitude']:.4f}°")
            st.metric("Longitude", f"{station['longitude']:.4f}°")

        with col2:
            st.map(
                stations_df.rename(
                    columns={
                        "latitude": "lat",
                        "longitude": "lon",
                    }
                ),
                width="stretch",
                height="stretch"
            )

    # =========================
    # Measurements
    # =========================

    meas_df = (
        results_df[results_df["city"] == tmp_station]
        .sort_values("lead")
        .head(5)
        .reset_index(drop=True)
    )

    def metric_values(column, multiplier=1):

        values = meas_df[column].to_numpy() * multiplier
        value = values[0]
        delta = round(values[1] - values[0], 2)
        chart_data = np.round(values, 2).tolist()
        return value, delta, chart_data

    temperature_value, temperature_delta, temperature_chart = metric_values(
        "temperature_c"
    )
    humidity_value, humidity_delta, humidity_chart = metric_values(
        "humidity_pct"
    )
    rain_value, rain_delta, rain_chart = metric_values(
        "rain_proba", multiplier=100
    )

    with st.container(border=True):

        st.subheader("📡 Actual Measurements -- From Sensor Data")
        row = st.container(horizontal=True)

        with row:
            st.metric(
                "Temperature",
                f"{temperature_value:.1f} °C",
                f"{temperature_delta:.1f} °C",
                icon="🌡️",
                chart_data=temperature_chart,
                chart_type="area",
                border=True
            )
            st.metric(
                "Humidity",
                f"{humidity_value:.0f} %",
                f"{humidity_delta:.0f} %",
                icon="💧",
                chart_data=humidity_chart,
                chart_type="area",
                border=True
            )
            st.metric(
                "Rain",
                f"{rain_value:.0f} %",
                f"{rain_delta:.0f} %",
                icon="🌧️",
                chart_data=rain_chart,
                chart_type="area",
                border=True
            )

    # =========================
    # Nowcasting
    # =========================

    with st.container(border=True):
    
        st.subheader("📊 Weather Nowcasting")
        row = st.container(horizontal=True)
        # with row:
        #     st.metric(
        #         "30m",
        #         f"{temperature_value:.1f} °C",
        #         temperature_delta,
        #         icon="🕒",
        #         chart_data=temperature_chart,
        #         chart_type="area",
        #         border=True
        #     )
        #     st.metric(
        #         "60m",
        #         f"{humidity_value:.0f} %",
        #         humidity_delta,
        #         icon="🕞",
        #         chart_data=humidity_chart,
        #         chart_type="area",
        #         border=True
        #     )
        #     st.metric(
        #         "90m",
        #         f"{rain_value:.0f} %",
        #         rain_delta,
        #         icon="🕓",
        #         chart_data=rain_chart,
        #         chart_type="area",
        #         border=True
        #     )
        #     st.metric(
        #         "120m",
        #         f"{rain_value:.0f} %",
        #         rain_delta,
        #         icon="🕟",
        #         chart_data=rain_chart,
        #         chart_type="area",
        #         border=True
        #     )

    # =========================
    # Table
    # =========================

    with st.container(border=True):

        st.subheader("📋 Raw Predictions")
        st.dataframe(
            (
                results_df[results_df["city"] == tmp_station]
                .sort_values(["city", "station", "lead"])
                .reset_index(drop=True)
            ),
            width="stretch",
            hide_index=True
        )

    # =========================
    # Footer
    # =========================

    with st.container(border=False):

        st.divider()
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="text-align: center;">
                Copyright &copy; {datetime.now().year} <a href="https://www.robertovicario.com" target="_blank"><strong>Roberto Vicario</strong></a>. All rights reserved.
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()

# -------------------------
