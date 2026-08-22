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
import plotly.graph_objects as go
import pydeck as pdk
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

    # -------------------------

    # Feature Engineering -- Classification
    clf_df = inf_df.copy()
    clf_df, _ = the_utils.feature_engineering_clf(
        inf_df, inference=True
    )
    clf_df = clf_df.assign(
        timestamp=lambda x: pd.to_datetime(
            x[["year", "month", "day", "hour", "minute"]]
        )
    )

    # -------------------------

    # Feature Engineering -- Regression
    reg_df = inf_df.copy()
    reg_df, _ = the_utils.feature_engineering_reg(
        inf_df, inference=True
    )
    reg_df = reg_df.assign(
        timestamp=lambda x: pd.to_datetime(
            x[["year", "month", "day", "hour", "minute"]]
        )
    )

    # -------------------------

    # Inference
    return the_utils.exec_inference(clf, reg, clf_df, reg_df)

def main():

    # =========================
    # Configurations
    # =========================

    # Settings
    st.set_page_config(
        page_title="LarioNow",
        page_icon=the_config.LOGO_SVG,
        layout="wide"
    )

    # Scripts
    stations_df = the_utils.build_stations_df()
    results_df = build_results_df()

    # =========================
    # Introduction
    # =========================

    logo_b64 = base64.b64encode(
        the_config.LOGO_SVG.read_bytes()
    ).decode()
    html_code = f"""
        <div style="margin-bottom: 24px;display: flex;justify-content: center; align-items: center;">
            <div style="display: flex;align-items: center;gap: 24px;margin-bottom: 24px;">
                <img src="data:image/svg+xml;base64,{logo_b64}" style="width: 64px; height: 64px;">
                <h1 style="margin: 0;font-size: 32px;line-height: 1;">LarioNow</h1>
            </div>
        </div>
    """

    with st.container():
        st.html(html_code)

    # =========================
    # Reference Station
    # =========================

    with st.container(border=True):
        st.subheader(
            ":material/map: Lake Como Area",
            anchor=False
        )

        col1, col2 = st.columns([1, 2])
        with col1:

            stations = stations_df["city"].tolist()
            if "selected_station" not in st.session_state:
                st.session_state.selected_station = "Como"

            tmp_station = st.selectbox(
                "Select the reference station:",
                options=stations,
                index=stations.index(st.session_state.selected_station),
                key="station_select",
            )
            st.session_state.selected_station = tmp_station
            station = stations_df[
                stations_df["city"] == st.session_state.selected_station
            ].iloc[0]

            st.metric(
                "Province", station["province"], icon=":material/location_city:"
            )
            st.metric(
                "City", station["city"], icon=":material/location_on:"
            )
            st.metric(
                "Latitude", f"{station['latitude']:.4f}°", icon=":material/my_location:"
            )
            st.metric(
                "Longitude", f"{station['longitude']:.4f}°", icon=":material/my_location:"
            )

        # -------------------------
        # Map
        # -------------------------

        with col2:

            map_df = stations_df
            map_df["selected"] = map_df["city"] == tmp_station
            layers = [
                pdk.Layer(
                    "ScatterplotLayer",
                    data=map_df[~map_df["selected"]],
                    get_position="[longitude, latitude]",
                    get_radius=500,
                    get_fill_color="[0, 100, 255, 200]",
                    pickable=True
                ),
                pdk.Layer(
                    "ScatterplotLayer",
                    data=map_df[map_df["selected"]],
                    get_position="[longitude, latitude]",
                    get_radius=500,
                    get_fill_color="[255, 0, 0, 255]",
                    pickable=True
                )
            ]
            view_state = pdk.ViewState(
                latitude=map_df["latitude"].mean(),
                longitude=map_df["longitude"].mean(),
                zoom=8.75
            )
            st.pydeck_chart(
                pdk.Deck(
                    layers=layers,
                    map_style=st.get_option("theme.base"),
                    initial_view_state=view_state,
                    tooltip={
                        "text": "{city}",
                        "style": {"color": "white"}
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

    def compute_actual_delta(column, multiplier=1):

        values = meas_df[column].to_numpy() * multiplier
        value = values[0]
        delta = round(values[1] - values[0], 2)
        chart_data = np.round(values, 2).tolist()
        return value, delta, chart_data

    temperature_value, temperature_delta, temperature_chart = (
        compute_actual_delta("temperature_c")
    )
    humidity_value, humidity_delta, humidity_chart = (
        compute_actual_delta("humidity_pct")
    )
    rain_value, rain_delta, rain_chart = (
        compute_actual_delta("rain_proba", multiplier=100)
    )

    with st.container(border=True):
        st.subheader(
            ":material/cell_tower: Actual Measurements",
            anchor=False
        )
        st.badge(
            f"Last update: {pd.to_datetime(meas_df["timestamp"].iloc[0]).strftime("%H:%M")}",
            icon=":material/schedule:",
            color="green"
        )

        row = st.container(horizontal=True)
        with row:

            # Temperature
            st.metric(
                "Temperature",
                f"{temperature_value:.1f}°C",
                f"{temperature_delta:.1f}°C",
                icon=":material/thermostat:",
                chart_data=temperature_chart,
                chart_type="area",
                border=True
            )

            # Humidity
            st.metric(
                "Humidity",
                f"{humidity_value:.0f}%",
                f"{humidity_delta:.0f}%",
                icon=":material/water_drop:",
                chart_data=humidity_chart,
                chart_type="area",
                border=True
            )

            # Rain
            st.metric(
                "Rain",
                f"{rain_value:.0f}%",
                f"{rain_delta:.0f}%",
                icon=":material/rainy:",
                chart_data=rain_chart,
                chart_type="area",
                border=True
            )

    # =========================
    # Nowcasting
    # =========================

    with st.container(border=True):
        st.subheader(
            ":material/partly_cloudy_day: Weather Nowcasting",
            anchor=False
        )
        st.badge(
            f"Last update: {pd.to_datetime(meas_df["timestamp"].iloc[0]).strftime("%H:%M")}",
            icon=":material/schedule:",
            color="green"
        )

        nowcast_df = (
            results_df[
                (results_df["city"] == tmp_station) &
                (results_df["lead"] > 0)
            ]
            .sort_values("lead")
            .reset_index(drop=True)
        )

        cols = st.columns(4)
        for col, lead, lead_color in zip(
            cols,
            [30, 60, 90, 120],
            ["blue", "violet", "orange", "red"]
        ):

            row = nowcast_df[nowcast_df["lead"] == lead]
            if row.empty:
                continue

            data = row.iloc[0]
            temperature = data["temperature_c"]
            humidity = data["humidity_pct"]
            dew_point = data["dew_point_c"]
            pressure = data["pressure_hpa"]
            rain_proba = data["rain_proba"] * 100
            wind_speed = data["wind_speed_kmh"]
            wind_dir = the_utils.get_wind_direction(
                data["wind_x"], data["wind_y"]
            )

            with col:
                with st.container(border=True):

                    # Cards
                    st.metric(
                        label="Time",
                        value=f"{pd.to_datetime(data["timestamp"]).strftime("%H:%M")}",
                        delta=f"+{lead} min",
                        delta_color=lead_color,
                        delta_arrow="off",
                        icon=":material/schedule:"
                    )
                    st.metric(
                        "Temperature", f"{temperature:.1f}°", icon=":material/thermostat:"
                    )
                    st.metric(
                        "Humidity", f"{humidity:.0f}%", icon=":material/water_drop:"
                    )
                    st.metric(
                        "Dew Point", f"{dew_point:.1f}°", icon=":material/dew_point:"
                    )
                    st.metric(
                        "Pressure", f"{pressure:.1f} hPa", icon=":material/speed:"
                    )
                    st.metric(
                        "Wind", f"{wind_speed:.1f} km/h {wind_dir}", icon=":material/air:"
                    )
                    st.metric(
                        "Rain", f"{rain_proba:.0f}%", icon=":material/rainy:"
                    )

    # =========================
    # Insights
    # =========================

    with st.container(border=True):
        st.subheader(
            ":material/search_insights: Insights",
            anchor=False
        )
        st.badge(
            f"Last update: {pd.to_datetime(meas_df["timestamp"].iloc[0]).strftime("%H:%M")}",
            icon=":material/schedule:",
            color="green"
        )

        plot_df = (
            results_df[
                results_df["city"] == tmp_station
            ]
            .sort_values("lead")
            .head(5)
            .reset_index(drop=True)
        )

        row = st.container(horizontal=True)
        with row:

            # Atmospheric Conditions
            with st.container(border=True):
                st.subheader(
                    ":material/thermostat: Atmospheric Conditions",
                    anchor=False
                )

                # Temperature
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=plot_df["lead"],
                    y=plot_df["temperature_c"],
                    mode="lines+markers",
                    name="Temperature",
                    hovertemplate=(
                        "Lead: %{x} min<br>"
                        "Temperature: %{y:.1f}°C"
                        "<extra></extra>"
                    )
                ))

                # Dew Point
                fig.add_trace(go.Scatter(
                    x=plot_df["lead"],
                    y=plot_df["dew_point_c"],
                    mode="lines+markers",
                    name="Dew Point",
                    hovertemplate=(
                        "Lead: %{x} min<br>"
                        "Dew Point: %{y:.1f}°C"
                        "<extra></extra>"
                    )
                ))

                # Humidity
                fig.add_trace(go.Scatter(
                    x=plot_df["lead"],
                    y=plot_df["humidity_pct"],
                    mode="lines+markers",
                    name="Humidity",
                    hovertemplate=(
                        "Lead: %{x} min<br>"
                        "Humidity: %{y:.0f}%"
                        "<extra></extra>"
                    ),
                    yaxis="y2"
                ))
                fig.update_layout(
                    xaxis_title="Minutes (lead)",
                    yaxis_title="Temperature / Dew Point (°C)",
                    yaxis2=dict(
                        title="Humidity (%)",
                        overlaying="y",
                        side="right",
                        range=[0, 100]
                    ),
                    hovermode="x unified",
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(
                    fig,
                    width="stretch",
                    config={"displayModeBar": False},
                )

            # Wind Rose
            with st.container(border=True):
                st.subheader(
                    ":material/air: Wind Rose",
                    anchor=False
                )

                wind_x = plot_df["wind_x"].to_numpy()
                wind_y = plot_df["wind_y"].to_numpy()
                wind_speed = plot_df["wind_speed_kmh"].to_numpy()
                wind_dir = (
                    np.degrees(np.arctan2(wind_x, wind_y)) + 360
                ) % 360
                direction_centers = np.arange(0, 360, 22.5)
                speed_bins = [0, 1.2, 2.4, 3.6, 4.8, 6.0, np.inf]
                speed_labels = [
                    "0.0–1.2",
                    "1.2–2.4",
                    "2.4–3.6",
                    "3.6–4.8",
                    "4.8–6.0",
                    "6.0+"
                ]

                traces = []
                for i in range(len(speed_bins) - 1):

                    lower = speed_bins[i]
                    upper = speed_bins[i + 1]

                    if np.isinf(upper):
                        mask = wind_speed >= lower

                    else:
                        mask = (
                            (wind_speed >= lower)
                            & (wind_speed < upper)
                        )
                    counts = np.zeros(
                        len(direction_centers)
                    )

                    for direction in wind_dir[mask]:

                        sector = int(
                            np.round(direction / 22.5)
                        ) % 16
                        counts[sector] += 1
                    traces.append(
                        go.Barpolar(
                            r=counts,
                            theta=direction_centers,
                            width=[20] * 16,
                            name=speed_labels[i],
                            hovertemplate=(
                                "<b>%{theta:.0f}°</b><br>"
                                f"Wind speed: {speed_labels[i]} km/h<br>"
                                "Observations: %{r}"
                                "<extra></extra>"
                            )
                        )
                    )

                # Wind Rose
                fig = go.Figure(data=traces)
                fig.update_layout(
                    polar=dict(
                        angularaxis=dict(
                            direction="clockwise",
                            rotation=90,
                            tickmode="array",
                            tickvals=list(the_config.FENG_WIND_DIR_MAP.values()),
                            ticktext=list(the_config.FENG_WIND_DIR_MAP.keys())
                        ),
                        radialaxis=dict(
                            showticklabels=True,
                            ticksuffix=""
                        )
                    ),
                    legend=dict(title="Wind speed"),
                    margin=dict(l=20, r=20, t=10, b=20)
                )
                st.plotly_chart(
                    fig,
                    width="stretch",
                    config={"displayModeBar": False},
                )

            # Rain & Pressure
            with st.container(border=True):
                st.subheader(
                    ":material/rainy: Rain & Pressure",
                    anchor=False
                )

                rain_probability = (
                    plot_df["rain_proba"] * 100
                )
                pressure = plot_df["pressure_hpa"]

                # Rain
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=plot_df["lead"],
                        y=rain_probability,
                        mode="lines+markers",
                        name="Rain",
                        line=dict(width=2),
                        marker=dict(size=8),
                        fill="tozeroy",
                        fillcolor="rgba(30, 144, 255, 0.12)",
                        hovertemplate=(
                            "Lead: %{x} min<br>"
                            "Rain: %{y:.0f}%"
                            "<extra></extra>"
                        )
                    )
                )

                # Pressure
                fig.add_trace(
                    go.Scatter(
                        x=plot_df["lead"],
                        y=pressure,
                        mode="lines+markers",
                        name="Pressure",
                        line=dict(width=2, dash="dash"),
                        marker=dict(size=8),
                        yaxis="y2",
                        hovertemplate=(
                            "Lead: %{x} min<br>"
                            "Pressure: %{y:.1f} hPa"
                            "<extra></extra>"
                        )
                    )
                )

                # Rain
                fig.update_layout(
                    xaxis=dict(
                        title="Minutes (lead)",
                        tickmode="array",
                        tickvals=plot_df["lead"].tolist()
                    ),
                    yaxis=dict(
                        title="Rain (%)",
                        range=[0, 100],
                        ticksuffix="%"
                    ),
                    yaxis2=dict(
                        title="Pressure (hPa)",
                        overlaying="y",
                        side="right",
                        showgrid=False
                    ),
                    hovermode="x unified",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    ),
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(
                    fig,
                    width="stretch",
                    config={"displayModeBar": False},
                )

    # =========================
    # Raw Predictions
    # =========================

    with st.container(border=True):

        st.subheader(":material/table: Raw Predictions", anchor=False)
        st.badge(
            f"Last update: {pd.to_datetime(meas_df["timestamp"].iloc[0]).strftime("%H:%M")}",
            icon=":material/schedule:",
            color="green"
        )
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

        html_code = f"""
            <br>
            <div style="text-align: center;">
                Copyright &copy; {datetime.now().year} <strong>Roberto Vicario</strong>. All rights reserved.
            </div>
        """

        st.divider()
        st.html(html_code)

if __name__ == "__main__":
    main()

# -------------------------
