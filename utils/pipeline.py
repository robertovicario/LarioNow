# =========================
# Dependencies
# =========================

import numpy as np
import pandas as pd

from config import config as the_config

# =========================
# Methods
# =========================

def feature_engineering_clf(df, inference=False):

    # Rain Flag
    if not inference:
        df[the_config.CLASSIFICATION["targets"][0]] = (df["rain_mm"] > 0).astype(int)
    return df, the_config.CLASSIFICATION["targets"]

def feature_engineering_reg(df, inference=False):

    # Data Cleaning
    int_cols = df.select_dtypes(include="Int64").columns
    df[int_cols] = df[int_cols].astype("float64")

    # Data Preparation
    df["timestamp"] = pd.to_datetime(
        dict(
            year=df["year"],
            month=df["month"],
            day=df["day"],
            hour=df["hour"],
            minute=df["minute"],
        )
    )
    df = (
        df
        .sort_values(["latitude", "longitude", "timestamp"])
        .reset_index(drop=True)
    )
    df.drop(columns=["timestamp"], inplace=True)

    # Wind X-Y
    wind_angle = df["wind_dir"].map(the_config.FENG_WIND_DIR_MAP)
    wind_angle_rad = np.deg2rad(wind_angle)
    df["wind_x"] = (
        df["wind_speed_kmh"] * np.cos(wind_angle_rad)
    )
    df["wind_y"] = (
        df["wind_speed_kmh"] * np.sin(wind_angle_rad)
    )

    # Lag Calculation
    lag_features = {}
    for feature in the_config.REGRESSION["targets"]:

        grouped = df.groupby(["latitude", "longitude"])[feature]
        for lag in the_config.FENG_LAGS:
            lag_features[f"{feature}_lag_{lag}"] = grouped.shift(lag)
    df = pd.concat(
        [df, pd.DataFrame(lag_features, index=df.index)],
        axis=1,
    )

    # Rolling Features
    rolling_features = {}
    for feature in the_config.REGRESSION["targets"]:

        grouped = df.groupby(["latitude", "longitude"])[feature]
        for window in the_config.FENG_ROLLING_WINDOWS:
            rolling_features[f"{feature}_mean_{window}"] = (
                grouped.transform(lambda x: x.rolling(window).mean())
            )
            rolling_features[f"{feature}_std_{window}"] = (
                grouped.transform(lambda x: x.rolling(window).std())
            )
            rolling_features[f"{feature}_min_{window}"] = (
                grouped.transform(lambda x: x.rolling(window).min())
            )
            rolling_features[f"{feature}_max_{window}"] = (
                grouped.transform(lambda x: x.rolling(window).max())
            )
    df = pd.concat(
        [df, pd.DataFrame(rolling_features, index=df.index)],
        axis=1,
    )

    # Lead Calculation
    leads = [
        f"{feature}_lead_{forecast}"
        for forecast in the_config.FENG_FORECASTS
        for feature in the_config.REGRESSION["targets"]
    ]
    if not inference:
        for forecast in the_config.FENG_FORECASTS:

            lead_steps = forecast // the_config.FENG_SAMPLING_MIN
            for feature in the_config.REGRESSION["targets"]:
                target = f"{feature}_lead_{forecast}"
                df[target] = (
                    df.groupby(["latitude", "longitude"])[feature]
                    .shift(-lead_steps)
                )
        df = df.dropna().reset_index(drop=True)

    # -------------------------

    return df, leads

def exec_inference(clf, reg, clf_df, reg_df):

    # Feature Selection -- Regression
    latest = (
        reg_df
        .sort_values(["station", "timestamp"])
        .groupby("station")
        .tail(1)
        .copy()
    )
    to_drop = [
        *the_config.REGRESSION["to_drop"],
        "timestamp"
    ]
    to_drop += [c for c in reg_df.columns if c.lower().startswith("conf_")]
    X_latest_reg = latest.drop(columns=[c for c in to_drop if c in latest.columns])

    reg_features = getattr(next(iter(reg.model_.values())), "feature_names_in_", None)
    if reg_features is not None:
        X_latest_reg = X_latest_reg[list(reg_features)]

    # Predictions -- Regression
    y_pred_reg = reg.predict(X_latest_reg)

    # Results -- Regression
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

    # -------------------------

    # Feature Selection -- Classification
    station = latest[
        ["station", "city", "latitude", "longitude", "timestamp"]
    ].rename_axis("row_index").reset_index()
    results_df = results_df.merge(station, on="row_index")
    results_df["timestamp"] += pd.to_timedelta(results_df["lead"], unit="m")

    clf_features = getattr(clf.model_, "feature_names_in_", None)
    if clf_features is None:

        to_drop = [
            *the_config.CLASSIFICATION["to_drop"],
            "timestamp"
        ]
        to_drop += [c for c in clf_df.columns if c.lower().startswith("conf_")]
        clf_features = (
            clf_df
            .drop(columns=[c for c in to_drop if c in clf_df.columns])
            .columns.tolist()
        )

    # Predictions -- Classification
    results_df["rain_proba"] = clf.predict_proba(
        results_df[list(clf_features)]
    )[:, 1]

    # Results -- Classification
    results_df = results_df[the_config.INF_RES_COLS]
    actual_df = (
        latest[the_config.INF_ACTUAL_COLS]
        .assign(
            lead=0,
            rain_proba=lambda x: x["rain_mm"].gt(0).astype(int),
        )
        .drop(columns=["rain_mm"])
    )

    # -------------------------

    return (
        pd.concat([actual_df, results_df], ignore_index=True)
        [the_config.INF_RES_COLS]
        .sort_values(["city", "station", "lead"])
        .reset_index(drop=True)
    )

# -------------------------
