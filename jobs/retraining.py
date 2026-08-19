# =========================
# Dependencies
# =========================

from google.cloud import bigquery, storage
from loguru import logger
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import joblib
import json
import numpy as np
import pandas as pd
import sys
import warnings

ROOT_PATH = Path(__file__).resolve().parent
if ROOT_PATH.name in ["jobs", "notebook"]:
    ROOT_PATH = ROOT_PATH.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from lib import config as the_config
from lib import utils as the_utils
from lib import ml_metrics as metrics_lib
from lib.automl import AutoMLClassifier, AutoMLRegressor

# -------------------------

# Settings
# pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)
# pd.set_option("display.max_colwidth", None)
# pd.set_option("display.width", None)
# pd.set_option("display.float_format", "{:.3f}".format)
warnings.filterwarnings("ignore")

# =========================
# Configurations
# =========================

# GCP
BQ_CLIENT = bigquery.Client(project=the_config.GCP_PROJECT)
GCS_CLIENT = storage.Client()
GCS_BUCKET = GCS_CLIENT.bucket(the_config.GCS_BUCKET)

# Paths
for path in the_config.PATHS:
    the_utils.ensure_path(path)

# =========================
# Pipeline
# =========================

def run_pipeline() -> None:

    # -------------------------
    # DataFrame
    # -------------------------

    query = f"""
    SELECT * FROM `{the_config.BQ_TABLE}`
    ORDER BY TIMESTAMP(
        DATETIME(year, month, day, hour, minute, 0)
    )
    """
    df = BQ_CLIENT.query(query).to_dataframe()
    logger.info(f"{'[LOAD]':<8}{'[BQ]':<6}{'Rows:':<10}{len(df)}")
    # display(df.info())

    # -------------------------
    # Feature Engineering
    # -------------------------

    # Feature Engineering -- Classification
    clf_df = df.iloc[-the_config.HYPER_SPACE["sampling"]:]
    clf_df, target_clf = the_utils.feature_engineering_clf(clf_df)
    logger.debug(f"[LIST] Targets ({len(target_clf)}): {target_clf}")

    # Feature Engineering -- Regression
    reg_df = df.iloc[-the_config.HYPER_SPACE["sampling"]:]
    reg_df, targets_reg = the_utils.feature_engineering_reg(reg_df)
    logger.debug(f"[LIST] Targets ({len(targets_reg)}): {targets_reg}")

    # -------------------------

    # Feature Selection -- Classification
    to_drop = [
        "date", "year", "month", "day", "hour", "minute",
        "quarter", "week_of_year", "day_of_year", "day_of_week",
        "province", "city", "station",
        "wind_speed_kmh", "wind_dir",
        "rain_mm", "rain_mmh",
        the_config.CLASSIFICATION["targets"][0]
    ]
    to_drop += [c for c in clf_df.columns if c.lower().startswith("conf_")]
    X_clf = clf_df.drop(columns=to_drop)
    y_clf = clf_df[the_config.CLASSIFICATION["targets"][0]]

    # Train-Test Split -- Classification
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf,
        test_size=the_config.CLASSIFICATION["test_size"],
        shuffle=the_config.CLASSIFICATION["shuffle"],
        random_state=the_config.CLASSIFICATION["seed"]
    )

    logger.debug(f"{'[LIST]':<8}{'[CLF]':<6}{'Features:':<10}{X_train_clf.columns.tolist()}")
    logger.debug(f"{'[LIST]':<8}{'[CLF]':<6}{'Labels:':<10}{y_train_clf.name}")
    logger.debug(f"{'[TRAIN]':<8}{'[CLF]':<6}{'Shape:':<10}{y_train_clf.shape}")
    logger.debug(f"{'[TEST]':<8}{'[CLF]':<6}{'Shape:':<10}{y_test_clf.shape}")

    # -------------------------

    # Feature Selection -- Regression
    to_drop = [
        "date",
        "province", "city", "station",
        "wind_speed_kmh", "wind_dir",
        "rain_mm", "rain_mmh",
        *targets_reg
    ]
    to_drop += [c for c in reg_df.columns if c.lower().startswith("conf_")]
    X_reg = reg_df.drop(columns=to_drop)
    y_reg = reg_df[targets_reg]

    # Train-Test Split -- Regression
    X_train_reg = X_reg.iloc[:-the_config.INF_ROWS]
    y_train_reg = y_reg.iloc[:-the_config.INF_ROWS]
    X_test_reg = X_reg.iloc[-the_config.INF_ROWS:]
    y_test_reg = y_reg.iloc[-the_config.INF_ROWS:]

    logger.debug(f"{'[LIST]':<8}{'[REG]':<6}{'Features:':<10}{X_train_reg.columns.tolist()}")
    logger.debug(f"{'[LIST]':<8}{'[REG]':<6}{'Labels:':<10}{y_train_reg.columns.tolist()}")
    logger.debug(f"{'[TRAIN]':<8}{'[REG]':<6}{'Shape:':<10}{y_train_reg.shape}")
    logger.debug(f"{'[TEST]':<8}{'[REG]':<6}{'Shape:':<10}{y_test_reg.shape}")

    # -------------------------
    # Training
    # -------------------------

    # AutoML Classifier -- LGBM
    lgbm_clf = AutoMLClassifier(
        **the_config.HYPER_SPACE["automl_clf"],
        estimator_list=["lgbm"],
        seed=the_config.CLASSIFICATION["seed"]
    )
    lgbm_clf.fit(X_train_clf, y_train_clf)
    # display(lgbm_clf)

    # AutoML Regressor -- LGBM
    lgbm_reg = AutoMLRegressor(
        **the_config.HYPER_SPACE["automl_reg"],
        estimator_list=["lgbm"],
        seed=the_config.REGRESSION["seed"]
    )
    lgbm_reg.fit(X_train_reg, y_train_reg)
    # display(lgbm_reg)

    # -------------------------
    # Evaluation
    # -------------------------

    # Evaluation -- Classification
    models_clf = {
        "LGBM Classifier": { "model": lgbm_clf, "metrics": {} }
    }
    for model_name, model_info in models_clf.items():

        logger.debug(f"Model: {model_name}")
        model = model_info["model"]
        model_metrics = model_info["metrics"]

        y_true_train_clf = y_train_clf
        y_true_test_clf = y_test_clf
        y_pred_train_clf = model.predict(X_train_clf)
        y_pred_test_clf = model.predict(X_test_clf)

        model_metrics[the_config.CLASSIFICATION["targets"][0]] = {}
        metrics_lib.compute_metrics_clf(
            model_metrics[the_config.CLASSIFICATION["targets"][0]],
            y_true_train_clf,
            y_pred_train_clf,
            model_classes=[0, 1],
            set="TRAIN"
        )
        metrics_lib.compute_metrics_clf(
            model_metrics[the_config.CLASSIFICATION["targets"][0]],
            y_true_test_clf,
            y_pred_test_clf,
            model_classes=[0, 1],
            set="TEST"
        )

        # Evaluation -- Regression
        models_reg = {
            "LGBM Regressor": { "model": lgbm_reg, "metrics": {} }
        }
        for model_name, model_info in models_reg.items():

            logger.debug(f"Model: {model_name}")
            model = model_info["model"]
            model_metrics = model_info["metrics"]

            y_true_train_reg = y_train_reg
            y_true_test_reg = y_test_reg
            y_pred_train_reg = pd.DataFrame(
                model.predict(X_train_reg),
                columns=y_train_reg.columns,
                index=y_train_reg.index
            )
            y_pred_test_reg = pd.DataFrame(
                model.predict(X_test_reg),
                columns=y_test_reg.columns,
                index=y_test_reg.index
            )

            for forecast in the_config.FORECASTS:

                model_metrics[forecast] = {}
                for feature in the_config.REGRESSION["targets"]:

                    col = f"{feature}_lead_{forecast}"
                    model_metrics[forecast][feature] = {}
                    metrics_lib.compute_metrics_reg(
                        model_metrics[forecast][feature],
                        y_true_train_reg[[col]],
                        y_pred_train_reg[[col]],
                        set="TRAIN"
                    )
                    metrics_lib.compute_metrics_reg(
                        model_metrics[forecast][feature],
                        y_true_test_reg[[col]],
                        y_pred_test_reg[[col]],
                        set="TEST"
                    )

    # -------------------------
    # Results
    # -------------------------

    # Metrics -- Classification
    rows_clf = []
    for model_name, model_info in models_clf.items():

        for target, sets in model_info["metrics"].items():
            for split in ["TRAIN", "TEST"]:

                size, accuracy, precision, recall, f1_score, cm = sets[split]
                rows_clf.append({
                    "Model": model_name,
                    "Target": target,
                    "Set": split,
                    "Size": size,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1_score,
                    "Confusion Matrix": cm
                })
    metrics_clf_df = (
        pd.DataFrame(rows_clf)
        .reset_index(drop=True)
    )
    

    # display(y_train_clf.describe())
    # display(metrics_clf_df)

    # Confusion Matrix -- Classification
    for model_name in metrics_clf_df["Model"].unique():

        model_results = metrics_clf_df[
            metrics_clf_df["Model"] == model_name
        ]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        for ax, (_, row) in zip(
            axes,
            (
                model_results
                .set_index("Set")
                .loc[["TRAIN", "TEST"]]
                .reset_index()
                .iterrows()
            )
        ):

            cm = np.array(row["Confusion Matrix"])
            cm = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["dry", "rain"]
            )
            cm.plot(
                ax=ax,
                cmap="Blues",
                values_format="d"
            )
            ax.set_title(row["Set"])

        fig.suptitle(f"Confusion Matrix: {model_name}")
        fig.tight_layout()
        fig.savefig(
            the_config.MODELS_LATEST_PATH /
            f"{model_name.lower().replace(' ', '_')}_cm.png",
            bbox_inches="tight",
            dpi=300
        )
        # plt.show()
        plt.close(fig)

    # Metrics -- Regression
    rows_reg = []
    for model_name, model_info in models_reg.items():
        for lead, targets in model_info["metrics"].items():
            for target, sets in targets.items():
                for split in ["TRAIN", "TEST"]:

                    size, mae, rmse, r2 = sets[split]
                    rows_reg.append({
                        "Model": model_name,
                        "Target": target,
                        "Lead": lead,
                        "Set": split,
                        "Size": size,
                        "MAE": mae,
                        "RMSE": rmse,
                        "R2": r2,
                    })                
    metrics_reg_df = (
        pd.DataFrame(rows_reg)
        .sort_values(["Target", "Lead"])
        .reset_index(drop=True)
    )

    # display(y_train_reg.describe())
    # display(metrics_reg_df)

    # Metrics -- Models
    for models_task in [models_clf, models_reg]:
        for model_name, model_info in models_task.items():

            model = model_info["model"]
            filename = model_name.lower().replace(" ", "_")
            joblib.dump(
                model,
                the_config.MODELS_LATEST_PATH / f"{filename}.joblib"
            )

    # Metrics -- Classification
    with open(the_config.MODELS_LATEST_PATH / "stats_clf.json", "w") as f:
        json.dump(
            (
                y_train_clf.to_frame().describe().T
                .to_dict(orient="index")
            ),
            f, indent=4
        )
    with open(the_config.MODELS_LATEST_PATH / "metrics_clf.json", "w") as f:
        json.dump(
            metrics_clf_df.to_dict(orient="records"),
            f, indent=4,
            default=lambda x: x.tolist()
        )

    # Metrics -- Regression
    with open(the_config.MODELS_LATEST_PATH / "stats_reg.json", "w") as f:
        json.dump(
            y_train_reg.describe().T
            .to_dict(orient="index"),
            f, indent=4
        )
    with open(the_config.MODELS_LATEST_PATH / "metrics_reg.json", "w") as f:
        json.dump(
            metrics_reg_df.to_dict(orient="records"),
            f, indent=4
        )

    # logger.debug(f"[OUT] {the_config.MODELS_LATEST_PATH}")

    # ---------------------------
    # Google Cloud Platform (GCP)
    # ---------------------------

    blobs = GCS_BUCKET.list_blobs(prefix=the_config.GCS_PREFIX_LATEST)
    for blob in blobs:
        blob.delete()

    for file_path in (the_config.MODELS_LATEST_PATH).iterdir():
        if file_path.is_file():

            blob = GCS_BUCKET.blob(f"{the_config.GCS_PREFIX_LATEST}{file_path.name}")
            blob.upload_from_filename(str(file_path))

if __name__ == "__main__":
    run_pipeline()

# -------------------------
