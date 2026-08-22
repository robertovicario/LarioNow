# =========================
# Dependencies
# =========================

from google.cloud import bigquery
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

ROOT_PATH = Path(__file__).resolve().parent
if ROOT_PATH.name in ["jobs", "notebook"]:
    ROOT_PATH = ROOT_PATH.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from lib import config as the_config
from lib import utils as the_utils
from lib import ml_metrics as metrics_lib
from lib.automl import AutoMLClassifier, AutoMLRegressor

# =========================
# Configurations
# =========================

# GCP
BQ_CLIENT = bigquery.Client(project=the_config.GCP_PROJECT)

# =========================
# Methods
# =========================

def train_model() -> None:

    logger.info(
f"""\n
# =========================
# (1) TRAINING
# =========================
"""
    )

    # -------------------------
    # DataFrame
    # -------------------------

    query = f"""
        SELECT * FROM `{the_config.BQ_TABLE}`
        ORDER BY TIMESTAMP(
            DATETIME(year, month, day, hour, minute, 0)
        )
    """
    train_df = BQ_CLIENT.query(query).to_dataframe()
    logger.info(f"{'[LOAD]':<8}{'[BQ]':<6}{'Rows:':<10}{len(train_df)}")

    # -------------------------
    # Feature Engineering
    # -------------------------

    # Feature Engineering -- Classification
    clf_df = train_df.copy()
    clf_df, target_clf = the_utils.feature_engineering_clf(clf_df)
    clf_df = clf_df.assign(
        timestamp=lambda x: pd.to_datetime(
            x[["year", "month", "day", "hour", "minute"]]
        )
    )
    logger.debug(f"[LIST] Targets ({len(target_clf)}): {target_clf}")

    # Feature Selection -- Classification
    to_drop = [
        *the_config.CLASSIFICATION["to_drop"],
        the_config.CLASSIFICATION["targets"][0],
        "timestamp"
    ]
    to_drop += [c for c in clf_df.columns if c.lower().startswith("conf_")]
    X_clf = clf_df.drop(columns=to_drop)
    y_clf = clf_df[the_config.CLASSIFICATION["targets"][0]]

    # Feature Engineering -- Regression
    reg_df = train_df.copy()
    reg_df, targets_reg = the_utils.feature_engineering_reg(reg_df)
    reg_df = reg_df.assign(
        timestamp=lambda x: pd.to_datetime(
            x[["year", "month", "day", "hour", "minute"]]
        )
    )
    logger.debug(f"[LIST] Targets ({len(targets_reg)}): {targets_reg}")

    # Feature Selection -- Regression
    to_drop = [
        *the_config.REGRESSION["to_drop"],
        *targets_reg,
        "timestamp"
    ]
    to_drop += [c for c in reg_df.columns if c.lower().startswith("conf_")]
    X_reg = reg_df.drop(columns=to_drop)
    y_reg = reg_df[targets_reg]

    # -------------------------

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

    # Classifier
    clf = AutoMLClassifier(
        **the_config.ML_HYPER_SPACE["automl_clf"],
        estimator_list=["lgbm"],
        seed=the_config.CLASSIFICATION["seed"]
    )
    clf.fit(X_train_clf, y_train_clf)

    # Regressor
    reg = AutoMLRegressor(
        **the_config.ML_HYPER_SPACE["automl_reg"],
        estimator_list=["lgbm"],
        seed=the_config.REGRESSION["seed"]
    )
    reg.fit(X_train_reg, y_train_reg)

    # -------------------------
    # Evaluation
    # -------------------------

    # Evaluation -- Classification
    models_clf = {
        # "Random Forest Classifier": { "model": clf, "metrics": {} },
        # "LGBM Classifier": { "model": clf, "metrics": {} },
        "XGBoost Classifier": { "model": clf, "metrics": {} },
        # "AutoML Classifier": { "model": clf, "metrics": {} }
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
            # "Random Forest Regressor": { "model": reg, "metrics": {} },
            # "LGBM Regressor": { "model": reg, "metrics": {} },
            "XGBoost Regressor": { "model": reg, "metrics": {} },
            # "AutoML Regressor": { "model": reg, "metrics": {} }
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

            for forecast in the_config.FENG_FORECASTS:

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

# -------------------------
