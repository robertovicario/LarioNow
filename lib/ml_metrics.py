# =========================
# Dependencies
# =========================

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_absolute_error, root_mean_squared_error, r2_score
)
import numpy as np

# =========================
# Methods
# =========================

def compute_metrics_clf(
    metrics_out: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_classes: list,
    set: str = "test"
) -> None:

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=model_classes)
    metrics_out[set] = [
        int(len(y_true)),
        float(acc),
        float(prec),
        float(rec),
        float(f1),
        list(cm)
    ]

def compute_metrics_reg(
    metrics_out: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    set: str = "test"
) -> None:

	mae = mean_absolute_error(y_true, y_pred)
	rmse = root_mean_squared_error(y_true, y_pred)
	r2 = r2_score(y_true, y_pred)
	metrics_out[set] = [
        int(len(y_true)), float(mae), float(rmse), float(r2)
    ]

# -------------------------
