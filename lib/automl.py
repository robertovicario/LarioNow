# =========================
# Dependencies
# =========================

from flaml import AutoML
import pandas as pd

# =========================
# Class
# =========================

class AutoMLClassifier:

    def __init__(
        self,
        time_budget,
        metric,
        estimator_list=None,
        seed=42,
        **kwargs
    ):

        self.time_budget = time_budget
        self.metric = metric
        self.estimator_list = estimator_list
        self.seed = seed
        self.kwargs = kwargs
        self.model_ = None

    def fit(self, X, y):

        automl = AutoML()
        automl.fit(
            X_train=X,
            y_train=y,
            task="classification",
            metric=self.metric,
            time_budget=self.time_budget,
            estimator_list=self.estimator_list,
            seed=self.seed,
            verbose=3,
            **self.kwargs
        )
        self.model_ = automl
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

class AutoMLRegressor:

    def __init__(
        self,
        time_budget,
        metric,
        estimator_list=None,
        seed=42,
        **kwargs
    ):

        self.time_budget = time_budget
        self.metric = metric
        self.estimator_list = estimator_list
        self.seed = seed
        self.kwargs = kwargs
        self.model_ = {}
        self.cols_ = None

    def fit(self, X, y):

        self.cols_ = list(y.columns)
        for col in self.cols_:

            automl = AutoML()
            automl.fit(
                X_train=X,
                y_train=y[col],
                task="regression",
                metric=self.metric,
                time_budget=self.time_budget,
                estimator_list=self.estimator_list,
                seed=self.seed,
                verbose=3,
                **self.kwargs
            )
            self.model_[col] = automl
        return self

    def predict(self, X):

        y_pred = pd.DataFrame(index=X.index)
        for col in self.cols_:
            y_pred[col] = self.model_[col].predict(X)
        return y_pred

# -------------------------
