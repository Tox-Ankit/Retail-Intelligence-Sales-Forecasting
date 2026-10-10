
"""Train and compare baseline machine-learning models."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.machine_learning.preparation import (
    SUPPORTED_TASKS,
    build_preprocessor,
)


@dataclass
class ModelComparisonResult:
    """Fitted models, their evaluation metrics, and the selected winner."""

    task_type: str
    target_name: str | None
    metrics: list[dict] = field(default_factory=list)
    models: dict = field(default_factory=dict)
    best_model_name: str | None = None
    selection_metric: str = ""
    selection_direction: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def best_model(self):
        """Return the fitted best model, if comparison succeeded."""
        if self.best_model_name is None:
            return None

        return self.models[self.best_model_name]


def _validate_training_inputs(X_train, X_test, y_train, y_test, task_type):
    """Validate train/test data before model fitting."""
    if task_type not in SUPPORTED_TASKS:
        raise ValueError(
            "task_type must be 'classification' or 'regression'."
        )

    if not isinstance(X_train, pd.DataFrame) or not isinstance(
        X_test, pd.DataFrame
    ):
        raise TypeError("X_train and X_test must be pandas DataFrames.")

    if X_train.empty or X_test.empty:
        raise ValueError("Training and test features must not be empty.")

    if X_train.columns.has_duplicates or X_test.columns.has_duplicates:
        raise ValueError("Feature column names must be unique.")

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Training and test feature columns must match in the same order."
        )

    if not isinstance(y_train, (pd.Series, np.ndarray, list)):
        raise TypeError("y_train must be a Series, array, or list.")

    if not isinstance(y_test, (pd.Series, np.ndarray, list)):
        raise TypeError("y_test must be a Series, array, or list.")

    y_train = np.asarray(y_train).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same row count.")

    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test must have the same row count.")

    if len(y_train) == 0 or len(y_test) == 0:
        raise ValueError("Training and test targets must not be empty.")

    if pd.isna(y_train).any() or pd.isna(y_test).any():
        raise ValueError("Target values cannot contain missing values.")

    if task_type == "regression":
        try:
            y_train = pd.to_numeric(
                pd.Series(y_train), errors="raise"
            ).to_numpy(dtype=float)
            y_test = pd.to_numeric(
                pd.Series(y_test), errors="raise"
            ).to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Regression targets must contain numeric values."
            ) from exc

        if not np.isfinite(y_train).all() or not np.isfinite(y_test).all():
            raise ValueError("Regression targets must be finite numbers.")

    else:
        if pd.Series(y_train).nunique() < 2:
            raise ValueError(
                "Classification training data must contain at least two classes."
            )

        train_classes = set(pd.Series(y_train).tolist())
        test_classes = set(pd.Series(y_test).tolist())

        if not test_classes.issubset(train_classes):
            raise ValueError(
                "The test set contains a class not present in the training set."
            )

    return y_train, y_test


def _regression_metrics(y_true, predictions):
    """Calculate regression evaluation metrics."""
    rmse = float(
        np.sqrt(mean_squared_error(y_true, predictions))
    )

    try:
        r2 = float(r2_score(y_true, predictions))
        if not np.isfinite(r2):
            r2 = None
    except (ValueError, FloatingPointError):
        r2 = None

    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": rmse,
        "r2": r2,
    }


def _classification_metrics(y_true, predictions):
    """Calculate classification evaluation metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "precision_weighted": float(
            precision_score(
                y_true, predictions, average="weighted", zero_division=0
            )
        ),
        "recall_weighted": float(
            recall_score(
                y_true, predictions, average="weighted", zero_division=0
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_true, predictions, average="weighted", zero_division=0
            )
        ),
    }


def train_and_compare_models(
    X_train,
    X_test,
    y_train,
    y_test,
    task_type,
    target_name=None,
    random_state=42,
):
    """Train baseline models and compare their held-out test performance.

    Every model receives its own cloned preprocessing pipeline. Imputation,
    scaling and categorical encoding are fitted only on the training data.

    The best model is selected using:
      - Regression: lowest RMSE.
      - Classification: highest weighted F1 score.

    The test set is used for model comparison, so a final unbiased estimate
    should ideally come from a separate untouched evaluation set.
    """
    y_train, y_test = _validate_training_inputs(
        X_train, X_test, y_train, y_test, task_type
    )

    preprocessor = build_preprocessor(X_train)

    if task_type == "regression":
        estimators = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=150,
                random_state=random_state,
                n_jobs=-1,
            ),
            "Extra Trees": ExtraTreesRegressor(
                n_estimators=150,
                random_state=random_state,
                n_jobs=-1,
            ),
        }
        selection_metric = "rmse"
        selection_direction = "minimize"
    else:
        estimators = {
            "Logistic Regression": LogisticRegression(
                max_iter=2000,
                random_state=random_state,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=150,
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced",
            ),
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=150,
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced",
            ),
        }
        selection_metric = "f1_weighted"
        selection_direction = "maximize"

    fitted_models = {}
    results = []

    for model_name, estimator in estimators.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        if task_type == "regression":
            model_metrics = _regression_metrics(y_test, predictions)
        else:
            model_metrics = _classification_metrics(y_test, predictions)

        results.append({
            "model": model_name,
            "task_type": task_type,
            **model_metrics,
        })
        fitted_models[model_name] = pipeline

    if task_type == "regression":
        eligible = [
            item for item in results if item["rmse"] is not None
        ]
        best = min(eligible, key=lambda item: item["rmse"]) if eligible else None
    else:
        eligible = [
            item for item in results if item["f1_weighted"] is not None
        ]
        best = (
            max(eligible, key=lambda item: item["f1_weighted"])
            if eligible
            else None
        )

    # Sort so the leading model is the best according to the chosen metric.
    if task_type == "regression":
        results.sort(key=lambda item: item["rmse"])
    else:
        results.sort(
            key=lambda item: item["f1_weighted"],
            reverse=True,
        )

    return ModelComparisonResult(
        task_type=task_type,
        target_name=target_name,
        metrics=results,
        models=fitted_models,
        best_model_name=best["model"] if best else None,
        selection_metric=selection_metric,
        selection_direction=selection_direction,
        notes=[
            "Models were evaluated on the supplied test set.",
            "Preprocessing was fitted separately within each model pipeline.",
            (
                "The best model is selected using a single metric; "
                "review all metrics before choosing a model for deployment."
            ),
            (
                "For a final unbiased performance estimate, use an untouched "
                "evaluation set after model selection."
            ),
        ],
    )
