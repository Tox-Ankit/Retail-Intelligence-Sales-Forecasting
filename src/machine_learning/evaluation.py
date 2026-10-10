
"""Cross-validation, overfitting checks, and model explainability."""

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.machine_learning.preparation import SUPPORTED_TASKS


def _validate_task_type(task_type):
    if task_type not in SUPPORTED_TASKS:
        raise ValueError(
            "task_type must be 'classification' or 'regression'."
        )


def _validate_xy(X, y, name="dataset"):
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"{name} features must be a pandas DataFrame.")

    if X.empty:
        raise ValueError(f"{name} features cannot be empty.")

    if X.columns.has_duplicates:
        raise ValueError("Feature column names must be unique.")

    if not isinstance(y, (pd.Series, np.ndarray, list)):
        raise TypeError(f"{name} target must be a Series, array, or list.")

    target = np.asarray(y).reshape(-1)

    if len(target) != len(X):
        raise ValueError(
            f"{name} features and target must have the same number of rows."
        )

    if len(target) == 0:
        raise ValueError(f"{name} target cannot be empty.")

    if pd.isna(target).any():
        raise ValueError(f"{name} target cannot contain missing values.")

    return target


def cross_validate_model(
    model,
    X_train,
    y_train,
    task_type,
    cv=5,
    random_state=42,
):
    """Evaluate a model with cross-validation on training data only.

    The model should be a scikit-learn estimator or Pipeline. If preprocessing
    is required, include it inside that Pipeline so every fold fits its own
    transformations.

    Returns a dictionary containing individual fold scores and their summary.
    """
    _validate_task_type(task_type)
    target = _validate_xy(X_train, y_train, "training")

    if isinstance(cv, bool) or not isinstance(cv, int) or cv < 2:
        raise ValueError("cv must be an integer of at least 2.")

    if not hasattr(model, "fit") or not hasattr(model, "predict"):
        raise TypeError("model must provide fit() and predict() methods.")

    if task_type == "classification":
        class_counts = pd.Series(target).value_counts()

        if class_counts.min() < cv:
            raise ValueError(
                "Each class must have at least cv observations for "
                "stratified cross-validation. Reduce cv or provide more data."
            )

        splitter = StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=random_state,
        )

        scoring = {
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1_weighted": "f1_weighted",
        }

        primary_metric = "test_f1_weighted"

    else:
        if len(target) < cv:
            raise ValueError(
                "The training dataset must contain at least cv observations."
            )

        splitter = KFold(
            n_splits=cv,
            shuffle=True,
            random_state=random_state,
        )

        scoring = {
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        }

        primary_metric = "test_rmse"

    results = cross_validate(
        clone(model),
        X_train,
        target,
        cv=splitter,
        scoring=scoring,
        return_train_score=True,
        error_score="raise",
        n_jobs=1,
    )

    fold_scores = {}

    for metric_name in scoring:
        train_key = f"train_{metric_name}"
        test_key = f"test_{metric_name}"

        train_values = np.asarray(results[train_key], dtype=float)
        test_values = np.asarray(results[test_key], dtype=float)

        # sklearn returns negative losses so they can be maximized.
        # Convert loss metrics back to their normal positive scale.
        if metric_name in {"rmse", "mae"}:
            train_values = -train_values
            test_values = -test_values

        fold_scores[metric_name] = {
            "train_scores": train_values.tolist(),
            "validation_scores": test_values.tolist(),
            "mean_train_score": float(np.mean(train_values)),
            "mean_validation_score": float(np.mean(test_values)),
            "std_validation_score": float(np.std(test_values, ddof=1))
            if len(test_values) > 1
            else 0.0,
        }

    primary_values = (
        fold_scores["rmse"]["validation_scores"]
        if task_type == "regression"
        else fold_scores["f1_weighted"]["validation_scores"]
    )

    return {
        "task_type": task_type,
        "folds": cv,
        "primary_metric": (
            "rmse" if task_type == "regression" else "f1_weighted"
        ),
        "primary_mean": float(np.mean(primary_values)),
        "primary_std": float(np.std(primary_values, ddof=1))
        if len(primary_values) > 1
        else 0.0,
        "metrics": fold_scores,
        "notes": [
            "Cross-validation was run on the supplied training data.",
            "Validation scores are fold-level estimates, not final test scores.",
            "A larger fold-to-fold standard deviation indicates less stable results.",
        ],
    }


def compare_train_validation_performance(
    model,
    X_train,
    y_train,
    X_validation,
    y_validation,
    task_type,
):
    """Compare fitted model performance on training and validation data.

    The model must already be fitted. Pass a validation set here, not the
    final held-out test set during model selection.
    """
    _validate_task_type(task_type)

    y_train = _validate_xy(X_train, y_train, "training")
    y_validation = _validate_xy(
        X_validation, y_validation, "validation"
    )

    if list(X_train.columns) != list(X_validation.columns):
        raise ValueError(
            "Training and validation feature columns must match in the same order."
        )

    if not hasattr(model, "predict"):
        raise TypeError("model must provide a predict() method.")

    train_predictions = model.predict(X_train)
    validation_predictions = model.predict(X_validation)

    if task_type == "regression":
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        def metrics(actual, predicted):
            return {
                "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
                "mae": float(mean_absolute_error(actual, predicted)),
                "r2": float(r2_score(actual, predicted))
                if len(actual) >= 2
                else None,
            }

        train_metrics = metrics(y_train, train_predictions)
        validation_metrics = metrics(y_validation, validation_predictions)

        # Lower RMSE is better.
        gap = (
            validation_metrics["rmse"] - train_metrics["rmse"]
        )
        possible_overfitting = (
            gap > 0
            and train_metrics["rmse"] < validation_metrics["rmse"]
        )

        gap_description = (
            "Validation RMSE is higher than training RMSE; inspect the gap "
            "for possible overfitting."
            if possible_overfitting
            else "No positive RMSE gap was detected by this simple check."
        )

        primary_metric = "rmse"

    else:
        from sklearn.metrics import accuracy_score, f1_score

        def metrics(actual, predicted):
            return {
                "accuracy": float(accuracy_score(actual, predicted)),
                "f1_weighted": float(
                    f1_score(
                        actual,
                        predicted,
                        average="weighted",
                        zero_division=0,
                    )
                ),
            }

        train_metrics = metrics(y_train, train_predictions)
        validation_metrics = metrics(
            y_validation, validation_predictions
        )

        # Higher weighted F1 is better.
        gap = (
            train_metrics["f1_weighted"]
            - validation_metrics["f1_weighted"]
        )
        possible_overfitting = gap > 0.10

        gap_description = (
            "Training weighted F1 exceeds validation weighted F1 by more "
            "than 0.10; inspect for possible overfitting."
            if possible_overfitting
            else "No large weighted F1 gap was detected by this simple check."
        )

        primary_metric = "f1_weighted"

    return {
        "task_type": task_type,
        "primary_metric": primary_metric,
        "training_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "primary_metric_gap": float(gap),
        "possible_overfitting": bool(possible_overfitting),
        "interpretation": gap_description,
        "notes": [
            "This is a diagnostic, not a definitive overfitting test.",
            "Consider cross-validation and domain context before drawing conclusions.",
        ],
    }


def extract_feature_importance(model):
    """Extract feature importance from a fitted scikit-learn Pipeline.

    Supports tree-based feature_importances_ and linear coefficients.
    For multiclass linear models, coefficient magnitudes are averaged
    across classes. Results are sorted by importance descending.

    This describes model reliance, not causation.
    """
    if not isinstance(model, Pipeline):
        raise TypeError(
            "model must be a fitted scikit-learn Pipeline."
        )

    if len(model.steps) < 2:
        raise ValueError(
            "Pipeline must contain a preprocessing step and a model step."
        )

    preprocessor = model.steps[-2][1]
    estimator = model.steps[-1][1]

    if not hasattr(preprocessor, "get_feature_names_out"):
        raise ValueError(
            "The preprocessing step does not expose feature names."
        )

    try:
        feature_names = preprocessor.get_feature_names_out()
    except (AttributeError, ValueError) as exc:
        raise ValueError(
            "Could not retrieve transformed feature names."
        ) from exc

    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(
            estimator.feature_importances_,
            dtype=float,
        )
        method = "feature_importances_"

    elif hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=float)

        if coefficients.ndim == 1:
            importance = np.abs(coefficients)
        else:
            importance = np.mean(np.abs(coefficients), axis=0)

        method = "absolute_coefficients"

    else:
        raise ValueError(
            "This estimator does not expose supported feature importance "
            "or coefficient values."
        )

    if len(feature_names) != len(importance):
        raise ValueError(
            "Feature-name count does not match importance-value count."
        )

    if not np.isfinite(importance).all():
        raise ValueError("Feature importance values are not all finite.")

    result = pd.DataFrame({
        "feature": feature_names.astype(str),
        "importance": importance,
    })

    result = result.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    result["method"] = method

    return result
