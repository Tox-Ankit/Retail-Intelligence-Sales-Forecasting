
import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.machine_learning.preparation import build_preprocessor
from src.machine_learning.evaluation import (
    cross_validate_model,
    compare_train_validation_performance,
    extract_feature_importance,
)


def make_regression_data():
    X = pd.DataFrame({
        "quantity": np.arange(1, 61, dtype=float),
        "price": np.arange(10, 70, dtype=float),
        "region": ["North", "South", "West"] * 20,
    })
    y = X["quantity"] * 5 + X["price"] * 2
    return X, y


def make_classification_data():
    X = pd.DataFrame({
        "spend": np.arange(1, 101, dtype=float),
        "visits": np.arange(10, 110, dtype=float),
        "region": ["North", "South"] * 50,
    })
    y = pd.Series(
        ["High" if value % 2 == 0 else "Low" for value in range(100)]
    )
    return X, y


def make_regression_pipeline(X):
    return Pipeline([
        ("preprocessor", build_preprocessor(X)),
        ("model", RandomForestRegressor(
            n_estimators=20,
            random_state=42,
        )),
    ])


def test_regression_cross_validation():
    X, y = make_regression_data()
    model = make_regression_pipeline(X)

    result = cross_validate_model(
        model, X, y, task_type="regression", cv=5
    )

    assert result["folds"] == 5
    assert result["primary_metric"] == "rmse"
    assert result["primary_mean"] >= 0
    assert len(result["metrics"]["rmse"]["validation_scores"]) == 5


def test_classification_cross_validation():
    X, y = make_classification_data()

    from sklearn.linear_model import LogisticRegression

    model = Pipeline([
        ("preprocessor", build_preprocessor(X)),
        ("model", LogisticRegression(max_iter=1000)),
    ])

    result = cross_validate_model(
        model, X, y, task_type="classification", cv=5
    )

    assert result["primary_metric"] == "f1_weighted"
    assert 0 <= result["primary_mean"] <= 1
    assert len(result["metrics"]["f1_weighted"]["validation_scores"]) == 5


def test_cross_validation_rejects_too_many_folds_for_rare_class():
    X = pd.DataFrame({"feature": range(10)})
    y = pd.Series(["A"] * 8 + ["B"] * 2)

    model = Pipeline([
        ("preprocessor", build_preprocessor(X)),
        ("model", RandomForestRegressor(n_estimators=5, random_state=42)),
    ])

    with pytest.raises(ValueError, match="Each class"):
        cross_validate_model(
            model, X, y, task_type="classification", cv=3
        )


def test_cross_validation_rejects_invalid_fold_count():
    X, y = make_regression_data()
    model = make_regression_pipeline(X)

    with pytest.raises(ValueError, match="cv"):
        cross_validate_model(
            model, X, y, task_type="regression", cv=1
        )


def test_train_validation_comparison_for_regression():
    X, y = make_regression_data()

    X_train, X_validation = X.iloc[:45], X.iloc[45:]
    y_train, y_validation = y.iloc[:45], y.iloc[45:]

    model = make_regression_pipeline(X_train)
    model.fit(X_train, y_train)

    result = compare_train_validation_performance(
        model,
        X_train,
        y_train,
        X_validation,
        y_validation,
        task_type="regression",
    )

    assert result["primary_metric"] == "rmse"
    assert result["training_metrics"]["rmse"] >= 0
    assert result["validation_metrics"]["rmse"] >= 0
    assert isinstance(result["possible_overfitting"], bool)


def test_train_validation_comparison_for_classification():
    X, y = make_classification_data()

    X_train, X_validation = X.iloc[:80], X.iloc[80:]
    y_train, y_validation = y.iloc[:80], y.iloc[80:]

    from sklearn.linear_model import LogisticRegression

    model = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    model.fit(X_train, y_train)

    result = compare_train_validation_performance(
        model,
        X_train,
        y_train,
        X_validation,
        y_validation,
        task_type="classification",
    )

    assert result["primary_metric"] == "f1_weighted"
    assert 0 <= result["training_metrics"]["f1_weighted"] <= 1
    assert 0 <= result["validation_metrics"]["f1_weighted"] <= 1


def test_feature_importance_from_random_forest_pipeline():
    X, y = make_regression_data()
    model = make_regression_pipeline(X)
    model.fit(X, y)

    result = extract_feature_importance(model)

    assert not result.empty
    assert {"feature", "importance", "method"}.issubset(result.columns)
    assert result["importance"].ge(0).all()
    assert result["importance"].is_monotonic_decreasing


def test_feature_importance_from_linear_pipeline():
    X, y = make_regression_data()

    model = Pipeline([
        ("preprocessor", build_preprocessor(X)),
        ("model", LinearRegression()),
    ])
    model.fit(X, y)

    result = extract_feature_importance(model)

    assert not result.empty
    assert (result["method"] == "absolute_coefficients").all()


def test_feature_importance_rejects_unfitted_pipeline():
    X, _ = make_regression_data()
    model = make_regression_pipeline(X)

    with pytest.raises(ValueError, match="Could not retrieve"):
        extract_feature_importance(model)


def test_evaluation_rejects_mismatched_target_length():
    X, y = make_regression_data()
    model = make_regression_pipeline(X)

    with pytest.raises(ValueError, match="same number of rows"):
        cross_validate_model(
            model,
            X,
            y.iloc[:-1],
            task_type="regression",
            cv=5,
        )


def test_evaluation_does_not_modify_input():
    X, y = make_regression_data()
    original_X = X.copy(deep=True)
    original_y = y.copy(deep=True)

    model = make_regression_pipeline(X)
    cross_validate_model(model, X, y, task_type="regression", cv=3)

    pd.testing.assert_frame_equal(X, original_X)
    pd.testing.assert_series_equal(y, original_y)
