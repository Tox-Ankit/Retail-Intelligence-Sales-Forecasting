
import numpy as np
import pandas as pd
import pytest

from src.machine_learning.preparation import (
    suggest_task_type,
    validate_target,
    split_dataset,
    build_preprocessor,
)


def test_suggest_classification_for_categorical_target():
    target = pd.Series(["Yes", "No", "Yes", "No"])

    assert suggest_task_type(target) == "classification"


def test_suggest_classification_for_boolean_target():
    target = pd.Series([True, False, True, False])

    assert suggest_task_type(target) == "classification"


def test_suggest_regression_for_continuous_target():
    target = pd.Series([120.5, 150.2, 200.7, 250.9])

    assert suggest_task_type(target) == "regression"


def test_validate_target_separates_features_and_target():
    data = pd.DataFrame({
        "sales": [100, 200, 300],
        "quantity": [1, 2, 3],
        "region": ["North", "South", "West"],
    })

    X, y = validate_target(data, "sales")

    assert "sales" not in X.columns
    assert list(X.columns) == ["quantity", "region"]
    assert y.tolist() == [100, 200, 300]


def test_validate_target_rejects_missing_target_values():
    data = pd.DataFrame({
        "sales": [100, np.nan, 300],
        "quantity": [1, 2, 3],
    })

    with pytest.raises(ValueError, match="missing values"):
        validate_target(data, "sales")


def test_validate_target_rejects_unknown_target():
    data = pd.DataFrame({
        "sales": [100, 200, 300],
        "quantity": [1, 2, 3],
    })

    with pytest.raises(ValueError, match="not found"):
        validate_target(data, "unknown")


def test_regression_split():
    data = pd.DataFrame({
        "sales": range(100, 120),
        "quantity": range(1, 21),
        "region": ["North", "South"] * 10,
    })

    X_train, X_test, y_train, y_test = split_dataset(
        data,
        target_column="sales",
        task_type="regression",
        test_size=0.2,
        random_state=42,
    )

    assert len(X_train) == 16
    assert len(X_test) == 4
    assert len(y_train) == 16
    assert len(y_test) == 4
    assert "sales" not in X_train.columns


def test_classification_split_preserves_classes():
    data = pd.DataFrame({
        "customer_type": ["Regular"] * 10 + ["Premium"] * 10,
        "spend": list(range(20)),
    })

    X_train, X_test, y_train, y_test = split_dataset(
        data,
        target_column="customer_type",
        task_type="classification",
        test_size=0.2,
        random_state=42,
    )

    assert set(y_train) == {"Regular", "Premium"}
    assert set(y_test) == {"Regular", "Premium"}
    assert len(X_train) == 16
    assert len(X_test) == 4


def test_classification_split_rejects_rare_classes():
    data = pd.DataFrame({
        "label": ["A"] * 8 + ["B"],
        "feature": list(range(9)),
    })

    with pytest.raises(ValueError, match="two rows"):
        split_dataset(data, "label", "classification")


def test_preprocessor_handles_missing_numeric_and_categorical_values():
    X_train = pd.DataFrame({
        "sales": [100.0, np.nan, 300.0, 400.0],
        "region": ["North", "South", None, "North"],
    })

    preprocessor = build_preprocessor(X_train)
    transformed = preprocessor.fit_transform(X_train)

    assert transformed.shape[0] == 4
    assert np.isfinite(transformed.toarray()).all() if hasattr(
        transformed, "toarray"
    ) else np.isfinite(transformed).all()


def test_preprocessor_handles_unseen_categories():
    X_train = pd.DataFrame({
        "region": ["North", "South", "North", "South"],
        "sales": [100, 200, 150, 250],
    })

    X_test = pd.DataFrame({
        "region": ["East"],
        "sales": [175],
    })

    preprocessor = build_preprocessor(X_train)
    preprocessor.fit(X_train)
    transformed = preprocessor.transform(X_test)

    assert transformed.shape[0] == 1


def test_preprocessor_rejects_unprocessed_datetime_columns():
    X = pd.DataFrame({
        "order_date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "sales": [100, 200],
    })

    with pytest.raises(ValueError, match="Datetime columns"):
        build_preprocessor(X)


def test_invalid_task_type_is_rejected():
    data = pd.DataFrame({
        "sales": [100, 200, 300, 400, 500],
        "quantity": [1, 2, 3, 4, 5],
    })

    with pytest.raises(ValueError, match="task_type"):
        split_dataset(data, "sales", "forecasting")


def test_input_is_not_modified():
    data = pd.DataFrame({
        "sales": [100, 200, 300, 400, 500],
        "quantity": [1, 2, 3, 4, 5],
    })
    original = data.copy(deep=True)

    validate_target(data, "sales")

    pd.testing.assert_frame_equal(data, original)
