
import numpy as np
import pandas as pd
import pytest

from src.machine_learning.training import train_and_compare_models


def test_regression_models_train_and_compare():
    data = pd.DataFrame({
        "quantity": np.arange(1, 61),
        "price": np.arange(10, 70),
        "region": ["North", "South", "West"] * 20,
    })
    target = data["quantity"] * 5.0 + data["price"] * 2.0

    X_train = data.iloc[:45].copy()
    X_test = data.iloc[45:].copy()
    y_train = target.iloc[:45]
    y_test = target.iloc[45:]

    result = train_and_compare_models(
        X_train,
        X_test,
        y_train,
        y_test,
        task_type="regression",
        target_name="sales",
    )

    assert len(result.metrics) == 3
    assert result.best_model_name is not None
    assert result.best_model is not None
    assert result.selection_metric == "rmse"
    assert result.selection_direction == "minimize"

    for row in result.metrics:
        assert row["rmse"] >= 0
        assert row["mae"] >= 0
        assert row["model"] in result.models



def test_classification_models_train_and_compare():
    data = pd.DataFrame({
        "spend": np.arange(1, 101),
        "visits": np.arange(10, 110),
        "region": ["North", "South"] * 50,
    })

    # Alternate labels so both classes appear in train and test sets.
    target = pd.Series(
        ["High" if value % 2 == 0 else "Low" for value in range(100)]
    )

    X_train = data.iloc[:80].copy()
    X_test = data.iloc[80:].copy()
    y_train = target.iloc[:80]
    y_test = target.iloc[80:]

    result = train_and_compare_models(
        X_train,
        X_test,
        y_train,
        y_test,
        task_type="classification",
        target_name="customer_segment",
    )

    assert len(result.metrics) == 3
    assert result.best_model_name is not None
    assert result.selection_metric == "f1_weighted"
    assert result.selection_direction == "maximize"

    assert set(y_test) == {"High", "Low"}

    for row in result.metrics:
        assert 0 <= row["accuracy"] <= 1
        assert 0 <= row["f1_weighted"] <= 1
        assert row["model"] in result.models


def test_training_rejects_invalid_task():
    X_train = pd.DataFrame({"feature": [1, 2, 3]})
    X_test = pd.DataFrame({"feature": [4, 5]})

    with pytest.raises(ValueError, match="task_type"):
        train_and_compare_models(
            X_train,
            X_test,
            [1, 2, 3],
            [4, 5],
            task_type="forecasting",
        )


def test_training_rejects_mismatched_feature_columns():
    X_train = pd.DataFrame({"sales": [1, 2, 3]})
    X_test = pd.DataFrame({"quantity": [4, 5]})

    with pytest.raises(ValueError, match="feature columns must match"):
        train_and_compare_models(
            X_train,
            X_test,
            [1, 2, 3],
            [4, 5],
            task_type="regression",
        )


def test_training_rejects_mismatched_target_lengths():
    X_train = pd.DataFrame({"feature": [1, 2, 3, 4]})
    X_test = pd.DataFrame({"feature": [5, 6]})

    with pytest.raises(ValueError, match="same row count"):
        train_and_compare_models(
            X_train,
            X_test,
            [1, 2, 3],
            [5, 6],
            task_type="regression",
        )


def test_training_rejects_missing_target_values():
    X_train = pd.DataFrame({"feature": [1, 2, 3, 4]})
    X_test = pd.DataFrame({"feature": [5, 6]})

    with pytest.raises(ValueError, match="missing values"):
        train_and_compare_models(
            X_train,
            X_test,
            [1, np.nan, 3, 4],
            [5, 6],
            task_type="regression",
        )


def test_training_does_not_modify_features():
    X_train = pd.DataFrame({
        "sales": [10, 20, np.nan, 40, 50, 60],
        "region": ["North", "South", "North", "West", "South", "West"],
    })
    X_test = pd.DataFrame({
        "sales": [70, 80],
        "region": ["North", "South"],
    })

    train_original = X_train.copy(deep=True)
    test_original = X_test.copy(deep=True)

    result = train_and_compare_models(
        X_train,
        X_test,
        [10, 20, 30, 40, 50, 60],
        [70, 80],
        task_type="regression",
    )

    pd.testing.assert_frame_equal(X_train, train_original)
    pd.testing.assert_frame_equal(X_test, test_original)
    assert result.best_model is not None


def test_fitted_best_model_can_predict():
    X_train = pd.DataFrame({
        "quantity": np.arange(1, 31),
        "region": ["North", "South", "West"] * 10,
    })
    X_test = pd.DataFrame({
        "quantity": [31, 32],
        "region": ["East", "North"],
    })

    y_train = np.arange(1, 31) * 10
    y_test = np.array([310, 320])

    result = train_and_compare_models(
        X_train,
        X_test,
        y_train,
        y_test,
        task_type="regression",
    )

    predictions = result.best_model.predict(X_test)

    assert len(predictions) == 2
    assert np.isfinite(predictions).all()
