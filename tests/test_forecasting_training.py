
import numpy as np
import pandas as pd
import pytest

from src.forecasting.models import ForecastComparisonResult
from src.forecasting.training import train_and_compare_forecasters


@pytest.fixture
def train_test_data():
    train_dates = pd.date_range(
        "2025-01-31",
        periods=12,
        freq="ME",
        tz="UTC",
    )

    test_dates = pd.date_range(
        "2026-01-31",
        periods=3,
        freq="ME",
        tz="UTC",
    )

    train = pd.Series(
        [100, 110, 120, 115, 130, 140, 135, 150, 160, 155, 170, 180],
        index=train_dates,
        name="sales",
    )

    test = pd.Series(
        [185, 190, 200],
        index=test_dates,
        name="sales",
    )

    return train, test


def test_three_forecasting_models_are_compared(train_test_data):
    train, test = train_test_data

    result = train_and_compare_forecasters(train, test)

    assert len(result.results) == 3
    assert result.best_model_name in {
        "Naive",
        "Moving Average",
        "Exponential Smoothing",
    }


def test_predictions_have_correct_length(train_test_data):
    train, test = train_test_data

    result = train_and_compare_forecasters(train, test)

    for model_result in result.results:
        assert len(model_result.predictions) == len(test)
        assert model_result.predictions.index.equals(test.index)


def test_predictions_are_finite(train_test_data):
    train, test = train_test_data

    result = train_and_compare_forecasters(train, test)

    for model_result in result.results:
        assert np.isfinite(model_result.predictions.to_numpy()).all()
        assert np.isfinite(model_result.mae)
        assert np.isfinite(model_result.rmse)


def test_naive_forecast_uses_last_training_value(train_test_data):
    train, test = train_test_data

    result = train_and_compare_forecasters(train, test)
    naive = next(r for r in result.results if r.model_name == "Naive")

    assert (naive.predictions == train.iloc[-1]).all()


def test_moving_average_uses_recent_training_values(train_test_data):
    train, test = train_test_data

    result = train_and_compare_forecasters(
        train,
        test,
        moving_average_window=3,
    )

    moving_average = next(
        r for r in result.results if r.model_name == "Moving Average"
    )

    expected = train.iloc[-3:].mean()

    assert np.allclose(moving_average.predictions.to_numpy(), expected)


def test_comparison_table_has_expected_columns(train_test_data):
    train, test = train_test_data

    result = train_and_compare_forecasters(train, test)
    table = result.to_dataframe()

    assert list(table.columns) == ["Model", "MAE", "RMSE", "MAPE (%)"]
    assert len(table) == 3
    assert table["RMSE"].is_monotonic_increasing


def test_best_model_has_lowest_rmse(train_test_data):
    train, test = train_test_data

    result = train_and_compare_forecasters(train, test)

    best = result.best_result

    assert best.rmse == min(r.rmse for r in result.results)


def test_mape_is_unavailable_when_actual_contains_zero():
    dates = pd.date_range(
        "2025-01-31",
        periods=6,
        freq="ME",
        tz="UTC",
    )

    train = pd.Series([10, 12, 14, 15], index=dates[:4])
    test = pd.Series([0, 16], index=dates[4:])

    result = train_and_compare_forecasters(train, test)

    assert all(r.mape is None for r in result.results)
    assert all(r.notes for r in result.results)


def test_invalid_smoothing_alpha_is_rejected(train_test_data):
    train, test = train_test_data

    with pytest.raises(ValueError, match="smoothing_alpha"):
        train_and_compare_forecasters(
            train,
            test,
            smoothing_alpha=0,
        )


def test_invalid_moving_average_window_is_rejected(train_test_data):
    train, test = train_test_data

    with pytest.raises(ValueError, match="moving_average_window"):
        train_and_compare_forecasters(
            train,
            test,
            moving_average_window=100,
        )


def test_missing_values_are_rejected(train_test_data):
    train, test = train_test_data
    train.iloc[3] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        train_and_compare_forecasters(train, test)


def test_future_training_data_is_rejected():
    dates = pd.date_range(
        "2025-01-31",
        periods=6,
        freq="ME",
        tz="UTC",
    )

    train = pd.Series([10, 11, 12, 13], index=dates[:4])
    test = pd.Series([14, 15], index=dates[2:4])

    with pytest.raises(ValueError, match="strictly after"):
        train_and_compare_forecasters(train, test)


def test_comparison_result_rejects_missing_best_model():
    result = ForecastComparisonResult(
        results=[],
        best_model_name="Missing",
    )

    with pytest.raises(ValueError, match="not found"):
        _ = result.best_result
