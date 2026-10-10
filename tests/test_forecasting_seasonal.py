
import numpy as np
import pandas as pd
import pytest

from src.forecasting.seasonal import (
    seasonal_naive_forecast,
    holt_winters_forecast,
    train_and_compare_seasonal_forecasters,
)


@pytest.fixture
def seasonal_data():
    dates = pd.date_range(
        "2022-01-31",
        periods=36,
        freq="ME",
        tz="UTC",
    )

    # Repeating annual pattern plus a gradual upward trend.
    seasonal_pattern = [
        80, 75, 90, 100, 115, 130,
        145, 140, 120, 110, 150, 190,
    ]

    values = [
        seasonal_pattern[i % 12] + 2 * i
        for i in range(36)
    ]

    series = pd.Series(values, index=dates, name="sales")

    return series.iloc[:30], series.iloc[30:]


def test_seasonal_naive_repeats_last_season(seasonal_data):
    train, test = seasonal_data

    predictions = seasonal_naive_forecast(
        train,
        test.index,
        season_length=12,
    )

    expected = np.resize(train.to_numpy()[-12:], len(test))

    assert np.allclose(predictions.to_numpy(), expected)


def test_seasonal_naive_preserves_test_index(seasonal_data):
    train, test = seasonal_data

    predictions = seasonal_naive_forecast(
        train,
        test.index,
        season_length=12,
    )

    assert predictions.index.equals(test.index)


def test_holt_winters_returns_finite_predictions(seasonal_data):
    train, test = seasonal_data

    predictions = holt_winters_forecast(
        train,
        test.index,
        season_length=12,
    )

    assert len(predictions) == len(test)
    assert np.isfinite(predictions.to_numpy()).all()


def test_holt_winters_preserves_test_index(seasonal_data):
    train, test = seasonal_data

    predictions = holt_winters_forecast(
        train,
        test.index,
        season_length=12,
    )

    assert predictions.index.equals(test.index)


def test_seasonal_models_can_be_compared(seasonal_data):
    train, test = seasonal_data

    result = train_and_compare_seasonal_forecasters(
        train,
        test,
        season_length=12,
    )

    assert len(result.results) == 2
    assert result.best_model_name in {
        "Seasonal Naive",
        "Holt-Winters",
    }


def test_seasonal_comparison_has_finite_metrics(seasonal_data):
    train, test = seasonal_data

    result = train_and_compare_seasonal_forecasters(
        train,
        test,
        season_length=12,
    )

    for model_result in result.results:
        assert np.isfinite(model_result.mae)
        assert np.isfinite(model_result.rmse)


def test_seasonal_comparison_selects_lowest_rmse(seasonal_data):
    train, test = seasonal_data

    result = train_and_compare_seasonal_forecasters(
        train,
        test,
        season_length=12,
    )

    assert result.best_result.rmse == min(
        item.rmse for item in result.results
    )


def test_seasonal_naive_rejects_short_training_data():
    dates = pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC")
    train = pd.Series(range(5), index=dates)

    with pytest.raises(ValueError, match="shorter than one complete season"):
        seasonal_naive_forecast(
            train,
            pd.date_range("2025-01-06", periods=2, freq="D", tz="UTC"),
            season_length=7,
        )


def test_holt_winters_requires_two_seasons():
    dates = pd.date_range("2025-01-31", periods=12, freq="ME", tz="UTC")
    train = pd.Series(range(12), index=dates)

    with pytest.raises(ValueError, match="two complete seasons"):
        holt_winters_forecast(
            train,
            pd.date_range("2026-01-31", periods=2, freq="ME", tz="UTC"),
            season_length=12,
        )


def test_multiplicative_seasonality_rejects_zero_values():
    dates = pd.date_range("2022-01-31", periods=24, freq="ME", tz="UTC")
    train = pd.Series([0] + list(range(1, 24)), index=dates)

    with pytest.raises(ValueError, match="strictly positive"):
        holt_winters_forecast(
            train,
            pd.date_range("2024-01-31", periods=2, freq="ME", tz="UTC"),
            season_length=12,
            seasonal="mul",
        )


def test_seasonal_comparison_rejects_missing_values(seasonal_data):
    train, test = seasonal_data
    train = train.copy()
    train.iloc[3] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        train_and_compare_seasonal_forecasters(
            train,
            test,
            season_length=12,
        )


def test_seasonal_comparison_rejects_overlapping_dates(seasonal_data):
    train, test = seasonal_data
    overlapping_test = test.copy()
    overlapping_test.index = train.index[-len(test):]

    with pytest.raises(ValueError, match="strictly after"):
        train_and_compare_seasonal_forecasters(
            train,
            overlapping_test,
            season_length=12,
        )
