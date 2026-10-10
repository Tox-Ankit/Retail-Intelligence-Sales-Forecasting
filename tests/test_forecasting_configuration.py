
import numpy as np
import pandas as pd
import pytest

from src.forecasting.configuration import (
    recommend_forecasting_configuration,
)


@pytest.fixture
def monthly_series():
    dates = pd.date_range(
        "2022-01-31",
        periods=36,
        freq="ME",
        tz="UTC",
    )
    return pd.Series(range(100, 136), index=dates, name="sales")


def test_monthly_configuration_recommends_annual_seasonality(monthly_series):
    config = recommend_forecasting_configuration(
        monthly_series,
        frequency="ME",
    )

    assert config.frequency == "ME"
    assert config.season_length == 12
    assert config.seasonal_models_recommended is True


def test_monthly_configuration_has_default_horizon(monthly_series):
    config = recommend_forecasting_configuration(monthly_series, "ME")

    assert config.forecast_horizon == 3


def test_configuration_respects_custom_horizon(monthly_series):
    config = recommend_forecasting_configuration(
        monthly_series,
        "ME",
        forecast_horizon=6,
    )

    assert config.forecast_horizon == 6


def test_configuration_respects_custom_training_window(monthly_series):
    config = recommend_forecasting_configuration(
        monthly_series,
        "ME",
        forecast_horizon=3,
        initial_train_size=24,
    )

    assert config.initial_train_size == 24


def test_configuration_respects_custom_season_length(monthly_series):
    config = recommend_forecasting_configuration(
        monthly_series,
        "ME",
        season_length=6,
    )

    assert config.season_length == 6
    assert config.seasonal_models_recommended is True


def test_short_history_disables_seasonal_recommendation():
    dates = pd.date_range(
        "2025-01-31",
        periods=10,
        freq="ME",
        tz="UTC",
    )
    series = pd.Series(range(10), index=dates)

    config = recommend_forecasting_configuration(series, "ME")

    assert config.seasonal_models_recommended is False


def test_configuration_reports_missing_values(monthly_series):
    series = monthly_series.copy()
    series.iloc[3] = np.nan

    config = recommend_forecasting_configuration(series, "ME")

    assert config.missing_values == 1
    assert config.ready_for_forecasting is False
    assert any("missing values" in note for note in config.notes)


def test_configuration_rejects_unsorted_series(monthly_series):
    with pytest.raises(ValueError, match="sorted"):
        recommend_forecasting_configuration(
            monthly_series.iloc[::-1],
            "ME",
        )


def test_configuration_rejects_duplicate_timestamps(monthly_series):
    series = monthly_series.copy()
    series.index = list(series.index[:-1]) + [series.index[-2]]

    with pytest.raises(ValueError, match="unique timestamps"):
        recommend_forecasting_configuration(series, "ME")


def test_configuration_rejects_unsupported_frequency(monthly_series):
    with pytest.raises(ValueError, match="Unsupported frequency"):
        recommend_forecasting_configuration(monthly_series, "H")


def test_configuration_rejects_invalid_horizon(monthly_series):
    with pytest.raises(ValueError, match="forecast_horizon"):
        recommend_forecasting_configuration(
            monthly_series,
            "ME",
            forecast_horizon=0,
        )


def test_configuration_rejects_invalid_season_length(monthly_series):
    with pytest.raises(ValueError, match="season_length"):
        recommend_forecasting_configuration(
            monthly_series,
            "ME",
            season_length=1,
        )


def test_configuration_rejects_empty_series():
    dates = pd.DatetimeIndex([], tz="UTC")
    series = pd.Series([], index=dates, dtype=float)

    with pytest.raises(ValueError, match="empty"):
        recommend_forecasting_configuration(series, "ME")


def test_yearly_data_has_no_default_season_length():
    dates = pd.date_range(
        "2010-12-31",
        periods=20,
        freq="YE",
        tz="UTC",
    )
    series = pd.Series(range(20), index=dates)

    config = recommend_forecasting_configuration(series, "YE")

    assert config.season_length is None
    assert config.seasonal_models_recommended is False
