
import numpy as np
import pandas as pd
import pytest

from src.forecasting.engine import run_forecasting


@pytest.fixture
def monthly_sales():
    dates = pd.date_range(
        "2022-01-31",
        periods=48,
        freq="ME",
        tz="UTC",
    )

    seasonal_pattern = [
        80, 75, 90, 100, 115, 130,
        145, 140, 120, 110, 150, 190,
    ]

    values = [
        seasonal_pattern[i % 12] + 2 * i
        for i in range(48)
    ]

    return pd.Series(values, index=dates, name="sales")


def test_engine_returns_forecast(monthly_sales):
    result = run_forecasting(monthly_sales, frequency="ME")

    assert len(result.forecast) == 3
    assert np.isfinite(result.forecast.to_numpy()).all()


def test_engine_forecast_dates_are_in_the_future(monthly_sales):
    result = run_forecasting(monthly_sales, frequency="ME")

    assert result.forecast.index.min() > monthly_sales.index.max()


def test_engine_returns_selected_model(monthly_sales):
    result = run_forecasting(monthly_sales, frequency="ME")

    assert result.selected_model_name in set(
        result.backtest.summary["Model"]
    )


def test_engine_returns_backtest_results(monthly_sales):
    result = run_forecasting(monthly_sales, frequency="ME")

    assert not result.backtest.fold_results.empty
    assert not result.backtest.summary.empty


def test_engine_returns_configuration(monthly_sales):
    result = run_forecasting(monthly_sales, frequency="ME")

    assert result.configuration.frequency == "ME"
    assert result.configuration.ready_for_forecasting is True


def test_engine_respects_custom_horizon(monthly_sales):
    result = run_forecasting(
        monthly_sales,
        frequency="ME",
        forecast_horizon=6,
    )

    assert len(result.forecast) == 6


def test_engine_can_enable_seasonal_models(monthly_sales):
    result = run_forecasting(
        monthly_sales,
        frequency="ME",
        initial_train_size=24,
        forecast_horizon=3,
        season_length=12,
        include_seasonal=True,
    )

    assert {
        "Seasonal Naive",
        "Holt-Winters",
    }.issubset(set(result.backtest.fold_results["Model"]))


def test_engine_can_disable_seasonal_models(monthly_sales):
    result = run_forecasting(
        monthly_sales,
        frequency="ME",
        include_seasonal=False,
    )

    assert "Seasonal Naive" not in set(
        result.backtest.fold_results["Model"]
    )


def test_engine_rejects_missing_values(monthly_sales):
    series = monthly_sales.copy()
    series.iloc[4] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        run_forecasting(series, frequency="ME")


def test_engine_rejects_unsorted_series(monthly_sales):
    with pytest.raises(ValueError, match="sorted"):
        run_forecasting(monthly_sales.iloc[::-1], frequency="ME")


def test_engine_rejects_invalid_frequency(monthly_sales):
    with pytest.raises(ValueError, match="Unsupported frequency"):
        run_forecasting(monthly_sales, frequency="H")


def test_engine_rejects_invalid_smoothing_alpha(monthly_sales):
    with pytest.raises(ValueError, match="smoothing_alpha"):
        run_forecasting(
            monthly_sales,
            frequency="ME",
            smoothing_alpha=0,
        )


def test_engine_exports_forecast_dataframe(monthly_sales):
    result = run_forecasting(monthly_sales, frequency="ME")
    table = result.to_dataframe()

    assert list(table.columns) == [
        "Forecast Date",
        "Forecast",
        "Selected Model",
    ]
    assert len(table) == 3


def test_engine_provides_explanatory_notes(monthly_sales):
    result = run_forecasting(monthly_sales, frequency="ME")

    assert len(result.notes) >= 2
