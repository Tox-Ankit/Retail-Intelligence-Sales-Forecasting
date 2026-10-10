
import numpy as np
import pandas as pd
import pytest

from src.forecasting.workflow import run_forecasting_workflow


@pytest.fixture
def monthly_retail_data():
    dates = pd.date_range(
        "2022-01-31",
        periods=48,
        freq="ME",
    )

    seasonal_pattern = [
        80, 75, 90, 100, 115, 130,
        145, 140, 120, 110, 150, 190,
    ]

    sales = [
        seasonal_pattern[i % 12] + 2 * i
        for i in range(48)
    ]

    return pd.DataFrame(
        {
            "Order Date": dates,
            "Sales": sales,
            "Profit": [value * 0.2 for value in sales],
        }
    )


def test_workflow_returns_future_forecast(monthly_retail_data):
    result = run_forecasting_workflow(
        monthly_retail_data,
        date_column="Order Date",
        value_column="Sales",
        frequency="ME",
        include_seasonal=False,
    )

    assert len(result.forecast) == 3
    assert np.isfinite(result.forecast.to_numpy()).all()


def test_workflow_returns_detection_suggestions(monthly_retail_data):
    result = run_forecasting_workflow(
        monthly_retail_data,
        "Order Date",
        "Sales",
        "ME",
        include_seasonal=False,
    )

    assert "Order Date" in [
        item.column for item in result.detection.date_candidates
    ]
    assert "Sales" in [
        item.column for item in result.detection.target_candidates
    ]


def test_workflow_returns_model_comparison(monthly_retail_data):
    result = run_forecasting_workflow(
        monthly_retail_data,
        "Order Date",
        "Sales",
        "ME",
        include_seasonal=False,
    )

    assert not result.model_comparison.empty
    assert result.selected_model_name in set(
        result.model_comparison["Model"]
    )


def test_workflow_returns_downloadable_forecast_table(monthly_retail_data):
    result = run_forecasting_workflow(
        monthly_retail_data,
        "Order Date",
        "Sales",
        "ME",
        include_seasonal=False,
    )

    table = result.forecast_dataframe()

    assert list(table.columns) == [
        "Forecast Date",
        "Forecast",
        "Selected Model",
    ]
    assert len(table) == 3


def test_workflow_respects_custom_horizon(monthly_retail_data):
    result = run_forecasting_workflow(
        monthly_retail_data,
        "Order Date",
        "Sales",
        "ME",
        forecast_horizon=5,
        include_seasonal=False,
    )

    assert len(result.forecast) == 5


def test_workflow_respects_aggregation():
    data = pd.DataFrame(
        {
            "date": [
                "2025-01-01",
                "2025-01-15",
                "2025-02-01",
                "2025-03-01",
                "2025-04-01",
                "2025-05-01",
                "2025-06-01",
                "2025-07-01",
            ],
            "sales": [10, 30, 40, 50, 60, 70, 80, 90],
        }
    )

    result = run_forecasting_workflow(
        data,
        "date",
        "sales",
        "ME",
        aggregation="mean",
        include_seasonal=False,
    )

    assert result.preparation.series.iloc[0] == 20


def test_workflow_rejects_missing_periods_by_default():
    dates = pd.date_range("2022-01-31", periods=48, freq="ME")
    dates = dates.delete(12)

    data = pd.DataFrame(
        {
            "date": dates,
            "sales": range(len(dates)),
        }
    )

    with pytest.raises(ValueError, match="missing periods"):
        run_forecasting_workflow(
            data,
            "date",
            "sales",
            "ME",
            missing_period_strategy="none",
            include_seasonal=False,
        )


def test_workflow_can_fill_missing_periods_explicitly():
    dates = pd.date_range("2022-01-31", periods=48, freq="ME")
    dates = dates.delete(12)

    data = pd.DataFrame(
        {
            "date": dates,
            "sales": range(len(dates)),
        }
    )

    result = run_forecasting_workflow(
        data,
        "date",
        "sales",
        "ME",
        missing_period_strategy="zero",
        include_seasonal=False,
    )

    assert not result.forecast.isna().any()
    assert result.preparation.filled_periods == 1


def test_workflow_rejects_invalid_date_column(monthly_retail_data):
    with pytest.raises(ValueError, match="was not found"):
        run_forecasting_workflow(
            monthly_retail_data,
            "Unknown Date",
            "Sales",
            "ME",
            include_seasonal=False,
        )


def test_workflow_rejects_invalid_target_column(monthly_retail_data):
    with pytest.raises(ValueError, match="was not found"):
        run_forecasting_workflow(
            monthly_retail_data,
            "Order Date",
            "Unknown Target",
            "ME",
            include_seasonal=False,
        )


def test_workflow_exposes_configuration(monthly_retail_data):
    result = run_forecasting_workflow(
        monthly_retail_data,
        "Order Date",
        "Sales",
        "ME",
        forecast_horizon=4,
        include_seasonal=False,
    )

    assert result.configuration.forecast_horizon == 4
    assert result.configuration.ready_for_forecasting is True


def test_workflow_exposes_preparation_metadata(monthly_retail_data):
    result = run_forecasting_workflow(
        monthly_retail_data,
        "Order Date",
        "Sales",
        "ME",
        include_seasonal=False,
    )

    assert result.preparation.input_rows == 48
    assert result.preparation.dropped_rows == 0
    assert result.preparation.valid_rows == 48
