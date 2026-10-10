
import pandas as pd
import pytest

from src.forecasting.preparation import (
    prepare_time_series,
    split_time_series,
)


@pytest.fixture
def monthly_data():
    return pd.DataFrame(
        {
            "date": [
                "2025-01-05",
                "2025-01-20",
                "2025-02-10",
                "2025-04-15",
                "2025-05-01",
            ],
            "sales": [100, 150, 200, 400, 500],
        }
    )


def test_prepare_time_series_aggregates_monthly_sales(monthly_data):
    result = prepare_time_series(
        monthly_data,
        "date",
        "sales",
        frequency="ME",
        aggregation="sum",
    )

    assert len(result.series) == 5
    assert result.series.iloc[0] == 250
    assert result.series.iloc[1] == 200
    assert result.series.iloc[2] == 0 or pd.isna(result.series.iloc[2])
    assert result.frequency == "ME"


def test_prepare_time_series_preserves_missing_periods(monthly_data):
    result = prepare_time_series(
        monthly_data,
        "date",
        "sales",
        frequency="ME",
        missing_period_strategy="none",
    )

    assert result.missing_periods == 1
    assert pd.isna(result.series.loc["2025-03-31 00:00:00+00:00"])


def test_prepare_time_series_can_fill_missing_periods_with_zero(monthly_data):
    result = prepare_time_series(
        monthly_data,
        "date",
        "sales",
        frequency="ME",
        missing_period_strategy="zero",
    )

    assert result.series.loc["2025-03-31 00:00:00+00:00"] == 0
    assert result.filled_periods == 1


def test_prepare_time_series_mean_aggregation():
    data = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-15"],
            "sales": [10, 30],
        }
    )

    result = prepare_time_series(
        data, "date", "sales", frequency="ME", aggregation="mean"
    )

    assert result.series.iloc[0] == 20


def test_prepare_time_series_count_aggregation():
    data = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-15", "2025-02-01"],
            "sales": [10, 30, 40],
        }
    )

    result = prepare_time_series(
        data, "date", "sales", frequency="ME", aggregation="count"
    )

    assert result.series.iloc[0] == 2
    assert result.series.iloc[1] == 1


def test_prepare_time_series_drops_invalid_rows():
    data = pd.DataFrame(
        {
            "date": ["2025-01-01", "invalid", "2025-02-01"],
            "sales": [10, 20, "not a number"],
        }
    )

    result = prepare_time_series(data, "date", "sales")

    assert result.input_rows == 3
    assert result.valid_rows == 1
    assert result.dropped_rows == 2


def test_prepare_time_series_rejects_missing_column(monthly_data):
    with pytest.raises(ValueError, match="was not found"):
        prepare_time_series(monthly_data, "missing_date", "sales")


def test_prepare_time_series_rejects_invalid_aggregation(monthly_data):
    with pytest.raises(ValueError, match="aggregation"):
        prepare_time_series(
            monthly_data, "date", "sales", aggregation="median"
        )


def test_prepare_time_series_rejects_empty_data():
    with pytest.raises(ValueError, match="empty dataset"):
        prepare_time_series(pd.DataFrame(), "date", "sales")


def test_split_time_series_preserves_chronological_order():
    index = pd.date_range("2025-01-31", periods=10, freq="ME", tz="UTC")
    series = pd.Series(range(10), index=index, name="sales")

    train, test = split_time_series(series, test_size=0.2)

    assert len(train) == 8
    assert len(test) == 2
    assert train.index.max() < test.index.min()
    assert train.iloc[-1] == 7
    assert test.iloc[0] == 8


def test_split_time_series_rejects_missing_values():
    index = pd.date_range("2025-01-31", periods=5, freq="ME", tz="UTC")
    series = pd.Series([1, 2, None, 4, 5], index=index)

    with pytest.raises(ValueError, match="missing values"):
        split_time_series(series)


def test_split_time_series_rejects_too_few_observations():
    index = pd.date_range("2025-01-31", periods=4, freq="ME", tz="UTC")
    series = pd.Series(range(4), index=index)

    with pytest.raises(ValueError, match="At least 5"):
        split_time_series(series)


def test_split_time_series_rejects_invalid_test_size():
    index = pd.date_range("2025-01-31", periods=10, freq="ME", tz="UTC")
    series = pd.Series(range(10), index=index)

    with pytest.raises(ValueError, match="test_size"):
        split_time_series(series, test_size=1.0)
