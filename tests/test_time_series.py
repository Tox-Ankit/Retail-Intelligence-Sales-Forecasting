
import pandas as pd
import pytest

from src.analytics.time_series import analyze_time_series


def test_monthly_sales_summary():
    data = pd.DataFrame({
        "date": [
            "2025-01-10",
            "2025-02-10",
            "2025-03-10",
            "2025-04-10",
        ],
        "sales": [100, 200, 300, 400],
    })

    result = analyze_time_series(
        data, "date", "sales", frequency="M"
    )

    assert result.observations == 4
    assert result.total == 1000
    assert result.trend == "increasing"
    assert result.period_change == 300


def test_invalid_dates_are_excluded():
    data = pd.DataFrame({
        "date": ["2025-01-01", "invalid", "2025-02-01"],
        "sales": [100, 500, 200],
    })

    result = analyze_time_series(data, "date", "sales")

    assert result.observations == 2
    assert result.total == 300


def test_empty_valid_data():
    data = pd.DataFrame({
        "date": ["invalid", None],
        "sales": [100, 200],
    })

    result = analyze_time_series(data, "date", "sales")

    assert result.observations == 0


def test_invalid_frequency():
    data = pd.DataFrame({
        "date": ["2025-01-01", "2025-02-01"],
        "sales": [100, 200],
    })

    with pytest.raises(ValueError):
        analyze_time_series(
            data, "date", "sales", frequency="INVALID"
        )


def test_missing_column():
    data = pd.DataFrame({
        "date": ["2025-01-01"],
        "sales": [100],
    })

    with pytest.raises(ValueError):
        analyze_time_series(data, "missing", "sales")


def test_input_is_not_modified():
    data = pd.DataFrame({
        "date": ["2025-01-01", "2025-02-01"],
        "sales": [100, 200],
    })

    original = data.copy(deep=True)

    analyze_time_series(data, "date", "sales")

    pd.testing.assert_frame_equal(data, original)


def test_zero_first_period_percentage_change():
    data = pd.DataFrame({
        "date": ["2025-01-01", "2025-02-01"],
        "sales": [0, 100],
    })

    result = analyze_time_series(data, "date", "sales")

    assert result.period_change == 100
    assert result.period_change_percentage is None
