
import pandas as pd
import pytest

from src.forecasting.detection import detect_forecasting_columns


@pytest.fixture
def retail_data():
    return pd.DataFrame(
        {
            "Order Date": pd.date_range(
                "2025-01-01",
                periods=6,
                freq="D",
            ).astype(str),
            "Sales": [100, 120, 90, 140, 160, 150],
            "Profit": [10, 15, 8, 20, 25, 22],
            "Store": ["A", "B", "A", "B", "A", "B"],
        }
    )


def test_detects_date_column(retail_data):
    result = detect_forecasting_columns(retail_data)

    assert "Order Date" in [item.column for item in result.date_candidates]


def test_detects_numerical_targets(retail_data):
    result = detect_forecasting_columns(retail_data)

    targets = [item.column for item in result.target_candidates]

    assert "Sales" in targets
    assert "Profit" in targets


def test_business_target_name_ranks_sales_above_generic_numeric_column():
    data = pd.DataFrame(
        {
            "Order Date": pd.date_range("2025-01-01", periods=5).astype(str),
            "Sales": [10, 20, 30, 40, 50],
            "Metric": [2, 4, 6, 8, 10],
        }
    )

    result = detect_forecasting_columns(data)

    assert result.target_candidates[0].column == "Sales"


def test_suggests_daily_frequency(retail_data):
    result = detect_forecasting_columns(retail_data)

    assert result.suggested_frequency == "D"


def test_datetime_dtype_is_detected():
    data = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=5),
            "sales": [10, 20, 30, 40, 50],
        }
    )

    result = detect_forecasting_columns(data)

    assert result.date_candidates[0].column == "date"


def test_numeric_identifiers_are_not_dates():
    data = pd.DataFrame(
        {
            "date_id": [20250101, 20250102, 20250103, 20250104],
            "sales": [10, 20, 30, 40],
        }
    )

    result = detect_forecasting_columns(data)

    assert "date_id" not in [item.column for item in result.date_candidates]


def test_constant_numeric_columns_are_excluded():
    data = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=5).astype(str),
            "sales": [10, 20, 30, 40, 50],
            "constant": [1, 1, 1, 1, 1],
        }
    )

    result = detect_forecasting_columns(data)

    assert "constant" not in [item.column for item in result.target_candidates]


def test_all_missing_columns_are_ignored():
    data = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=5).astype(str),
            "sales": [10, 20, 30, 40, 50],
            "empty": [None] * 5,
        }
    )

    result = detect_forecasting_columns(data)

    assert "empty" not in [item.column for item in result.target_candidates]
    assert "empty" not in [item.column for item in result.date_candidates]


def test_rejects_empty_dataset():
    with pytest.raises(ValueError, match="empty dataset"):
        detect_forecasting_columns(pd.DataFrame())


def test_rejects_duplicate_column_names():
    data = pd.DataFrame(
        [[1, 2], [3, 4], [5, 6]],
        columns=["sales", "sales"],
    )

    with pytest.raises(ValueError, match="duplicate column"):
        detect_forecasting_columns(data)


def test_rejects_invalid_date_threshold(retail_data):
    with pytest.raises(ValueError, match="date_parse_threshold"):
        detect_forecasting_columns(
            retail_data,
            date_parse_threshold=0,
        )


def test_suggestions_include_confidence_and_reason(retail_data):
    result = detect_forecasting_columns(retail_data)

    for item in result.date_candidates + result.target_candidates:
        assert 0 < item.confidence <= 1
        assert item.reason


def test_missing_date_column_produces_note():
    data = pd.DataFrame(
        {
            "Sales": [10, 20, 30, 40, 50],
            "Profit": [2, 3, 4, 5, 6],
        }
    )

    result = detect_forecasting_columns(data)

    assert result.date_candidates == []
    assert any("No suitable date column" in note for note in result.notes)


def test_numeric_text_column_is_not_target():
    data = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=5).astype(str),
            "sales": ["10", "20", "30", "40", "50"],
        }
    )

    result = detect_forecasting_columns(data)

    assert "sales" not in [item.column for item in result.target_candidates]
