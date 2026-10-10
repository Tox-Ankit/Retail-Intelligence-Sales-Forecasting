
import pandas as pd
import pytest

from src.analytics.descriptive import (
    summarize_dataset,
    summarize_numerical_column,
    summarize_categorical_column,
)


def test_numerical_summary():
    series = pd.Series([100, 200, 300, 400, 500])

    result = summarize_numerical_column(series, "sales")

    assert result.count == 5
    assert result.mean == 300
    assert result.median == 300
    assert result.minimum == 100
    assert result.maximum == 500
    assert result.range_value == 400
    assert result.q1 == 200
    assert result.q3 == 400
    assert result.iqr == 200


def test_numerical_missing_values():
    series = pd.Series([10.0, 20.0, None, 40.0])

    result = summarize_numerical_column(series, "sales")

    assert result.count == 3
    assert result.missing_count == 1
    assert result.mean == pytest.approx(70 / 3)


def test_empty_numerical_column():
    series = pd.Series([None, None, None])

    result = summarize_numerical_column(series, "sales")

    assert result.count == 0
    assert result.missing_count == 3
    assert result.mean is None
    assert result.median is None


def test_constant_numerical_column():
    series = pd.Series([5, 5, 5, 5])

    result = summarize_numerical_column(series, "quantity")

    assert result.mean == 5
    assert result.standard_deviation == 0
    assert result.iqr == 0


def test_categorical_frequencies():
    series = pd.Series(["East", "West", "East", "North", "West"])

    result = summarize_categorical_column(series, "region")

    assert result.count == 5
    assert result.unique_count == 3
    assert result.frequencies["East"] == 2
    assert result.frequencies["West"] == 2
    assert result.frequencies["North"] == 1


def test_categorical_missing_values():
    series = pd.Series(["East", None, "West", "East"])

    result = summarize_categorical_column(series, "region")

    assert result.count == 3
    assert result.missing_count == 1
    assert result.unique_count == 2


def test_dataset_summary():
    data = pd.DataFrame({
        "sales": [100, 200, 300],
        "region": ["East", "West", "East"],
        "active": [True, False, True],
    })

    result = summarize_dataset(data, "retail")

    assert result.dataset_name == "retail"
    assert result.rows == 3
    assert result.columns == 3
    assert result.numerical_column_count == 1
    assert result.categorical_column_count == 2


def test_input_dataframe_is_not_modified():
    data = pd.DataFrame({
        "sales": [100, 200, None],
        "region": ["East", "West", "East"],
    })

    original = data.copy(deep=True)

    summarize_dataset(data)

    pd.testing.assert_frame_equal(data, original)


def test_invalid_input():
    with pytest.raises(TypeError):
        summarize_dataset([100, 200, 300])


def test_invalid_top_n():
    with pytest.raises(ValueError):
        summarize_categorical_column(
            pd.Series(["East", "West"]),
            "region",
            top_n=0,
        )
