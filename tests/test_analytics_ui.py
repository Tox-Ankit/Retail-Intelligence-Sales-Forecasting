
import pandas as pd
import pytest

from src.analytics.correlation import analyze_correlations
from src.analytics.descriptive import summarize_dataset
from src.analytics.ui import (
    _get_categorical_columns,
    _get_numeric_columns,
    build_frequency_dataframe,
    categorical_summaries_to_dataframe,
    correlation_matrix_to_dataframe,
    distributions_to_dataframe,
    numerical_summaries_to_dataframe,
)
from src.analytics.distribution import analyze_dataset_distributions


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "sales": [100, 200, 300, 400, 500],
            "profit": [10, 20, 30, 40, 50],
            "region": ["North", "South", "North", "West", "South"],
            "active": [True, False, True, True, False],
        }
    )


def test_numeric_column_detection(sample_data):
    assert _get_numeric_columns(sample_data) == ["sales", "profit"]


def test_categorical_column_detection(sample_data):
    assert _get_categorical_columns(sample_data) == ["region", "active"]


def test_numerical_summary_table(sample_data):
    summary = summarize_dataset(sample_data, dataset_name="test")
    table = numerical_summaries_to_dataframe(summary)

    assert set(table["column_name"]) == {"sales", "profit"}
    assert "mean" in table.columns
    assert "standard_deviation" in table.columns


def test_categorical_summary_table(sample_data):
    summary = summarize_dataset(sample_data, dataset_name="test")
    table = categorical_summaries_to_dataframe(summary)

    assert set(table["Column"]) == {"region", "active"}
    assert "Top frequencies" in table.columns


def test_distribution_table(sample_data):
    results = analyze_dataset_distributions(
        sample_data,
        columns=["sales", "profit"],
    )

    table = distributions_to_dataframe(results)

    assert len(table) == 2
    assert "shape" in table.columns
    assert "Interpretation" in table.columns


def test_correlation_matrix_table(sample_data):
    result = analyze_correlations(
        sample_data,
        columns=["sales", "profit"],
    )

    matrix = correlation_matrix_to_dataframe(result)

    assert list(matrix.index) == ["sales", "profit"]
    assert matrix.loc["sales", "profit"] == pytest.approx(1.0)


def test_frequency_table(sample_data):
    table = build_frequency_dataframe(sample_data, "region", top_n=2)

    assert len(table) == 2
    assert list(table.columns) == ["Category", "Count", "Percentage"]
    assert table.iloc[0]["Category"] == "North"
    assert table.iloc[0]["Count"] == 2


def test_frequency_table_includes_missing_values():
    data = pd.DataFrame({"region": ["North", None, "North", "South"]})

    table = build_frequency_dataframe(data, "region")

    missing_row = table[table["Category"] == "(Missing)"].iloc[0]

    assert missing_row["Count"] == 1
    assert missing_row["Percentage"] == pytest.approx(25.0)


def test_frequency_table_rejects_unknown_column(sample_data):
    with pytest.raises(ValueError, match="Column not found"):
        build_frequency_dataframe(sample_data, "unknown")


def test_frequency_table_rejects_invalid_top_n(sample_data):
    with pytest.raises(ValueError, match="at least 1"):
        build_frequency_dataframe(sample_data, "region", top_n=0)
