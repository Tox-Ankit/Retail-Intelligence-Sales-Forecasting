"""Tests for the deterministic business insights engine."""

import pandas as pd
import pytest

from src.insights.engine import generate_business_insights


@pytest.fixture
def retail_data():
    return pd.DataFrame(
        {
            "sales": [100, 120, 130, 140, 150, 900],
            "profit": [10, 12, 13, 14, 15, 90],
            "region": ["North", "North", "South", "North", "West", "North"],
            "channel": ["Online", "Store", "Online", "Online", "Store", "Online"],
        }
    )


def test_generates_report_with_dataset_metadata(retail_data):
    report = generate_business_insights(retail_data, "retail")

    assert report.dataset_name == "retail"
    assert report.rows_analyzed == 6
    assert report.columns_analyzed == 4
    assert report.total_insights > 0


def test_reports_missing_values():
    data = pd.DataFrame({"sales": [10, None, 20], "region": ["A", "B", None]})

    report = generate_business_insights(data)
    missing = [item for item in report.insights if item.title == "Missing values detected"]

    assert len(missing) == 1
    assert missing[0].evidence["missing_cells"] == 2


def test_reports_duplicate_rows():
    data = pd.DataFrame({"sales": [10, 10, 20], "region": ["A", "A", "B"]})

    report = generate_business_insights(data)
    duplicates = [
        item for item in report.insights if item.title == "Duplicate rows detected"
    ]

    assert len(duplicates) == 1
    assert duplicates[0].evidence["duplicate_rows"] == 1


def test_reports_constant_columns():
    data = pd.DataFrame({"sales": [10, 20, 30], "store": ["A", "A", "A"]})

    report = generate_business_insights(data)
    constant = [
        item for item in report.insights if item.title == "Constant columns detected"
    ]

    assert len(constant) == 1
    assert "store" in constant[0].evidence["columns"]


def test_reports_numeric_summary_and_potential_outliers(retail_data):
    report = generate_business_insights(retail_data)
    titles = [item.title for item in report.insights]

    assert "Range and average for sales" in titles
    assert "Potential outliers in sales" in titles


def test_reports_most_common_category(retail_data):
    report = generate_business_insights(retail_data)
    category_insights = [
        item for item in report.insights if item.category == "Category pattern"
    ]

    assert any(item.evidence["column"] == "region" for item in category_insights)
    region = next(item for item in category_insights if item.evidence["column"] == "region")
    assert region.evidence["top_category"] == "North"


def test_reports_strong_correlation(retail_data):
    report = generate_business_insights(retail_data)
    relationships = [
        item for item in report.insights if item.category == "Relationship"
    ]

    assert any(
        {item.evidence["column_a"], item.evidence["column_b"]} == {"sales", "profit"}
        for item in relationships
    )


def test_empty_dataframe_returns_explanatory_insight():
    report = generate_business_insights(pd.DataFrame())

    assert report.rows_analyzed == 0
    assert report.total_insights == 1
    assert report.insights[0].category == "Dataset overview"


def test_non_dataframe_input_is_rejected():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        generate_business_insights([1, 2, 3])


def test_duplicate_column_names_are_rejected():
    data = pd.DataFrame([[1, 2]], columns=["sales", "sales"])

    with pytest.raises(ValueError, match="unique column names"):
        generate_business_insights(data)


def test_input_data_is_not_modified(retail_data):
    original = retail_data.copy(deep=True)

    generate_business_insights(retail_data)

    pd.testing.assert_frame_equal(retail_data, original)


def test_report_can_be_converted_to_dataframe(retail_data):
    report = generate_business_insights(retail_data)
    table = report.to_dataframe()

    assert list(table.columns) == [
        "Category",
        "Severity",
        "Title",
        "Description",
        "Evidence",
        "Suggested next step",
    ]
    assert len(table) == report.total_insights


def test_data_without_numeric_or_categorical_columns_has_fallback():
    data = pd.DataFrame({"nothing": [None, None]})

    report = generate_business_insights(data)

    assert report.total_insights >= 1
