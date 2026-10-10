"""Tests for Business Insights UI helper functions."""

from src.insights.engine import BusinessInsight, InsightsReport
from src.insights.ui import filter_insights_dataframe


def sample_report():
    return InsightsReport(
        dataset_name="retail.csv",
        rows_analyzed=10,
        columns_analyzed=3,
        insights=[
            BusinessInsight(
                category="Data Quality",
                severity="warning",
                title="Missing values found",
                description="Some cells are missing.",
                evidence={"missing_cells": 2},
                recommendation="Review missing values.",
            ),
            BusinessInsight(
                category="Relationships",
                severity="info",
                title="Strong relationship found",
                description="Two numeric columns are strongly related.",
                evidence={"correlation": 0.82},
            ),
            BusinessInsight(
                category="Data Quality",
                severity="info",
                title="Duplicate rows found",
                description="Some rows are duplicated.",
                evidence={"duplicate_rows": 1},
            ),
        ],
    )


def test_all_findings_are_returned_without_filters():
    result = filter_insights_dataframe(sample_report())

    assert len(result) == 3
    assert "Title" in result.columns


def test_filter_by_severity():
    result = filter_insights_dataframe(
        sample_report(),
        severities=["warning"],
    )

    assert len(result) == 1
    assert result.iloc[0]["Title"] == "Missing values found"


def test_filter_by_category():
    result = filter_insights_dataframe(
        sample_report(),
        categories=["Relationships"],
    )

    assert len(result) == 1
    assert result.iloc[0]["Title"] == "Strong relationship found"


def test_combined_filters():
    result = filter_insights_dataframe(
        sample_report(),
        severities=["info"],
        categories=["Data Quality"],
    )

    assert len(result) == 1
    assert result.iloc[0]["Title"] == "Duplicate rows found"


def test_no_matching_findings():
    result = filter_insights_dataframe(
        sample_report(),
        severities=["warning"],
        categories=["Relationships"],
    )

    assert result.empty


def test_filter_does_not_modify_original_report():
    report = sample_report()

    filter_insights_dataframe(report, severities=["warning"])

    assert report.total_insights == 3