"""Tests for PDF business report generation."""

import pytest

from src.insights.engine import BusinessInsight, InsightsReport
from src.reporting.pdf_report import build_business_insights_pdf


def sample_report():
    return InsightsReport(
        dataset_name="retail_sales.csv",
        rows_analyzed=100,
        columns_analyzed=5,
        insights=[
            BusinessInsight(
                category="Data Quality",
                severity="warning",
                title="Missing values detected",
                description="Some dataset cells contain missing values.",
                evidence={"missing_cells": 4},
                recommendation="Review the affected columns.",
            ),
            BusinessInsight(
                category="Relationships",
                severity="info",
                title="Strong correlation detected",
                description="Two numeric columns have a strong relationship.",
                evidence={"correlation": 0.85},
                recommendation="Investigate the business relationship.",
            ),
        ],
    )


def test_pdf_report_returns_valid_pdf_bytes():
    pdf_bytes = build_business_insights_pdf(sample_report())

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000


def test_pdf_report_does_not_modify_original_report():
    report = sample_report()
    original_count = report.total_insights

    build_business_insights_pdf(report)

    assert report.total_insights == original_count
    assert report.dataset_name == "retail_sales.csv"


def test_pdf_report_handles_no_findings():
    report = InsightsReport(
        dataset_name="empty_findings.csv",
        rows_analyzed=0,
        columns_analyzed=3,
        insights=[],
    )

    pdf_bytes = build_business_insights_pdf(report)

    assert pdf_bytes.startswith(b"%PDF")


def test_pdf_report_rejects_invalid_input():
    with pytest.raises(TypeError):
        build_business_insights_pdf("not a report")