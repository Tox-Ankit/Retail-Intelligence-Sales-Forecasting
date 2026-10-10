"""Deterministic, evidence-based business insights for tabular datasets."""

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.analytics.correlation import analyze_correlations
from src.analytics.distribution import analyze_dataset_distributions


@dataclass
class BusinessInsight:
    """One evidence-based finding generated from a dataset."""

    category: str
    severity: str
    title: str
    description: str
    evidence: dict[str, Any] = field(default_factory=dict)
    recommendation: str | None = None


@dataclass
class InsightsReport:
    """Collection of insights for one dataset."""

    dataset_name: str
    rows_analyzed: int
    columns_analyzed: int
    insights: list[BusinessInsight] = field(default_factory=list)

    @property
    def total_insights(self) -> int:
        return len(self.insights)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert insights into a download-friendly table."""
        rows = []
        for insight in self.insights:
            rows.append(
                {
                    "Category": insight.category,
                    "Severity": insight.severity,
                    "Title": insight.title,
                    "Description": insight.description,
                    "Evidence": str(insight.evidence),
                    "Suggested next step": insight.recommendation or "",
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "Category",
                "Severity",
                "Title",
                "Description",
                "Evidence",
                "Suggested next step",
            ],
        )


def _safe_number(value: Any) -> float | None:
    """Return a finite float when possible."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if np.isfinite(number) else None


def _numeric_columns(data: pd.DataFrame) -> list[str]:
    """Return numeric columns while excluding booleans."""
    return [
        column
        for column in data.select_dtypes(include="number").columns
        if not pd.api.types.is_bool_dtype(data[column].dtype)
    ]


def _categorical_columns(data: pd.DataFrame) -> list[str]:
    """Return text, category, and Boolean columns."""
    return [
        column
        for column in data.columns
        if (
            pd.api.types.is_object_dtype(data[column].dtype)
            or isinstance(data[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(data[column].dtype)
            or pd.api.types.is_bool_dtype(data[column].dtype)
        )
    ]


def _add_quality_insights(report: InsightsReport, data: pd.DataFrame) -> None:
    """Report missing data, duplicates, and constant columns."""
    row_count = len(data)

    missing_cells = int(data.isna().sum().sum())
    total_cells = int(data.shape[0] * data.shape[1])
    if missing_cells:
        percentage = (missing_cells / total_cells * 100) if total_cells else 0.0
        report.insights.append(
            BusinessInsight(
                category="Data quality",
                severity="warning" if percentage >= 5 else "info",
                title="Missing values detected",
                description=(
                    f"The dataset contains {missing_cells:,} missing cells "
                    f"({percentage:.2f}% of all cells)."
                ),
                evidence={
                    "missing_cells": missing_cells,
                    "total_cells": total_cells,
                    "missing_percentage": round(percentage, 2),
                    "columns": {
                        str(column): int(count)
                        for column, count in data.isna().sum().items()
                        if count > 0
                    },
                },
                recommendation=(
                    "Review missingness by column and decide whether to impute, "
                    "retain, or remove affected records before modeling."
                ),
            )
        )

    duplicate_count = int(data.duplicated().sum())
    if duplicate_count:
        duplicate_percentage = (
            duplicate_count / row_count * 100 if row_count else 0.0
        )
        report.insights.append(
            BusinessInsight(
                category="Data quality",
                severity="warning",
                title="Duplicate rows detected",
                description=(
                    f"{duplicate_count:,} rows are exact duplicates "
                    f"({duplicate_percentage:.2f}% of rows)."
                ),
                evidence={
                    "duplicate_rows": duplicate_count,
                    "duplicate_percentage": round(duplicate_percentage, 2),
                },
                recommendation=(
                    "Check whether repeated rows represent valid repeated events "
                    "or accidental duplication before removing them."
                ),
            )
        )

    constant_columns = [
        str(column)
        for column in data.columns
        if data[column].nunique(dropna=False) <= 1
    ]
    if constant_columns:
        report.insights.append(
            BusinessInsight(
                category="Data quality",
                severity="info",
                title="Constant columns detected",
                description=(
                    "These columns have one distinct value and may provide "
                    "little analytical value: " + ", ".join(constant_columns) + "."
                ),
                evidence={"columns": constant_columns},
                recommendation=(
                    "Confirm whether these fields are required for reporting; "
                    "consider excluding them from predictive features if they "
                    "carry no useful variation."
                ),
            )
        )


def _add_numerical_insights(
    report: InsightsReport, data: pd.DataFrame, numeric_columns: list[str]
) -> None:
    """Report numerical ranges, skew, and potential outliers."""
    if not numeric_columns:
        return

    try:
        summaries = analyze_dataset_distributions(data, columns=numeric_columns)
    except (ValueError, TypeError, KeyError):
        summaries = {}

    for column in numeric_columns:
        series = pd.to_numeric(data[column], errors="coerce")
        finite = series.replace([np.inf, -np.inf], np.nan).dropna()
        if finite.empty:
            continue

        minimum = _safe_number(finite.min())
        maximum = _safe_number(finite.max())
        mean = _safe_number(finite.mean())
        median = _safe_number(finite.median())

        if minimum is not None and maximum is not None and mean is not None:
            report.insights.append(
                BusinessInsight(
                    category="Numerical summary",
                    severity="info",
                    title=f"Range and average for {column}",
                    description=(
                        f"{column} ranges from {minimum:,.4g} to {maximum:,.4g}; "
                        f"its mean is {mean:,.4g}"
                        + (
                            f" and its median is {median:,.4g}."
                            if median is not None
                            else "."
                        )
                    ),
                    evidence={
                        "column": str(column),
                        "valid_values": int(finite.size),
                        "minimum": minimum,
                        "maximum": maximum,
                        "mean": mean,
                        "median": median,
                    },
                    recommendation=(
                        "Compare the mean and median and review the distribution "
                        "before choosing summary statistics or transformations."
                    ),
                )
            )

        summary = summaries.get(column)
        if summary is not None and getattr(summary, "outlier_count", 0):
            outlier_count = int(summary.outlier_count)
            outlier_percentage = float(summary.outlier_percentage)
            report.insights.append(
                BusinessInsight(
                    category="Potential anomaly",
                    severity="warning" if outlier_percentage >= 5 else "info",
                    title=f"Potential outliers in {column}",
                    description=(
                        f"{outlier_count:,} values in {column} fall outside "
                        f"the distribution module's IQR-based bounds "
                        f"({outlier_percentage:.2f}% of valid observations)."
                    ),
                    evidence={
                        "column": str(column),
                        "outlier_count": outlier_count,
                        "outlier_percentage": round(outlier_percentage, 2),
                        "lower_bound": _safe_number(
                            getattr(summary, "lower_bound", None)
                        ),
                        "upper_bound": _safe_number(
                            getattr(summary, "upper_bound", None)
                        ),
                    },
                    recommendation=(
                        "Inspect flagged records in context. IQR flags are "
                        "potential outliers, not proof of data errors."
                    ),
                )
            )


def _add_categorical_insights(
    report: InsightsReport, data: pd.DataFrame, categorical_columns: list[str]
) -> None:
    """Describe dominant categories without implying business causation."""
    for column in categorical_columns:
        series = data[column].dropna()
        if series.empty:
            continue

        counts = series.value_counts(dropna=True)
        if counts.empty:
            continue

        top_value = counts.index[0]
        top_count = int(counts.iloc[0])
        valid_count = int(series.size)
        top_percentage = top_count / valid_count * 100 if valid_count else 0.0

        report.insights.append(
            BusinessInsight(
                category="Category pattern",
                severity="info",
                title=f"Most common value in {column}",
                description=(
                    f"{top_value!s} is the most frequent value in {column}, "
                    f"appearing {top_count:,} times "
                    f"({top_percentage:.2f}% of non-missing values)."
                ),
                evidence={
                    "column": str(column),
                    "top_category": str(top_value),
                    "top_count": top_count,
                    "valid_values": valid_count,
                    "top_percentage": round(top_percentage, 2),
                    "distinct_non_missing_values": int(counts.size),
                },
                recommendation=(
                    "Compare this category with relevant outcome metrics "
                    "before deciding whether its prominence is beneficial."
                ),
            )
        )


def _add_correlation_insights(
    report: InsightsReport, data: pd.DataFrame, numeric_columns: list[str]
) -> None:
    """Report the strongest observed numerical relationships."""
    if len(numeric_columns) < 2:
        return

    try:
        result = analyze_correlations(
            data,
            columns=numeric_columns,
            method="pearson",
            minimum_periods=3,
            top_n=5,
        )
    except (ValueError, TypeError, KeyError):
        return

    seen: set[tuple[str, str]] = set()
    for first_column, row in result.matrix.items():
        if not isinstance(row, dict):
            continue
        for second_column, raw_value in row.items():
            if first_column == second_column:
                continue
            value = _safe_number(raw_value)
            if value is None:
                continue
            pair = tuple(sorted((str(first_column), str(second_column))))
            if pair in seen:
                continue
            seen.add(pair)

            strength = abs(value)
            if strength < 0.7:
                continue

            direction = "positive" if value > 0 else "negative"
            report.insights.append(
                BusinessInsight(
                    category="Relationship",
                    severity="info",
                    title=(
                        f"Strong {direction} correlation: "
                        f"{first_column} and {second_column}"
                    ),
                    description=(
                        f"{first_column} and {second_column} have a Pearson "
                        f"correlation of {value:.3f}, indicating a strong "
                        f"{direction} linear association in this dataset."
                    ),
                    evidence={
                        "column_a": str(first_column),
                        "column_b": str(second_column),
                        "pearson_correlation": round(value, 4),
                        "absolute_correlation": round(strength, 4),
                    },
                    recommendation=(
                        "Investigate this relationship using domain knowledge "
                        "and visualizations. Correlation alone does not establish "
                        "causation."
                    ),
                )
            )


def generate_business_insights(
    data: pd.DataFrame, dataset_name: str = "dataset"
) -> InsightsReport:
    """Generate deterministic insights from a pandas DataFrame.

    The function does not modify the input data. Findings are descriptive
    signals for review, not causal conclusions or financial advice.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if data.columns.duplicated().any():
        raise ValueError("The dataset must have unique column names.")

    report = InsightsReport(
        dataset_name=str(dataset_name),
        rows_analyzed=int(len(data)),
        columns_analyzed=int(len(data.columns)),
    )

    if data.empty or data.shape[1] == 0:
        report.insights.append(
            BusinessInsight(
                category="Dataset overview",
                severity="warning",
                title="No data available for analysis",
                description=(
                    "The selected dataset has no rows or no columns, "
                    "so business insights could not be generated."
                ),
                evidence={
                    "rows": int(len(data)),
                    "columns": int(len(data.columns)),
                },
                recommendation="Upload a dataset containing rows and columns.",
            )
        )
        return report

    _add_quality_insights(report, data)

    numeric_columns = _numeric_columns(data)
    categorical_columns = _categorical_columns(data)

    _add_numerical_insights(report, data, numeric_columns)
    _add_categorical_insights(report, data, categorical_columns)
    _add_correlation_insights(report, data, numeric_columns)

    if not report.insights:
        report.insights.append(
            BusinessInsight(
                category="Dataset overview",
                severity="info",
                title="No notable patterns detected",
                description=(
                    "The current rules did not identify any findings to report. "
                    "This does not guarantee that the dataset has no issues "
                    "or meaningful patterns."
                ),
                evidence={
                    "rows": int(len(data)),
                    "columns": int(len(data.columns)),
                },
                recommendation=(
                    "Review the descriptive statistics and choose analyses "
                    "that match the dataset's business context."
                ),
            )
        )

    return report
