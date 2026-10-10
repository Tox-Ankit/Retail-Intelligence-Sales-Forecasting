import numpy as np
import pandas as pd

from src.quality.models import QualityFinding


IQR_MULTIPLIER = 1.5
Z_SCORE_THRESHOLD = 3.0
MODIFIED_Z_THRESHOLD = 3.5


def _get_numeric_columns(data: pd.DataFrame) -> list[str]:
    """Return numerical columns suitable for outlier analysis."""

    return data.select_dtypes(include="number").columns.tolist()


def check_iqr_outliers(
    data: pd.DataFrame,
) -> list[QualityFinding]:
    """
    Detect outliers using the Interquartile Range method.
    """

    findings = []

    for column in _get_numeric_columns(data):
        series = data[column].dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        # No variation means there is nothing to detect.
        if iqr == 0:
            continue

        lower_bound = q1 - IQR_MULTIPLIER * iqr
        upper_bound = q3 + IQR_MULTIPLIER * iqr

        outlier_mask = (
            (series < lower_bound)
            | (series > upper_bound)
        )

        outlier_count = int(outlier_mask.sum())

        if outlier_count == 0:
            continue

        percentage = (
            outlier_count / len(series) * 100
        )

        findings.append(
            QualityFinding(
                issue_type="iqr_outliers",
                severity="info",
                message=(
                    f"Column '{column}' contains "
                    f"{outlier_count} potential outliers "
                    f"using the IQR method ({percentage:.2f}%)."
                ),
                column_name=column,
                affected_rows=outlier_count,
                affected_percentage=percentage,
                suggested_action=(
                    "Review these observations before deciding "
                    "whether to retain, transform, cap, or remove them."
                ),
                details={
                    "method": "IQR",
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(iqr),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                },
            )
        )

    return findings


def check_z_score_outliers(
    data: pd.DataFrame,
) -> list[QualityFinding]:
    """
    Detect outliers using the standard Z-score method.
    """

    findings = []

    for column in _get_numeric_columns(data):
        series = data[column].dropna()

        if len(series) < 4:
            continue

        mean = series.mean()
        std = series.std()

        if std == 0 or pd.isna(std):
            continue

        z_scores = (series - mean) / std

        outlier_mask = z_scores.abs() > Z_SCORE_THRESHOLD
        outlier_count = int(outlier_mask.sum())

        if outlier_count == 0:
            continue

        percentage = (
            outlier_count / len(series) * 100
        )

        findings.append(
            QualityFinding(
                issue_type="z_score_outliers",
                severity="info",
                message=(
                    f"Column '{column}' contains "
                    f"{outlier_count} potential outliers "
                    f"using the Z-score method ({percentage:.2f}%)."
                ),
                column_name=column,
                affected_rows=outlier_count,
                affected_percentage=percentage,
                suggested_action=(
                    "Review these observations and consider "
                    "whether the distribution is approximately normal."
                ),
                details={
                    "method": "Z-score",
                    "threshold": Z_SCORE_THRESHOLD,
                    "mean": float(mean),
                    "std": float(std),
                },
            )
        )

    return findings


def check_modified_z_score_outliers(
    data: pd.DataFrame,
) -> list[QualityFinding]:
    """
    Detect outliers using the Modified Z-score method.

    This uses the median and MAD, making it more robust to
    extreme observations.
    """

    findings = []

    for column in _get_numeric_columns(data):
        series = data[column].dropna()

        if len(series) < 4:
            continue

        median = series.median()

        absolute_deviation = (series - median).abs()
        mad = absolute_deviation.median()

        if mad == 0 or pd.isna(mad):
            continue

        modified_z_scores = (
            0.6745 * (series - median) / mad
        )

        outlier_mask = (
            modified_z_scores.abs()
            > MODIFIED_Z_THRESHOLD
        )

        outlier_count = int(outlier_mask.sum())

        if outlier_count == 0:
            continue

        percentage = (
            outlier_count / len(series) * 100
        )

        findings.append(
            QualityFinding(
                issue_type="modified_z_score_outliers",
                severity="info",
                message=(
                    f"Column '{column}' contains "
                    f"{outlier_count} potential outliers "
                    f"using the Modified Z-score method "
                    f"({percentage:.2f}%)."
                ),
                column_name=column,
                affected_rows=outlier_count,
                affected_percentage=percentage,
                suggested_action=(
                    "Review these observations using the robust "
                    "Modified Z-score method before treatment."
                ),
                details={
                    "method": "Modified Z-score",
                    "threshold": MODIFIED_Z_THRESHOLD,
                    "median": float(median),
                    "mad": float(mad),
                },
            )
        )

    return findings


def check_outliers(
    data: pd.DataFrame,
) -> list[QualityFinding]:
    """Run all supported outlier detection methods."""

    findings = []

    findings.extend(
        check_iqr_outliers(data)
    )

    findings.extend(
        check_z_score_outliers(data)
    )

    findings.extend(
        check_modified_z_score_outliers(data)
    )

    return findings