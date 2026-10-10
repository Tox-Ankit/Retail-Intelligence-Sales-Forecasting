import pandas as pd

from src.quality.models import QualityFinding


PERCENTAGE_KEYWORDS = {
    "percentage",
    "percent",
    "pct",
    "rate",
    "ratio",
    "margin",
}


AGE_KEYWORDS = {
    "age",
}


def _normalized_name(column_name: str) -> str:
    """Normalize a column name for heuristic checks."""

    return (
        column_name
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def check_suspicious_negative_values(
    data: pd.DataFrame,
) -> list[QualityFinding]:
    """
    Identify negative values in columns where negative values
    are commonly suspicious.

    This is a warning only. Negative values can be valid in
    many business datasets.
    """

    findings = []

    for column in data.select_dtypes(include="number").columns:
        name = _normalized_name(column)

        suspicious_column = any(
            keyword in name
            for keyword in {
                "age",
                "quantity",
                "count",
                "units",
                "discount",
                "percentage",
                "percent",
                "rate",
            }
        )

        if not suspicious_column:
            continue

        negative_count = int((data[column] < 0).sum())

        if negative_count == 0:
            continue

        percentage = (
            negative_count / len(data) * 100
            if len(data) > 0
            else 0.0
        )

        findings.append(
            QualityFinding(
                issue_type="suspicious_negative_values",
                severity="warning",
                message=(
                    f"Column '{column}' contains {negative_count} "
                    f"negative values ({percentage:.2f}%)."
                ),
                column_name=column,
                affected_rows=negative_count,
                affected_percentage=percentage,
                suggested_action=(
                    "Review whether negative values are valid "
                    "for this business variable."
                ),
                details={
                    "negative_count": negative_count,
                    "negative_percentage": percentage,
                },
            )
        )

    return findings


def check_percentage_ranges(
    data: pd.DataFrame,
) -> list[QualityFinding]:
    """
    Identify percentage-like columns containing values
    outside the conventional 0-100 range.
    """

    findings = []

    for column in data.select_dtypes(include="number").columns:
        name = _normalized_name(column)

        is_percentage_column = any(
            keyword in name
            for keyword in PERCENTAGE_KEYWORDS
        )

        if not is_percentage_column:
            continue

        invalid_mask = (data[column] < 0) | (data[column] > 100)
        invalid_count = int(invalid_mask.sum())

        if invalid_count == 0:
            continue

        percentage = (
            invalid_count / len(data) * 100
            if len(data) > 0
            else 0.0
        )

        findings.append(
            QualityFinding(
                issue_type="invalid_percentage_range",
                severity="warning",
                message=(
                    f"Column '{column}' contains {invalid_count} "
                    f"values outside the 0-100 range "
                    f"({percentage:.2f}% of rows)."
                ),
                column_name=column,
                affected_rows=invalid_count,
                affected_percentage=percentage,
                suggested_action=(
                    "Verify whether this column is truly a "
                    "percentage and review the out-of-range values."
                ),
                details={
                    "invalid_count": invalid_count,
                    "expected_min": 0,
                    "expected_max": 100,
                },
            )
        )

    return findings


def check_suspicious_age_values(
    data: pd.DataFrame,
) -> list[QualityFinding]:
    """
    Identify obviously suspicious age values.

    This intentionally uses a broad range rather than claiming
    that every value outside it is definitely invalid.
    """

    findings = []

    for column in data.select_dtypes(include="number").columns:
        name = _normalized_name(column)

        is_age_column = (
            name in AGE_KEYWORDS
            or name.endswith("_age")
            or name.startswith("age_")
        )

        if not is_age_column:
            continue

        invalid_mask = (data[column] < 0) | (data[column] > 120)
        invalid_count = int(invalid_mask.sum())

        if invalid_count == 0:
            continue

        percentage = (
            invalid_count / len(data) * 100
            if len(data) > 0
            else 0.0
        )

        findings.append(
            QualityFinding(
                issue_type="suspicious_age_values",
                severity="warning",
                message=(
                    f"Column '{column}' contains {invalid_count} "
                    f"values outside the broad expected age range "
                    f"of 0-120."
                ),
                column_name=column,
                affected_rows=invalid_count,
                affected_percentage=percentage,
                suggested_action=(
                    "Review these values for possible data-entry "
                    "errors or confirm that the column has a "
                    "different meaning."
                ),
                details={
                    "invalid_count": invalid_count,
                    "expected_min": 0,
                    "expected_max": 120,
                },
            )
        )

    return findings


def check_numerical_rules(
    data: pd.DataFrame,
) -> list[QualityFinding]:
    """Run all numerical quality rules."""

    findings = []

    findings.extend(
        check_suspicious_negative_values(data)
    )

    findings.extend(
        check_percentage_ranges(data)
    )

    findings.extend(
        check_suspicious_age_values(data)
    )

    return findings