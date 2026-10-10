import pandas as pd

from src.quality.models import QualityFinding


EXCESSIVE_MISSING_THRESHOLD = 50.0


def check_missing_values(
    data: pd.DataFrame,
) -> list[QualityFinding]:
    """Check every column for missing values."""

    findings = []

    total_rows = len(data)

    if total_rows == 0:
        return findings

    for column in data.columns:
        null_count = int(data[column].isna().sum())

        if null_count == 0:
            continue

        percentage = (null_count / total_rows) * 100

        if percentage >= EXCESSIVE_MISSING_THRESHOLD:
            severity = "critical"

            suggested_action = (
                "Investigate the column and consider whether it "
                "should be retained, dropped, or carefully imputed."
            )
        else:
            severity = "warning"

            suggested_action = (
                "Consider an appropriate missing-value treatment "
                "based on the column's analytical meaning."
            )

        findings.append(
            QualityFinding(
                issue_type="missing_values",
                severity=severity,
                message=(
                    f"Column '{column}' contains {null_count} "
                    f"missing values ({percentage:.2f}%)."
                ),
                column_name=column,
                affected_rows=null_count,
                affected_percentage=percentage,
                suggested_action=suggested_action,
                details={
                    "missing_count": null_count,
                    "missing_percentage": percentage,
                },
            )
        )

    return findings