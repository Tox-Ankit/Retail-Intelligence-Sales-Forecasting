import pandas as pd

from src.quality.models import QualityFinding


def check_constant_columns(
    data: pd.DataFrame,
) -> list[QualityFinding]:
    """Check for columns containing one or zero unique values."""

    findings = []

    for column in data.columns:
        unique_count = int(
            data[column].nunique(dropna=True)
        )

        if unique_count > 1:
            continue

        findings.append(
            QualityFinding(
                issue_type="constant_column",
                severity="info",
                message=(
                    f"Column '{column}' contains only "
                    f"{unique_count} unique non-null value."
                ),
                column_name=column,
                affected_rows=len(data),
                affected_percentage=100.0,
                suggested_action=(
                    "Consider removing the column if it provides "
                    "no useful analytical information."
                ),
                details={
                    "unique_count": unique_count,
                },
            )
        )

    return findings