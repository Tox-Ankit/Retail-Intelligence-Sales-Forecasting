import pandas as pd

from src.quality.models import QualityFinding


def check_empty_dataset(
    data: pd.DataFrame,
) -> list[QualityFinding]:
    """Check whether the dataset contains no usable data."""

    if len(data) > 0 and len(data.columns) > 0:
        return []

    return [
        QualityFinding(
            issue_type="empty_dataset",
            severity="critical",
            message=(
                "Dataset contains no usable rows or columns."
            ),
            affected_rows=len(data),
            affected_percentage=100.0,
            suggested_action=(
                "Verify that the uploaded file contains "
                "valid tabular data."
            ),
        )
    ]