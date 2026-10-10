import pandas as pd

from src.quality.models import QualityFinding


def check_duplicate_rows(
    data: pd.DataFrame,
) -> list[QualityFinding]:
    """Check for completely duplicated rows."""

    duplicate_count = int(data.duplicated().sum())

    if duplicate_count == 0:
        return []

    total_rows = len(data)

    percentage = (
        (duplicate_count / total_rows) * 100
        if total_rows > 0
        else 0.0
    )

    return [
        QualityFinding(
            issue_type="duplicate_rows",
            severity="warning",
            message=(
                f"Dataset contains {duplicate_count} "
                f"duplicate rows ({percentage:.2f}%)."
            ),
            affected_rows=duplicate_count,
            affected_percentage=percentage,
            suggested_action=(
                "Review duplicate records before deciding "
                "whether they should be removed."
            ),
            details={
                "duplicate_count": duplicate_count,
                "duplicate_percentage": percentage,
            },
        )
    ]