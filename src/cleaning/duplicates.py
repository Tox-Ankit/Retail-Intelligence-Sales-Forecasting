import pandas as pd

from src.cleaning.models import CleaningAction


def remove_duplicate_rows(
    data: pd.DataFrame,
    keep: str = "first",
) -> tuple[pd.DataFrame, list[CleaningAction]]:
    """
    Remove completely duplicated rows without modifying
    the original DataFrame.

    keep:
        'first'  -> keep first occurrence
        'last'   -> keep last occurrence
        False    -> remove all duplicated occurrences
    """

    if keep not in {"first", "last", False}:
        raise ValueError(
            "keep must be 'first', 'last', or False."
        )

    cleaned_data = data.copy(deep=True)

    duplicate_count = int(data.duplicated(keep=keep).sum())

    if duplicate_count == 0:
        return cleaned_data, []

    cleaned_data = cleaned_data.drop_duplicates(
        keep=keep
    ).reset_index(drop=True)

    action = CleaningAction(
        action_type="duplicate_removal",
        description=(
            f"Removed {duplicate_count} duplicate rows."
        ),
        affected_rows=duplicate_count,
        parameters={
            "keep": keep,
        },
        reason=(
            "Duplicate records can distort statistics, "
            "aggregations, and machine-learning models."
        ),
    )

    return cleaned_data, [action]