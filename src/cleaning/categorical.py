import pandas as pd

from src.cleaning.models import CleaningAction


def standardize_categorical_values(
    data: pd.DataFrame,
    strip_whitespace: bool = True,
    normalize_case: bool = False,
) -> tuple[pd.DataFrame, list[CleaningAction]]:
    """
    Standardize categorical/text values without modifying
    the original DataFrame.

    By default:
        - leading/trailing whitespace is removed
        - original capitalization is preserved

    normalize_case=True converts values to lowercase.
    """

    cleaned_data = data.copy(deep=True)
    actions: list[CleaningAction] = []

    categorical_columns = cleaned_data.select_dtypes(
        include=["object", "category", "string"]
    ).columns

    for column in categorical_columns:
        original = cleaned_data[column].copy()

        standardized = original.astype("string")

        if strip_whitespace:
            standardized = standardized.str.strip()

        if normalize_case:
            standardized = standardized.str.casefold()

        changed_mask = (
            original.astype("string")
            != standardized
        )

        changed_mask = changed_mask.fillna(False)

        changed_count = int(changed_mask.sum())

        if changed_count == 0:
            continue

        cleaned_data[column] = standardized

        actions.append(
            CleaningAction(
                action_type="categorical_standardization",
                description=(
                    f"Standardized {changed_count} values "
                    f"in column '{column}'."
                ),
                column_name=column,
                affected_rows=changed_count,
                parameters={
                    "strip_whitespace": strip_whitespace,
                    "normalize_case": normalize_case,
                },
                reason=(
                    "Standardizing categorical representations "
                    "helps prevent equivalent values from being "
                    "treated as separate categories."
                ),
            )
        )

    return cleaned_data, actions