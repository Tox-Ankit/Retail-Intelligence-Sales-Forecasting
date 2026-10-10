import pandas as pd

from src.cleaning.models import CleaningAction, CleaningResult
from src.cleaning.missing import fill_missing_values
from src.cleaning.duplicates import remove_duplicate_rows
from src.cleaning.categorical import standardize_categorical_values
from src.cleaning.outliers import (
    cap_iqr_outliers,
    remove_iqr_outliers,
    remove_z_score_outliers,
)


def clean_dataset(
    data: pd.DataFrame,
    dataset_name: str = "dataset",
    apply_missing_values: bool = True,
    missing_strategy: str = "auto",
    remove_duplicates: bool = True,
    standardize_whitespace: bool = True,
    normalize_case: bool = False,
    outlier_strategy: str = "none",
    outlier_columns: list[str] | None = None,
    iqr_multiplier: float = 1.5,
    z_score_threshold: float = 3.0,
) -> CleaningResult:
    """
    Apply the configured cleaning operations to a dataset.

    The original DataFrame is never modified.

    Parameters
    ----------
    data:
        Input DataFrame.

    dataset_name:
        Name used in the CleaningResult.

    apply_missing_values:
        Whether missing values should be treated.

    missing_strategy:
        Missing-value strategy:
        "auto", "mean", "median", "mode", or "constant".

    remove_duplicates:
        Whether duplicate rows should be removed.

    standardize_whitespace:
        Whether leading/trailing whitespace in categorical
        values should be removed.

    normalize_case:
        Whether categorical values should be normalized
        to lowercase.

    outlier_strategy:
        "none", "cap_iqr", "remove_iqr", or "remove_zscore".

    outlier_columns:
        Numerical columns on which outlier treatment is applied.
        If None, numerical columns are automatically selected.

    iqr_multiplier:
        IQR multiplier used for IQR-based treatment.

    z_score_threshold:
        Z-score threshold used for Z-score removal.

    Returns
    -------
    CleaningResult
        Contains original data, cleaned data, and audit actions.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if outlier_strategy not in {
        "none",
        "cap_iqr",
        "remove_iqr",
        "remove_zscore",
    }:
        raise ValueError(
            "outlier_strategy must be one of: "
            "'none', 'cap_iqr', 'remove_iqr', 'remove_zscore'."
        )

    if missing_strategy not in {
        "auto",
        "mean",
        "median",
        "mode",
        "constant",
    }:
        raise ValueError(
            "missing_strategy must be one of: "
            "'auto', 'mean', 'median', 'mode', 'constant'."
        )

    original_data = data.copy(deep=True)
    cleaned_data = data.copy(deep=True)

    all_actions: list[CleaningAction] = []

    # ---------------------------------------------------------
    # 1. Missing-value treatment
    # ---------------------------------------------------------

    if apply_missing_values:

        cleaned_data, actions = fill_missing_values(
            cleaned_data,
            strategy=missing_strategy,
        )

        all_actions.extend(actions)

    # ---------------------------------------------------------
    # 2. Duplicate-row removal
    # ---------------------------------------------------------

    if remove_duplicates:

        cleaned_data, actions = remove_duplicate_rows(
            cleaned_data
        )

        all_actions.extend(actions)

    # ---------------------------------------------------------
    # 3. Categorical standardization
    # ---------------------------------------------------------

    if standardize_whitespace or normalize_case:

        cleaned_data, actions = standardize_categorical_values(
            cleaned_data,
            strip_whitespace=standardize_whitespace,
            normalize_case=normalize_case,
        )

        all_actions.extend(actions)

    # ---------------------------------------------------------
    # 4. Outlier treatment
    # ---------------------------------------------------------

    if outlier_strategy == "cap_iqr":

        cleaned_data, actions = cap_iqr_outliers(
            cleaned_data,
            columns=outlier_columns,
            multiplier=iqr_multiplier,
        )

        all_actions.extend(actions)

    elif outlier_strategy == "remove_iqr":

        cleaned_data, actions = remove_iqr_outliers(
            cleaned_data,
            columns=outlier_columns,
            multiplier=iqr_multiplier,
        )

        all_actions.extend(actions)

    elif outlier_strategy == "remove_zscore":

        cleaned_data, actions = remove_z_score_outliers(
            cleaned_data,
            columns=outlier_columns,
            threshold=z_score_threshold,
        )

        all_actions.extend(actions)

    # ---------------------------------------------------------
    # 5. Build final cleaning result
    # ---------------------------------------------------------

    return CleaningResult(
        dataset_name=dataset_name,
        original_data=original_data,
        cleaned_data=cleaned_data,
        actions=all_actions,
    )