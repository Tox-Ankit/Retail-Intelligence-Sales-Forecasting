import pandas as pd

from src.cleaning.models import CleaningAction


DEFAULT_IQR_MULTIPLIER = 1.5
DEFAULT_Z_SCORE_THRESHOLD = 3.0


def cap_iqr_outliers(
    data,
    columns=None,
    multiplier=DEFAULT_IQR_MULTIPLIER,
):
    """
    Cap numerical outliers using the IQR method.

    Instead of removing outlier rows, values below the lower
    IQR boundary are replaced with the lower boundary, and
    values above the upper boundary are replaced with the
    upper boundary.

    Returns:
        cleaned_data: DataFrame
        actions: list of CleaningAction
    """

    cleaned_data = data.copy(deep=True)
    actions = []

    if columns is None:
        columns = cleaned_data.select_dtypes(
            include="number"
        ).columns.tolist()

    for column in columns:

        if column not in cleaned_data.columns:
            continue

        if not pd.api.types.is_numeric_dtype(
            cleaned_data[column]
        ):
            continue

        series = cleaned_data[column]

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        mask = (
            (series < lower_bound)
            | (series > upper_bound)
        )

        count = int(mask.sum())

        if count == 0:
            continue

        # IQR boundaries can be decimal values.
        # Convert integer columns to float before capping.
        cleaned_data[column] = cleaned_data[column].astype(float)

        cleaned_data.loc[
            cleaned_data[column] < lower_bound,
            column,
        ] = lower_bound

        cleaned_data.loc[
            cleaned_data[column] > upper_bound,
            column,
        ] = upper_bound

        actions.append(
            CleaningAction(
                action_type="iqr_outlier_capping",
                description=(
                    f"Capped {count} outliers "
                    f"in column '{column}'."
                ),
                column_name=column,
                affected_rows=count,
                parameters={
                    "method": "IQR",
                    "multiplier": multiplier,
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                },
                reason=(
                    "Extreme observations were capped at "
                    "statistical IQR boundaries instead of "
                    "being removed."
                ),
            )
        )

    return cleaned_data, actions


def remove_iqr_outliers(
    data,
    columns=None,
    multiplier=DEFAULT_IQR_MULTIPLIER,
):
    """
    Remove rows containing IQR-based numerical outliers.

    Returns:
        cleaned_data: DataFrame
        actions: list of CleaningAction
    """

    cleaned_data = data.copy(deep=True)
    actions = []

    if columns is None:
        columns = cleaned_data.select_dtypes(
            include="number"
        ).columns.tolist()

    rows_to_remove = pd.Series(
        False,
        index=cleaned_data.index,
    )

    column_details = {}

    for column in columns:

        if column not in cleaned_data.columns:
            continue

        if not pd.api.types.is_numeric_dtype(
            cleaned_data[column]
        ):
            continue

        series = cleaned_data[column]

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        mask = (
            (series < lower_bound)
            | (series > upper_bound)
        )

        rows_to_remove |= mask

        column_details[column] = {
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outliers": int(mask.sum()),
        }

    count = int(rows_to_remove.sum())

    if count == 0:
        return cleaned_data, actions

    cleaned_data = (
        cleaned_data.loc[~rows_to_remove]
        .reset_index(drop=True)
    )

    actions.append(
        CleaningAction(
            action_type="iqr_outlier_removal",
            description=(
                f"Removed {count} rows "
                "containing IQR outliers."
            ),
            affected_rows=count,
            parameters={
                "method": "IQR",
                "multiplier": multiplier,
                "columns": column_details,
            },
            reason=(
                "Rows containing selected statistical "
                "outliers were removed according to the "
                "chosen IQR rule."
            ),
        )
    )

    return cleaned_data, actions


def remove_z_score_outliers(
    data,
    columns=None,
    threshold=DEFAULT_Z_SCORE_THRESHOLD,
):
    """
    Remove rows containing Z-score based numerical outliers.

    Returns:
        cleaned_data: DataFrame
        actions: list of CleaningAction
    """

    cleaned_data = data.copy(deep=True)
    actions = []

    if columns is None:
        columns = cleaned_data.select_dtypes(
            include="number"
        ).columns.tolist()

    rows_to_remove = pd.Series(
        False,
        index=cleaned_data.index,
    )

    column_details = {}

    for column in columns:

        if column not in cleaned_data.columns:
            continue

        if not pd.api.types.is_numeric_dtype(
            cleaned_data[column]
        ):
            continue

        series = cleaned_data[column]

        mean = series.mean()
        std = series.std()

        if pd.isna(std) or std == 0:
            continue

        z_scores = (series - mean) / std

        mask = z_scores.abs() > threshold

        rows_to_remove |= mask

        column_details[column] = {
            "threshold": threshold,
            "outliers": int(mask.sum()),
        }

    count = int(rows_to_remove.sum())

    if count == 0:
        return cleaned_data, actions

    cleaned_data = (
        cleaned_data.loc[~rows_to_remove]
        .reset_index(drop=True)
    )

    actions.append(
        CleaningAction(
            action_type="z_score_outlier_removal",
            description=(
                f"Removed {count} rows containing "
                "Z-score outliers."
            ),
            affected_rows=count,
            parameters={
                "method": "Z-score",
                "threshold": threshold,
                "columns": column_details,
            },
            reason=(
                "Rows exceeding the configured Z-score "
                "threshold were removed."
            ),
        )
    )

    return cleaned_data, actions