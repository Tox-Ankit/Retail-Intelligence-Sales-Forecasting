import pandas as pd

from src.cleaning.models import CleaningAction


def recommend_missing_strategy(
    series: pd.Series,
) -> tuple[str, str]:
    """
    Recommend a missing-value treatment based on the
    column's data type and distribution.
    """

    missing_count = int(series.isna().sum())

    if missing_count == 0:
        return "none", "Column contains no missing values."

    if pd.api.types.is_numeric_dtype(series):
        non_null = series.dropna()

        if non_null.empty:
            return (
                "constant",
                "Column contains only missing values.",
            )

        skewness = non_null.skew()

        if abs(skewness) > 1:
            return (
                "median",
                "Numerical column is strongly skewed; "
                "median is more robust to extreme values.",
            )

        return (
            "mean",
            "Numerical column has no strong skew; "
            "mean is a reasonable default.",
        )

    # Categorical / text / other non-numerical data
    return (
        "mode",
        "Non-numerical column; most frequent category "
        "is a reasonable default.",
    )


def fill_missing_values(
    data: pd.DataFrame,
    strategy: str = "auto",
    constant_value: object = "Unknown",
) -> tuple[pd.DataFrame, list[CleaningAction]]:
    """
    Fill missing values without modifying the original DataFrame.

    Supported strategies:
        auto
        mean
        median
        mode
        constant
    """

    cleaned_data = data.copy(deep=True)
    actions: list[CleaningAction] = []

    for column in cleaned_data.columns:
        missing_count = int(cleaned_data[column].isna().sum())

        if missing_count == 0:
            continue

        selected_strategy = strategy
        reason = ""

        if strategy == "auto":
            selected_strategy, reason = recommend_missing_strategy(
                cleaned_data[column]
            )

        if selected_strategy == "none":
            continue

        if selected_strategy == "mean":
            if not pd.api.types.is_numeric_dtype(
                cleaned_data[column]
            ):
                continue

            fill_value = cleaned_data[column].mean()

            if pd.isna(fill_value):
                continue

            cleaned_data[column] = cleaned_data[column].fillna(
                fill_value
            )

            description = (
                f"Filled {missing_count} missing values "
                f"with mean ({fill_value:.4f})."
            )

            parameters = {
                "strategy": "mean",
                "fill_value": float(fill_value),
            }

        elif selected_strategy == "median":
            if not pd.api.types.is_numeric_dtype(
                cleaned_data[column]
            ):
                continue

            fill_value = cleaned_data[column].median()

            if pd.isna(fill_value):
                continue

            cleaned_data[column] = cleaned_data[column].fillna(
                fill_value
            )

            description = (
                f"Filled {missing_count} missing values "
                f"with median ({fill_value:.4f})."
            )

            parameters = {
                "strategy": "median",
                "fill_value": float(fill_value),
            }

        elif selected_strategy == "mode":
            mode_values = cleaned_data[column].mode()

            if mode_values.empty:
                continue

            fill_value = mode_values.iloc[0]

            cleaned_data[column] = cleaned_data[column].fillna(
                fill_value
            )

            description = (
                f"Filled {missing_count} missing values "
                f"with mode ({fill_value})."
            )

            parameters = {
                "strategy": "mode",
                "fill_value": str(fill_value),
            }

        elif selected_strategy == "constant":
            cleaned_data[column] = cleaned_data[column].fillna(
                constant_value
            )

            description = (
                f"Filled {missing_count} missing values "
                f"with constant '{constant_value}'."
            )

            parameters = {
                "strategy": "constant",
                "fill_value": constant_value,
            }

        else:
            raise ValueError(
                f"Unsupported missing-value strategy: "
                f"{selected_strategy}"
            )

        actions.append(
            CleaningAction(
                action_type="missing_value_imputation",
                description=description,
                column_name=column,
                affected_rows=missing_count,
                parameters=parameters,
                reason=reason,
            )
        )

    return cleaned_data, actions