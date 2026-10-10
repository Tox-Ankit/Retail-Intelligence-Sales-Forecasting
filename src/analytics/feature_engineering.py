
"""Reusable feature engineering utilities for analytics datasets.

Functions return a new DataFrame and do not modify the original input.
Model-specific preprocessing should be fitted inside an ML pipeline
after splitting training and test data to prevent data leakage.
"""

import numpy as np
import pandas as pd


DATE_FEATURES = {
    "year",
    "month",
    "quarter",
    "day",
    "dayofyear",
    "weekofyear",
}


def _validate_dataframe(data):
    """Ensure the input is a pandas DataFrame."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")


def _validate_columns(data, columns):
    """Ensure all requested input columns exist."""
    missing = [column for column in columns if column not in data.columns]

    if missing:
        raise ValueError(f"Columns not found in dataset: {missing}")


def add_date_features(
    data,
    date_columns,
    features=None,
    drop_original=False,
):
    """Extract useful calendar features from date columns.

    Supported features:
    year, month, quarter, day, dayofyear, weekofyear.

    Invalid dates produce missing values in generated features.
    """
    _validate_dataframe(data)

    if isinstance(date_columns, str):
        date_columns = [date_columns]
    else:
        date_columns = list(date_columns)

    if not date_columns:
        raise ValueError("At least one date column must be provided.")

    if len(set(date_columns)) != len(date_columns):
        raise ValueError("date_columns must not contain duplicates.")

    _validate_columns(data, date_columns)

    if features is None:
        features = ["year", "month", "quarter", "day"]
    else:
        features = list(features)

    if not features:
        raise ValueError("At least one date feature must be requested.")

    if len(set(features)) != len(features):
        raise ValueError("features must not contain duplicates.")

    unsupported = sorted(set(features) - DATE_FEATURES)

    if unsupported:
        raise ValueError(f"Unsupported date features: {unsupported}")

    if not isinstance(drop_original, bool):
        raise TypeError("drop_original must be a boolean.")

    result = data.copy(deep=True)

    for column in date_columns:
        parsed_dates = pd.to_datetime(
            result[column],
            errors="coerce",
            utc=True,
        )

        for feature in features:
            new_column = f"{column}_{feature}"

            if new_column in result.columns:
                raise ValueError(
                    f"Generated column '{new_column}' already exists."
                )

            if feature == "year":
                values = parsed_dates.dt.year
            elif feature == "month":
                values = parsed_dates.dt.month
            elif feature == "quarter":
                values = parsed_dates.dt.quarter
            elif feature == "day":
                values = parsed_dates.dt.day
            elif feature == "dayofyear":
                values = parsed_dates.dt.dayofyear
            else:  # weekofyear
                values = parsed_dates.dt.isocalendar().week

            result[new_column] = values

        if drop_original:
            result = result.drop(columns=[column])

    return result


def add_retail_metrics(
    data,
    quantity_column=None,
    unit_price_column=None,
    revenue_column=None,
    unit_cost_column=None,
    cost_column=None,
    profit_column=None,
    profit_margin_column="profit_margin",
):
    """Create explicitly requested retail metrics.

    Revenue = quantity * unit price
    Cost = quantity * unit cost
    Profit = revenue - cost
    Profit margin (%) = profit / revenue * 100

    Only the metrics requested through output column names are generated.
    Existing columns are never silently overwritten.
    """
    _validate_dataframe(data)

    result = data.copy(deep=True)

    if profit_margin_column is not None and not isinstance(
        profit_margin_column, str
    ):
        raise TypeError("profit_margin_column must be a string or None.")

    required_columns = []

    if revenue_column is not None:
        if quantity_column is None or unit_price_column is None:
            raise ValueError(
                "Calculating revenue requires quantity_column "
                "and unit_price_column."
            )
        required_columns.extend([quantity_column, unit_price_column])

    if cost_column is not None:
        if quantity_column is None or unit_cost_column is None:
            raise ValueError(
                "Calculating cost requires quantity_column "
                "and unit_cost_column."
            )
        required_columns.extend([quantity_column, unit_cost_column])

    if profit_column is not None:
        if revenue_column is None or cost_column is None:
            raise ValueError(
                "Calculating profit requires revenue_column "
                "and cost_column."
            )

    _validate_columns(data, list(dict.fromkeys(required_columns)))

    output_columns = []

    if revenue_column is not None:
        output_columns.append(revenue_column)

    if cost_column is not None:
        output_columns.append(cost_column)

    if profit_column is not None:
        output_columns.append(profit_column)

    if (
        profit_margin_column is not None
        and profit_column is not None
        and revenue_column is not None
    ):
        output_columns.append(profit_margin_column)

    existing_outputs = [
        column for column in output_columns if column in data.columns
    ]

    if existing_outputs:
        raise ValueError(
            f"Output columns already exist: {existing_outputs}"
        )

    def numeric_column(column):
        values = pd.to_numeric(result[column], errors="coerce")
        return values.replace([np.inf, -np.inf], np.nan)

    if revenue_column is not None:
        result[revenue_column] = (
            numeric_column(quantity_column)
            * numeric_column(unit_price_column)
        )

    if cost_column is not None:
        result[cost_column] = (
            numeric_column(quantity_column)
            * numeric_column(unit_cost_column)
        )

    if profit_column is not None:
        result[profit_column] = (
            numeric_column(revenue_column)
            - numeric_column(cost_column)
        )

    if (
        profit_margin_column is not None
        and profit_column is not None
        and revenue_column is not None
    ):
        revenue = numeric_column(revenue_column)
        profit = numeric_column(profit_column)

        result[profit_margin_column] = (
            profit / revenue.where(revenue != 0)
        ) * 100

    return result


def log_transform_numeric(data, columns, suffix="_log1p"):
    """Apply log1p to non-negative numeric columns.

    Creates new columns and preserves the original columns.
    Missing values remain missing; negative values are rejected.
    """
    _validate_dataframe(data)

    if isinstance(columns, str):
        columns = [columns]
    else:
        columns = list(columns)

    if not columns:
        raise ValueError("At least one column must be provided.")

    if len(set(columns)) != len(columns):
        raise ValueError("columns must not contain duplicates.")

    if not isinstance(suffix, str) or not suffix:
        raise ValueError("suffix must be a non-empty string.")

    _validate_columns(data, columns)

    result = data.copy(deep=True)

    for column in columns:
        new_column = f"{column}{suffix}"

        if new_column in result.columns:
            raise ValueError(
                f"Output column '{new_column}' already exists."
            )

        values = pd.to_numeric(result[column], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan)

        if values.lt(0).any():
            raise ValueError(
                f"Column '{column}' contains negative values. "
                "log1p requires values greater than or equal to zero."
            )

        result[new_column] = np.log1p(values)

    return result


def encode_categorical_features(
    data,
    columns=None,
    drop_first=False,
    dummy_na=False,
    prefix_sep="=",
):
    """One-hot encode selected categorical columns.

    For machine learning, prefer fitting the encoder within a
    scikit-learn Pipeline after splitting the data.
    """
    _validate_dataframe(data)

    if not isinstance(drop_first, bool):
        raise TypeError("drop_first must be a boolean.")

    if not isinstance(dummy_na, bool):
        raise TypeError("dummy_na must be a boolean.")

    if not isinstance(prefix_sep, str):
        raise TypeError("prefix_sep must be a string.")

    if columns is None:
        selected = data.select_dtypes(
            include=["object", "category", "string", "bool"]
        ).columns.tolist()
    elif isinstance(columns, str):
        selected = [columns]
    else:
        selected = list(columns)

    if not selected:
        return data.copy(deep=True)

    if len(set(selected)) != len(selected):
        raise ValueError("columns must not contain duplicates.")

    _validate_columns(data, selected)

    unsupported = [
        column
        for column in selected
        if not (
            pd.api.types.is_object_dtype(data[column].dtype)
            or isinstance(data[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(data[column].dtype)
            or pd.api.types.is_bool_dtype(data[column].dtype)
        )
    ]

    if unsupported:
        raise ValueError(
            "Only categorical, string, and boolean columns can be encoded. "
            f"Unsupported columns: {unsupported}"
        )

    return pd.get_dummies(
        data.copy(deep=True),
        columns=selected,
        drop_first=drop_first,
        dummy_na=dummy_na,
        prefix_sep=prefix_sep,
        dtype=int,
    )
