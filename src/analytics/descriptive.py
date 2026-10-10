
import math

import numpy as np
import pandas as pd

from src.analytics.models import (
    CategoricalSummary,
    DatasetSummary,
    NumericalSummary,
)


def _safe_float(value):
    """Convert a value to a regular float, or None if invalid."""
    try:
        if pd.isna(value):
            return None

        result = float(value)

        return result if math.isfinite(result) else None

    except (TypeError, ValueError, OverflowError):
        return None


def _safe_mode(series: pd.Series) -> list:
    """Return modes safely, including when a column has no values."""
    if series.empty:
        return []

    modes = series.mode(dropna=True)

    if modes.empty:
        return []

    return modes.tolist()


def summarize_numerical_column(
    series: pd.Series,
    column_name: str,
) -> NumericalSummary:
    """Calculate descriptive statistics for a numerical column."""

    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)

    valid = numeric.dropna()
    count = int(valid.count())
    missing_count = int(len(series) - count)

    if count == 0:
        return NumericalSummary(
            column_name=column_name,
            count=0,
            missing_count=missing_count,
        )

    minimum = _safe_float(valid.min())
    maximum = _safe_float(valid.max())
    q1 = _safe_float(valid.quantile(0.25))
    median = _safe_float(valid.median())
    q3 = _safe_float(valid.quantile(0.75))
    mean = _safe_float(valid.mean())
    variance = _safe_float(valid.var(ddof=1))
    standard_deviation = _safe_float(valid.std(ddof=1))

    # Coefficient of variation is undefined when the mean is zero.
    coefficient_of_variation = None

    if mean is not None and standard_deviation is not None and mean != 0:
        coefficient_of_variation = (
            standard_deviation / abs(mean)
        ) * 100

    percentiles = {
        "10": _safe_float(valid.quantile(0.10)),
        "25": q1,
        "50": median,
        "75": q3,
        "90": _safe_float(valid.quantile(0.90)),
        "95": _safe_float(valid.quantile(0.95)),
        "99": _safe_float(valid.quantile(0.99)),
    }

    return NumericalSummary(
        column_name=column_name,
        count=count,
        missing_count=missing_count,
        mean=mean,
        median=median,
        mode=_safe_mode(valid),
        minimum=minimum,
        maximum=maximum,
        range_value=(
            maximum - minimum
            if minimum is not None and maximum is not None
            else None
        ),
        variance=variance,
        standard_deviation=standard_deviation,
        q1=q1,
        q3=q3,
        iqr=(
            q3 - q1
            if q1 is not None and q3 is not None
            else None
        ),
        skewness=_safe_float(valid.skew()),
        kurtosis=_safe_float(valid.kurt()),
        coefficient_of_variation=_safe_float(
            coefficient_of_variation
        ),
        percentiles=percentiles,
    )


def summarize_categorical_column(
    series: pd.Series,
    column_name: str,
    top_n: int = 20,
) -> CategoricalSummary:
    """Calculate frequencies and percentages for a categorical column."""

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    valid = series.dropna()
    count = int(len(valid))
    missing_count = int(series.isna().sum())
    unique_count = int(valid.nunique())

    modes = _safe_mode(valid)

    # Use value_counts to avoid converting the full column to strings.
    counts = valid.value_counts().head(top_n)

    frequencies = {
        str(category): int(frequency)
        for category, frequency in counts.items()
    }

    percentages = {
        category: (frequency / count) * 100
        for category, frequency in frequencies.items()
    } if count > 0 else {}

    return CategoricalSummary(
        column_name=column_name,
        count=count,
        missing_count=missing_count,
        unique_count=unique_count,
        mode=modes,
        frequencies=frequencies,
        percentages=percentages,
    )


def summarize_dataset(
    data: pd.DataFrame,
    dataset_name: str = "dataset",
    top_n_categories: int = 20,
) -> DatasetSummary:
    """
    Calculate descriptive statistics for a complete DataFrame.

    Numerical and categorical summaries are returned separately.
    The input DataFrame is never modified.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if top_n_categories < 1:
        raise ValueError("top_n_categories must be at least 1.")

    numerical_summaries = {}
    categorical_summaries = {}

    for column in data.columns:
        series = data[column]

        # Boolean columns are categorical, not numerical,
        # for this descriptive summary.
        if pd.api.types.is_bool_dtype(series):
            categorical_summaries[str(column)] = (
                summarize_categorical_column(
                    series,
                    str(column),
                    top_n=top_n_categories,
                )
            )

        elif pd.api.types.is_numeric_dtype(series):
            numerical_summaries[str(column)] = (
                summarize_numerical_column(
                    series,
                    str(column),
                )
            )

        elif (
            pd.api.types.is_object_dtype(series)
            or isinstance(
                series.dtype,
                pd.CategoricalDtype,
            )
            or pd.api.types.is_string_dtype(series)
        ):
            categorical_summaries[str(column)] = (
                summarize_categorical_column(
                    series,
                    str(column),
                    top_n=top_n_categories,
                )
            )

    return DatasetSummary(
        dataset_name=dataset_name,
        rows=len(data),
        columns=len(data.columns),
        numerical_summaries=numerical_summaries,
        categorical_summaries=categorical_summaries,
        metadata={
            "missing_cells": int(data.isna().sum().sum()),
            "memory_usage_bytes": int(
                data.memory_usage(deep=True).sum()
            ),
        },
    )
