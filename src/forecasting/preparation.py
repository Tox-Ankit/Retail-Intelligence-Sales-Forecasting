
"""Prepare time-series data for forecasting."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class PreparedTimeSeries:
    """Store a prepared time series and its preparation metadata."""

    series: pd.Series
    date_column: str
    value_column: str
    frequency: str
    aggregation: str
    input_rows: int
    valid_rows: int
    dropped_rows: int
    missing_periods: int
    filled_periods: int
    notes: list[str] = field(default_factory=list)


def _normalize_frequency(frequency: str) -> str:
    """Support common frequency aliases across pandas versions."""

    aliases = {
        "M": "ME",
        "Q": "QE",
        "Y": "YE",
        "A": "YE",
    }

    return aliases.get(frequency, frequency)


def prepare_time_series(
    data: pd.DataFrame,
    date_column: str,
    value_column: str,
    frequency: str = "ME",
    aggregation: str = "sum",
    missing_period_strategy: str = "none",
) -> PreparedTimeSeries:
    """
    Convert raw tabular data into a regular time series.

    Supported aggregations: sum, mean, count.
    Missing-period strategies: none, zero, ffill, interpolate.

    The default strategy preserves missing periods as NaN rather than
    silently inventing values.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("Cannot prepare a time series from an empty dataset.")

    if not data.columns.is_unique:
        raise ValueError("Dataset contains duplicate column names.")

    if date_column not in data.columns:
        raise ValueError(f"Date column '{date_column}' was not found.")

    if value_column not in data.columns:
        raise ValueError(f"Value column '{value_column}' was not found.")

    if date_column == value_column:
        raise ValueError("Date and value columns must be different.")

    if not isinstance(frequency, str) or not frequency.strip():
        raise ValueError("frequency must be a non-empty pandas frequency string.")

    aggregation = aggregation.lower()
    if aggregation not in {"sum", "mean", "count"}:
        raise ValueError("aggregation must be 'sum', 'mean', or 'count'.")

    missing_period_strategy = missing_period_strategy.lower()
    valid_strategies = {"none", "zero", "ffill", "interpolate"}

    if missing_period_strategy not in valid_strategies:
        raise ValueError(
            "missing_period_strategy must be 'none', 'zero', "
            "'ffill', or 'interpolate'."
        )

    work = data[[date_column, value_column]].copy()
    input_rows = len(work)

    work[date_column] = pd.to_datetime(
        work[date_column],
        errors="coerce",
        utc=True,
    )
    work[value_column] = pd.to_numeric(
        work[value_column],
        errors="coerce",
    )

    # Replace infinite values with missing values.
    work[value_column] = work[value_column].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Keep only rows with valid dates and numeric values.
    work = work.dropna(subset=[date_column, value_column])
    valid_rows = len(work)
    dropped_rows = input_rows - valid_rows

    if work.empty:
        raise ValueError(
            "No valid rows remain. Check the date and numeric value columns."
        )

    work = work.sort_values(date_column).set_index(date_column)
    normalized_frequency = _normalize_frequency(frequency)

    # Aggregate observations into regular time periods.
    grouped = work[value_column].resample(normalized_frequency)

    if aggregation == "sum":
        series = grouped.sum(min_count=1)
    elif aggregation == "mean":
        series = grouped.mean()
    else:
        series = grouped.count().astype(float)

    series.name = value_column
    series.index.name = date_column

    if series.empty:
        raise ValueError("No time periods could be created from the data.")

    # Count periods with no valid observations before filling.
    missing_periods = int(series.isna().sum())

    if missing_periods and missing_period_strategy == "zero":
        series = series.fillna(0)

    elif missing_periods and missing_period_strategy == "ffill":
        series = series.ffill()

        if series.isna().any():
            raise ValueError(
                "Forward fill cannot fill missing periods at the beginning "
                "of the series. Choose 'zero' or 'interpolate' if appropriate."
            )

    elif missing_periods and missing_period_strategy == "interpolate":
        if series.notna().sum() == 0:
            raise ValueError(
                "Cannot interpolate because every time period is missing."
            )

        # Interpolation can use future observations. Avoid applying it to
        # the full dataset before a forecasting train/test split.
        series = series.interpolate(
            method="time",
            limit_direction="both",
        )

    filled_periods = missing_periods - int(series.isna().sum())

    notes = [
        f"Aggregated observations using '{aggregation}'.",
        f"Used frequency '{normalized_frequency}'.",
    ]

    if dropped_rows:
        notes.append(
            f"Dropped {dropped_rows} rows with invalid dates or values."
        )

    if series.isna().any():
        notes.append(
            "Missing periods remain. Choose a suitable filling strategy "
            "before training a forecasting model."
        )

    if missing_period_strategy == "interpolate" and missing_periods:
        notes.append(
            "Interpolation may use future observations; use caution "
            "when evaluating forecasting performance."
        )

    return PreparedTimeSeries(
        series=series,
        date_column=date_column,
        value_column=value_column,
        frequency=normalized_frequency,
        aggregation=aggregation,
        input_rows=input_rows,
        valid_rows=valid_rows,
        dropped_rows=dropped_rows,
        missing_periods=missing_periods,
        filled_periods=filled_periods,
        notes=notes,
    )


def split_time_series(
    series: pd.Series,
    test_size: float = 0.2,
) -> tuple[pd.Series, pd.Series]:
    """
    Split a time series chronologically.

    Earlier observations go into training; later observations go into testing.
    This function never shuffles the data.
    """

    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series.")

    if len(series) < 5:
        raise ValueError("At least 5 observations are required for splitting.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("The series must have a DatetimeIndex.")

    if not series.index.is_monotonic_increasing:
        raise ValueError("The series must be sorted by date.")

    if not series.index.is_unique:
        raise ValueError("The series must have unique timestamps.")

    if series.isna().any():
        raise ValueError(
            "The series contains missing values. Handle them before splitting."
        )

    test_count = max(1, int(np.ceil(len(series) * test_size)))
    train_count = len(series) - test_count

    if train_count < 2:
        raise ValueError(
            "Not enough observations for training. Reduce test_size."
        )

    train = series.iloc[:train_count].copy()
    test = series.iloc[train_count:].copy()

    return train, test
