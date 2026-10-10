
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class TimeSeriesResult:
    date_column: str
    value_column: str
    frequency: str
    observations: int

    total: float | None = None
    mean: float | None = None
    minimum: float | None = None
    maximum: float | None = None

    trend: str = "insufficient_data"
    period_change: float | None = None
    period_change_percentage: float | None = None

    rolling_average: dict = field(default_factory=dict)
    period_summary: list = field(default_factory=list)
    interpretation: list = field(default_factory=list)


def _safe_float(value):
    """Convert a value to a finite Python float, or None."""
    try:
        if pd.isna(value):
            return None

        result = float(value)

        if not np.isfinite(result):
            return None

        return result

    except (TypeError, ValueError, OverflowError):
        return None


def analyze_time_series(
    data,
    date_column,
    value_column,
    frequency="M",
    rolling_window=3,
    aggregation="sum",
):
    """
    Analyze a time series using a date column and a numerical value column.

    Parameters
    ----------
    data : pandas.DataFrame
        Source dataset. The original DataFrame is not modified.

    date_column : str
        Column containing dates or timestamps.

    value_column : str
        Column containing numerical values.

    frequency : str
        D = daily
        W = weekly
        M or ME = month-end
        Q or QE = quarter-end
        Y or YE = year-end

    rolling_window : int
        Number of aggregated periods used for the rolling average.

    aggregation : str
        One of "sum", "mean", or "count".

    Returns
    -------
    TimeSeriesResult
        Summary statistics, trend, period changes, rolling averages,
        period-level results, and interpretations.
    """

    # ---------------------------------------------------------
    # 1. Validate inputs
    # ---------------------------------------------------------
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if date_column not in data.columns:
        raise ValueError(
            f"Date column '{date_column}' was not found in the dataset."
        )

    if value_column not in data.columns:
        raise ValueError(
            f"Value column '{value_column}' was not found in the dataset."
        )

    if date_column == value_column:
        raise ValueError(
            "date_column and value_column must be different."
        )

    if not isinstance(rolling_window, int) or isinstance(
        rolling_window, bool
    ):
        raise TypeError("rolling_window must be an integer.")

    if rolling_window < 1:
        raise ValueError("rolling_window must be at least 1.")

    if aggregation not in {"sum", "mean", "count"}:
        raise ValueError(
            "aggregation must be 'sum', 'mean', or 'count'."
        )

    # ---------------------------------------------------------
    # 2. Normalize frequency aliases for Pandas compatibility
    # ---------------------------------------------------------
    frequency_aliases = {
        "D": "D",
        "W": "W",
        "M": "ME",
        "ME": "ME",
        "Q": "QE",
        "QE": "QE",
        "Y": "YE",
        "YE": "YE",
    }

    if frequency not in frequency_aliases:
        raise ValueError(
            "frequency must be D, W, M/ME, Q/QE, or Y/YE."
        )

    normalized_frequency = frequency_aliases[frequency]

    # ---------------------------------------------------------
    # 3. Validate and prepare the selected columns
    # ---------------------------------------------------------
    working_data = data[[date_column, value_column]].copy()

    # Parse dates and standardize them to UTC.
    working_data[date_column] = pd.to_datetime(
        working_data[date_column],
        errors="coerce",
        utc=True,
    )

    # Convert values to numeric; invalid values become missing.
    working_data[value_column] = pd.to_numeric(
        working_data[value_column],
        errors="coerce",
    )

    # Treat positive and negative infinity as missing values.
    working_data[value_column] = working_data[
        value_column
    ].replace([np.inf, -np.inf], np.nan)

    # Remove rows with invalid dates or values.
    working_data = working_data.dropna(
        subset=[date_column, value_column]
    )

    # Sort chronologically.
    working_data = working_data.sort_values(
        by=date_column
    )

    # ---------------------------------------------------------
    # 4. Handle empty valid data
    # ---------------------------------------------------------
    if working_data.empty:
        return TimeSeriesResult(
            date_column=date_column,
            value_column=value_column,
            frequency=normalized_frequency,
            observations=0,
            interpretation=[
                "No valid date-and-value records were available "
                "for time-series analysis."
            ],
        )

    # ---------------------------------------------------------
    # 5. Resample the data
    # ---------------------------------------------------------
    indexed_data = working_data.set_index(date_column)

    value_series = indexed_data[value_column]

    if aggregation == "sum":
        aggregated = value_series.resample(
            normalized_frequency
        ).sum(min_count=1)

    elif aggregation == "mean":
        aggregated = value_series.resample(
            normalized_frequency
        ).mean()

    else:
        # Count valid numerical observations in each period.
        aggregated = value_series.resample(
            normalized_frequency
        ).count()

    aggregated = aggregated.dropna()

    # ---------------------------------------------------------
    # 6. Handle empty aggregation results
    # ---------------------------------------------------------
    if aggregated.empty:
        return TimeSeriesResult(
            date_column=date_column,
            value_column=value_column,
            frequency=normalized_frequency,
            observations=0,
            interpretation=[
                "No valid aggregated periods were available."
            ],
        )

    # ---------------------------------------------------------
    # 7. Calculate summary statistics
    # ---------------------------------------------------------
    total = _safe_float(aggregated.sum())
    mean = _safe_float(aggregated.mean())
    minimum = _safe_float(aggregated.min())
    maximum = _safe_float(aggregated.max())

    first_value = _safe_float(aggregated.iloc[0])
    last_value = _safe_float(aggregated.iloc[-1])

    period_change = None
    period_change_percentage = None

    if first_value is not None and last_value is not None:
        period_change = _safe_float(last_value - first_value)

        if first_value != 0:
            period_change_percentage = _safe_float(
                ((last_value - first_value) / abs(first_value)) * 100
            )

    # ---------------------------------------------------------
    # 8. Estimate the overall trend
    # ---------------------------------------------------------
    trend = "insufficient_data"

    finite_values = aggregated.to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]

    if len(finite_values) >= 2:
        x_values = np.arange(len(finite_values), dtype=float)

        try:
            slope = float(
                np.polyfit(x_values, finite_values, 1)[0]
            )

            scale = max(
                float(np.max(np.abs(finite_values))),
                1.0,
            )

            # Treat very small slopes as approximately stable.
            tolerance = 1e-10 * scale

            if slope > tolerance:
                trend = "increasing"
            elif slope < -tolerance:
                trend = "decreasing"
            else:
                trend = "stable"

        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            trend = "insufficient_data"

    # ---------------------------------------------------------
    # 9. Calculate rolling averages
    # ---------------------------------------------------------
    rolling_series = aggregated.rolling(
        window=rolling_window,
        min_periods=rolling_window,
    ).mean()

    rolling_average = {
        timestamp.isoformat(): _safe_float(value)
        for timestamp, value in rolling_series.items()
        if pd.notna(value)
    }

    # ---------------------------------------------------------
    # 10. Build period-level summary
    # ---------------------------------------------------------
    period_summary = []

    for timestamp, value in aggregated.items():
        period_summary.append(
            {
                "period": timestamp.isoformat(),
                "value": _safe_float(value),
            }
        )

    # ---------------------------------------------------------
    # 11. Generate interpretations
    # ---------------------------------------------------------
    interpretation = []

    interpretation.append(
        f"Analyzed {len(aggregated)} valid time periods "
        f"using {normalized_frequency} frequency."
    )

    if aggregation == "sum":
        interpretation.append(
            f"Total across all periods: {total}."
        )
    elif aggregation == "mean":
        interpretation.append(
            f"Average of the period-level means: {mean}."
        )
    else:
        interpretation.append(
            f"Total valid observations counted across periods: {total}."
        )

    if trend == "increasing":
        interpretation.append(
            "The overall linear trend is increasing."
        )
    elif trend == "decreasing":
        interpretation.append(
            "The overall linear trend is decreasing."
        )
    elif trend == "stable":
        interpretation.append(
            "The overall linear trend is approximately stable."
        )
    else:
        interpretation.append(
            "There is insufficient data to estimate a reliable trend."
        )

    if period_change is not None:
        interpretation.append(
            f"Change from the first to the last period: {period_change}."
        )

    if period_change_percentage is not None:
        interpretation.append(
            "Percentage change from the first to the last period: "
            f"{period_change_percentage:.2f}%."
        )
    elif first_value == 0 and last_value is not None:
        interpretation.append(
            "Percentage change was not calculated because "
            "the first period's value is zero."
        )

    # ---------------------------------------------------------
    # 12. Return the complete result
    # ---------------------------------------------------------
    return TimeSeriesResult(
        date_column=date_column,
        value_column=value_column,
        frequency=normalized_frequency,
        observations=len(aggregated),
        total=total,
        mean=mean,
        minimum=minimum,
        maximum=maximum,
        trend=trend,
        period_change=period_change,
        period_change_percentage=period_change_percentage,
        rolling_average=rolling_average,
        period_summary=period_summary,
        interpretation=interpretation,
    )
