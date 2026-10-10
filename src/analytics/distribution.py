
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DistributionSummary:
    column_name: str
    count: int
    missing_count: int

    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    median: float | None = None

    skewness: float | None = None
    kurtosis: float | None = None
    shape: str = "insufficient_data"

    q1: float | None = None
    q3: float | None = None
    iqr: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None

    outlier_count: int = 0
    outlier_percentage: float = 0.0

    percentiles: dict[str, float | None] = field(
        default_factory=dict
    )
    interpretation: list[str] = field(default_factory=list)


def _finite_float(value):
    """Return a finite float, or None if undefined."""
    if pd.isna(value):
        return None

    result = float(value)

    return result if np.isfinite(result) else None


def analyze_distribution(
    series: pd.Series,
    column_name: str,
    iqr_multiplier: float = 1.5,
) -> DistributionSummary:
    """Analyze the distribution of one numerical column."""

    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series.")

    if iqr_multiplier <= 0 or not np.isfinite(iqr_multiplier):
        raise ValueError("iqr_multiplier must be finite and positive.")

    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)

    valid = numeric.dropna()
    count = len(valid)
    missing_count = len(series) - count

    if count == 0:
        return DistributionSummary(
            column_name=column_name,
            count=0,
            missing_count=missing_count,
            interpretation=[
                "No valid numerical values are available."
            ],
        )

    minimum = _finite_float(valid.min())
    maximum = _finite_float(valid.max())
    mean = _finite_float(valid.mean())
    median = _finite_float(valid.median())

    q1 = _finite_float(valid.quantile(0.25))
    q3 = _finite_float(valid.quantile(0.75))
    iqr = (
        q3 - q1
        if q1 is not None and q3 is not None
        else None
    )

    lower_bound = (
        q1 - iqr_multiplier * iqr
        if q1 is not None and iqr is not None
        else None
    )
    upper_bound = (
        q3 + iqr_multiplier * iqr
        if q3 is not None and iqr is not None
        else None
    )

    if lower_bound is not None and upper_bound is not None:
        outlier_mask = (
            (valid < lower_bound) | (valid > upper_bound)
        )
        outlier_count = int(outlier_mask.sum())
    else:
        outlier_count = 0

    outlier_percentage = (
        outlier_count / count * 100 if count else 0.0
    )

    skewness = _finite_float(valid.skew())
    kurtosis = _finite_float(valid.kurt())

    # These are descriptive heuristics, not formal normality tests.
    if count < 3:
        shape = "insufficient_data"
    elif iqr == 0 and valid.nunique() == 1:
        shape = "constant"
    elif skewness is None:
        shape = "undetermined"
    elif skewness > 0.5:
        shape = "right_skewed"
    elif skewness < -0.5:
        shape = "left_skewed"
    else:
        shape = "approximately_symmetric"

    interpretation = []

    if shape == "right_skewed":
        interpretation.append(
            "The distribution has a longer right tail."
        )
    elif shape == "left_skewed":
        interpretation.append(
            "The distribution has a longer left tail."
        )
    elif shape == "approximately_symmetric":
        interpretation.append(
            "The distribution is approximately symmetric by skewness."
        )
    elif shape == "constant":
        interpretation.append(
            "All valid values are identical."
        )
    else:
        interpretation.append(
            "There is insufficient information to classify the shape."
        )

    if mean is not None and median is not None:
        if mean > median:
            interpretation.append(
                "The mean exceeds the median; high values may be "
                "pulling the average upward."
            )
        elif mean < median:
            interpretation.append(
                "The mean is below the median; low values may be "
                "pulling the average downward."
            )

    if outlier_count:
        interpretation.append(
            f"{outlier_count} values ({outlier_percentage:.2f}%) "
            "fall outside the IQR outlier boundaries."
        )
    else:
        interpretation.append(
            "No values fall outside the IQR outlier boundaries."
        )

    percentiles = {
        "5": _finite_float(valid.quantile(0.05)),
        "10": _finite_float(valid.quantile(0.10)),
        "25": q1,
        "50": median,
        "75": q3,
        "90": _finite_float(valid.quantile(0.90)),
        "95": _finite_float(valid.quantile(0.95)),
    }

    return DistributionSummary(
        column_name=column_name,
        count=count,
        missing_count=missing_count,
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        median=median,
        skewness=skewness,
        kurtosis=kurtosis,
        shape=shape,
        q1=q1,
        q3=q3,
        iqr=iqr,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        outlier_count=outlier_count,
        outlier_percentage=outlier_percentage,
        percentiles=percentiles,
        interpretation=interpretation,
    )


def analyze_dataset_distributions(
    data: pd.DataFrame,
    columns: list[str] | None = None,
    iqr_multiplier: float = 1.5,
) -> dict[str, DistributionSummary]:
    """Analyze selected numerical columns in a DataFrame."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if columns is None:
        columns = data.select_dtypes(
            include="number"
        ).columns.tolist()

    results = {}

    for column in columns:
        if column not in data.columns:
            raise ValueError(f"Column not found: {column}")

        if not pd.api.types.is_numeric_dtype(data[column]):
            continue

        results[str(column)] = analyze_distribution(
            data[column],
            column_name=str(column),
            iqr_multiplier=iqr_multiplier,
        )

    return results
