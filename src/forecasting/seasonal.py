
"""Seasonal forecasting models for retail time-series data."""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.forecasting.models import (
    ForecastComparisonResult,
    ForecastModelResult,
)


def _validate_inputs(
    train: pd.Series,
    test: pd.Series,
    season_length: int,
) -> None:
    """Validate the inputs for seasonal forecasting."""

    if not isinstance(train, pd.Series) or not isinstance(test, pd.Series):
        raise TypeError("train and test must be pandas Series.")

    if not isinstance(season_length, int) or isinstance(season_length, bool):
        raise TypeError("season_length must be an integer.")

    if season_length < 2:
        raise ValueError("season_length must be at least 2.")

    if len(train) < 2 * season_length:
        raise ValueError(
            "At least two complete seasons are required in training data."
        )

    if len(test) < 1:
        raise ValueError("Test data must contain at least one observation.")

    if not isinstance(train.index, pd.DatetimeIndex):
        raise ValueError("Training data must have a DatetimeIndex.")

    if not isinstance(test.index, pd.DatetimeIndex):
        raise ValueError("Test data must have a DatetimeIndex.")

    if not train.index.is_monotonic_increasing:
        raise ValueError("Training dates must be sorted.")

    if not test.index.is_monotonic_increasing:
        raise ValueError("Test dates must be sorted.")

    if not train.index.is_unique or not test.index.is_unique:
        raise ValueError("Training and test timestamps must be unique.")

    if train.index.max() >= test.index.min():
        raise ValueError("Test data must occur strictly after training data.")

    for name, values in (("Training", train), ("Test", test)):
        if values.isna().any():
            raise ValueError(f"{name} data contains missing values.")

        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{name} data contains invalid or infinite values.")


def _build_result(
    name: str,
    actual: pd.Series,
    predictions: pd.Series,
) -> ForecastModelResult:
    """Calculate metrics and build a forecasting result."""

    actual_values = actual.to_numpy(dtype=float)
    predicted_values = predictions.to_numpy(dtype=float)
    errors = actual_values - predicted_values

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))

    if np.any(actual_values == 0):
        mape = None
        notes = ["MAPE is unavailable because actual values include zero."]
    else:
        mape = float(
            np.mean(np.abs(errors / actual_values)) * 100
        )
        notes = []

    return ForecastModelResult(
        model_name=name,
        predictions=predictions,
        mae=mae,
        rmse=rmse,
        mape=mape,
        notes=notes,
    )


def seasonal_naive_forecast(
    train: pd.Series,
    test_index: pd.DatetimeIndex,
    season_length: int,
) -> pd.Series:
    """
    Forecast by repeating the last observed season.

    Example: for monthly data with season_length=12, the last 12
    training values are repeated to predict future months.
    """

    if not isinstance(train, pd.Series):
        raise TypeError("train must be a pandas Series.")

    if not isinstance(test_index, pd.DatetimeIndex):
        raise TypeError("test_index must be a pandas DatetimeIndex.")

    if not isinstance(season_length, int) or isinstance(season_length, bool):
        raise TypeError("season_length must be an integer.")

    if season_length < 2:
        raise ValueError("season_length must be at least 2.")

    if len(train) < season_length:
        raise ValueError("Training data is shorter than one complete season.")

    if len(test_index) == 0:
        raise ValueError("test_index must contain at least one timestamp.")

    if not test_index.is_monotonic_increasing or not test_index.is_unique:
        raise ValueError("Test timestamps must be sorted and unique.")

    if not isinstance(train.index, pd.DatetimeIndex):
        raise ValueError("Training data must have a DatetimeIndex.")

    if train.index.max() >= test_index.min():
        raise ValueError("Test data must occur strictly after training data.")

    if train.isna().any():
        raise ValueError("Training data contains missing values.")

    values = train.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Training data contains invalid or infinite values.")

    last_season = values[-season_length:]
    repeated = np.resize(last_season, len(test_index))

    return pd.Series(
        repeated,
        index=test_index,
        name="Seasonal Naive",
    )


def holt_winters_forecast(
    train: pd.Series,
    test_index: pd.DatetimeIndex,
    season_length: int,
    trend: str | None = "add",
    seasonal: str = "add",
) -> pd.Series:
    """
    Fit Holt-Winters exponential smoothing on training data only.

    trend can be 'add', 'mul', or None.
    seasonal can be 'add' or 'mul'.
    """

    if not isinstance(train, pd.Series):
        raise TypeError("train must be a pandas Series.")

    if not isinstance(test_index, pd.DatetimeIndex):
        raise TypeError("test_index must be a pandas DatetimeIndex.")

    if not isinstance(season_length, int) or isinstance(season_length, bool):
        raise TypeError("season_length must be an integer.")

    if season_length < 2:
        raise ValueError("season_length must be at least 2.")

    if len(train) < 2 * season_length:
        raise ValueError(
            "At least two complete seasons are required in training data."
        )

    if len(test_index) == 0:
        raise ValueError("test_index must contain at least one timestamp.")

    if not isinstance(train.index, pd.DatetimeIndex):
        raise ValueError("Training data must have a DatetimeIndex.")

    if train.index.max() >= test_index.min():
        raise ValueError("Test data must occur strictly after training data.")

    if train.isna().any():
        raise ValueError("Training data contains missing values.")

    if not np.isfinite(train.to_numpy(dtype=float)).all():
        raise ValueError("Training data contains invalid or infinite values.")

    if trend not in {"add", "mul", None}:
        raise ValueError("trend must be 'add', 'mul', or None.")

    if seasonal not in {"add", "mul"}:
        raise ValueError("seasonal must be 'add' or 'mul'.")

    values = train.to_numpy(dtype=float)

    if seasonal == "mul" and np.any(values <= 0):
        raise ValueError(
            "Multiplicative seasonality requires strictly positive values."
        )

    if trend == "mul" and np.any(values <= 0):
        raise ValueError(
            "Multiplicative trend requires strictly positive values."
        )

    model = ExponentialSmoothing(
        values,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=season_length,
        initialization_method="estimated",
    )

    fitted_model = model.fit(optimized=True)
    predictions = fitted_model.forecast(len(test_index))

    return pd.Series(
        np.asarray(predictions, dtype=float),
        index=test_index,
        name="Holt-Winters",
    )


def train_and_compare_seasonal_forecasters(
    train: pd.Series,
    test: pd.Series,
    season_length: int,
    trend: str | None = "add",
    seasonal: str = "add",
) -> ForecastComparisonResult:
    """Compare seasonal naive and Holt-Winters on a held-out test set."""

    _validate_inputs(train, test, season_length)

    seasonal_naive_predictions = seasonal_naive_forecast(
        train,
        test.index,
        season_length,
    )

    holt_winters_predictions = holt_winters_forecast(
        train,
        test.index,
        season_length,
        trend=trend,
        seasonal=seasonal,
    )

    results = [
        _build_result(
            "Seasonal Naive",
            test,
            seasonal_naive_predictions,
        ),
        _build_result(
            "Holt-Winters",
            test,
            holt_winters_predictions,
        ),
    ]

    best_result = min(results, key=lambda result: result.rmse)

    return ForecastComparisonResult(
        results=results,
        best_model_name=best_result.model_name,
        selection_metric="RMSE",
    )
