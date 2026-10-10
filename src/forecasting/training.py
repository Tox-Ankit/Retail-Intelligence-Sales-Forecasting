
"""Train and compare baseline statistical forecasting models."""

import numpy as np
import pandas as pd

from src.forecasting.models import (
    ForecastComparisonResult,
    ForecastModelResult,
)


def _validate_series(
    train: pd.Series,
    test: pd.Series,
) -> None:
    """Validate time-series inputs before forecasting."""

    if not isinstance(train, pd.Series) or not isinstance(test, pd.Series):
        raise TypeError("train and test must be pandas Series.")

    if len(train) < 2:
        raise ValueError("Training data must contain at least 2 observations.")

    if len(test) < 1:
        raise ValueError("Test data must contain at least 1 observation.")

    if not isinstance(train.index, pd.DatetimeIndex):
        raise ValueError("Training data must have a DatetimeIndex.")

    if not isinstance(test.index, pd.DatetimeIndex):
        raise ValueError("Test data must have a DatetimeIndex.")

    if not train.index.is_monotonic_increasing:
        raise ValueError("Training dates must be sorted.")

    if not test.index.is_monotonic_increasing:
        raise ValueError("Test dates must be sorted.")

    if not train.index.is_unique or not test.index.is_unique:
        raise ValueError("Training and test dates must be unique.")

    if train.index.max() >= test.index.min():
        raise ValueError(
            "Test data must occur strictly after the training data."
        )

    if train.isna().any() or test.isna().any():
        raise ValueError(
            "Training and test data cannot contain missing values."
        )

    if not np.isfinite(train.to_numpy(dtype=float)).all():
        raise ValueError("Training data contains infinite or invalid values.")

    if not np.isfinite(test.to_numpy(dtype=float)).all():
        raise ValueError("Test data contains infinite or invalid values.")


def _calculate_metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> tuple[float, float, float | None]:
    """Calculate MAE, RMSE and MAPE."""

    actual_values = actual.to_numpy(dtype=float)
    predicted_values = predicted.to_numpy(dtype=float)

    errors = actual_values - predicted_values

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # MAPE is undefined when any actual observation is zero.
    if np.any(actual_values == 0):
        mape = None
    else:
        mape = float(
            np.mean(np.abs(errors / actual_values)) * 100
        )

    return mae, rmse, mape


def _create_result(
    model_name: str,
    predictions: pd.Series,
    actual: pd.Series,
) -> ForecastModelResult:
    """Build a result object for one forecasting model."""

    mae, rmse, mape = _calculate_metrics(actual, predictions)

    notes = []

    if mape is None:
        notes.append(
            "MAPE is unavailable because the test data contains zero values."
        )

    return ForecastModelResult(
        model_name=model_name,
        predictions=predictions,
        mae=mae,
        rmse=rmse,
        mape=mape,
        notes=notes,
    )


def train_and_compare_forecasters(
    train: pd.Series,
    test: pd.Series,
    moving_average_window: int = 3,
    smoothing_alpha: float = 0.3,
) -> ForecastComparisonResult:
    """
    Compare three baseline forecasting approaches.

    All predictions cover the full test horizon. Each method is fitted
    using training data only; test values are never used to make forecasts.
    """

    _validate_series(train, test)

    if not isinstance(moving_average_window, int):
        raise TypeError("moving_average_window must be an integer.")

    if not 1 <= moving_average_window <= len(train):
        raise ValueError(
            "moving_average_window must be between 1 and the training length."
        )

    if not 0 < smoothing_alpha <= 1:
        raise ValueError("smoothing_alpha must be greater than 0 and at most 1.")

    horizon = len(test)
    forecast_index = test.index

    # 1. Naive forecast: repeat the last training observation.
    naive_value = float(train.iloc[-1])

    naive_predictions = pd.Series(
        naive_value,
        index=forecast_index,
        name="Naive",
    )

    # 2. Moving-average forecast: repeat the recent training average.
    moving_average_value = float(
        train.iloc[-moving_average_window:].mean()
    )

    moving_average_predictions = pd.Series(
        moving_average_value,
        index=forecast_index,
        name="Moving Average",
    )

    # 3. Simple exponential smoothing:
    # Calculate the smoothed level from training data only.
    smoothed_level = (
        train.astype(float)
        .ewm(
            alpha=smoothing_alpha,
            adjust=False,
        )
        .mean()
        .iloc[-1]
    )

    exponential_predictions = pd.Series(
        float(smoothed_level),
        index=forecast_index,
        name="Exponential Smoothing",
    )

    results = [
        _create_result("Naive", naive_predictions, test),
        _create_result(
            "Moving Average",
            moving_average_predictions,
            test,
        ),
        _create_result(
            "Exponential Smoothing",
            exponential_predictions,
            test,
        ),
    ]

    # Select the model with the lowest RMSE on the held-out test period.
    best_result = min(results, key=lambda result: result.rmse)

    return ForecastComparisonResult(
        results=results,
        best_model_name=best_result.model_name,
        selection_metric="RMSE",
    )
