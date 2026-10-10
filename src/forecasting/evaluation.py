
"""Unified walk-forward backtesting for forecasting models."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.forecasting.training import train_and_compare_forecasters
from src.forecasting.seasonal import (
    seasonal_naive_forecast,
    holt_winters_forecast,
)


@dataclass
class BacktestResult:
    """Store evaluation results across multiple chronological folds."""

    fold_results: pd.DataFrame
    summary: pd.DataFrame
    best_model_name: str
    initial_train_size: int
    horizon: int
    step_size: int


def _calculate_metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict[str, float | None]:
    """Calculate MAE, RMSE and MAPE safely."""

    actual_values = actual.to_numpy(dtype=float)
    predicted_values = predicted.to_numpy(dtype=float)
    errors = actual_values - predicted_values

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))

    mape = (
        None
        if np.any(actual_values == 0)
        else float(np.mean(np.abs(errors / actual_values)) * 100)
    )

    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}


def walk_forward_backtest(
    series: pd.Series,
    initial_train_size: int,
    horizon: int = 1,
    step_size: int = 1,
    moving_average_window: int = 3,
    smoothing_alpha: float = 0.3,
    season_length: int | None = None,
    include_seasonal: bool = False,
    seasonal_trend: str | None = "add",
    seasonal_type: str = "add",
) -> BacktestResult:
    """
    Evaluate baseline and optionally seasonal forecasting models.

    Seasonal methods are included only when explicitly requested.
    Every model is trained on the same expanding chronological window.
    """

    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series.")

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("series must have a DatetimeIndex.")

    if not series.index.is_monotonic_increasing:
        raise ValueError("series must be sorted by date.")

    if not series.index.is_unique:
        raise ValueError("series must have unique timestamps.")

    if series.isna().any():
        raise ValueError("series cannot contain missing values.")

    if not np.isfinite(series.to_numpy(dtype=float)).all():
        raise ValueError("series cannot contain infinite or invalid values.")

    if not isinstance(initial_train_size, int) or isinstance(
        initial_train_size, bool
    ):
        raise TypeError("initial_train_size must be an integer.")

    if not isinstance(horizon, int) or isinstance(horizon, bool):
        raise TypeError("horizon must be an integer.")

    if not isinstance(step_size, int) or isinstance(step_size, bool):
        raise TypeError("step_size must be an integer.")

    if initial_train_size < 2:
        raise ValueError("initial_train_size must be at least 2.")

    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    if step_size < 1:
        raise ValueError("step_size must be at least 1.")

    if initial_train_size + horizon > len(series):
        raise ValueError(
            "Not enough observations for the initial training window "
            "and first test horizon."
        )

    if not isinstance(include_seasonal, bool):
        raise TypeError("include_seasonal must be a boolean.")

    if include_seasonal:
        if season_length is None:
            raise ValueError(
                "season_length is required when include_seasonal=True."
            )
        if not isinstance(season_length, int) or isinstance(
            season_length, bool
        ):
            raise TypeError("season_length must be an integer.")
        if season_length < 2:
            raise ValueError("season_length must be at least 2.")
        if initial_train_size < 2 * season_length:
            raise ValueError(
                "initial_train_size must contain at least two complete "
                "seasons when seasonal models are enabled."
            )

    fold_records = []
    fold_number = 1
    train_end = initial_train_size

    while train_end + horizon <= len(series):
        train = series.iloc[:train_end]
        test = series.iloc[train_end : train_end + horizon]

        comparison = train_and_compare_forecasters(
            train=train,
            test=test,
            moving_average_window=moving_average_window,
            smoothing_alpha=smoothing_alpha,
        )

        fold_predictions = [
            (item.model_name, item.predictions)
            for item in comparison.results
        ]

        if include_seasonal:
            seasonal_naive = seasonal_naive_forecast(
                train,
                test.index,
                season_length,
            )
            fold_predictions.append(("Seasonal Naive", seasonal_naive))

            try:
                holt_winters = holt_winters_forecast(
                    train,
                    test.index,
                    season_length,
                    trend=seasonal_trend,
                    seasonal=seasonal_type,
                )
                fold_predictions.append(("Holt-Winters", holt_winters))
            except (ValueError, np.linalg.LinAlgError) as exc:
                # A model may fail on a particular training window.
                # Record the failure rather than silently hiding it.
                fold_records.append(
                    {
                        "Fold": fold_number,
                        "Model": "Holt-Winters",
                        "Train End": train.index[-1],
                        "Test Start": test.index[0],
                        "Test End": test.index[-1],
                        "MAE": np.nan,
                        "RMSE": np.nan,
                        "MAPE (%)": np.nan,
                        "Error": str(exc),
                    }
                )

        for model_name, predictions in fold_predictions:
            metrics = _calculate_metrics(test, predictions)

            fold_records.append(
                {
                    "Fold": fold_number,
                    "Model": model_name,
                    "Train End": train.index[-1],
                    "Test Start": test.index[0],
                    "Test End": test.index[-1],
                    **metrics,
                    "Error": None,
                }
            )

        fold_number += 1
        train_end += step_size

    if not fold_records:
        raise ValueError("No backtesting folds were created.")

    fold_results = pd.DataFrame(fold_records)

    successful = fold_results.dropna(subset=["RMSE"])

    if successful.empty:
        raise ValueError("No forecasting models completed successfully.")

    summary = (
        successful.groupby("Model", sort=False)
        .agg(
            **{
                "Mean MAE": ("MAE", "mean"),
                "Mean RMSE": ("RMSE", "mean"),
                "Folds Evaluated": ("RMSE", "count"),
            }
        )
        .reset_index()
    )

    mape_summary = (
        successful.groupby("Model", sort=False)["MAPE (%)"]
        .agg(lambda values: values.mean() if values.notna().all() else np.nan)
        .rename("Mean MAPE (%)")
        .reset_index()
    )

    summary = summary.merge(mape_summary, on="Model", how="left")
    summary = summary.sort_values(
        "Mean RMSE",
        ascending=True,
    ).reset_index(drop=True)

    best_model_name = str(summary.iloc[0]["Model"])

    return BacktestResult(
        fold_results=fold_results,
        summary=summary,
        best_model_name=best_model_name,
        initial_train_size=initial_train_size,
        horizon=horizon,
        step_size=step_size,
    )
