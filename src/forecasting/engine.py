
"""End-to-end forecasting orchestration."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.forecasting.configuration import (
    ForecastingConfiguration,
    recommend_forecasting_configuration,
)
from src.forecasting.evaluation import (
    BacktestResult,
    walk_forward_backtest,
)
from src.forecasting.seasonal import (
    seasonal_naive_forecast,
)


@dataclass
class ForecastEngineResult:
    """Complete output from a forecasting run."""

    forecast: pd.Series
    selected_model_name: str
    backtest: BacktestResult
    configuration: ForecastingConfiguration
    notes: list[str] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Return future predictions as a downloadable table."""

        return pd.DataFrame(
            {
                "Forecast Date": self.forecast.index,
                "Forecast": self.forecast.to_numpy(),
                "Selected Model": self.selected_model_name,
            }
        )


def _create_future_index(
    series: pd.Series,
    frequency: str,
    horizon: int,
) -> pd.DatetimeIndex:
    """Generate future timestamps after the final observation."""

    dates = pd.date_range(
        start=series.index[-1],
        periods=horizon + 1,
        freq=frequency,
    )[1:]

    if len(dates) != horizon or dates[0] <= series.index[-1]:
        raise ValueError(
            "Could not generate valid future timestamps. "
            "Check the frequency and the time-series index."
        )

    return dates


def _generate_forecast(
    model_name: str,
    series: pd.Series,
    future_index: pd.DatetimeIndex,
    moving_average_window: int,
    smoothing_alpha: float,
    season_length: int | None,
    seasonal_trend: str | None,
    seasonal_type: str,
) -> pd.Series:
    """Generate a future forecast using the selected model."""

    values = series.to_numpy(dtype=float)
    horizon = len(future_index)

    if model_name == "Naive":
        predictions = np.repeat(values[-1], horizon)

    elif model_name == "Moving Average":
        predictions = np.repeat(
            float(series.iloc[-moving_average_window:].mean()),
            horizon,
        )

    elif model_name == "Exponential Smoothing":
        smoothed = (
            series.astype(float)
            .ewm(alpha=smoothing_alpha, adjust=False)
            .mean()
            .iloc[-1]
        )
        predictions = np.repeat(float(smoothed), horizon)

    elif model_name == "Seasonal Naive":
        if season_length is None:
            raise ValueError(
                "Seasonal Naive requires a configured season_length."
            )

        return seasonal_naive_forecast(
            series,
            future_index,
            season_length,
        )

    elif model_name == "Holt-Winters":
        if season_length is None:
            raise ValueError(
                "Holt-Winters requires a configured season_length."
            )

        model = ExponentialSmoothing(
            values,
            trend=seasonal_trend,
            seasonal=seasonal_type,
            seasonal_periods=season_length,
            initialization_method="estimated",
        )
        fitted_model = model.fit(optimized=True)
        predictions = np.asarray(
            fitted_model.forecast(horizon),
            dtype=float,
        )

    else:
        raise ValueError(f"Unsupported forecasting model: {model_name}")

    if not np.isfinite(predictions).all():
        raise ValueError("The selected model generated invalid predictions.")

    return pd.Series(
        predictions,
        index=future_index,
        name="Forecast",
    )


def run_forecasting(
    series: pd.Series,
    frequency: str,
    forecast_horizon: int | None = None,
    season_length: int | None = None,
    initial_train_size: int | None = None,
    step_size: int | None = None,
    include_seasonal: bool | None = None,
    moving_average_window: int = 3,
    smoothing_alpha: float = 0.3,
    seasonal_trend: str | None = "add",
    seasonal_type: str = "add",
) -> ForecastEngineResult:
    """
    Configure, backtest, select and fit a forecasting model.

    Model selection uses chronological backtesting. The selected model
    is then fitted on the full historical series to forecast future periods.
    """

    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series.")

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("series must have a DatetimeIndex.")

    if len(series) < 5:
        raise ValueError("At least 5 observations are required.")

    if not series.index.is_monotonic_increasing:
        raise ValueError("series must be sorted by date.")

    if not series.index.is_unique:
        raise ValueError("series must have unique timestamps.")

    if series.isna().any():
        raise ValueError(
            "Handle missing values before running the forecasting engine."
        )

    if not np.isfinite(series.to_numpy(dtype=float)).all():
        raise ValueError("series contains invalid or infinite values.")

    if (
        not isinstance(moving_average_window, int)
        or isinstance(moving_average_window, bool)
    ):
        raise TypeError("moving_average_window must be an integer.")

    if not 1 <= moving_average_window <= len(series):
        raise ValueError(
            "moving_average_window must be between 1 and the series length."
        )

    if not 0 < smoothing_alpha <= 1:
        raise ValueError(
            "smoothing_alpha must be greater than 0 and at most 1."
        )

    if include_seasonal is not None and not isinstance(
        include_seasonal, bool
    ):
        raise TypeError("include_seasonal must be a boolean or None.")

    if seasonal_trend not in {"add", "mul", None}:
        raise ValueError("seasonal_trend must be 'add', 'mul', or None.")

    if seasonal_type not in {"add", "mul"}:
        raise ValueError("seasonal_type must be 'add' or 'mul'.")

    configuration = recommend_forecasting_configuration(
        series=series,
        frequency=frequency,
        forecast_horizon=forecast_horizon,
        season_length=season_length,
        initial_train_size=initial_train_size,
        step_size=step_size,
    )

    if not configuration.ready_for_forecasting:
        raise ValueError(
            "The series is not ready for forecasting. "
            + " ".join(configuration.notes)
        )

    # Keep the chosen seasonal configuration explicit.
    use_seasonal = (
        configuration.seasonal_models_recommended
        if include_seasonal is None
        else include_seasonal
    )

    if use_seasonal:
        if configuration.season_length is None:
            raise ValueError(
                "Seasonal models require an explicit season_length."
            )

        if configuration.initial_train_size < (
            2 * configuration.season_length
        ):
            raise ValueError(
                "The initial training window must contain at least "
                "two complete seasons."
            )

    window = min(moving_average_window, configuration.initial_train_size)

    backtest = walk_forward_backtest(
        series=series,
        initial_train_size=configuration.initial_train_size,
        horizon=configuration.forecast_horizon,
        step_size=configuration.step_size,
        moving_average_window=window,
        smoothing_alpha=smoothing_alpha,
        season_length=configuration.season_length,
        include_seasonal=use_seasonal,
        seasonal_trend=seasonal_trend,
        seasonal_type=seasonal_type,
    )

    selected_model = backtest.best_model_name

    future_index = _create_future_index(
        series,
        configuration.frequency,
        configuration.forecast_horizon,
    )

    try:
        forecast = _generate_forecast(
            model_name=selected_model,
            series=series,
            future_index=future_index,
            moving_average_window=window,
            smoothing_alpha=smoothing_alpha,
            season_length=configuration.season_length,
            seasonal_trend=seasonal_trend,
            seasonal_type=seasonal_type,
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise RuntimeError(
            f"The selected model '{selected_model}' could not be fitted "
            "to the full historical series."
        ) from exc

    notes = list(configuration.notes)
    notes.append(
        f"Selected '{selected_model}' using the lowest mean RMSE "
        "across successful backtesting folds."
    )
    notes.append(
        "Forecasts are estimates, not guarantees. Review historical "
        "performance before using them for business decisions."
    )

    if use_seasonal:
        notes.append(
            "Seasonal methods were enabled. The configured season length "
            "is an assumption, not proof of seasonality."
        )

    return ForecastEngineResult(
        forecast=forecast,
        selected_model_name=selected_model,
        backtest=backtest,
        configuration=configuration,
        notes=notes,
    )
