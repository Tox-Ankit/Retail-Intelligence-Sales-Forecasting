
"""Integrated workflow for preparing and forecasting tabular data."""

from dataclasses import dataclass, field

import pandas as pd

from src.forecasting.configuration import (
    ForecastingConfiguration,
)
from src.forecasting.detection import (
    ForecastingDetectionResult,
    detect_forecasting_columns,
)
from src.forecasting.engine import (
    ForecastEngineResult,
    run_forecasting,
)
from src.forecasting.preparation import (
    PreparedTimeSeries,
    prepare_time_series,
)


@dataclass
class ForecastingWorkflowResult:
    """Collect outputs from the complete forecasting workflow."""

    detection: ForecastingDetectionResult
    preparation: PreparedTimeSeries
    forecast_result: ForecastEngineResult
    notes: list[str] = field(default_factory=list)

    @property
    def forecast(self) -> pd.Series:
        """Return future predictions."""

        return self.forecast_result.forecast

    @property
    def selected_model_name(self) -> str:
        """Return the model selected by backtesting."""

        return self.forecast_result.selected_model_name

    @property
    def configuration(self) -> ForecastingConfiguration:
        """Return the forecasting configuration."""

        return self.forecast_result.configuration

    @property
    def model_comparison(self) -> pd.DataFrame:
        """Return the backtesting model comparison."""

        return self.forecast_result.backtest.summary.copy()

    def forecast_dataframe(self) -> pd.DataFrame:
        """Return predictions in a downloadable table."""

        return self.forecast_result.to_dataframe()


def run_forecasting_workflow(
    data: pd.DataFrame,
    date_column: str,
    value_column: str,
    frequency: str,
    aggregation: str = "sum",
    missing_period_strategy: str = "none",
    forecast_horizon: int | None = None,
    season_length: int | None = None,
    initial_train_size: int | None = None,
    step_size: int | None = None,
    include_seasonal: bool | None = None,
    moving_average_window: int = 3,
    smoothing_alpha: float = 0.3,
    seasonal_trend: str | None = "add",
    seasonal_type: str = "add",
) -> ForecastingWorkflowResult:
    """
    Detect candidates, prepare a selected series, and generate forecasts.

    The caller explicitly selects the date column, value column, frequency,
    aggregation and missing-period policy.
    """

    # 1. Validate the input and provide candidate suggestions.
    detection = detect_forecasting_columns(data)

    # 2. Prepare the explicitly selected time series.
    preparation = prepare_time_series(
        data=data,
        date_column=date_column,
        value_column=value_column,
        frequency=frequency,
        aggregation=aggregation,
        missing_period_strategy=missing_period_strategy,
    )

    series = preparation.series

    # Do not silently train a forecasting model on incomplete data.
    if series.isna().any():
        raise ValueError(
            f"The prepared time series contains {int(series.isna().sum())} "
            "missing periods. Choose an appropriate missing_period_strategy "
            "or provide more complete data before forecasting."
        )

    # 3. Configure, backtest, select, and forecast.
    forecast_result = run_forecasting(
        series=series,
        frequency=preparation.frequency,
        forecast_horizon=forecast_horizon,
        season_length=season_length,
        initial_train_size=initial_train_size,
        step_size=step_size,
        include_seasonal=include_seasonal,
        moving_average_window=moving_average_window,
        smoothing_alpha=smoothing_alpha,
        seasonal_trend=seasonal_trend,
        seasonal_type=seasonal_type,
    )

    notes = [
        "Date and target columns were explicitly selected by the caller.",
        "Column detection provides suggestions only; it does not choose "
        "the forecasting target.",
        *preparation.notes,
        *forecast_result.notes,
    ]

    return ForecastingWorkflowResult(
        detection=detection,
        preparation=preparation,
        forecast_result=forecast_result,
        notes=notes,
    )
