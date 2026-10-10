
"""Automatic configuration recommendations for forecasting."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ForecastingConfiguration:
    """Store recommended forecasting settings and validation findings."""

    frequency: str
    observation_count: int
    forecast_horizon: int
    initial_train_size: int
    step_size: int
    season_length: int | None
    seasonal_models_recommended: bool
    ready_for_forecasting: bool
    missing_values: int
    notes: list[str] = field(default_factory=list)


def _normalize_frequency(frequency: str) -> str:
    """Normalize common pandas frequency aliases."""

    aliases = {
        "D": "D",
        "W": "W",
        "M": "ME",
        "ME": "ME",
        "Q": "QE",
        "QE": "QE",
        "Y": "YE",
        "A": "YE",
        "YE": "YE",
    }

    if not isinstance(frequency, str):
        raise TypeError("frequency must be a string.")

    normalized = frequency.strip().upper()

    if normalized not in aliases:
        raise ValueError(
            "Unsupported frequency. Choose D, W, ME, QE, or YE."
        )

    return aliases[normalized]


def _default_horizon(frequency: str) -> int:
    """Recommend a practical starting forecast horizon."""

    horizons = {
        "D": 7,
        "W": 4,
        "ME": 3,
        "QE": 1,
        "YE": 1,
    }

    return horizons[frequency]


def _default_season_length(frequency: str) -> int | None:
    """Suggest a common seasonal period for the chosen frequency."""

    seasons = {
        "D": 7,    # Weekly pattern in daily data
        "W": 52,   # Annual pattern in weekly data
        "ME": 12,  # Annual pattern in monthly data
        "QE": 4,   # Annual pattern in quarterly data
        "YE": None,
    }

    return seasons[frequency]


def recommend_forecasting_configuration(
    series: pd.Series,
    frequency: str,
    forecast_horizon: int | None = None,
    season_length: int | None = None,
    initial_train_size: int | None = None,
    step_size: int | None = None,
) -> ForecastingConfiguration:
    """
    Recommend forecasting settings without modifying the input series.

    The caller explicitly supplies the data frequency. Seasonal settings
    are recommendations, not proof that a seasonal pattern exists.
    """

    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series.")

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("series must have a DatetimeIndex.")

    frequency = _normalize_frequency(frequency)
    count = len(series)

    if count == 0:
        raise ValueError("series cannot be empty.")

    if not series.index.is_monotonic_increasing:
        raise ValueError("series must be sorted by date.")

    if not series.index.is_unique:
        raise ValueError("series must have unique timestamps.")

    if forecast_horizon is not None:
        if (
            not isinstance(forecast_horizon, int)
            or isinstance(forecast_horizon, bool)
        ):
            raise TypeError("forecast_horizon must be an integer.")
        if forecast_horizon < 1:
            raise ValueError("forecast_horizon must be at least 1.")
        horizon = forecast_horizon
    else:
        horizon = _default_horizon(frequency)

    if season_length is not None:
        if not isinstance(season_length, int) or isinstance(
            season_length, bool
        ):
            raise TypeError("season_length must be an integer.")
        if season_length < 2:
            raise ValueError("season_length must be at least 2.")
        selected_season = season_length
    else:
        selected_season = _default_season_length(frequency)

    if initial_train_size is not None:
        if not isinstance(initial_train_size, int) or isinstance(
            initial_train_size, bool
        ):
            raise TypeError("initial_train_size must be an integer.")
        if initial_train_size < 2:
            raise ValueError("initial_train_size must be at least 2.")
        train_size = initial_train_size
    else:
        # Reserve the requested horizon for testing.
        available_train = count - horizon
        train_size = min(max(2, int(count * 0.7)), available_train)

    if step_size is not None:
        if not isinstance(step_size, int) or isinstance(step_size, bool):
            raise TypeError("step_size must be an integer.")
        if step_size < 1:
            raise ValueError("step_size must be at least 1.")
        selected_step = step_size
    else:
        selected_step = horizon

    missing_values = int(series.isna().sum())
    notes = []

    if missing_values:
        notes.append(
            f"The series contains {missing_values} missing values. "
            "Handle them before training or backtesting."
        )

    finite_values = series.dropna().to_numpy(dtype=float)
    invalid_values = not np.isfinite(finite_values).all()

    if invalid_values:
        notes.append(
            "The series contains infinite values. Clean them before forecasting."
        )

    if count < 5:
        notes.append(
            "Very little history is available. Forecast results may be unreliable."
        )

    if train_size < 2 or train_size + horizon > count:
        notes.append(
            "The selected training window and forecast horizon do not fit "
            "within the available observations."
        )

    seasonal_recommended = (
        selected_season is not None
        and train_size >= 2 * selected_season
    )

    if selected_season is None:
        notes.append(
            "No default seasonal period was suggested for this frequency."
        )
    elif not seasonal_recommended:
        notes.append(
            f"The training window has fewer than two complete seasons "
            f"of length {selected_season}. Seasonal models are not "
            "recommended with this training window."
        )
    else:
        notes.append(
            f"A seasonal period of {selected_season} is a candidate. "
            "Check whether the data actually exhibits recurring seasonality."
        )

    if not notes:
        notes.append(
            "The basic configuration checks passed. This does not guarantee "
            "that the data or forecasting model will be reliable."
        )

    ready = (
        count >= 5
        and train_size >= 2
        and train_size + horizon <= count
        and missing_values == 0
        and not invalid_values
    )

    return ForecastingConfiguration(
        frequency=frequency,
        observation_count=count,
        forecast_horizon=horizon,
        initial_train_size=train_size,
        step_size=selected_step,
        season_length=selected_season,
        seasonal_models_recommended=seasonal_recommended,
        ready_for_forecasting=ready,
        missing_values=missing_values,
        notes=notes,
    )
