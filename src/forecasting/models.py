
"""Data models for forecasting results."""

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ForecastModelResult:
    """Store the forecast and evaluation metrics for one model."""

    model_name: str
    predictions: pd.Series
    mae: float
    rmse: float
    mape: float | None
    notes: list[str] = field(default_factory=list)


@dataclass
class ForecastComparisonResult:
    """Store the comparison of multiple forecasting models."""

    results: list[ForecastModelResult]
    best_model_name: str
    selection_metric: str = "RMSE"

    @property
    def best_result(self) -> ForecastModelResult:
        """Return the result selected as the best model."""

        for result in self.results:
            if result.model_name == self.best_model_name:
                return result

        raise ValueError("The selected best model was not found.")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert model results into a comparison table."""

        return pd.DataFrame(
            [
                {
                    "Model": result.model_name,
                    "MAE": result.mae,
                    "RMSE": result.rmse,
                    "MAPE (%)": result.mape,
                }
                for result in self.results
            ]
        ).sort_values(
            self.selection_metric,
            ascending=True,
            na_position="last",
        ).reset_index(drop=True)
