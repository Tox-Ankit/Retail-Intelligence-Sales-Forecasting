
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CorrelationResult:
    method: str
    matrix: dict[str, dict[str, float | None]]
    strongest_pairs: list[dict] = field(default_factory=list)
    interpretation: list[str] = field(default_factory=list)


def analyze_correlations(
    data: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "pearson",
    minimum_periods: int = 3,
    top_n: int = 10,
) -> CorrelationResult:
    """
    Calculate pairwise correlations between numerical columns.

    Pearson measures linear association.
    Spearman measures monotonic association using ranks.

    Correlation does not establish causation.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if method not in {"pearson", "spearman"}:
        raise ValueError(
            "method must be 'pearson' or 'spearman'."
        )

    if minimum_periods < 2:
        raise ValueError("minimum_periods must be at least 2.")

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    if columns is None:
        columns = data.select_dtypes(
            include="number"
        ).columns.tolist()
    else:
        for column in columns:
            if column not in data.columns:
                raise ValueError(f"Column not found: {column}")
            if not pd.api.types.is_numeric_dtype(data[column]):
                raise TypeError(
                    f"Column '{column}' must be numerical."
                )

    numeric_data = data[columns].copy()
    numeric_data = numeric_data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    matrix_df = numeric_data.corr(
        method=method,
        min_periods=minimum_periods,
    )

    matrix = {}

    for row in matrix_df.index:
        matrix[row] = {}

        for column in matrix_df.columns:
            value = matrix_df.loc[row, column]

            matrix[row][column] = (
                float(value) if pd.notna(value) else None
            )

    pairs = []

    column_names = matrix_df.columns.tolist()

    for i, left in enumerate(column_names):
        for right in column_names[i + 1:]:
            value = matrix_df.loc[left, right]

            if pd.isna(value):
                continue

            pairs.append({
                "column_1": left,
                "column_2": right,
                "correlation": float(value),
                "absolute_correlation": abs(float(value)),
            })

    pairs.sort(
        key=lambda pair: pair["absolute_correlation"],
        reverse=True,
    )

    strongest_pairs = pairs[:top_n]
    interpretation = []

    for pair in strongest_pairs:
        value = pair["correlation"]

        if abs(value) >= 0.7:
            strength = "strong"
        elif abs(value) >= 0.4:
            strength = "moderate"
        elif abs(value) >= 0.2:
            strength = "weak"
        else:
            strength = "very weak"

        direction = (
            "positive" if value > 0
            else "negative" if value < 0
            else "zero"
        )

        interpretation.append(
            f"{pair['column_1']} and {pair['column_2']} "
            f"have a {strength} {direction} {method} "
            f"correlation ({value:.3f})."
        )

    return CorrelationResult(
        method=method,
        matrix=matrix,
        strongest_pairs=strongest_pairs,
        interpretation=interpretation,
    )
