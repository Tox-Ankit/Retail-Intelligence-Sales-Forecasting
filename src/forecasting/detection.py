
"""Detect potential date columns and forecasting targets."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ColumnSuggestion:
    """Describe one candidate column and why it was suggested."""

    column: str
    confidence: float
    reason: str
    data_type: str


@dataclass
class ForecastingDetectionResult:
    """Store candidate columns and configuration suggestions."""

    date_candidates: list[ColumnSuggestion] = field(default_factory=list)
    target_candidates: list[ColumnSuggestion] = field(default_factory=list)
    suggested_frequency: str | None = None
    notes: list[str] = field(default_factory=list)


_DATE_NAME_HINTS = (
    "date",
    "datetime",
    "timestamp",
    "time",
    "day",
)

_TARGET_NAME_HINTS = (
    "sales",
    "revenue",
    "quantity",
    "qty",
    "profit",
    "demand",
    "amount",
    "units",
    "orders",
    "transactions",
    "price",
    "income",
    "volume",
)


def _date_name_score(column: str) -> float:
    """Return a small ranking boost for date-like column names."""

    name = column.strip().lower().replace("_", " ").replace("-", " ")

    if any(hint in name for hint in _DATE_NAME_HINTS):
        return 0.15

    return 0.0


def _target_name_score(column: str) -> float:
    """Return a ranking boost for common business target names."""

    name = column.strip().lower().replace("_", " ").replace("-", " ")
    tokens = set(name.split())

    if any(hint in tokens for hint in _TARGET_NAME_HINTS):
        return 0.25

    if any(hint in name for hint in _TARGET_NAME_HINTS):
        return 0.15

    return 0.0



def _parse_dates(values: pd.Series) -> pd.Series:
    """Parse date-like values without format-inference warnings."""

    if pd.api.types.is_numeric_dtype(values.dtype):
        return pd.Series(
            pd.NaT,
            index=values.index,
            dtype="datetime64[ns, UTC]",
        )

    return pd.to_datetime(
        values,
        errors="coerce",
        utc=True,
        format="mixed",
    )



def _suggest_frequency(dates: pd.Series) -> str | None:
    """Suggest a common frequency from distinct, valid timestamps."""

    valid_dates = dates.dropna().drop_duplicates().sort_values()

    if len(valid_dates) < 3:
        return None

    # Use date-level gaps rather than intraday time differences.
    date_values = valid_dates.dt.normalize().drop_duplicates().sort_values()

    if len(date_values) < 3:
        return None

    gaps = date_values.diff().dropna().dt.total_seconds() / 86400
    median_gap = float(gaps.median())

    if median_gap <= 1.5:
        return "D"
    if median_gap <= 8:
        return "W"
    if median_gap <= 35:
        return "ME"
    if median_gap <= 100:
        return "QE"

    return "YE"


def detect_forecasting_columns(
    data: pd.DataFrame,
    date_parse_threshold: float = 0.8,
) -> ForecastingDetectionResult:
    """
    Rank likely date and numerical target columns.

    This function only produces suggestions. It does not choose a target
    or launch a forecasting model.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("Cannot detect forecasting columns in an empty dataset.")

    if not data.columns.is_unique:
        raise ValueError("Dataset contains duplicate column names.")

    if not 0 < date_parse_threshold <= 1:
        raise ValueError("date_parse_threshold must be greater than 0 and at most 1.")

    date_candidates = []
    target_candidates = []
    parsed_date_columns = {}

    for column in data.columns:
        values = data[column]
        non_missing = values.dropna()

        if non_missing.empty:
            continue

        # Detect date columns.
        is_datetime = pd.api.types.is_datetime64_any_dtype(values.dtype)

        if is_datetime:
            parsed = pd.to_datetime(values, errors="coerce", utc=True)
            parse_ratio = float(parsed.notna().mean())
        elif (
            pd.api.types.is_object_dtype(values.dtype)
            or pd.api.types.is_string_dtype(values.dtype)
        ):
            parsed = _parse_dates(values)
            parse_ratio = float(parsed.notna().sum() / len(non_missing))
        else:
            parsed = None
            parse_ratio = 0.0

        if (
            parsed is not None
            and parse_ratio >= date_parse_threshold
            and parsed.nunique(dropna=True) >= 3
        ):
            name_boost = _date_name_score(str(column))
            confidence = min(0.99, 0.75 + 0.20 * parse_ratio + name_boost)

            date_candidates.append(
                ColumnSuggestion(
                    column=str(column),
                    confidence=round(confidence, 3),
                    reason=(
                        "Datetime values were detected in this column."
                        if is_datetime
                        else "Most non-missing values can be parsed as dates."
                    ),
                    data_type="datetime",
                )
            )
            parsed_date_columns[column] = parsed

        # Detect numerical target candidates, excluding booleans.
        if (
            pd.api.types.is_numeric_dtype(values.dtype)
            and not pd.api.types.is_bool_dtype(values.dtype)
        ):
            numeric = pd.to_numeric(values, errors="coerce")
            numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()

            if len(numeric) >= 3 and numeric.nunique() >= 2:
                name_boost = _target_name_score(str(column))
                completeness = len(numeric) / len(data)

                confidence = min(
                    0.99,
                    0.55 + 0.20 * completeness + name_boost,
                )

                reason = "Numerical column with varying values."

                if name_boost:
                    reason += " Its name resembles a common business metric."

                target_candidates.append(
                    ColumnSuggestion(
                        column=str(column),
                        confidence=round(confidence, 3),
                        reason=reason,
                        data_type="numeric",
                    )
                )

    date_candidates.sort(
        key=lambda item: item.confidence,
        reverse=True,
    )
    target_candidates.sort(
        key=lambda item: item.confidence,
        reverse=True,
    )

    notes = []

    if not date_candidates:
        notes.append(
            "No suitable date column was detected. Check the source format "
            "or select and prepare a date column manually."
        )

    if not target_candidates:
        notes.append(
            "No varying numerical target candidate was detected."
        )

    suggested_frequency = None

    if date_candidates:
        top_date_column = date_candidates[0].column
        suggested_frequency = _suggest_frequency(
            parsed_date_columns[top_date_column]
        )

        if suggested_frequency is None:
            notes.append(
                "There are not enough distinct dates to suggest a frequency."
            )
        else:
            notes.append(
                f"Suggested frequency: {suggested_frequency}. "
                "Verify it against the actual business reporting cadence."
            )

    notes.append(
        "Candidate rankings are heuristics, not probabilities. "
        "Review the suggestions and choose the forecasting target yourself."
    )

    return ForecastingDetectionResult(
        date_candidates=date_candidates,
        target_candidates=target_candidates,
        suggested_frequency=suggested_frequency,
        notes=notes,
    )
