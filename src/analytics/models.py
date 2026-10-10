
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NumericalSummary:
    """Descriptive statistics for a numerical column."""

    column_name: str
    count: int
    missing_count: int

    mean: float | None = None
    median: float | None = None
    mode: list[Any] = field(default_factory=list)

    minimum: float | None = None
    maximum: float | None = None
    range_value: float | None = None

    variance: float | None = None
    standard_deviation: float | None = None

    q1: float | None = None
    q3: float | None = None
    iqr: float | None = None

    skewness: float | None = None
    kurtosis: float | None = None
    coefficient_of_variation: float | None = None

    percentiles: dict[str, float | None] = field(
        default_factory=dict
    )


@dataclass
class CategoricalSummary:
    """Frequency statistics for a categorical column."""

    column_name: str
    count: int
    missing_count: int
    unique_count: int

    mode: list[Any] = field(default_factory=list)
    frequencies: dict[str, int] = field(default_factory=dict)
    percentages: dict[str, float] = field(default_factory=dict)


@dataclass
class DatasetSummary:
    """Overall analytical summary of one dataset."""

    dataset_name: str
    rows: int
    columns: int

    numerical_summaries: dict[str, NumericalSummary] = field(
        default_factory=dict
    )

    categorical_summaries: dict[str, CategoricalSummary] = field(
        default_factory=dict
    )

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def numerical_column_count(self) -> int:
        return len(self.numerical_summaries)

    @property
    def categorical_column_count(self) -> int:
        return len(self.categorical_summaries)
