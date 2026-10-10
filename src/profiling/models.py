from dataclasses import dataclass, field
from typing import Any


@dataclass
class ColumnProfile:
    """Profile information for a single column."""

    name: str
    data_type: str
    inferred_type: str
    non_null_count: int
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    is_constant: bool
    is_potential_id: bool
    sample_values: list[Any] = field(default_factory=list)


@dataclass
class ProfileResult:
    """Complete profiling information for one dataset."""

    dataset_name: str
    rows: int
    columns: int
    duplicate_rows: int
    memory_usage_bytes: int
    column_profiles: list[ColumnProfile] = field(default_factory=list)

    @property
    def duplicate_percentage(self) -> float:
        """Return the percentage of duplicate rows."""
        if self.rows == 0:
            return 0.0

        return (self.duplicate_rows / self.rows) * 100

    @property
    def columns_with_missing_values(self) -> list[ColumnProfile]:
        """Return columns containing missing values."""
        return [
            column
            for column in self.column_profiles
            if column.null_count > 0
        ]

    @property
    def constant_columns(self) -> list[ColumnProfile]:
        """Return columns containing only one unique value."""
        return [
            column
            for column in self.column_profiles
            if column.is_constant
        ]

    @property
    def potential_id_columns(self) -> list[ColumnProfile]:
        """Return columns that may represent identifiers."""
        return [
            column
            for column in self.column_profiles
            if column.is_potential_id
        ]