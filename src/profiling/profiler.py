from typing import Any

import pandas as pd

from src.ingestion.models import Dataset
from src.profiling.models import ColumnProfile, ProfileResult


def infer_column_type(series: pd.Series) -> str:
    """
    Infer a practical analytical type for a column.

    Returns one of:
    numerical, categorical, datetime, boolean, text
    """

    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_numeric_dtype(series):
        return "numerical"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    non_null = series.dropna()

    if len(non_null) > 0:
        if pd.api.types.is_string_dtype(series):
            unique_ratio = non_null.nunique() / len(non_null)

            # High-cardinality strings are more likely to be text.
            if unique_ratio > 0.5:
                return "text"

            return "categorical"

    return "text"


def detect_potential_id(
    series: pd.Series,
    column_name: str,
) -> bool:
    """
    Determine whether a column is likely to be an identifier.

    This is a heuristic, not a definitive classification.
    """

    non_null = series.dropna()

    if len(non_null) == 0:
        return False

    unique_ratio = non_null.nunique() / len(non_null)

    normalized_name = column_name.strip().lower().replace("-", "_").replace(" ", "_")

    id_keywords = {
        "id",
        "_id",
        "code",
        "_code",
        "key",
        "_key",
    }

    name_suggests_id = (
        normalized_name == "id"
        or any(
            normalized_name.endswith(keyword)
            for keyword in id_keywords
        )
    )

    high_uniqueness = unique_ratio >= 0.95

    return name_suggests_id or high_uniqueness


def get_sample_values(
    series: pd.Series,
    limit: int = 5,
) -> list[Any]:
    """Return a small set of representative non-null values."""

    values = series.dropna().head(limit).tolist()

    return values


def profile_column(
    series: pd.Series,
    column_name: str,
) -> ColumnProfile:
    """Create a profile for one column."""

    row_count = len(series)

    null_count = int(series.isna().sum())
    non_null_count = row_count - null_count

    unique_count = int(series.nunique(dropna=True))

    if non_null_count > 0:
        unique_percentage = (
            unique_count / non_null_count
        ) * 100
    else:
        unique_percentage = 0.0

    null_percentage = (
        (null_count / row_count) * 100
        if row_count > 0
        else 0.0
    )

    return ColumnProfile(
        name=column_name,
        data_type=str(series.dtype),
        inferred_type=infer_column_type(series),
        non_null_count=non_null_count,
        null_count=null_count,
        null_percentage=null_percentage,
        unique_count=unique_count,
        unique_percentage=unique_percentage,
        is_constant=unique_count <= 1,
        is_potential_id=detect_potential_id(
            series,
            column_name,
        ),
        sample_values=get_sample_values(series),
    )


def profile_dataset(dataset: Dataset) -> ProfileResult:
    """
    Profile one Dataset without modifying its data.
    """

    data = dataset.data

    column_profiles = [
        profile_column(data[column], column)
        for column in data.columns
    ]

    duplicate_rows = int(data.duplicated().sum())

    memory_usage = int(
        data.memory_usage(deep=True).sum()
    )

    return ProfileResult(
        dataset_name=dataset.name,
        rows=len(data),
        columns=len(data.columns),
        duplicate_rows=duplicate_rows,
        memory_usage_bytes=memory_usage,
        column_profiles=column_profiles,
    )


def profile_collection(
    datasets: list[Dataset],
) -> list[ProfileResult]:
    """
    Profile multiple datasets.
    """

    return [
        profile_dataset(dataset)
        for dataset in datasets
    ]