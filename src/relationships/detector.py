import pandas as pd

from src.ingestion.models import Dataset, DatasetCollection
from src.relationships.models import (
    RelationshipCandidate,
    RelationshipResult,
)


def _normalize_column_name(column_name: str) -> str:
    """Normalize a column name for comparison."""
    return (
        str(column_name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def _columns_have_similar_names(
    left_column: str,
    right_column: str,
) -> bool:
    """Check whether two column names are likely related."""

    left = _normalize_column_name(left_column)
    right = _normalize_column_name(right_column)

    if left == right:
        return True

    # Handle common ID naming variations.
    left_without_id = left.removesuffix("_id").removesuffix("id")
    right_without_id = right.removesuffix("_id").removesuffix("id")

    return (
        left_without_id
        and right_without_id
        and left_without_id == right_without_id
    )


def _is_unique(series: pd.Series) -> bool:
    """Check whether non-null values in a column are unique."""

    non_null = series.dropna()

    if len(non_null) == 0:
        return False

    return bool(non_null.nunique() == len(non_null))


def _calculate_value_overlap(
    left_series: pd.Series,
    right_series: pd.Series,
) -> tuple[int, float]:
    """
    Calculate how many values from the left column
    appear in the right column.

    Returns:
        overlap_count
        overlap_percentage
    """

    left_values = set(left_series.dropna().unique())
    right_values = set(right_series.dropna().unique())

    if not left_values:
        return 0, 0.0

    overlap = left_values.intersection(right_values)

    percentage = (len(overlap) / len(left_values)) * 100

    return len(overlap), percentage


def _infer_relationship_type(
    left_unique: bool,
    right_unique: bool,
) -> str:
    """Infer cardinality from column uniqueness."""

    if left_unique and right_unique:
        return "one_to_one"

    if left_unique and not right_unique:
        return "one_to_many"

    if not left_unique and right_unique:
        return "many_to_one"

    return "many_to_many"


def _calculate_confidence(
    same_name: bool,
    compatible_dtype: bool,
    left_unique: bool,
    right_unique: bool,
    overlap_percentage: float,
) -> tuple[float, list[str]]:
    """
    Calculate a transparent relationship confidence score.

    Returns:
        confidence score
        evidence list
    """

    score = 0.0
    evidence = []

    # Column-name similarity
    if same_name:
        score += 30
        evidence.append("Column names are similar.")

    # Data type compatibility
    if compatible_dtype:
        score += 20
        evidence.append("Column data types are compatible.")

    # Primary/foreign-key characteristics
    if left_unique:
        score += 15
        evidence.append(
            "Left column contains unique non-null values."
        )

    if right_unique:
        score += 15
        evidence.append(
            "Right column contains unique non-null values."
        )

    # Value overlap
    if overlap_percentage >= 90:
        score += 20
        evidence.append(
            "Very high value overlap was detected."
        )
    elif overlap_percentage >= 70:
        score += 15
        evidence.append(
            "High value overlap was detected."
        )
    elif overlap_percentage >= 40:
        score += 10
        evidence.append(
            "Moderate value overlap was detected."
        )
    elif overlap_percentage > 0:
        score += 5
        evidence.append(
            "Some value overlap was detected."
        )

    return min(score, 100.0), evidence


def _columns_are_compatible(
    left_series: pd.Series,
    right_series: pd.Series,
) -> bool:
    """Check whether two columns have compatible data types."""

    if (
        pd.api.types.is_numeric_dtype(left_series)
        and pd.api.types.is_numeric_dtype(right_series)
    ):
        return True

    if (
        pd.api.types.is_string_dtype(left_series)
        and pd.api.types.is_string_dtype(right_series)
    ):
        return True

    if (
        pd.api.types.is_datetime64_any_dtype(left_series)
        and pd.api.types.is_datetime64_any_dtype(right_series)
    ):
        return True

    # Object columns may contain mixed/string-like identifiers.
    if left_series.dtype == object and right_series.dtype == object:
        return True

    return False


def detect_relationships(
    collection: DatasetCollection,
    minimum_confidence: float = 40.0,
) -> RelationshipResult:
    """
    Detect possible relationships between datasets.

    No data is merged or modified.

    Parameters
    ----------
    collection:
        DatasetCollection containing the datasets.

    minimum_confidence:
        Minimum confidence required for a candidate to be
        included in the result.

    Returns
    -------
    RelationshipResult
        Detected relationship candidates.
    """

    if not isinstance(collection, DatasetCollection):
        raise TypeError(
            "collection must be a DatasetCollection."
        )

    if not 0 <= minimum_confidence <= 100:
        raise ValueError(
            "minimum_confidence must be between 0 and 100."
        )

    result = RelationshipResult()

    datasets = collection.datasets

    # Compare every dataset pair exactly once.
    for left_index in range(len(datasets)):

        left_dataset: Dataset = datasets[left_index]

        for right_index in range(
            left_index + 1,
            len(datasets),
        ):

            right_dataset: Dataset = datasets[right_index]

            for left_column in left_dataset.data.columns:

                left_series = left_dataset.data[left_column]

                for right_column in right_dataset.data.columns:

                    right_series = right_dataset.data[right_column]

                    same_name = _columns_have_similar_names(
                        left_column,
                        right_column,
                    )

                    compatible_dtype = _columns_are_compatible(
                        left_series,
                        right_series,
                    )

                    # Completely incompatible columns are not
                    # useful relationship candidates.
                    if not compatible_dtype:
                        continue

                    overlap_count, overlap_percentage = (
                        _calculate_value_overlap(
                            left_series,
                            right_series,
                        )
                    )

                    left_unique = _is_unique(left_series)
                    right_unique = _is_unique(right_series)

                    confidence, evidence = _calculate_confidence(
                        same_name=same_name,
                        compatible_dtype=compatible_dtype,
                        left_unique=left_unique,
                        right_unique=right_unique,
                        overlap_percentage=overlap_percentage,
                    )

                    # A relationship with no matching values
                    # and unrelated column names is not useful.
                    if (
                        not same_name
                        and overlap_percentage == 0
                    ):
                        continue

                    if confidence < minimum_confidence:
                        continue

                    relationship_type = _infer_relationship_type(
                        left_unique,
                        right_unique,
                    )

                    candidate = RelationshipCandidate(
                        left_dataset=left_dataset.name,
                        left_column=left_column,
                        right_dataset=right_dataset.name,
                        right_column=right_column,
                        relationship_type=relationship_type,
                        confidence=confidence,
                        evidence=evidence,
                        left_unique=left_unique,
                        right_unique=right_unique,
                        left_non_null_count=int(
                            left_series.notna().sum()
                        ),
                        right_non_null_count=int(
                            right_series.notna().sum()
                        ),
                        value_overlap_count=overlap_count,
                        value_overlap_percentage=overlap_percentage,
                        details={
                            "left_dtype": str(left_series.dtype),
                            "right_dtype": str(right_series.dtype),
                        },
                    )

                    result.add_candidate(candidate)

    # Highest-confidence relationships first.
    result.candidates.sort(
        key=lambda candidate: candidate.confidence,
        reverse=True,
    )

    return result