import pandas as pd

from src.quality.models import QualityFinding


RARE_CATEGORY_THRESHOLD = 0.01
MAX_RARE_CATEGORY_UNIQUES = 100


def _normalize_category(value: object) -> str:
    """Normalize a categorical value for comparison."""

    return str(value).strip().casefold()


def check_whitespace_inconsistencies(
    data: pd.DataFrame,
) -> list[QualityFinding]:
    """
    Detect categorical values containing leading or trailing
    whitespace.
    """

    findings = []

    categorical_columns = data.select_dtypes(
        include=["object", "category", "string"]
    ).columns

    for column in categorical_columns:
        series = data[column].dropna()

        if series.empty:
            continue

        string_values = series.astype(str)

        whitespace_mask = (
            string_values != string_values.str.strip()
        )

        affected_count = int(whitespace_mask.sum())

        if affected_count == 0:
            continue

        percentage = (
            affected_count / len(data) * 100
            if len(data) > 0
            else 0.0
        )

        findings.append(
            QualityFinding(
                issue_type="categorical_whitespace",
                severity="warning",
                message=(
                    f"Column '{column}' contains {affected_count} "
                    f"values with leading or trailing whitespace."
                ),
                column_name=column,
                affected_rows=affected_count,
                affected_percentage=percentage,
                suggested_action=(
                    "Review and standardize whitespace before "
                    "categorical analysis."
                ),
                details={
                    "affected_count": affected_count,
                },
            )
        )

    return findings


def check_casing_inconsistencies(
    data: pd.DataFrame,
) -> list[QualityFinding]:
    """
    Detect categories that differ only by capitalization.

    Example:
        Ahmedabad
        ahmedabad
        AHMEDABAD
    """

    findings = []

    categorical_columns = data.select_dtypes(
        include=["object", "category", "string"]
    ).columns

    for column in categorical_columns:
        series = data[column].dropna()

        if series.empty:
            continue

        groups: dict[str, set[str]] = {}

        for value in series.astype(str):
            normalized = _normalize_category(value)

            if normalized not in groups:
                groups[normalized] = set()

            groups[normalized].add(value)

        inconsistent_groups = {
            normalized: values
            for normalized, values in groups.items()
            if len(values) > 1
        }

        if not inconsistent_groups:
            continue

        affected_values = sum(
            len(values)
            for values in inconsistent_groups.values()
        )

        findings.append(
            QualityFinding(
                issue_type="categorical_casing_inconsistency",
                severity="warning",
                message=(
                    f"Column '{column}' contains "
                    f"{len(inconsistent_groups)} groups of "
                    "categories that differ by casing or whitespace."
                ),
                column_name=column,
                affected_rows=affected_values,
                affected_percentage=(
                    affected_values / len(data) * 100
                    if len(data) > 0
                    else 0.0
                ),
                suggested_action=(
                    "Review equivalent category values and "
                    "standardize them before analysis."
                ),
                details={
                    "inconsistent_groups": {
                        key: sorted(values)
                        for key, values in inconsistent_groups.items()
                    }
                },
            )
        )

    return findings


def check_rare_categories(
    data: pd.DataFrame,
) -> list[QualityFinding]:
    """
    Detect very rare categories.

    Rare categories are reported for review and are not
    automatically considered errors.
    """

    findings = []

    categorical_columns = data.select_dtypes(
        include=["object", "category", "string"]
    ).columns

    for column in categorical_columns:
        series = data[column].dropna()

        if series.empty:
            continue

        unique_count = series.nunique()

        if unique_count > MAX_RARE_CATEGORY_UNIQUES:
            continue

        frequencies = series.value_counts()

        rare_categories = frequencies[
            frequencies / len(series) < RARE_CATEGORY_THRESHOLD
        ]

        if rare_categories.empty:
            continue

        rare_count = int(rare_categories.sum())

        findings.append(
            QualityFinding(
                issue_type="rare_categories",
                severity="info",
                message=(
                    f"Column '{column}' contains "
                    f"{len(rare_categories)} rare categories "
                    f"representing {rare_count} rows."
                ),
                column_name=column,
                affected_rows=rare_count,
                affected_percentage=(
                    rare_count / len(data) * 100
                    if len(data) > 0
                    else 0.0
                ),
                suggested_action=(
                    "Review rare categories for possible data-entry "
                    "errors or legitimate low-frequency groups."
                ),
                details={
                    "rare_categories": {
                        str(category): int(count)
                        for category, count
                        in rare_categories.items()
                    }
                },
            )
        )

    return findings


def check_categorical_rules(
    data: pd.DataFrame,
) -> list[QualityFinding]:
    """Run all categorical quality rules."""

    findings = []

    findings.extend(
        check_whitespace_inconsistencies(data)
    )

    findings.extend(
        check_casing_inconsistencies(data)
    )

    findings.extend(
        check_rare_categories(data)
    )

    return findings