from src.ingestion.models import Dataset
from src.quality.models import QualityReport

from src.quality.rules.constant_columns import (
    check_constant_columns,
)
from src.quality.rules.duplicates import (
    check_duplicate_rows,
)
from src.quality.rules.empty_dataset import (
    check_empty_dataset,
)
from src.quality.rules.missing_values import (
    check_missing_values,
)
from src.quality.rules.numerical import (
    check_numerical_rules,
)
from src.quality.rules.categorical import (
    check_categorical_rules,
)
from src.quality.rules.outliers import check_outliers

def check_quality(dataset: Dataset) -> QualityReport:
    """
    Run all registered quality rules for one dataset.

    The dataset is never modified.
    """

    report = QualityReport(
        dataset_name=dataset.name,
    )

    data = dataset.data

    rules = [
        check_empty_dataset,
        check_missing_values,
        check_duplicate_rows,
        check_constant_columns,
        check_numerical_rules,
        check_categorical_rules,
        check_outliers,
    ]

    for rule in rules:
        findings = rule(data)

        for finding in findings:
            report.add_finding(finding)

    return report


def check_collection(
    datasets: list[Dataset],
) -> list[QualityReport]:
    """Run quality checks across multiple datasets."""

    return [
        check_quality(dataset)
        for dataset in datasets
    ]