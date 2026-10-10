
from io import BytesIO

import pandas as pd
import pytest

from src.cleaning.cleaner import clean_dataset
from src.ingestion.models import Dataset
from src.profiling.profiler import profile_dataset
from src.quality.checker import check_quality
from src.workspace_ui import (
    cleaning_actions_to_dataframe,
    load_uploaded_files,
    profiles_to_dataframe,
    quality_findings_to_dataframe,
)


def make_csv_upload(name="retail.csv"):
    content = (
        b"date,sales,region\n"
        b"2025-01-01,100,North\n"
        b"2025-01-02,200,South\n"
        b"2025-01-02,200,South\n"
    )
    uploaded = BytesIO(content)
    uploaded.name = name
    return uploaded


def make_excel_upload(name="retail.xlsx"):
    buffer = BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(
            {"sales": [100, 200]}
        ).to_excel(writer, sheet_name="Sales", index=False)

        pd.DataFrame(
            {"orders": [2, 3]}
        ).to_excel(writer, sheet_name="Orders", index=False)

    buffer.seek(0)
    buffer.name = name
    return buffer


def make_dataset():
    data = pd.DataFrame(
        {
            "sales": [100.0, 200.0, 200.0, None],
            "region": ["North", "South", "South", "North"],
        }
    )

    return Dataset(
        name="retail",
        source_file="retail.csv",
        data=data,
        source_type="csv",
    )


def test_load_uploaded_csv():
    datasets = load_uploaded_files([make_csv_upload()])

    assert len(datasets) == 1
    assert datasets[0].source_type == "csv"
    assert len(datasets[0].data) == 3
    assert "sales" in datasets[0].data.columns


def test_load_uploaded_excel_sheets():
    datasets = load_uploaded_files([make_excel_upload()])

    assert len(datasets) == 2
    assert datasets[0].sheet_name == "Sales"
    assert datasets[1].sheet_name == "Orders"


def test_load_multiple_uploaded_files():
    datasets = load_uploaded_files(
        [make_csv_upload(), make_excel_upload()]
    )

    assert len(datasets) == 3


def test_reject_empty_upload_list():
    with pytest.raises(ValueError, match="at least one"):
        load_uploaded_files([])


def test_reject_unsupported_file():
    uploaded = BytesIO(b"invalid")
    uploaded.name = "data.json"

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_uploaded_files([uploaded])


def test_profiles_to_dataframe():
    profile = profile_dataset(make_dataset())
    result = profiles_to_dataframe(profile)

    assert "Column" in result.columns
    assert "Missing" in result.columns
    assert "sales" in result["Column"].tolist()


def test_quality_findings_to_dataframe():
    report = check_quality(make_dataset())
    result = quality_findings_to_dataframe(report)

    assert "Severity" in result.columns
    assert "Message" in result.columns
    assert len(result) == report.total_findings


def test_cleaning_actions_to_dataframe():
    dataset = make_dataset()

    result = clean_dataset(
        dataset.data,
        dataset_name=dataset.name,
        apply_missing_values=True,
        remove_duplicates=True,
    )

    audit_table = cleaning_actions_to_dataframe(result)

    assert "Action" in audit_table.columns
    assert "Description" in audit_table.columns
    assert len(audit_table) == result.total_actions


def test_empty_quality_report_has_expected_columns():
    from src.quality.models import QualityReport

    report = QualityReport(dataset_name="empty")
    result = quality_findings_to_dataframe(report)

    assert result.empty
    assert "Severity" in result.columns
