
from io import BytesIO

import pandas as pd
import pytest

from src.forecasting.ui import (
    get_excel_sheet_names,
    load_uploaded_data,
)


def make_csv_upload(name="retail.csv"):
    content = b"date,sales\n2025-01-01,100\n2025-01-02,120\n"
    uploaded = BytesIO(content)
    uploaded.name = name
    return uploaded


def make_excel_upload(name="retail.xlsx"):
    buffer = BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(
            {
                "date": ["2025-01-01", "2025-01-02"],
                "sales": [100, 120],
            }
        ).to_excel(writer, sheet_name="Sales", index=False)

        pd.DataFrame(
            {
                "customer": ["A", "B"],
                "orders": [2, 3],
            }
        ).to_excel(writer, sheet_name="Customers", index=False)

    buffer.seek(0)
    buffer.name = name
    return buffer


def test_load_csv():
    data = load_uploaded_data(make_csv_upload())

    assert list(data.columns) == ["date", "sales"]
    assert len(data) == 2
    assert data["sales"].tolist() == [100, 120]


def test_load_excel_selected_sheet():
    data = load_uploaded_data(make_excel_upload(), "Customers")

    assert list(data.columns) == ["customer", "orders"]
    assert data["customer"].tolist() == ["A", "B"]


def test_excel_sheet_names():
    names = get_excel_sheet_names(make_excel_upload())

    assert names == ["Sales", "Customers"]


def test_excel_requires_sheet_when_multiple_exist():
    with pytest.raises(ValueError, match="multiple sheets"):
        load_uploaded_data(make_excel_upload())


def test_load_rejects_unsupported_extension():
    uploaded = BytesIO(b"some data")
    uploaded.name = "data.json"

    with pytest.raises(ValueError, match="Only CSV and XLSX"):
        load_uploaded_data(uploaded)


def test_load_rejects_missing_upload():
    with pytest.raises(ValueError, match="Upload a CSV"):
        load_uploaded_data(None)


def test_load_rejects_unknown_sheet():
    with pytest.raises(ValueError, match="was not found"):
        load_uploaded_data(make_excel_upload(), "Unknown")
