"""Unit tests for Power BI UI helpers and workflow."""

import pandas as pd
from unittest.mock import MagicMock

from src.ingestion.models import Dataset
from src.reporting.powerbi.ui import _get_dataset_name


def test_get_dataset_name_with_named_dataset():
    ds = Dataset(name="Quarterly_Sales", source_file="sales.csv", data=pd.DataFrame({"a": [1]}), source_type="csv")
    assert _get_dataset_name(ds, 0) == "Quarterly_Sales"


def test_get_dataset_name_fallback():
    mock_obj = MagicMock()
    del mock_obj.name
    assert _get_dataset_name(mock_obj, 2) == "Dataset 3"
