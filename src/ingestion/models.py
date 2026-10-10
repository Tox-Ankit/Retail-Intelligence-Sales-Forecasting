from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Dataset:
    """
    Represents one loaded dataset.

    A dataset can originate from a CSV file
    or one sheet of an Excel workbook.
    """

    name: str
    source_file: str
    data: pd.DataFrame
    source_type: str
    sheet_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rows(self) -> int:
        """Return the number of rows."""
        return len(self.data)

    @property
    def columns(self) -> int:
        """Return the number of columns."""
        return len(self.data.columns)

    @property
    def column_names(self) -> list[str]:
        """Return column names."""
        return self.data.columns.tolist()


@dataclass
class DatasetCollection:
    """
    Represents all datasets uploaded during one session.
    """

    datasets: list[Dataset] = field(default_factory=list)

    def add(self, dataset: Dataset) -> None:
        """Add a dataset to the collection."""
        self.datasets.append(dataset)

    def __len__(self) -> int:
        """Return the number of datasets."""
        return len(self.datasets)

    def get(self, name: str) -> Dataset | None:
        """Find a dataset by name."""
        for dataset in self.datasets:
            if dataset.name == name:
                return dataset

        return None

    @property
    def names(self) -> list[str]:
        """Return all dataset names."""
        return [dataset.name for dataset in self.datasets]