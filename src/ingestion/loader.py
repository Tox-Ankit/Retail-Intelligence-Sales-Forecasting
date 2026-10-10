from pathlib import Path

import pandas as pd

from src.ingestion.models import Dataset, DatasetCollection
from src.ingestion.validators import validate_file_path


def load_csv(file_path: str | Path) -> Dataset:
    """
    Load a CSV file into a Dataset object.
    """

    path = Path(file_path)

    data = pd.read_csv(path)

    return Dataset(
        name=path.stem,
        source_file=path.name,
        data=data,
        source_type="csv",
    )



def load_excel(file_path: str | Path) -> list[Dataset]:
    """
    Load every sheet from an Excel workbook.

    Each sheet becomes an independent Dataset.
    The workbook is closed automatically after loading.
    """

    path = Path(file_path)
    datasets = []

    # The context manager guarantees the workbook is closed,
    # including when an error occurs while reading a worksheet.
    with pd.ExcelFile(path) as workbook:
        for sheet_name in workbook.sheet_names:
            data = pd.read_excel(workbook, sheet_name=sheet_name)

            dataset_name = f"{path.stem}_{sheet_name}"

            datasets.append(
                Dataset(
                    name=dataset_name,
                    source_file=path.name,
                    data=data,
                    source_type="excel",
                    sheet_name=sheet_name,
                )
            )

    return datasets



def load_file(file_path: str | Path) -> list[Dataset]:
    """
    Validate and load a CSV or Excel file.

    Returns a list because one Excel file can contain
    multiple sheets.
    """

    is_valid, message = validate_file_path(file_path)

    if not is_valid:
        raise ValueError(message)

    path = Path(file_path)
    extension = path.suffix.lower()

    if extension == ".csv":
        return [load_csv(path)]

    if extension in {".xlsx", ".xls"}:
        return load_excel(path)

    raise ValueError(f"Unsupported file type: {extension}")


def load_files(file_paths: list[str | Path]) -> DatasetCollection:
    """
    Load multiple CSV and Excel files into one DatasetCollection.
    """

    collection = DatasetCollection()

    for file_path in file_paths:
        datasets = load_file(file_path)

        for dataset in datasets:
            collection.add(dataset)

    return collection