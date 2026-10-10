from pathlib import Path


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

MAX_FILE_SIZE_MB = 100
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def validate_file_path(file_path: str | Path) -> tuple[bool, str]:
    """
    Validate whether a file is suitable for ingestion.

    Returns:
        (True, message) when valid.
        (False, reason) when invalid.
    """

    path = Path(file_path)

    # Check whether the path exists
    if not path.exists():
        return False, f"File does not exist: {path}"

    # Make sure it is actually a file
    if not path.is_file():
        return False, f"Path is not a file: {path}"

    # Check extension
    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        return False, (
            f"Unsupported file type '{extension}'. "
            f"Supported types: {supported}"
        )

    # Check file size
    file_size = path.stat().st_size

    if file_size == 0:
        return False, "File is empty."

    if file_size > MAX_FILE_SIZE_BYTES:
        return False, (
            f"File exceeds the maximum allowed size "
            f"of {MAX_FILE_SIZE_MB} MB."
        )

    return True, "File passed validation."