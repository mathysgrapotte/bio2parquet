"""CSV file processing and conversion to Parquet datasets."""

import csv
from pathlib import Path

from datasets import Dataset, load_dataset

from bio2parquet.errors import FileProcessingError, InvalidFormatError


def _validate_file_exists(filepath: Path) -> None:
    """Validates that the file exists and is a file.

    Args:
        filepath: Path to validate

    Raises:
        FileProcessingError: If file doesn't exist or isn't a file
    """
    if not filepath.exists():
        raise FileProcessingError(f"Input file does not exist: {filepath}", str(filepath))
    if not filepath.is_file():
        raise FileProcessingError(f"Input path is not a file: {filepath}", str(filepath))


def _validate_csv_content(filepath: Path) -> None:
    """Validates that the CSV file has the required format and content.

    Args:
        filepath: Path to the CSV file

    Raises:
        InvalidFormatError: If the file is empty or missing required columns
    """
    try:
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                _header = next(reader)
            except StopIteration as err:
                raise InvalidFormatError("File is empty: no CSV records found.", str(filepath)) from err

            # Check if there are any data rows
            try:
                _first_row = next(reader)
            except StopIteration as err:
                raise InvalidFormatError("File contains no data rows", str(filepath)) from err

    except InvalidFormatError:
        raise
    except Exception as e:
        raise FileProcessingError(f"Error reading file {filepath}: {e}", str(filepath)) from e


def create_dataset_from_csv(filepath: Path) -> Dataset:
    """Creates a Hugging Face Dataset from a CSV file with dynamic columns.

    Args:
        filepath: Path to the CSV file.

    Returns:
        A Hugging Face Dataset with columns matching the CSV file.

    Raises:
        FileProcessingError: If the input filepath does not exist or is not a file.
        InvalidFormatError: If the file is not in valid CSV format or contains no data rows.
    """
    _validate_file_exists(filepath)
    _validate_csv_content(filepath)

    try:
        # Load the dataset using Hugging Face's load_dataset
        result = load_dataset("csv", data_files=str(filepath))["train"]
    except Exception as e:
        raise FileProcessingError(f"Error reading file {filepath}: {e}", str(filepath)) from e
    else:
        return result
