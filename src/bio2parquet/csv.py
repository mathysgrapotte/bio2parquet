"""CSV file processing and conversion to Parquet datasets."""

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset, Features, Value

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


def _validate_csv_header(header: list[str]) -> None:
    """Validates that the CSV header contains required columns.
    
    Args:
        header: List of column names from the CSV header
        
    Raises:
        InvalidFormatError: If required columns are missing
    """
    required_columns = {"header", "sequence"}
    missing_columns = required_columns - set(header)
    if missing_columns:
        raise InvalidFormatError(
            f"Missing required columns: {', '.join(missing_columns)}",
            "CSV file",
        )


def _raise_invalid_format_error(message: str, filepath_str: str) -> None:
    """Raises an InvalidFormatError with the given message.
    
    Args:
        message: The error message
        filepath_str: The filepath as a string
        
    Raises:
        InvalidFormatError: Always raises this exception
    """
    raise InvalidFormatError(message, filepath_str)


def read_csv_file(filepath: Path) -> Iterator[dict[str, str]]:
    """Reads a CSV file and yields records as dictionaries.

    Validates file format and content before processing.

    Args:
        filepath: Path to the CSV file.

    Yields:
        A dictionary with column names as keys for each record.

    Raises:
        FileProcessingError: If the file cannot be read.
        InvalidFormatError: If the file is not in valid CSV format.
    """
    _validate_file_exists(filepath)

    try:
        with open(filepath, newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                _raise_invalid_format_error("File is empty: no CSV records found.", str(filepath))
            
            _validate_csv_header(header)
            
            for row in reader:
                if len(row) != len(header):
                    _raise_invalid_format_error(
                        f"Row has {len(row)} columns, expected {len(header)}",
                        str(filepath),
                    )
                yield dict(zip(header, row))
                
    except FileNotFoundError:
        raise FileProcessingError(f"File not found: {filepath}", str(filepath)) from None
    except InvalidFormatError:
        raise
    except Exception as e:
        raise FileProcessingError(f"Error reading file {filepath}: {e}", str(filepath)) from e


def create_dataset_from_csv(filepath: Path) -> Dataset:
    """Creates a Hugging Face Dataset from a CSV file.

    Args:
        filepath: Path to the CSV file.

    Returns:
        A Hugging Face Dataset with columns matching the CSV file.

    Raises:
        FileProcessingError: If the input filepath does not exist or is not a file.
        InvalidFormatError: If the file is not in valid CSV format.
    """
    _validate_file_exists(filepath)

    # Read all records into memory
    records = list(read_csv_file(filepath))

    if not records:
        # Create empty dataset with correct features
        features = Features(
            {
                "header": Value("string"),
                "sequence": Value("string"),
                "description": Value("string"),
            },
        )
        return Dataset.from_dict({"header": [], "sequence": [], "description": []}, features=features)

    # Create features from the first record
    features = Features(
        {
            key: Value("string")
            for key in records[0].keys()
        },
    )

    def gen() -> Iterator[dict[str, str]]:
        yield from records

    return Dataset.from_generator(gen, features=features) 