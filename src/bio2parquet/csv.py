"""CSV file processing and conversion to Parquet datasets."""

import gzip
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional, TextIO

import pandas as pd
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


def _validate_csv_content(handle: TextIO, filepath: Path) -> None:
    """Validates the initial content of a CSV file.

    Args:
        handle: File handle to validate
        filepath: Path to the file being validated

    Raises:
        InvalidFormatError: If file is empty or doesn't have a header
    """
    first_line = handle.readline()
    if first_line == "":
        _raise_invalid_format_error("File is empty: no CSV records found.", str(filepath))
    if "," not in first_line and ";" not in first_line and "\t" not in first_line:
        _raise_invalid_format_error("File does not appear to be in CSV format (no valid delimiter found)", str(filepath))
    handle.seek(0)


def _raise_invalid_format_error(message: str, filepath_str: str) -> None:
    """Raises an InvalidFormatError with the given message.

    Args:
        message: The error message
        filepath_str: The filepath as a string

    Raises:
        InvalidFormatError: Always raises this exception
    """
    raise InvalidFormatError(message, filepath_str)


def read_csv_file(filepath: Path, **kwargs: Any) -> pd.DataFrame:
    """Reads a CSV file into a pandas DataFrame.

    Handles both .csv and .csv.gz files using pandas for efficient parsing.
    Validates file format and content before processing.

    Args:
        filepath: Path to the CSV file.
        **kwargs: Additional arguments to pass to pd.read_csv()

    Returns:
        A pandas DataFrame containing the CSV data.

    Raises:
        FileProcessingError: If the file cannot be read.
        InvalidFormatError: If the file is not in valid CSV format.
    """
    _validate_file_exists(filepath)

    try:
        # Handle gzip files explicitly
        if filepath.name.endswith(".gz"):
            with gzip.open(filepath, "rt", encoding="utf-8") as handle:
                _validate_csv_content(handle, filepath)
                return pd.read_csv(handle, **kwargs)
        else:
            with open(filepath, encoding="utf-8") as handle:
                _validate_csv_content(handle, filepath)
                return pd.read_csv(handle, **kwargs)
    except FileNotFoundError:
        raise FileProcessingError(f"File not found: {filepath}", str(filepath)) from None
    except gzip.BadGzipFile:
        raise FileProcessingError(f"File is not a valid GZIP file: {filepath}", str(filepath)) from None
    except InvalidFormatError:
        raise
    except Exception as e:
        raise FileProcessingError(f"Error reading file {filepath}: {e}", str(filepath)) from e


def _process_chunk(chunk: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Process a chunk of CSV records in parallel.

    Args:
        chunk: List of CSV records to process.

    Returns:
        Processed records.
    """
    return chunk


def create_dataset_from_csv(
    filepath: Path,
    chunk_size: int = 1000,
    max_workers: Optional[int] = None,
    **kwargs: Any,
) -> Dataset:
    """Creates a Hugging Face Dataset from a CSV file using parallel processing.

    Args:
        filepath: Path to the CSV file.
        chunk_size: Number of records to process in each chunk.
        max_workers: Maximum number of worker processes. If None, uses CPU count.
        **kwargs: Additional arguments to pass to pd.read_csv()

    Returns:
        A Hugging Face Dataset with columns matching the CSV headers.

    Raises:
        FileProcessingError: If the input filepath does not exist or is not a file.
    """
    _validate_file_exists(filepath)

    # Read the CSV file into a pandas DataFrame
    df = read_csv_file(filepath, **kwargs)

    if df.empty:
        return Dataset.from_dict({col: [] for col in df.columns})

    # Create features based on DataFrame columns
    features = Features({col: Value("string") for col in df.columns})

    # Convert DataFrame to list of dictionaries
    records = df.to_dict("records")

    # Process records in parallel chunks
    chunks = [records[i : i + chunk_size] for i in range(0, len(records), chunk_size)]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        processed_chunks = list(executor.map(_process_chunk, chunks))

    # Flatten the processed chunks
    all_records = [record for chunk in processed_chunks for record in chunk]

    def gen() -> Iterator[dict[str, Any]]:
        yield from all_records

    return Dataset.from_generator(gen, features=features)


def csv_to_parquet(
    input_filepath: Path,
    output_filepath: Path,
    chunk_size: int = 1000,
    max_workers: Optional[int] = None,
    **kwargs: Any,
) -> None:
    """Converts a CSV file to Parquet format using HuggingFace's datasets.

    This function reads a CSV file, converts it to a HuggingFace Dataset,
    and saves it as a Parquet file. It handles both regular CSV files and
    gzipped CSV files.

    Args:
        input_filepath: Path to the input CSV file
        output_filepath: Path where the Parquet file will be saved
        chunk_size: Number of records to process in each chunk
        max_workers: Maximum number of worker processes. If None, uses CPU count
        **kwargs: Additional arguments to pass to pd.read_csv()

    Raises:
        FileProcessingError: If there are issues reading the input file or writing the output file
        InvalidFormatError: If the input file is not in valid CSV format
    """
    # Create dataset from CSV
    dataset = create_dataset_from_csv(
        input_filepath,
        chunk_size=chunk_size,
        max_workers=max_workers,
        **kwargs
    )

    try:
        # Create output directory if it doesn't exist
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save dataset as Parquet
        dataset.save_to_disk(str(output_filepath))
    except Exception as e:
        raise FileProcessingError(
            f"Error writing Parquet file {output_filepath}: {e}",
            str(output_filepath)
        ) from e 