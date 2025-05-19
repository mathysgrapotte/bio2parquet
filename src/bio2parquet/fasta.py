"""FASTA file processing and conversion to Parquet datasets."""

import gzip
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional, TextIO

from Bio import SeqIO
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


def _validate_fasta_content(handle: TextIO, filepath: Path) -> None:
    """Validates the initial content of a FASTA file.

    Args:
        handle: File handle to validate
        filepath: Path to the file being validated

    Raises:
        InvalidFormatError: If file is empty or doesn't start with a header
    """
    first_line = handle.readline()
    if first_line == "":
        _raise_invalid_format_error("File is empty: no FASTA records found.", str(filepath))
    if not first_line.strip().startswith(">"):
        _raise_invalid_format_error("Sequence data found before a header line", str(filepath))
    handle.seek(0)


def _validate_record(record: Any, filepath: Path) -> None:
    """Validates a single FASTA record.

    Args:
        record: The FASTA record to validate
        filepath: Path to the file being processed

    Raises:
        InvalidFormatError: If record is invalid
    """
    if not record.seq:
        _raise_invalid_format_error("Sequence missing for header.", str(filepath))
    if not record.id:
        _raise_invalid_format_error("Header missing for sequence.", str(filepath))


def _raise_invalid_format_error(message: str, filepath_str: str) -> None:
    """Raises an InvalidFormatError with the given message.

    Args:
        message: The error message
        filepath_str: The filepath as a string

    Raises:
        InvalidFormatError: Always raises this exception
    """
    raise InvalidFormatError(message, filepath_str)


def read_fasta_file(filepath: Path) -> Iterator[dict[str, str]]:
    """Reads a FASTA file and yields records as dictionaries.

    Handles both .fasta and .fasta.gz files using Bio.SeqIO for efficient parsing.
    Validates file format and content before processing.

    Args:
        filepath: Path to the FASTA file.

    Yields:
        A dictionary with 'header' and 'sequence' keys for each record.

    Raises:
        FileProcessingError: If the file cannot be read.
        InvalidFormatError: If the file is not in valid FASTA format.
    """
    _validate_file_exists(filepath)

    try:
        # Handle gzip files explicitly
        if filepath.name.endswith(".gz"):
            with gzip.open(filepath, "rt", encoding="utf-8") as handle:
                _validate_fasta_content(handle, filepath)
                for record in SeqIO.parse(handle, "fasta"):
                    _validate_record(record, filepath)
                    yield {"header": record.id, "sequence": str(record.seq)}
        else:
            with open(filepath, encoding="utf-8") as handle:
                _validate_fasta_content(handle, filepath)
                for record in SeqIO.parse(handle, "fasta"):
                    _validate_record(record, filepath)
                    yield {"header": record.id, "sequence": str(record.seq)}
    except FileNotFoundError:
        raise FileProcessingError(f"File not found: {filepath}", str(filepath)) from None
    except gzip.BadGzipFile:
        raise FileProcessingError(f"File is not a valid GZIP file: {filepath}", str(filepath)) from None
    except InvalidFormatError:
        raise
    except Exception as e:
        raise FileProcessingError(f"Error reading file {filepath}: {e}", str(filepath)) from e


def _process_chunk(chunk: list[dict[str, str]]) -> list[dict[str, str]]:
    """Process a chunk of FASTA records in parallel.

    Args:
        chunk: List of FASTA records to process.

    Returns:
        Processed records.
    """
    return chunk


def create_dataset_from_fasta(filepath: Path, chunk_size: int = 1000, max_workers: Optional[int] = None) -> Dataset:
    """Creates a Hugging Face Dataset from a FASTA file using parallel processing.

    Args:
        filepath: Path to the FASTA file.
        chunk_size: Number of records to process in each chunk.
        max_workers: Maximum number of worker processes. If None, uses CPU count.

    Returns:
        A Hugging Face Dataset with 'header' and 'sequence' columns.

    Raises:
        FileProcessingError: If the input filepath does not exist or is not a file.
    """
    _validate_file_exists(filepath)

    features = Features(
        {
            "header": Value("string"),
            "sequence": Value("string"),
        },
    )

    # Read all records into memory (Bio.SeqIO is efficient for this)
    records = list(read_fasta_file(filepath))

    if not records:
        return Dataset.from_dict({"header": [], "sequence": []}, features=features)

    # Process records in parallel chunks
    chunks = [records[i : i + chunk_size] for i in range(0, len(records), chunk_size)]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        processed_chunks = list(executor.map(_process_chunk, chunks))

    # Flatten the processed chunks
    all_records = [record for chunk in processed_chunks for record in chunk]

    def gen() -> Iterator[dict[str, str]]:
        yield from all_records

    return Dataset.from_generator(gen, features=features)
