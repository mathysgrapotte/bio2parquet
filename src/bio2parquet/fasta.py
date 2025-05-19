"""FASTA file processing and conversion to Parquet datasets."""

import gzip
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable, Optional

from datasets import Dataset, Features, Value

from bio2parquet.errors import FileProcessingError, InvalidFormatError


def _handle_fasta_parsing_error(*, condition: bool, message: str, filepath_str: str) -> None:
    """Helper function to raise InvalidFormatError if a condition is met."""
    if condition:
        raise InvalidFormatError(message, filepath_str)


def open_gz_file_text(fp: Path, m: str) -> Any:
    """Open a GZIP file with text mode."""
    return gzip.open(fp, m, encoding="utf-8")


def read_fasta_file(filepath: Path) -> Iterator[dict[str, str]]:
    """Reads a FASTA file and yields records as dictionaries.

    Handles both .fasta and .fasta.gz files.

    Args:
        filepath: Path to the FASTA file.

    Yields:
        A dictionary with 'header' and 'sequence' keys for each record.

    Raises:
        FileProcessingError: If the file cannot be read.
        InvalidFormatError: If the file is not in valid FASTA format.
    """
    if not filepath.exists():
        raise FileProcessingError(f"Input file does not exist: {filepath}", str(filepath))
    if not filepath.is_file():
        raise FileProcessingError(f"Input path is not a file: {filepath}", str(filepath))

    file_opener: Callable
    if filepath.name.endswith(".gz"):
        file_opener = open_gz_file_text
        mode = "rt"
    else:
        file_opener = open
        mode = "r"

    try:
        with file_opener(filepath, mode) as f:
            header: Optional[str] = None
            sequence_parts: list[str] = []
            for current_line in f:
                line = current_line.strip()
                if not line:
                    continue  # Skip empty lines
                if line.startswith(">"):
                    if header is not None:
                        # Yield previous record
                        if not sequence_parts:
                            _handle_fasta_parsing_error(
                                condition=True,
                                message="Sequence missing for header.",
                                filepath_str=str(filepath),
                            )
                        yield {"header": header, "sequence": "".join(sequence_parts)}
                    header = line[1:]  # Remove '>'
                    sequence_parts = []
                elif header is not None:  # sequence line
                    sequence_parts.append(line)
                else:
                    # This means a sequence line appeared before any header
                    _handle_fasta_parsing_error(
                        condition=True,
                        message="Sequence data found before a header line.",
                        filepath_str=str(filepath),
                    )

            # Yield the last record in the file
            if header is not None and sequence_parts:
                yield {"header": header, "sequence": "".join(sequence_parts)}
            elif header is not None and not sequence_parts:
                _handle_fasta_parsing_error(
                    condition=True,
                    message="Sequence missing for the last header.",
                    filepath_str=str(filepath),
                )
            elif header is None and not sequence_parts and filepath.stat().st_size > 0:
                # File is not empty but no headers or sequences were found
                _handle_fasta_parsing_error(
                    condition=True,
                    message="File does not appear to be in FASTA format.",
                    filepath_str=str(filepath),
                )

    except FileNotFoundError:
        raise FileProcessingError(f"File not found: {filepath}", str(filepath)) from None
    except gzip.BadGzipFile:
        raise FileProcessingError(f"File is not a valid GZIP file: {filepath}", str(filepath)) from None
    except (InvalidFormatError, FileProcessingError):  # Re-raise specific errors directly
        raise
    except Exception as e:
        raise FileProcessingError(f"Error reading file {filepath}: {e}", str(filepath)) from e


def create_dataset_from_fasta(filepath: Path) -> Dataset:
    """Creates a Hugging Face Dataset from a FASTA file.

    Args:
        filepath: Path to the FASTA file.

    Returns:
        A Hugging Face Dataset with 'header' and 'sequence' columns.

    Raises:
        AssertionError: If the input filepath does not exist or is not a file.
    """
    if not filepath.exists():
        raise FileProcessingError(f"Input file does not exist: {filepath}", str(filepath))
    if not filepath.is_file():
        raise FileProcessingError(f"Input path is not a file: {filepath}", str(filepath))

    features = Features(
        {
            "header": Value("string"),
            "sequence": Value("string"),
        },
    )

    # Check if the generator will be empty to avoid issues with Dataset.from_generator
    # This is a bit inefficient as it might iterate twice in the worst case (once for check, once for generation)
    # but ensures robustness for empty or malformed-but-not-raising files.
    # A more optimized way might involve peeking the generator or modifying read_fasta_file.

    # Convert iterator to list to check if it's empty and to reuse it.
    records = list(read_fasta_file(filepath))

    if not records:
        # Return an empty dataset with the correct schema
        return Dataset.from_dict({"header": [], "sequence": []}, features=features)

    def gen() -> Iterator[dict[str, str]]:
        yield from records

    return Dataset.from_generator(gen, features=features)
