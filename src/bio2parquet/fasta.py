"""FASTA file processing and conversion to Parquet datasets."""

import gzip
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional, List

from Bio import SeqIO
from datasets import Dataset, Features, Value

from bio2parquet.errors import FileProcessingError, InvalidFormatError


def _handle_fasta_parsing_error(*, condition: bool, message: str, filepath_str: str) -> None:
    """Helper function to raise InvalidFormatError if a condition is met."""
    if condition:
        raise InvalidFormatError(message, filepath_str)


def read_fasta_file(filepath: Path) -> Iterator[dict[str, str]]:
    """Reads a FASTA file and yields records as dictionaries.

    Handles both .fasta and .fasta.gz files using Bio.SeqIO for efficient parsing.

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

    try:
        # Handle gzip files explicitly
        if filepath.name.endswith(".gz"):
            with gzip.open(filepath, "rt", encoding="utf-8") as handle:
                first_line = handle.readline()
                if first_line == "":
                    raise InvalidFormatError("Sequence data found before a header line", str(filepath))
                if not first_line.strip().startswith(">"):
                    raise InvalidFormatError("Sequence data found before a header line", str(filepath))
                handle.seek(0)
                for record in SeqIO.parse(handle, "fasta"):
                    if not record.seq:
                        raise InvalidFormatError("Sequence missing for header.", str(filepath))
                    yield {"header": record.id, "sequence": str(record.seq)}
        else:
            with open(filepath, "r", encoding="utf-8") as handle:
                first_line = handle.readline()
                if first_line == "":
                    raise InvalidFormatError("Sequence data found before a header line", str(filepath))
                if not first_line.strip().startswith(">"):
                    raise InvalidFormatError("Sequence data found before a header line", str(filepath))
                handle.seek(0)
                for record in SeqIO.parse(handle, "fasta"):
                    if not record.seq:
                        raise InvalidFormatError("Sequence missing for header.", str(filepath))
                    yield {"header": record.id, "sequence": str(record.seq)}
    except FileNotFoundError:
        raise FileProcessingError(f"File not found: {filepath}", str(filepath)) from None
    except gzip.BadGzipFile:
        raise FileProcessingError(f"File is not a valid GZIP file: {filepath}", str(filepath)) from None
    except InvalidFormatError:
        raise
    except Exception as e:
        raise FileProcessingError(f"Error reading file {filepath}: {e}", str(filepath)) from e


def _process_chunk(chunk: List[dict[str, str]]) -> List[dict[str, str]]:
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

    # Read all records into memory (Bio.SeqIO is efficient for this)
    records = list(read_fasta_file(filepath))

    if not records:
        return Dataset.from_dict({"header": [], "sequence": []}, features=features)

    # Process records in parallel chunks
    chunks = [records[i:i + chunk_size] for i in range(0, len(records), chunk_size)]
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        processed_chunks = list(executor.map(_process_chunk, chunks))

    # Flatten the processed chunks
    all_records = [record for chunk in processed_chunks for record in chunk]

    def gen() -> Iterator[dict[str, str]]:
        yield from all_records

    return Dataset.from_generator(gen, features=features)
