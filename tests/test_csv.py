"""Tests for CSV file processing and conversion."""

import csv
import gzip
import logging
from pathlib import Path

import pytest
from _pytest.logging import LogCaptureFixture
from datasets import Dataset

from bio2parquet.csv import create_dataset_from_csv
from bio2parquet.errors import FileProcessingError, InvalidFormatError

# Suppress ResourceWarning for unclosed files caused by upstream pandas/datasets bug.
pytestmark = pytest.mark.filterwarnings(
    "ignore:unclosed file <_io.BufferedReader.*:ResourceWarning",
)

logger = logging.getLogger(__name__)


@pytest.fixture
def sample_csv_path(tmp_path: Path) -> Path:
    """Create a sample CSV file for testing.

    Args:
        tmp_path: Pytest fixture providing a temporary directory

    Returns:
        Path to the created CSV file
    """
    csv_path = tmp_path / "test.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["header", "sequence", "description"])
        writer.writerow(["seq1", "ATCGATCGATCGATCGATCGATCGATCGATCG", "First test sequence"])
        writer.writerow(["seq2", "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA", "Second test sequence"])
    return csv_path


@pytest.fixture
def sample_gzipped_csv_path(tmp_path: Path) -> Path:
    """Create a sample gzipped CSV file for testing.

    Args:
        tmp_path: Pytest fixture providing a temporary directory

    Returns:
        Path to the created gzipped CSV file
    """
    csv_path = tmp_path / "test.csv.gz"
    with gzip.open(csv_path, "wt", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["header", "sequence", "description"])
        writer.writerow(["seq1", "ATCGATCGATCGATCGATCGATCGATCGATCG", "First test sequence"])
        writer.writerow(["seq2", "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA", "Second test sequence"])
    return csv_path


def test_create_dataset_from_csv_valid_file(sample_csv_path: Path) -> None:
    """Test creating a dataset from a valid CSV file."""
    dataset = create_dataset_from_csv(sample_csv_path)

    assert isinstance(dataset, Dataset)
    assert len(dataset) == 2
    assert set(dataset.features.keys()) == {"header", "sequence", "description"}

    # Check first record
    assert dataset[0]["header"] == "seq1"
    assert dataset[0]["sequence"] == "ATCGATCGATCGATCGATCGATCGATCGATCG"
    assert dataset[0]["description"] == "First test sequence"


def test_create_dataset_from_csv_missing_file() -> None:
    """Test creating a dataset from a non-existent file."""
    with pytest.raises(FileProcessingError):
        create_dataset_from_csv(Path("nonexistent.csv"))


def test_create_dataset_from_csv_invalid_format(tmp_path: Path) -> None:
    """Test creating a dataset from an invalid CSV file."""
    csv_path = tmp_path / "invalid.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["invalid", "columns"])

    with pytest.raises(InvalidFormatError):
        create_dataset_from_csv(csv_path)


def test_create_dataset_from_csv_empty_file(tmp_path: Path) -> None:
    """Test creating a dataset from an empty CSV file."""
    csv_path = tmp_path / "empty.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["header", "sequence", "description"])

    with pytest.raises(InvalidFormatError):
        create_dataset_from_csv(csv_path)


def test_create_dataset_from_csv_missing_required_columns(tmp_path: Path) -> None:
    """Test creating a dataset from a CSV file missing required columns."""
    csv_path = tmp_path / "missing_columns.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["header", "description"])  # Missing sequence column

    with pytest.raises(InvalidFormatError):
        create_dataset_from_csv(csv_path)


def test_create_dataset_from_csv_only_required_columns(tmp_path: Path) -> None:
    """Test creating a dataset from a CSV file with only required columns."""
    csv_path = tmp_path / "only_required.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["header", "sequence"])
        writer.writerow(["seq1", "ATCGATCGATCGATCGATCGATCGATCGATCG"])
    dataset = create_dataset_from_csv(csv_path)
    assert set(dataset.features.keys()) == {"header", "sequence"}
    assert dataset[0]["header"] == "seq1"
    assert dataset[0]["sequence"] == "ATCGATCGATCGATCGATCGATCGATCGATCG"


def test_create_dataset_from_gzipped_csv(caplog: LogCaptureFixture) -> None:
    """Test creating a dataset from a valid gzipped CSV file."""
    caplog.set_level(logging.INFO)

    gzipped_csv_path = Path("tests/data/csv/sample.csv.gz")
    dataset = create_dataset_from_csv(gzipped_csv_path)

    # Log dataset contents
    logger.info("Dataset contents:")
    for i in range(len(dataset)):
        logger.info(f"Row {i}: {dataset[i]}")

    assert isinstance(dataset, Dataset)
    assert len(dataset) == 4
    assert set(dataset.features.keys()) == {"header", "sequence", "description"}

    # Check first record
    assert dataset[0]["header"] == "seq1"
    assert dataset[0]["sequence"] == "ATCGATCGATCGATCGATCGATCGATCGATCG"
    assert dataset[0]["description"] == "First test sequence"

    # Check second record
    assert dataset[1]["header"] == "seq2"
    assert dataset[1]["sequence"] == "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA"
    assert dataset[1]["description"] == "Second test sequence"

    # Check third record
    assert dataset[2]["header"] == "seq3"
    assert dataset[2]["sequence"] == "TATATATATATATATATATATATATATATATA"
    assert dataset[2]["description"] == "Third test sequence"

    # Check fourth record
    assert dataset[3]["header"] == "seq4"
    assert dataset[3]["sequence"] == "CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG"
    assert dataset[3]["description"].strip() == "Fourth test sequence"
