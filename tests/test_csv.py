"""Tests for CSV file processing and conversion."""

import csv
from pathlib import Path

import pytest
from datasets import Dataset

from bio2parquet.csv import create_dataset_from_csv
from bio2parquet.errors import FileProcessingError, InvalidFormatError


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


def test_create_dataset_from_csv_valid_file(sample_csv_path: Path) -> None:
    """Test creating a dataset from a valid CSV file."""
    dataset = create_dataset_from_csv(sample_csv_path)
    
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 2
    assert dataset.features["header"].dtype == "string"
    assert dataset.features["sequence"].dtype == "string"
    assert dataset.features["description"].dtype == "string"
    
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
    
    dataset = create_dataset_from_csv(csv_path)
    assert len(dataset) == 0


def test_create_dataset_from_csv_missing_required_columns(tmp_path: Path) -> None:
    """Test creating a dataset from a CSV file missing required columns."""
    csv_path = tmp_path / "missing_columns.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["header", "description"])  # Missing sequence column
    
    with pytest.raises(InvalidFormatError):
        create_dataset_from_csv(csv_path) 