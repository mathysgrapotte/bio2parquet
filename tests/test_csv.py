"""Tests for CSV to Parquet conversion."""

import gzip
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from datasets import Dataset

from bio2parquet.csv import create_dataset_from_csv, read_csv_file
from bio2parquet.errors import FileProcessingError, InvalidFormatError


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Create a sample CSV file for testing.

    Args:
        tmp_path: Temporary directory provided by pytest

    Returns:
        Path to the sample CSV file
    """
    csv_path = tmp_path / "sample.csv"
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "value": [1.5, 2.5, 3.5],
        }
    )
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_gzipped_csv(tmp_path: Path) -> Path:
    """Create a sample gzipped CSV file for testing.

    Args:
        tmp_path: Temporary directory provided by pytest

    Returns:
        Path to the sample gzipped CSV file
    """
    csv_path = tmp_path / "sample.csv.gz"
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "value": [1.5, 2.5, 3.5],
        }
    )
    with gzip.open(csv_path, "wt", encoding="utf-8") as f:
        df.to_csv(f, index=False)
    return csv_path


def test_read_csv_file(sample_csv: Path) -> None:
    """Test reading a CSV file.

    Args:
        sample_csv: Path to a sample CSV file
    """
    df = read_csv_file(sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == ["id", "name", "value"]
    assert df["id"].tolist() == [1, 2, 3]
    assert df["name"].tolist() == ["Alice", "Bob", "Charlie"]
    assert df["value"].tolist() == [1.5, 2.5, 3.5]


def test_read_gzipped_csv(sample_gzipped_csv: Path) -> None:
    """Test reading a gzipped CSV file.

    Args:
        sample_gzipped_csv: Path to a sample gzipped CSV file
    """
    df = read_csv_file(sample_gzipped_csv)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == ["id", "name", "value"]
    assert df["id"].tolist() == [1, 2, 3]
    assert df["name"].tolist() == ["Alice", "Bob", "Charlie"]
    assert df["value"].tolist() == [1.5, 2.5, 3.5]


def test_read_csv_file_with_kwargs(sample_csv: Path) -> None:
    """Test reading a CSV file with additional kwargs.

    Args:
        sample_csv: Path to a sample CSV file
    """
    df = read_csv_file(sample_csv, sep=",", encoding="utf-8")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3


def test_read_nonexistent_file(tmp_path: Path) -> None:
    """Test reading a nonexistent file.

    Args:
        tmp_path: Temporary directory provided by pytest
    """
    with pytest.raises(FileProcessingError):
        read_csv_file(tmp_path / "nonexistent.csv")


def test_read_invalid_csv(tmp_path: Path) -> None:
    """Test reading an invalid CSV file.

    Args:
        tmp_path: Temporary directory provided by pytest
    """
    invalid_csv = tmp_path / "invalid.csv"
    invalid_csv.write_text("This is not a CSV file")
    with pytest.raises(InvalidFormatError):
        read_csv_file(invalid_csv)


def test_create_dataset_from_csv(sample_csv: Path) -> None:
    """Test creating a dataset from a CSV file.

    Args:
        sample_csv: Path to a sample CSV file
    """
    dataset = create_dataset_from_csv(sample_csv)
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 3
    assert dataset.column_names == ["id", "name", "value"]
    assert dataset["id"] == ["1", "2", "3"]  # Note: CSV reader converts to strings
    assert dataset["name"] == ["Alice", "Bob", "Charlie"]
    assert dataset["value"] == ["1.5", "2.5", "3.5"]  # Note: CSV reader converts to strings


def test_create_dataset_from_gzipped_csv(sample_gzipped_csv: Path) -> None:
    """Test creating a dataset from a gzipped CSV file.

    Args:
        sample_gzipped_csv: Path to a sample gzipped CSV file
    """
    dataset = create_dataset_from_csv(sample_gzipped_csv)
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 3
    assert dataset.column_names == ["id", "name", "value"]
    assert dataset["id"] == ["1", "2", "3"]  # Note: CSV reader converts to strings
    assert dataset["name"] == ["Alice", "Bob", "Charlie"]
    assert dataset["value"] == ["1.5", "2.5", "3.5"]  # Note: CSV reader converts to strings


def test_create_dataset_from_empty_csv(tmp_path: Path) -> None:
    """Test creating a dataset from an empty CSV file.

    Args:
        tmp_path: Temporary directory provided by pytest
    """
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("id,name,value\n")
    dataset = create_dataset_from_csv(empty_csv)
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 0
    assert dataset.column_names == ["id", "name", "value"]


def test_create_dataset_with_chunk_size(sample_csv: Path) -> None:
    """Test creating a dataset with a specific chunk size.

    Args:
        sample_csv: Path to a sample CSV file
    """
    dataset = create_dataset_from_csv(sample_csv, chunk_size=2)
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 3
    assert dataset.column_names == ["id", "name", "value"]


def test_create_dataset_with_max_workers(sample_csv: Path) -> None:
    """Test creating a dataset with a specific number of workers.

    Args:
        sample_csv: Path to a sample CSV file
    """
    dataset = create_dataset_from_csv(sample_csv, max_workers=2)
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 3
    assert dataset.column_names == ["id", "name", "value"] 