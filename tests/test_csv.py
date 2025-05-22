"""Tests for CSV file processing and conversion."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from datasets import load_from_disk

from bio2parquet.csv import csv_to_parquet
from bio2parquet.errors import FileProcessingError, InvalidFormatError


def test_csv_to_parquet_basic():
    """Test basic CSV to Parquet conversion."""
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as csv_file:
        csv_file.write("col1,col2\n1,a\n2,b\n3,c")
        csv_path = Path(csv_file.name)

    # Create a temporary directory for the Parquet output
    with tempfile.TemporaryDirectory() as temp_dir:
        parquet_path = Path(temp_dir) / "output.parquet"

        # Convert CSV to Parquet
        csv_to_parquet(csv_path, parquet_path)

        # Load the Parquet file and verify contents
        dataset = load_from_disk(str(parquet_path))
        assert len(dataset) == 3
        assert dataset.column_names == ["col1", "col2"]
        assert dataset[0]["col1"] == "1"
        assert dataset[0]["col2"] == "a"
        assert dataset[1]["col1"] == "2"
        assert dataset[1]["col2"] == "b"
        assert dataset[2]["col1"] == "3"
        assert dataset[2]["col2"] == "c"

    # Clean up the temporary CSV file
    csv_path.unlink()


def test_csv_to_parquet_empty_file():
    """Test CSV to Parquet conversion with an empty file."""
    # Create an empty CSV file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as csv_file:
        csv_file.write("col1,col2\n")
        csv_path = Path(csv_file.name)

    # Create a temporary directory for the Parquet output
    with tempfile.TemporaryDirectory() as temp_dir:
        parquet_path = Path(temp_dir) / "output.parquet"

        # Convert CSV to Parquet
        csv_to_parquet(csv_path, parquet_path)

        # Load the Parquet file and verify contents
        dataset = load_from_disk(str(parquet_path))
        assert len(dataset) == 0
        assert dataset.column_names == ["col1", "col2"]

    # Clean up the temporary CSV file
    csv_path.unlink()


def test_csv_to_parquet_nonexistent_file():
    """Test CSV to Parquet conversion with a nonexistent file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        parquet_path = Path(temp_dir) / "output.parquet"
        nonexistent_path = Path(temp_dir) / "nonexistent.csv"

        with pytest.raises(FileProcessingError):
            csv_to_parquet(nonexistent_path, parquet_path)


def test_csv_to_parquet_invalid_format():
    """Test CSV to Parquet conversion with an invalid CSV format."""
    # Create an invalid CSV file (no commas)
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as csv_file:
        csv_file.write("col1 col2\n1 a\n2 b\n3 c")
        csv_path = Path(csv_file.name)

    # Create a temporary directory for the Parquet output
    with tempfile.TemporaryDirectory() as temp_dir:
        parquet_path = Path(temp_dir) / "output.parquet"

        with pytest.raises(InvalidFormatError):
            csv_to_parquet(csv_path, parquet_path)

    # Clean up the temporary CSV file
    csv_path.unlink()


def test_csv_to_parquet_with_pandas_options():
    """Test CSV to Parquet conversion with pandas options."""
    # Create a CSV file with custom delimiter
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as csv_file:
        csv_file.write("col1;col2\n1;a\n2;b\n3;c")
        csv_path = Path(csv_file.name)

    # Create a temporary directory for the Parquet output
    with tempfile.TemporaryDirectory() as temp_dir:
        parquet_path = Path(temp_dir) / "output.parquet"

        # Convert CSV to Parquet with custom delimiter
        csv_to_parquet(csv_path, parquet_path, sep=";")

        # Load the Parquet file and verify contents
        dataset = load_from_disk(str(parquet_path))
        assert len(dataset) == 3
        assert dataset.column_names == ["col1", "col2"]
        assert dataset[0]["col1"] == "1"
        assert dataset[0]["col2"] == "a"

    # Clean up the temporary CSV file
    csv_path.unlink() 