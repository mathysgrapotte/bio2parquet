from pathlib import Path

import pytest
from datasets import Dataset

from bio2parquet.errors import FileProcessingError, InvalidFormatError
from bio2parquet.fasta import create_dataset_from_fasta, read_fasta_file


@pytest.fixture
def sample_fasta_file(tmp_path: Path) -> Path:
    content = ">Seq1;info1\nATGCATGC\nCGTACGTA\n>Seq2;info2\nGATTACAGATTACA\n>Seq3;info3\nCCCCGGGGTTTTAAAA"
    file_path = tmp_path / "test.fasta"
    file_path.write_text(content)
    assert file_path.exists(), "Test FASTA file was not created."
    return file_path


@pytest.fixture
def sample_fasta_gz_file(tmp_path: Path, sample_fasta_file: Path) -> Path:
    import gzip

    gz_file_path = tmp_path / "test.fasta.gz"
    with open(sample_fasta_file, "rb") as f_in, gzip.open(gz_file_path, "wb") as f_out:
        f_out.write(f_in.read())
    assert gz_file_path.exists(), "Test gzipped FASTA file was not created."
    return gz_file_path


@pytest.fixture
def empty_fasta_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "empty.fasta"
    file_path.touch()
    assert file_path.exists(), "Empty test FASTA file was not created."
    return file_path


@pytest.fixture
def malformed_fasta_file_no_header(tmp_path: Path) -> Path:
    content = "ATGCATGC\nCGTACGTA"
    file_path = tmp_path / "no_header.fasta"
    file_path.write_text(content)
    assert file_path.exists(), "Malformed (no header) FASTA file was not created."
    return file_path


@pytest.fixture
def malformed_fasta_file_no_sequence(tmp_path: Path) -> Path:
    content = ">Seq1\n>Seq2"
    file_path = tmp_path / "no_sequence.fasta"
    file_path.write_text(content)
    assert file_path.exists(), "Malformed (no sequence) FASTA file was not created."
    return file_path


@pytest.mark.parametrize("file_fixture_name", ["sample_fasta_file", "sample_fasta_gz_file"])
def test_read_fasta_file_valid(file_fixture_name: str, request: pytest.FixtureRequest) -> None:
    fasta_file = request.getfixturevalue(file_fixture_name)
    records = list(read_fasta_file(fasta_file))

    assert len(records) == 3, "Incorrect number of records read."
    assert records[0]["header"] == "Seq1;info1", "Header for Seq1 is incorrect."
    assert records[0]["sequence"] == "ATGCATGCCGTACGTA", "Sequence for Seq1 is incorrect."
    assert records[1]["header"] == "Seq2;info2", "Header for Seq2 is incorrect."
    assert records[1]["sequence"] == "GATTACAGATTACA", "Sequence for Seq2 is incorrect."
    assert records[2]["header"] == "Seq3;info3", "Header for Seq3 is incorrect."
    assert records[2]["sequence"] == "CCCCGGGGTTTTAAAA", "Sequence for Seq3 is incorrect."


def test_read_fasta_file_empty(empty_fasta_file: Path) -> None:
    with pytest.raises(InvalidFormatError) as excinfo:
        list(read_fasta_file(empty_fasta_file))
    assert "File is empty: no FASTA records found." in str(excinfo.value), "Incorrect error for empty FASTA file."


def test_read_fasta_file_no_header_error(malformed_fasta_file_no_header: Path) -> None:
    with pytest.raises(InvalidFormatError) as excinfo:
        list(read_fasta_file(malformed_fasta_file_no_header))
    assert "Sequence data found before a header line" in str(excinfo.value), "Incorrect error for FASTA without header."
    assert str(malformed_fasta_file_no_header) in str(excinfo.value)


def test_read_fasta_file_no_sequence_error(malformed_fasta_file_no_sequence: Path) -> None:
    with pytest.raises(InvalidFormatError) as excinfo:
        list(read_fasta_file(malformed_fasta_file_no_sequence))
    assert "Sequence missing for header." in str(excinfo.value) or "Sequence missing for the last header." in str(
        excinfo.value,
    ), "Incorrect error for FASTA with missing sequence."
    assert str(malformed_fasta_file_no_sequence) in str(excinfo.value)


def test_read_fasta_file_non_existent() -> None:
    with pytest.raises(FileProcessingError) as excinfo:
        list(read_fasta_file(Path("non_existent_file.fasta")))
    assert "Input file does not exist" in str(excinfo.value), "Incorrect error for non-existent file."


@pytest.mark.parametrize("file_fixture_name", ["sample_fasta_file", "sample_fasta_gz_file"])
def test_create_dataset_from_fasta_valid(file_fixture_name: str, request: pytest.FixtureRequest) -> None:
    fasta_file = request.getfixturevalue(file_fixture_name)
    dataset = create_dataset_from_fasta(fasta_file)

    assert isinstance(dataset, Dataset), "The result should be a Hugging Face Dataset."
    assert len(dataset) == 3, "Dataset has incorrect number of rows."
    assert list(dataset.column_names) == ["header", "sequence"], "Dataset has incorrect column names."

    assert dataset[0]["header"] == "Seq1;info1", "Dataset header for Seq1 is incorrect."
    assert dataset[0]["sequence"] == "ATGCATGCCGTACGTA", "Dataset sequence for Seq1 is incorrect."
    assert dataset[2]["header"] == "Seq3;info3", "Dataset header for Seq3 is incorrect."
    assert dataset[2]["sequence"] == "CCCCGGGGTTTTAAAA", "Dataset sequence for Seq3 is incorrect."


def test_create_dataset_from_empty_fasta(empty_fasta_file: Path) -> None:
    with pytest.raises(InvalidFormatError) as excinfo:
        create_dataset_from_fasta(empty_fasta_file)
    assert "File is empty: no FASTA records found." in str(excinfo.value), "Incorrect error for empty FASTA file."


def test_create_dataset_from_fasta_non_existent() -> None:
    with pytest.raises(FileProcessingError) as excinfo:
        create_dataset_from_fasta(Path("non_existent_file.fasta"))
    assert "Input file does not exist" in str(excinfo.value), (
        "Incorrect error for non-existent file when creating dataset."
    )


def test_fasta_assertions() -> None:
    """Test assertions within fasta functions."""
    # Test read_fasta_file assertions
    with pytest.raises(FileProcessingError, match="Input file does not exist"):
        list(read_fasta_file(Path("nonexistent.fasta")))

    with pytest.raises(FileProcessingError, match="Input path is not a file"):
        list(read_fasta_file(Path(".")))  # Current directory

    # Test create_dataset_from_fasta assertions
    with pytest.raises(FileProcessingError, match="Input file does not exist"):
        create_dataset_from_fasta(Path("nonexistent.fasta"))

    with pytest.raises(FileProcessingError, match="Input path is not a file"):
        create_dataset_from_fasta(Path("."))  # Current directory
