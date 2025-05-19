"""Command Line Interface for bio2parquet.

This module provides the CLI for converting bioinformatics files to Parquet format.
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import click

if TYPE_CHECKING:
    from datasets import Dataset

from bio2parquet.csv import create_dataset_from_csv
from bio2parquet.errors import Bio2ParquetError, print_error
from bio2parquet.fasta import create_dataset_from_fasta


def _handle_empty_dataset(dataset: "Dataset") -> None:
    """Helper function to raise Bio2ParquetError if the dataset is empty.

    Args:
        dataset: The dataset to check

    Raises:
        Bio2ParquetError: If the dataset is empty
    """
    if len(dataset) == 0:
        raise Bio2ParquetError(
            "The input file seems to be empty or could not be parsed correctly, resulting in an empty dataset.",
        )


def _validate_fasta_extension(filepath: Path) -> None:
    """Validates that the input file has a valid FASTA extension.

    Args:
        filepath: Path to validate

    Raises:
        click.BadParameter: If file extension is not valid
    """
    valid_extensions = (".fasta", ".fa", ".fna", ".fasta.gz", ".fa.gz", ".fna.gz")
    if not filepath.name.endswith(valid_extensions):
        raise click.BadParameter(
            f"Input file must be a FASTA file with one of these extensions: {', '.join(valid_extensions)}",
            param_hint="fasta_file",
        )


def _validate_csv_extension(filepath: Path) -> None:
    """Validates that the input file has a valid CSV extension.

    Args:
        filepath: Path to validate

    Raises:
        click.BadParameter: If file extension is not valid
    """
    if not filepath.name.endswith(".csv"):
        raise click.BadParameter(
            "Input file must be a CSV file with .csv extension",
            param_hint="csv_file",
        )


def _get_output_path(input_file: Path, output_file: Optional[Path]) -> Path:
    """Determines the output file path.

    Args:
        input_file: The input file path
        output_file: Optional output file path

    Returns:
        The output file path
    """
    if output_file is None:
        return Path.cwd() / f"{input_file.stem}.parquet"
    return output_file


def _handle_hf_upload(dataset: "Dataset", repo_id: str, token: Optional[str]) -> None:
    """Handles uploading the dataset to Hugging Face Hub.

    Args:
        dataset: The dataset to upload
        repo_id: The Hugging Face repository ID
        token: The Hugging Face token

    Raises:
        SystemExit: If token is missing or upload fails
    """
    if not token:
        click.secho(
            "Error: --hf-repo-id was provided, but --hf-token is missing. "
            "Please provide a Hugging Face token to upload the dataset.",
            fg="red",
        )
        sys.exit(1)

    click.echo(f"Pushing dataset to Hugging Face Hub: {repo_id}")
    dataset.push_to_hub(repo_id=repo_id, token=token)
    click.echo("Dataset pushed successfully.")


def _process_fasta_file(
    fasta_file: Path,
    output_file: Optional[Path],
    hf_token: Optional[str],
    hf_repo_id: Optional[str],
) -> None:
    """Process a FASTA file and convert it to Parquet format.

    Args:
        fasta_file: Path to the input FASTA file
        output_file: Optional path to the output Parquet file
        hf_token: Optional Hugging Face token
        hf_repo_id: Optional Hugging Face repository ID

    Raises:
        Bio2ParquetError: If there's an error processing the file
        click.ClickException: If there's a CLI error
        RuntimeError: For unexpected runtime errors
    """
    _validate_fasta_extension(fasta_file)
    click.echo(f"Processing FASTA file: {fasta_file}")

    dataset: Dataset = create_dataset_from_fasta(fasta_file)
    _handle_empty_dataset(dataset)

    output_path = _get_output_path(fasta_file, output_file)
    dataset.to_parquet(output_path)
    click.echo(f"Successfully converted to Parquet: {output_path}")

    if hf_repo_id:
        _handle_hf_upload(dataset, hf_repo_id, hf_token)


def _process_csv_file(
    csv_file: Path,
    output_file: Optional[Path],
    hf_token: Optional[str],
    hf_repo_id: Optional[str],
) -> None:
    """Process a CSV file and convert it to Parquet format.

    Args:
        csv_file: Path to the input CSV file
        output_file: Optional path to the output Parquet file
        hf_token: Optional Hugging Face token
        hf_repo_id: Optional Hugging Face repository ID

    Raises:
        Bio2ParquetError: If there's an error processing the file
        click.ClickException: If there's a CLI error
        RuntimeError: For unexpected runtime errors
    """
    _validate_csv_extension(csv_file)
    click.echo(f"Processing CSV file: {csv_file}")

    dataset: Dataset = create_dataset_from_csv(csv_file)
    _handle_empty_dataset(dataset)

    output_path = _get_output_path(csv_file, output_file)
    dataset.to_parquet(output_path)
    click.echo(f"Successfully converted to Parquet: {output_path}")

    if hf_repo_id:
        _handle_hf_upload(dataset, hf_repo_id, hf_token)


@click.group()
@click.version_option()  # Reads version from pyproject.toml
def main() -> None:
    """Bio2Parquet: Convert bioinformatics files to Parquet.

    Currently supports FASTA and CSV to Parquet conversion.
    """


@main.command()
@click.argument(
    "fasta_file",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "-o",
    "--output-file",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Path to the output Parquet file. If not provided, defaults to a .parquet file with the same name as the input in the current directory.",
)
@click.option(
    "--hf-token",
    envvar="HF_TOKEN",
    help="Hugging Face Hub token for uploading the dataset. Can also be set via HF_TOKEN environment variable.",
)
@click.option(
    "--hf-repo-id",
    help="Hugging Face Hub repository ID to push the dataset to (e.g., 'username/my_fasta_dataset'). Requires --hf-token.",
)
def fasta(
    fasta_file: Path,
    output_file: Optional[Path],
    hf_token: Optional[str],
    hf_repo_id: Optional[str],
) -> None:
    """Converts a FASTA file to Parquet format.

    FASTA_FILE: Path to the input FASTA file (.fasta or .fasta.gz).
    """
    try:
        _process_fasta_file(fasta_file, output_file, hf_token, hf_repo_id)
    except Bio2ParquetError as e:
        print_error(e)
        sys.exit(1)
    except click.ClickException as e:
        e.show()
        sys.exit(e.exit_code)
    except RuntimeError as e:
        print_error(e)
        sys.exit(1)


@main.command()
@click.argument(
    "csv_file",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "-o",
    "--output-file",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Path to the output Parquet file. If not provided, defaults to a .parquet file with the same name as the input in the current directory.",
)
@click.option(
    "--hf-token",
    envvar="HF_TOKEN",
    help="Hugging Face Hub token for uploading the dataset. Can also be set via HF_TOKEN environment variable.",
)
@click.option(
    "--hf-repo-id",
    help="Hugging Face Hub repository ID to push the dataset to (e.g., 'username/my_csv_dataset'). Requires --hf-token.",
)
def csv(
    csv_file: Path,
    output_file: Optional[Path],
    hf_token: Optional[str],
    hf_repo_id: Optional[str],
) -> None:
    """Converts a CSV file to Parquet format.

    CSV_FILE: Path to the input CSV file (.csv).
    The CSV file must have at least 'header' and 'sequence' columns.
    """
    try:
        _process_csv_file(csv_file, output_file, hf_token, hf_repo_id)
    except Bio2ParquetError as e:
        print_error(e)
        sys.exit(1)
    except click.ClickException as e:
        e.show()
        sys.exit(e.exit_code)
    except RuntimeError as e:
        print_error(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
