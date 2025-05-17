"""Command Line Interface for bio2parquet.

This module provides the CLI for converting bioinformatics files to Parquet format.
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from datasets import Dataset

from bio2parquet.errors import Bio2ParquetError, print_error
from bio2parquet.fasta_processor import create_dataset_from_fasta


def _handle_empty_dataset(dataset: "Dataset") -> None:
    """Helper function to raise Bio2ParquetError if the dataset is empty."""
    if len(dataset) == 0:
        raise Bio2ParquetError(
            "The FASTA file seems to be empty or could not be parsed correctly, resulting in an empty dataset.",
        )


@click.group()
@click.version_option()  # Reads version from pyproject.toml
def main() -> None:
    """Bio2Parquet: Convert bioinformatics files to Parquet.

    Currently supports FASTA to Parquet conversion.
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
    output_file: Path | None,
    hf_token: str | None,
    hf_repo_id: str | None,
) -> None:
    """Converts a FASTA file to Parquet format.

    FASTA_FILE: Path to the input FASTA file (.fasta or .fasta.gz).
    """
    try:
        if not fasta_file.name.endswith((".fasta", ".fa", ".fna", ".fasta.gz", ".fa.gz", ".fna.gz")):
            raise click.BadParameter(
                "Input file must be a FASTA file (e.g., .fasta, .fa, .fna, .fasta.gz)",
                param_hint="fasta_file",
            )

        click.echo(f"Processing FASTA file: {fasta_file}")

        dataset: Dataset = create_dataset_from_fasta(fasta_file)

        _handle_empty_dataset(dataset)

        if output_file is None:
            output_file = Path.cwd() / f"{fasta_file.stem}.parquet"

        dataset.to_parquet(output_file)
        click.echo(f"Successfully converted to Parquet: {output_file}")

        if hf_repo_id:
            if not hf_token:
                click.secho(
                    "Error: --hf-repo-id was provided, but --hf-token is missing. "
                    "Please provide a Hugging Face token to upload the dataset.",
                    fg="red",
                )
                sys.exit(1)

            click.echo(f"Pushing dataset to Hugging Face Hub: {hf_repo_id}")

            # TODO: Add dataset card content if provided
            # from huggingface_hub import HfApi, create_repo, CommitOperationAdd
            # from datasets.utils.metadata import DatasetMetadata
            # card_content = ""
            # if dataset_card:
            #     with open(dataset_card, 'r') as f:
            #         card_content = f.read()
            #     # Potentially parse and validate card_content here

            dataset.push_to_hub(repo_id=hf_repo_id, token=hf_token)
            click.echo("Dataset pushed successfully.")

    except Bio2ParquetError as e:
        print_error(e)
        sys.exit(1)
    except click.ClickException as e:  # Catch click exceptions
        e.show()
        sys.exit(e.exit_code)
    except RuntimeError as e:  # Catch unexpected runtime errors
        print_error(e)  # Fallback for unexpected errors
        sys.exit(1)


if __name__ == "__main__":
    main()
