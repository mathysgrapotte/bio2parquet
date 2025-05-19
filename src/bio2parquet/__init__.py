"""bio2parquet.

Convert common bioinformatics file formats to parquet.
"""

import importlib.metadata

__all__ = (
    "Bio2ParquetError",
    "FileProcessingError",
    "InvalidFormatError",
    "__version__",
    "create_dataset_from_fasta",
    "main",
    "print_error",
    "read_fasta_file",
)


try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0.dev0"  # Fallback version

from bio2parquet.cli import main
from bio2parquet.errors import Bio2ParquetError, FileProcessingError, InvalidFormatError, print_error
from bio2parquet.fasta import create_dataset_from_fasta, read_fasta_file
