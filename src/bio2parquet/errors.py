"""Custom error classes and error handling utilities for bio2parquet."""

import traceback

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


class Bio2ParquetError(Exception):
    """Base class for exceptions in bio2parquet."""


class FileProcessingError(Bio2ParquetError):
    """Exception raised for errors in file processing."""

    def __init__(self, message: str, filepath: str):
        """Initialize FileProcessingError.

        Args:
            message: The error message.
            filepath: The path to the file that caused the error.
        """
        self.message = message
        self.filepath = filepath
        super().__init__(f"Error processing file '{filepath}': {message}")


class InvalidFormatError(Bio2ParquetError):
    """Exception raised for invalid file formats."""

    def __init__(self, message: str, filepath: str):
        """Initialize InvalidFormatError.

        Args:
            message: The error message.
            filepath: The path to the file that caused the error.
        """
        self.message = message
        self.filepath = filepath
        super().__init__(f"Invalid format in file '{filepath}': {message}")


def print_error(exception: BaseException) -> None:
    """Prints a formatted error message and traceback using Rich.

    Args:
        exception: The exception to print.
    """
    console = Console(stderr=True)

    error_message = Text(f"An error occurred: {exception!s}\n", style="bold red")

    tb_str = traceback.format_exc()
    traceback_panel = Panel(
        Text(tb_str, style="dim white"),
        title="[bold yellow]Error Traceback[/bold yellow]",
        border_style="red",
        expand=False,
    )

    console.print(error_message)
    console.print(traceback_panel)

    if hasattr(exception, "__cause__") and exception.__cause__:
        console.print("\n[bold yellow]Underlying Cause:[/bold yellow]")
        print_error(exception.__cause__)
