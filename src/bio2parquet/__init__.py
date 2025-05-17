"""bio2parquet package.

Convert common bioinformatics file formats to parquet
"""

from __future__ import annotations

from bio2parquet._internal.cli import get_parser, main

__all__: list[str] = ["get_parser", "main"]
