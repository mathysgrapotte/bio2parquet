"""Module that can be run directly.

When run as `python -m bio2parquet`, it will invoke the main CLI group.
"""

from __future__ import annotations

import sys

if __name__ == "__main__":
    from bio2parquet.cli import main  # Updated import

    sys.exit(main())  # Call the click group
