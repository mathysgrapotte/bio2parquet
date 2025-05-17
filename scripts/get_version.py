# Get current project version from Git tags or changelog.

import os
import re
from contextlib import suppress
from pathlib import Path

from pdm.backend.hooks.version import SCMVersion, Version, default_version_formatter, get_version_from_scm

_root = Path(__file__).parent.parent
_changelog = _root / "CHANGELOG.md"
_changelog_version_re = re.compile(r"^## \[(?P<version>\d+\.\d+\.\d+)\].*$")
_default_scm_version = SCMVersion(Version("0.0.0"), None, False, None, None)  # noqa: FBT003
_version_file_path = _root / "src" / "bio2parquet" / "_version.py"


def get_version() -> str:
    """Determines the project version and writes it to _version.py."""
    scm_version = get_version_from_scm(_root) or _default_scm_version
    if scm_version.version <= Version("0.1"):  # Missing Git tags?
        with suppress(OSError, StopIteration), _changelog.open("r", encoding="utf-8") as file:
            for line in file:
                match = _changelog_version_re.match(line)
                if match:
                    scm_version = scm_version._replace(version=Version(match.group("version")))
                    break

    version_str = default_version_formatter(scm_version)

    # Ensure the src/bio2parquet directory exists
    _version_file_path.parent.mkdir(parents=True, exist_ok=True)
    # Write the version to the _version.py file
    with open(_version_file_path, "w", encoding="utf-8") as vf:
        vf.write(f"# SPDX-FileCopyrightText: 2023-present {os.getenv('GIT_AUTHOR_NAME', 'Unknown Author')}\n")
        vf.write("# SPDX-License-Identifier: MIT\n")
        vf.write(f'__version__ = "{version_str}"\n')

    return version_str


if __name__ == "__main__":
    print(get_version())
