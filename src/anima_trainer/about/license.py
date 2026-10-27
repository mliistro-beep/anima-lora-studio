"""The app's license: GPL v3 or later, with the notice the GPL asks interactive programs to show."""

from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version

DISTRIBUTION = "anima-trainer"
LICENSE_ID = "GPL-3.0-or-later"
COPYRIGHT = "Copyright (C) 2026 Mauro Liistro"
GPL_URL = "https://www.gnu.org/licenses/gpl-3.0.html"
NOTICE_PARAGRAPHS = (
    "This program is free software: you can redistribute it and/or modify it under the terms "
    "of the GNU General Public License as published by the Free Software Foundation, either "
    "version 3 of the License, or (at your option) any later version.",
    "This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; "
    "without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. "
    "See the GNU General Public License for more details.",
    "You should have received a copy of the GNU General Public License along with this "
    "program (the LICENSE file). If not, see the link below.",
)
UNKNOWN_VERSION = "unknown"


def app_version(read_version: Callable[[str], str] = version) -> str:
    """Version of the installed package, or "unknown" when run from a plain source tree."""
    try:
        return read_version(DISTRIBUTION)
    except PackageNotFoundError:
        return UNKNOWN_VERSION
