"""The release metadata agree: pyproject.toml, the package and CHANGELOG.md."""

import tomllib
from pathlib import Path

from anima_trainer import __version__

REPO = Path(__file__).resolve().parents[2]
RELEASE = "0.3.0"


def test_the_package_and_pyproject_have_the_release_version():
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert project["version"] == RELEASE
    assert __version__ == RELEASE


def test_the_changelog_has_a_section_for_each_release():
    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")

    for release in (RELEASE, "0.1.0"):
        assert f"\n## [{release}] - " in changelog
        assert f"\n[{release}]: https://" in changelog


def test_the_readme_names_the_release_version():
    readme = (REPO / "README.md").read_text(encoding="utf-8")

    assert f"Version {RELEASE}, " in readme
