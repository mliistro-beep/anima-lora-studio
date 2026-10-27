from importlib.metadata import PackageNotFoundError

from anima_trainer.about import license as app_license


def test_app_version_reads_the_installed_package():
    assert app_license.app_version(lambda name: {"anima-trainer": "1.2.3"}[name]) == "1.2.3"


def test_app_version_is_unknown_when_not_installed():
    def missing(name: str) -> str:
        raise PackageNotFoundError(name)

    assert app_license.app_version(missing) == "unknown"


def test_notice_is_the_gpl_v3_or_later_notice_with_the_author():
    text = " ".join(app_license.NOTICE_PARAGRAPHS)

    assert app_license.COPYRIGHT == "Copyright (C) 2026 Mauro Liistro"
    assert "either version 3 of the License, or" in text
    assert "WITHOUT ANY WARRANTY" in text
    assert app_license.GPL_URL.startswith("https://www.gnu.org/licenses/gpl-3.0")
