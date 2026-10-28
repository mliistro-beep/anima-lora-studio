from pathlib import Path
from xml.etree import ElementTree

from anima_trainer.ui.app_icon import APP_ICON_FILE, DESKTOP_FILE_NAME, app_icon
from anima_trainer.ui.main_window import APP_NAME

SVG_NAMESPACE = "{http://www.w3.org/2000/svg}"
LAUNCHER = Path(__file__).resolve().parents[2] / "run_linux.sh"


def test_the_icon_file_is_a_square_svg_named_like_the_desktop_entry():
    root = ElementTree.parse(APP_ICON_FILE).getroot()

    assert APP_ICON_FILE.name == f"{DESKTOP_FILE_NAME}.svg"
    assert root.tag == f"{SVG_NAMESPACE}svg"
    _, _, width, height = root.get("viewBox").split()
    assert width == height


def test_the_desktop_file_name_and_the_title_match_run_linux_sh():
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert f'readonly APP_ID="{DESKTOP_FILE_NAME}"\n' in launcher
    assert f'readonly APP_TITLE="{APP_NAME}"\n' in launcher


def test_the_icon_renders_at_menu_and_title_bar_sizes(qapp):
    icon = app_icon()

    assert not icon.isNull()
    for size in (16, 48, 256):
        assert not icon.pixmap(size, size).isNull()
