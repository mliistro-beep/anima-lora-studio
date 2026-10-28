"""The app icon, shared by the window and the menu entry that run_linux.sh installs."""

from pathlib import Path

from PySide6.QtGui import QIcon

# Name of the .desktop entry (and of the installed icon); KDE on Wayland matches the window to
# the menu entry through QGuiApplication.setDesktopFileName. Kept equal to run_linux.sh.
DESKTOP_FILE_NAME = "anima-lora-studio"
APP_ICON_FILE = Path(__file__).parent / "assets" / f"{DESKTOP_FILE_NAME}.svg"


def app_icon() -> QIcon:
    """The icon for the window title bar and the task bar (scalable, from the SVG)."""
    return QIcon(str(APP_ICON_FILE))
