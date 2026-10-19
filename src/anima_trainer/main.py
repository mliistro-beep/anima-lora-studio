"""Application entry point."""

import sys

from PySide6.QtWidgets import QApplication, QMainWindow

APP_NAME = "Anima LoRA Trainer"


def create_window() -> QMainWindow:
    window = QMainWindow()
    window.setWindowTitle(APP_NAME)
    window.resize(1200, 800)
    return window


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    window = create_window()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
