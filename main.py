import sys

from PySide6.QtWidgets import QApplication

from app.gui import InterfaceMatcherWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Interface Matcher")
    window = InterfaceMatcherWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
