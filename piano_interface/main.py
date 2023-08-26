from piano_window import PianoWindow
import sys
import os

from PyQt5.QtWidgets import QApplication


def main():
    os.environ["DML_VISIBLE_DEVICES"] = "0"
    app = QApplication(sys.argv)
    win = PianoWindow()

    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

