import sys
from data_collector import DataCollectorWindow

from PyQt5.QtWidgets import QApplication


def main():
    app = QApplication(sys.argv)
    win = DataCollectorWindow()

    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

