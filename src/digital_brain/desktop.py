from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from digital_brain.engine import DigitalBrain, make_text_stimulus


class DesktopUnavailableError(RuntimeError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Digital Brain PySide6 desktop GUI.")
    parser.add_argument("--database", default=".brain/brain.sqlite3")
    args = parser.parse_args()
    try:
        run_desktop(Path(args.database))
    except DesktopUnavailableError as error:
        raise SystemExit(str(error)) from error


def run_desktop(database_path: Path) -> None:
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QApplication,
            QGridLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QListWidget,
            QMainWindow,
            QPushButton,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as error:
        raise DesktopUnavailableError(
            "PySide6 is not installed. Run: python -m pip install -e \".[windows]\""
        ) from error

    class BrainWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.brain = DigitalBrain(database_path)
            self.setWindowTitle("Digital Brain Cognitive OS")
            self.resize(1180, 760)

            self.chat_log = QTextEdit()
            self.chat_log.setReadOnly(True)
            self.chat_input = QLineEdit()
            self.chat_input.setPlaceholderText("Type a goal, question, or task...")
            self.send_button = QPushButton("Send")
            self.state_view = QTextEdit()
            self.state_view.setReadOnly(True)
            self.memory_list = QListWidget()

            root = QWidget()
            layout = QGridLayout(root)
            chat_panel = QVBoxLayout()
            chat_panel.addWidget(QLabel("Agent Chat"))
            chat_panel.addWidget(self.chat_log)
            input_row = QHBoxLayout()
            input_row.addWidget(self.chat_input)
            input_row.addWidget(self.send_button)
            chat_panel.addLayout(input_row)

            side_panel = QVBoxLayout()
            side_panel.addWidget(QLabel("Cognitive State"))
            side_panel.addWidget(self.state_view)
            side_panel.addWidget(QLabel("Recent Episodes"))
            side_panel.addWidget(self.memory_list)

            layout.addLayout(chat_panel, 0, 0)
            layout.addLayout(side_panel, 0, 1)
            layout.setColumnStretch(0, 3)
            layout.setColumnStretch(1, 2)
            self.setCentralWidget(root)

            self.send_button.clicked.connect(self.send_message)
            self.chat_input.returnPressed.connect(self.send_message)
            self.refresh()

        def closeEvent(self, event) -> None:
            self.brain.close()
            event.accept()

        def send_message(self) -> None:
            text = self.chat_input.text().strip()
            if not text:
                return
            self.chat_input.clear()
            self.chat_log.append(f"<b>User:</b> {text}")
            cycle = self.brain.cycle(make_text_stimulus(text))
            self.chat_log.append(f"<b>Digital Brain:</b> {cycle.response}")
            self.refresh()

        def refresh(self) -> None:
            self.state_view.setPlainText(json.dumps(self.brain.state(), ensure_ascii=False, indent=2))
            self.memory_list.clear()
            for episode in self.brain.memory.recent_rows("episodes", 20):
                self.memory_list.addItem(
                    f"{episode['timestamp']} | {episode['active_goal']} | reward={episode['reward']}"
                )

    app = QApplication(sys.argv)
    app.setStyleSheet(
        """
        QMainWindow, QWidget { background: #090b16; color: #edf4ff; font-size: 14px; }
        QTextEdit, QLineEdit, QListWidget {
            background: #141b33; color: #edf4ff; border: 1px solid #28314f;
            border-radius: 10px; padding: 8px;
        }
        QPushButton {
            background: #705cff; color: white; border: 0; border-radius: 10px; padding: 10px 16px;
        }
        QLabel { color: #9fb0cc; font-weight: 700; }
        """
    )
    window = BrainWindow()
    window.setWindowFlag(Qt.Window)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
