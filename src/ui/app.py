"""PySide6 application entry point."""

from __future__ import annotations

import sys
import logging
from pathlib import Path
import tempfile


SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PySide6.QtWidgets import QApplication  # noqa: E402
from PySide6.QtGui import QIcon  # noqa: E402
from resource_paths import resource_path, user_data_dir  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402


def _configure_file_logging() -> Path:
    """Write GUI runtime logs outside the read-only application bundle."""

    log_path = user_data_dir("logs") / "UniversalUpscaler.log"
    try:
        handler = logging.FileHandler(log_path, encoding="utf-8")
    except OSError:
        fallback_directory = Path(tempfile.gettempdir()) / "UniversalUpscaler"
        fallback_directory.mkdir(parents=True, exist_ok=True)
        log_path = fallback_directory / "UniversalUpscaler.log"
        handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    return log_path


def main() -> int:
    _configure_file_logging()
    application = QApplication.instance() or QApplication(sys.argv)
    application.setApplicationName("UniversalUpscaler")
    icon_path = resource_path("assets", "UniversalUpscaler.png")
    if icon_path.is_file():
        application.setWindowIcon(QIcon(str(icon_path)))
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
