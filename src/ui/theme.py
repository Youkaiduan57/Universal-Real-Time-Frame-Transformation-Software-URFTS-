"""Neutral light and dark themes for the native Windows utility UI."""

from __future__ import annotations


ACCENT = "#525295"
ACCENT_HOVER = "#5555B8"
DANGER = "#B84A4A"


def stylesheet(theme: str) -> str:
    dark = theme == "dark"
    if dark:
        background = "#202124"
        topbar = "#252629"
        sidebar = "#242528"
        surface = "#2A2B2F"
        input_background = "#242529"
        text = "#F1F1F2"
        soft = "#C7C8CC"
        muted = "#92949B"
        border = "#3A3B40"
        hover = "#323339"
        selected = "#34343E"
    else:
        background = "#F7F7F8"
        topbar = "#FCFCFC"
        sidebar = "#FAFAFA"
        surface = "#FFFFFF"
        input_background = "#FFFFFF"
        text = "#29292D"
        soft = "#4D4D53"
        muted = "#76767E"
        border = "#E4E4E7"
        hover = "#F1F1F3"
        selected = "#EEEEF2"

    return f"""
        QWidget {{
            color: {text};
            font-family: "Segoe UI";
            font-size: 9.5pt;
            background: transparent;
        }}
        QMainWindow, QWidget#appShell, QWidget#workspacePage,
        QWidget#generalPage, QWidget#logsPage, QWidget#aboutPage {{
            background: {background};
        }}
        QWidget#topBar {{
            background: {topbar};
            border-bottom: 1px solid {border};
        }}
        QWidget#sidebar {{
            background: {sidebar};
            border-right: 1px solid {border};
        }}
        QLabel#brandLabel {{ font-size: 12pt; font-weight: 600; }}
        QLabel#versionLabel, QLabel#sidebarCaption, QLabel#pageDescription,
        QLabel#rowDescription, QLabel#mutedLabel {{ color: {muted}; }}
        QLabel#sidebarCaption {{ font-size: 9pt; font-weight: 600; }}
        QLabel#pageTitle {{ font-size: 18pt; font-weight: 600; }}
        QLabel#cardTitle {{ font-size: 11.5pt; font-weight: 600; }}
        QLabel#rowTitle {{ color: {soft}; }}
        QLabel#statusKey {{ color: {muted}; }}
        QLabel#statusValue {{ color: {text}; font-weight: 600; }}
        QLabel#errorLabel {{ color: {DANGER}; font-size: 9pt; }}
        QLabel#compatibilityLabel {{ color: {DANGER}; font-size: 9pt; }}

        QScrollArea#settingsScroll {{ background: {background}; border: none; }}
        QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
        QScrollBar::handle:vertical {{ background: {border}; border-radius: 4px; min-height: 36px; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

        QFrame#settingsCard {{
            background: {surface};
            border: 1px solid {border};
            border-radius: 8px;
        }}
        QFrame#divider {{ background: {border}; border: none; min-height: 1px; max-height: 1px; }}

        QPushButton {{
            background: {surface}; color: {text}; border: 1px solid {border};
            border-radius: 6px; padding: 5px 10px; min-height: 22px;
        }}
        QPushButton:hover {{ background: {hover}; }}
        QPushButton:pressed {{ background: {selected}; }}
        QPushButton:disabled {{ color: {muted}; background: {hover}; }}
        QPushButton#accentButton {{
            color: white; background: {ACCENT}; border-color: {ACCENT};
            font-weight: 600; min-width: 82px; padding: 6px 17px;
        }}
        QPushButton#accentButton:hover {{ background: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; }}
        QPushButton#accentButton[danger="true"] {{ background: {DANGER}; border-color: {DANGER}; }}
        QPushButton#quietButton {{ background: transparent; color: {soft}; }}
        QPushButton#sidebarItem {{
            background: transparent; border: none; border-left: 2px solid transparent;
            border-radius: 3px; text-align: left; padding: 7px 10px;
        }}
        QPushButton#sidebarItem:hover {{ background: {hover}; }}
        QPushButton#sidebarItem:checked {{
            background: {selected}; border-left: 2px solid {ACCENT}; color: {text};
        }}
        QPushButton#sidebarAction {{
            background: transparent; border: 1px solid {border}; color: {soft}; padding: 4px 7px;
        }}

        QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
            background: {input_background}; color: {text}; border: 1px solid {border};
            border-radius: 5px; padding: 4px 8px; min-height: 22px; min-width: 118px;
            selection-background-color: {ACCENT};
        }}
        QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{ border-color: {muted}; }}
        QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{ border-color: {ACCENT}; }}
        QComboBox:disabled, QLineEdit:disabled {{ background: {hover}; color: {muted}; }}
        QComboBox::drop-down {{ border: none; width: 24px; }}
        QComboBox QAbstractItemView {{
            background: {surface}; color: {text}; border: 1px solid {border};
            selection-background-color: {selected}; selection-color: {text}; outline: none;
        }}
        QPlainTextEdit#logView {{
            background: {input_background}; color: {soft}; border: 1px solid {border};
            border-radius: 6px; padding: 8px; font-family: "Cascadia Mono"; font-size: 9pt;
        }}
        QToolTip {{ background: {surface}; color: {text}; border: 1px solid {border}; padding: 4px; }}
    """


def apply_theme(application, theme: str) -> None:
    application.setStyle("Fusion")
    application.setStyleSheet(stylesheet(theme))
