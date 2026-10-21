"""Semantic actions the gesture engine asks for. macos.py turns them into real events."""
from dataclasses import dataclass


@dataclass
class MoveCursor:
    x: float
    y: float


@dataclass
class ButtonDown:
    button: str  # "left" or "right"
    x: float
    y: float
    clicks: int = 1  # 2 = second press of a double click, etc.


@dataclass
class ButtonUp:
    button: str
    x: float
    y: float
    clicks: int = 1


@dataclass
class Scroll:
    dx: int  # Pixels; positive = scroll left
    dy: int  # Pixels; positive = scroll up
    zoom: bool = False  # True = hold Ctrl while scrolling (zooms in many apps and in macOS screen zoom)


@dataclass
class ZoomStep:
    direction: int  # +1 = zoom in, -1 = zoom out


@dataclass
class Command:
    name: str  # A macos.SYSTEM_ACTIONS name, or "keys:cmd+shift+4" style shortcut
