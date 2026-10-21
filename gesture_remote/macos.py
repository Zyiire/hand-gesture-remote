"""The only module that talks to macOS: permissions, and turning engine actions into real events.

Events are posted through Quartz (Core Graphics), Apple's documented way to synthesize mouse,
keyboard and scroll events. There is no public API for injecting real multi-touch trackpad
gestures, so zoom and system gestures are sent as scroll events and keyboard shortcuts.
"""
import subprocess
import threading
import time

import AVFoundation
import Quartz
from AppKit import NSEvent

from .actions import ButtonDown, ButtonUp, Command, MoveCursor, Scroll, ZoomStep

# --- Permissions ---

_CAMERA_STATUS = {0: "not_determined", 1: "restricted", 2: "denied", 3: "authorized"}


def camera_permission():
    status = AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(AVFoundation.AVMediaTypeVideo)
    return _CAMERA_STATUS.get(status, "unknown")


def request_camera(timeout=60):
    """Show the macOS camera prompt and wait for the answer. Returns True if allowed."""
    answered = threading.Event()
    result = {}

    def handler(granted):
        result["granted"] = bool(granted)
        answered.set()

    AVFoundation.AVCaptureDevice.requestAccessForMediaType_completionHandler_(AVFoundation.AVMediaTypeVideo, handler)
    answered.wait(timeout)
    return result.get("granted", False)


def events_allowed():
    """Can this process post mouse/keyboard events? (System Settings > Privacy & Security > Accessibility)"""
    return bool(Quartz.CGPreflightPostEventAccess())


def request_events():
    """Ask macOS for event-posting access (opens the system prompt the first time)."""
    return bool(Quartz.CGRequestPostEventAccess())


def hotkey_allowed():
    """Can this process read the keyboard globally? (Privacy & Security > Input Monitoring)"""
    return bool(Quartz.CGPreflightListenEventAccess())


def screen_size():
    size = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID()).size
    return (size.width, size.height)


def real_cursor():
    loc = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
    return (loc.x, loc.y)


def double_click_time():
    return float(NSEvent.doubleClickInterval())


# --- Keyboard shortcuts ---

KEYCODES = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9, "b": 11, "q": 12,
    "w": 13, "e": 14, "r": 15, "y": 16, "t": 17, "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23,
    "=": 24, "9": 25, "7": 26, "-": 27, "8": 28, "0": 29, "]": 30, "o": 31, "u": 32, "[": 33, "i": 34,
    "p": 35, "return": 36, "l": 37, "j": 38, "'": 39, "k": 40, ";": 41, "\\": 42, ",": 43, "/": 44,
    "n": 45, "m": 46, ".": 47, "tab": 48, "space": 49, "`": 50, "delete": 51, "escape": 53,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97, "f7": 98, "f8": 100, "f9": 101,
    "f10": 109, "f11": 103, "f12": 111, "left": 123, "right": 124, "down": 125, "up": 126,
}
MODIFIERS = {
    "cmd": Quartz.kCGEventFlagMaskCommand, "command": Quartz.kCGEventFlagMaskCommand,
    "ctrl": Quartz.kCGEventFlagMaskControl, "control": Quartz.kCGEventFlagMaskControl,
    "alt": Quartz.kCGEventFlagMaskAlternate, "option": Quartz.kCGEventFlagMaskAlternate,
    "shift": Quartz.kCGEventFlagMaskShift, "fn": Quartz.kCGEventFlagMaskSecondaryFn,
}
# Real arrow keys carry these flags; without them some shortcuts (e.g. switching desktops) are ignored
ARROW_FLAGS = Quartz.kCGEventFlagMaskSecondaryFn | Quartz.kCGEventFlagMaskNumericPad

# Named actions for settings.json "mappings". Each uses a documented macOS shortcut or app.
SYSTEM_ACTIONS = {
    "mission_control": "open:Mission Control",
    "app_expose": "keys:ctrl+down",
    "next_desktop": "keys:ctrl+right",
    "previous_desktop": "keys:ctrl+left",
    "back": "keys:cmd+[",
    "forward": "keys:cmd+]",
}


def parse_shortcut(text):
    """"cmd+shift+4" -> (keycode, flags). Raises ValueError for unknown keys."""
    *mods, key = [part.strip().lower() for part in text.split("+")]
    if key not in KEYCODES:
        raise ValueError(f"unknown key '{key}' in shortcut '{text}'")
    flags = 0
    for mod in mods:
        if mod not in MODIFIERS:
            raise ValueError(f"unknown modifier '{mod}' in shortcut '{text}'")
        flags |= MODIFIERS[mod]
    if key in ("left", "right", "up", "down"):
        flags |= ARROW_FLAGS
    return KEYCODES[key], flags


def resolve(name):
    """A mapping value -> "keys:..." or "open:..." command string. Raises ValueError if invalid."""
    command = SYSTEM_ACTIONS.get(name, name)
    if command.startswith("keys:"):
        parse_shortcut(command[5:])
    elif not command.startswith("open:"):
        raise ValueError(f"unknown action '{name}' (use one of {', '.join(SYSTEM_ACTIONS)}, "
                         "or 'keys:cmd+shift+4' style shortcuts)")
    return command


def check_mappings(mappings):
    """Problems with settings.json mappings, as readable strings."""
    problems = []
    for gesture, name in mappings.items():
        if name:
            try:
                resolve(name)
            except ValueError as error:
                problems.append(f"mappings.{gesture}: {error}")
    return problems


# --- Posting events ---

_BUTTONS = {
    #         button id                    press                         release                     move while held
    "left":  (Quartz.kCGMouseButtonLeft,  Quartz.kCGEventLeftMouseDown,  Quartz.kCGEventLeftMouseUp,  Quartz.kCGEventLeftMouseDragged),
    "right": (Quartz.kCGMouseButtonRight, Quartz.kCGEventRightMouseDown, Quartz.kCGEventRightMouseUp, Quartz.kCGEventRightMouseDragged),
}


class Dispatcher:
    """Performs engine actions. Remembers held buttons so they can always be released."""

    def __init__(self, metrics, enabled=True):
        self.metrics = metrics
        self.enabled = enabled
        self.held = {}  # button -> click count, for buttons currently held down

    def set_enabled(self, enabled):
        if not enabled:
            self.release_all()
        self.enabled = enabled

    def run(self, actions, frame_t):
        if not actions or not self.enabled:
            return
        for action in actions:
            try:
                self._perform(action)
            except Exception as error:  # Never let one bad event stop the loop (or skip a release)
                print(f"Event failed: {action}: {error}")
        self.metrics.add_latency(time.monotonic() - frame_t)

    def release_all(self):
        """Release every held button. Called on pause, cancel, permission loss and shutdown."""
        if not self.held:
            return
        pos = real_cursor()
        for button, clicks in list(self.held.items()):
            try:
                self._mouse(_BUTTONS[button][2], button, pos, clicks)
            except Exception as error:
                print(f"Could not release {button} button: {error}")
        self.held.clear()

    def _perform(self, action):
        if isinstance(action, MoveCursor):
            # While a button is held, macOS expects "dragged" events instead of plain moves
            if self.held:
                button = next(iter(self.held))
                self._mouse(_BUTTONS[button][3], button, (action.x, action.y))
            else:
                self._mouse(Quartz.kCGEventMouseMoved, "left", (action.x, action.y))
        elif isinstance(action, ButtonDown):
            self._mouse(_BUTTONS[action.button][1], action.button, (action.x, action.y), action.clicks)
            self.held[action.button] = action.clicks
        elif isinstance(action, ButtonUp):
            self._mouse(_BUTTONS[action.button][2], action.button, (action.x, action.y), action.clicks)
            self.held.pop(action.button, None)
        elif isinstance(action, Scroll):
            event = Quartz.CGEventCreateScrollWheelEvent(None, Quartz.kCGScrollEventUnitPixel, 2, action.dy, action.dx)
            if action.zoom:
                Quartz.CGEventSetFlags(event, Quartz.kCGEventFlagMaskControl)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        elif isinstance(action, ZoomStep):
            self._keys("cmd+=" if action.direction > 0 else "cmd+-")
        elif isinstance(action, Command):
            command = resolve(action.name)
            if command.startswith("keys:"):
                self._keys(command[5:])
            else:
                subprocess.Popen(["open", "-a", command[5:]])

    def _mouse(self, event_type, button, pos, clicks=1):
        event = Quartz.CGEventCreateMouseEvent(None, event_type, pos, _BUTTONS[button][0])
        Quartz.CGEventSetIntegerValueField(event, Quartz.kCGMouseEventClickState, clicks)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)

    def _keys(self, shortcut):
        keycode, flags = parse_shortcut(shortcut)
        for is_down in (True, False):
            event = Quartz.CGEventCreateKeyboardEvent(None, keycode, is_down)
            Quartz.CGEventSetFlags(event, flags)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


class PauseHotkey:
    """Ctrl+Option+P toggles pause from any app (needs Input Monitoring permission)."""
    CTRL, OPTION, P = (59, 62), (58, 61), 35

    def __init__(self):
        self.was_down = False

    def pressed(self):
        state = Quartz.kCGEventSourceStateCombinedSessionState
        down = (any(Quartz.CGEventSourceKeyState(state, k) for k in self.CTRL)
                and any(Quartz.CGEventSourceKeyState(state, k) for k in self.OPTION)
                and Quartz.CGEventSourceKeyState(state, self.P))
        fired = down and not self.was_down
        self.was_down = down
        return fired
