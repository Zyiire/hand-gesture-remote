"""Every tunable value, loaded from and saved to settings.json.

Edit settings.json to change behaviour; delete it (or press R twice in the app) to go back to defaults.
"""
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

SETTINGS_PATH = Path(__file__).resolve().parent.parent / "settings.json"

# Opt-in system gestures. Each gesture maps to an action name from macos.SYSTEM_ACTIONS
# (e.g. "mission_control"), a custom shortcut like "keys:cmd+shift+4", or None to leave it off.
DEFAULT_MAPPINGS = {
    "two_finger_flick_right": None,
    "two_finger_flick_left": None,
    "spread_hold": None,
    "spread_flick_left": None,
    "spread_flick_right": None,
    "spread_flick_up": None,
    "spread_flick_down": None,
}

# A trackpad-like set of mappings, for turning everything on at once
TRACKPAD_MAPPINGS = {
    "two_finger_flick_right": "back",
    "two_finger_flick_left": "forward",
    "spread_hold": "mission_control",
    "spread_flick_left": "next_desktop",
    "spread_flick_right": "previous_desktop",
    "spread_flick_up": "mission_control",
    "spread_flick_down": "app_expose",
}


@dataclass
class Settings:
    calibrated: bool = False  # False = run the setup/calibration flow on next start

    # Activation
    arm_time: float = 1.0        # Seconds to hold an open hand before the hand takes control
    pointer_hand: str = "any"    # Which hand drives the pointer: "any", "left" or "right"
    min_hand_score: float = 0.5  # Ignore hand detections less confident than this (0-1)

    # Cursor movement: relative (like a trackpad) with acceleration.
    # Gain 1 = moving your hand across the whole webcam view moves the cursor across the whole screen.
    sensitivity: float = 1.0  # Overall speed multiplier (adjust live with + / -)
    slow_gain: float = 1.0    # Speed multiplier for slow, careful movements
    fast_gain: float = 4.0    # Speed multiplier for quick movements
    slow_speed: float = 0.1   # Hand speeds (webcam widths per second) at or below this use slow_gain
    fast_speed: float = 1.0   # Hand speeds at or above this use fast_gain

    # Cursor smoothing (One Euro filter)
    min_cutoff: float = 1.0  # Lower = steadier cursor when your hand is still, but more lag
    beta: float = 0.03       # Higher = less lag when you move fast, but more shake

    # Clicking: finger bend in degrees (0 = straight). Press above *_press, release below *_release.
    index_press: float = 50.0
    index_release: float = 35.0
    middle_press: float = 50.0
    middle_release: float = 35.0
    curled_bend: float = 50.0     # A finger bent more than this counts as curled (for poses)
    straight_bend: float = 35.0   # A finger bent less than this counts as straight
    click_confirm: float = 0.1    # Seconds neighbouring fingers must stay straight before a button is held down
    click_lookback: float = 0.15  # On a press, the cursor snaps back to where it was this long ago
    click_hold: float = 0.3       # Seconds the cursor stays locked after a click (for double clicks)
    drag_start: float = 40.0      # Px of movement from a locked click before it becomes a drag
    double_click_distance: float = 10.0
    double_click_time: float = 0.5  # Replaced by the Mac's own double-click speed at startup

    # Poses
    pose_time: float = 0.15   # Seconds a new pose must be held before it takes over
    spread_open: float = 1.1  # Index-to-pinky fingertip gap (hand sizes) above this = fingers spread
    thumb_out: float = 0.7    # Thumb tip at least this far (hand sizes) from the index knuckle = thumb open

    # Two-finger scrolling
    scroll_speed: float = 800.0  # Pixels scrolled per hand-size of hand movement
    natural_scroll: bool = True  # True = content follows your hand, like a Mac trackpad

    # Flicks (quick swipes) and holds for system gestures
    swipe_speed: float = 0.8     # Hand faster than this (webcam widths per second) counts as a flick
    swipe_distance: float = 0.12  # A flick must travel this far (fraction of webcam width)
    swipe_cooldown: float = 0.6
    system_hold: float = 0.4     # Seconds to hold the spread hand still for "spread_hold"

    # Opt-in features
    three_finger_drag: bool = False  # Three fingers up = hold the left button and drag
    mappings: dict = field(default_factory=lambda: dict(DEFAULT_MAPPINGS))

    # Two-hand zoom: hold both hands open, move them apart (zoom in) or together (zoom out).
    # Close both hands to clutch: zoom pauses so you can reposition, then open them to carry on.
    two_hand_zoom: bool = True
    zoom_output: str = "shortcut"  # "shortcut" = Cmd+= / Cmd+- steps, "ctrl_scroll" = continuous Ctrl+scroll
    zoom_step: float = 0.15        # Hand gap must change by this ratio for one zoom step
    zoom_hysteresis: float = 0.15  # Extra margin (in steps) before stepping, so zoom doesn't flicker at a boundary
    zoom_max_steps: int = 8        # Most steps in or out per zoom session
    zoom_scroll_pixels: float = 40.0  # ctrl_scroll only: scroll pixels per zoom step

    # Tracking reliability
    lost_grace: float = 0.3    # Seconds the hand can vanish before held buttons are released
    disarm_after: float = 3.0  # Seconds without a hand before control switches off (re-arm to continue)
    max_jump: float = 0.15     # Palm jump (fraction of webcam width) in one frame that counts as a bad detection
    face_every: int = 3        # Run the face detector every this many frames


def load(path=SETTINGS_PATH):
    """Settings from settings.json, falling back to defaults for anything missing."""
    settings = Settings()
    if not Path(path).exists():
        return settings
    data = json.loads(Path(path).read_text())
    known = {f.name for f in fields(Settings)}
    for key, value in data.items():
        if key not in known:
            print(f"settings.json: ignoring unknown setting '{key}'")
        elif key == "mappings":
            settings.mappings.update({k: v for k, v in value.items() if k in DEFAULT_MAPPINGS})
        else:
            setattr(settings, key, value)
    return settings


def save(settings, path=SETTINGS_PATH):
    Path(path).write_text(json.dumps(asdict(settings), indent=2) + "\n")


def reset(path=SETTINGS_PATH):
    """Back to defaults, keeping the user's opt-in choices (mappings, three-finger drag)."""
    old = load(path)
    fresh = Settings(mappings=old.mappings, three_finger_drag=old.three_finger_drag)
    save(fresh, path)
    return fresh
