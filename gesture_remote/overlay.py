"""Draws state, feedback and debug readings on the webcam image."""
import cv2

from .engine import State

# BGR colours per state
COLOURS = {
    State.PAUSED: (128, 128, 128), State.IDLE: (160, 160, 160), State.ARMING: (0, 200, 255),
    State.POINTER: (80, 200, 80), State.CLICKING: (255, 200, 0), State.DRAGGING: (0, 140, 255),
    State.PARKED: (200, 120, 40), State.SCROLLING: (200, 80, 160), State.SYSTEM: (220, 60, 220),
    State.ZOOMING: (230, 200, 80), State.LOST: (40, 40, 230),
}
HINTS = {
    State.PAUSED: "Paused. Press SPACE (or Ctrl+Option+P) to resume",
    State.IDLE: "Hold an open hand up to take control",
    State.ARMING: "Keep holding...",
    State.PARKED: "Parked: move your fist anywhere, open your hand to carry on",
    State.LOST: "Hand lost: bring it back into view",
}
# The 21-point hand skeleton
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
               (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
FONT = cv2.FONT_HERSHEY_SIMPLEX
HELP = "SPACE pause  C calibrate  D debug  +/- speed  R reset  Q quit   |   Ctrl+Option+P pauses from any app"


def text(img, s, pos, scale=0.55, colour=(255, 255, 255), thickness=1):
    cv2.putText(img, s, pos, FONT, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, s, pos, FONT, scale, colour, thickness, cv2.LINE_AA)


def draw_hand(img, pts, colour):
    for a, b in CONNECTIONS:
        cv2.line(img, (int(pts[a][0]), int(pts[a][1])), (int(pts[b][0]), int(pts[b][1])), colour, 2)
    for x, y in pts:
        cv2.circle(img, (int(x), int(y)), 3, (255, 255, 255), -1)


def draw(img, status, frame, info):
    """Draw everything. Returns the pause button's rectangle (x1, y1, x2, y2) for click handling."""
    height, width = img.shape[:2]
    colour = COLOURS.get(status.state, (255, 255, 255))

    for x, y, w, h in frame.faces:
        cv2.rectangle(img, (int(x * width), int(y * height)), (int((x + w) * width), int((y + h) * height)),
                      (0, 0, 255), 1)
    for pts in status.hands:
        draw_hand(img, pts, colour)

    # State banner
    cv2.rectangle(img, (0, 0), (width, 44), colour, -1)
    label = status.state.value + (f" - {status.detail}" if status.detail else "")
    text(img, label, (10, 30), 0.8, (255, 255, 255), 2)
    button = (width - 110, 6, width - 10, 38)
    cv2.rectangle(img, button[:2], button[2:], (40, 40, 40), -1)
    text(img, "RESUME" if status.state == State.PAUSED else "PAUSE", (button[0] + 12, 28), 0.6)

    if status.progress is not None:
        cv2.rectangle(img, (0, 44), (int(width * status.progress), 52), (255, 255, 255), -1)

    y = 75
    hint = HINTS.get(status.state)
    if hint:
        text(img, hint, (10, y))
        y += 24
    if status.note:
        text(img, status.note, (10, y), 0.5, (0, 200, 255))
        y += 22
    for warning in info.get("warnings", []):
        text(img, warning, (10, y), 0.5, (60, 60, 255))
        y += 22
    if info.get("message"):
        text(img, info["message"], (10, y), 0.5, (0, 255, 255))

    stats = f"{info.get('fps', 0):.0f} fps"
    if info.get("latency_ms") is not None:
        stats += f"  {info['latency_ms']:.0f} ms"
    stats += f"  speed x{info.get('sensitivity', 1):.1f}"
    if info.get("recording"):
        stats += "  REC"
    text(img, stats, (width - 300, 70), 0.5, (0, 255, 0))

    if info.get("debug") and status.readings:
        reading = "  ".join(f"{k} {v}" for k, v in status.readings.items())
        text(img, reading, (10, height - 40), 0.45, (0, 255, 255))
    text(img, HELP, (10, height - 15), 0.4, (220, 220, 220))
    return button


def draw_calibration(img, frame, calibrator, info):
    height, width = img.shape[:2]
    for hand in frame.hands:
        draw_hand(img, [(x * width, y * height) for x, y in hand.points], (0, 200, 255))
    cv2.rectangle(img, (0, 0), (width, 44), (0, 160, 255), -1)
    text(img, f"SETUP {calibrator.step_number}/{calibrator.total_steps}: {calibrator.instruction}", (10, 30), 0.6,
         (255, 255, 255), 2)
    if calibrator.progress:
        cv2.rectangle(img, (0, 44), (int(width * calibrator.progress), 52), (255, 255, 255), -1)
    text(img, calibrator.message, (10, 80), 0.55, (0, 255, 255))
    text(img, "Mouse control stays off during setup.  S skip (keep current values)  Q quit",
         (10, height - 15), 0.45, (220, 220, 220))
    for i, warning in enumerate(info.get("warnings", [])):
        text(img, warning, (10, 110 + 22 * i), 0.5, (60, 60, 255))
