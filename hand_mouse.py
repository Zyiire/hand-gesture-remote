import math
import subprocess
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import Quartz
from AppKit import NSEvent
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.vision import drawing_utils

# --- CONFIG ---
DOUBLE_CLICK_DISTANCE = 10  # Max px between two presses for them to count as a double click

# Cursor speed. The cursor moves by how much your hand moves (like a trackpad), with acceleration:
# slow hand = precise, fast hand = covers the screen. Gain 1 = moving your hand across the whole
# webcam view moves the cursor across the whole screen.
SLOW_GAIN = 1.0   # Speed multiplier for slow, careful movements
FAST_GAIN = 4.0   # Speed multiplier for quick movements
SLOW_SPEED = 0.1  # Hand speeds (webcam widths per second) at or below this use SLOW_GAIN
FAST_SPEED = 1.0  # Hand speeds at or above this use FAST_GAIN

# Cursor smoothing (One Euro filter)
MIN_CUTOFF = 1.0  # Lower = steadier cursor when your hand is still, but more lag
BETA = 0.03       # Higher = less lag when you move fast, but more shake

# Finger bend, in degrees (0 = straight in line with the back of the hand)
CLICK_PRESS = 50    # Bent more than this = button pressed / finger counts as curled
CLICK_RELEASE = 35  # Bent less than this = button released / finger counts as straight

# Keeping the cursor still around clicks
CLICK_LOOKBACK = 0.15  # Seconds: on a press, the cursor snaps back to where it was this long ago (before the finger started bending)
CLICK_HOLD = 0.3       # Seconds the cursor stays locked after a click, so you can double click in the same spot
DRAG_START = 40        # Move this many px away from a locked click and the cursor follows again (a drag)
CLICK_CONFIRM = 0.1    # Seconds a bent finger's neighbours must stay straight before the button is held down

# Poses
POSE_TIME = 0.15   # Seconds a new pose must be held before it takes over (stops flicker between poses)
SPREAD_OPEN = 1.1  # Index-to-pinky fingertip gap (in hand sizes) above this = hand spread open
THUMB_OUT = 0.7    # Thumb tip at least this far (in hand sizes) from the index knuckle = thumb open

# Two-finger scrolling
SCROLL_SPEED = 800    # Pixels scrolled per hand-size of hand movement
NATURAL_SCROLL = True # True = the page follows your hand, like a Mac trackpad. False = the opposite

# Swipes (quick flicks, in two-finger or open-hand pose)
SWIPE_SPEED = 0.8     # Hand faster than this (webcam widths per second) counts as a flick
SWIPE_DISTANCE = 0.12 # A flick must travel this far (fraction of the webcam width) to fire
SWIPE_COOLDOWN = 0.6  # Seconds between swipes
MISSION_CONTROL_HOLD = 0.4  # Seconds to hold an open hand still to open Mission Control

# Pinch to zoom (thumb + index out, other fingers curled)
ZOOM_STEP = 0.25  # Thumb-index gap must grow or shrink by this fraction to zoom one step

# Tracking reliability
LOST_GRACE = 0.3  # Seconds the hand can vanish (e.g. mid-click) before buttons are released and tracking resets
MAX_JUMP = 0.15   # A palm jump bigger than this (fraction of the webcam width) in one frame is a bad detection and is ignored
FACE_EVERY = 3    # Run the face detector every this many frames

# Hand landmark ids (21 points per hand)
WRIST, THUMB_TIP = 0, 4
INDEX_KNUCKLE, INDEX_TIP = 5, 8
MIDDLE_KNUCKLE, MIDDLE_TIP = 9, 12
RING_KNUCKLE, RING_TIP = 13, 16
PINKY_KNUCKLE, PINKY_TIP = 17, 20
PALM = (WRIST, INDEX_KNUCKLE, MIDDLE_KNUCKLE, RING_KNUCKLE, PINKY_KNUCKLE)
FINGERS = {"index": (INDEX_KNUCKLE, INDEX_TIP), "middle": (MIDDLE_KNUCKLE, MIDDLE_TIP),
           "ring": (RING_KNUCKLE, RING_TIP), "pinky": (PINKY_KNUCKLE, PINKY_TIP)}

# macOS key codes
KEY_LEFT, KEY_RIGHT, KEY_DOWN, KEY_UP = 123, 124, 125, 126
KEY_LEFT_BRACKET, KEY_RIGHT_BRACKET = 33, 30
KEY_EQUAL, KEY_MINUS = 24, 27
CTRL, CMD = Quartz.kCGEventFlagMaskControl, Quartz.kCGEventFlagMaskCommand
ARROW = Quartz.kCGEventFlagMaskSecondaryFn | Quartz.kCGEventFlagMaskNumericPad  # Real arrow keys carry these flags


class OneEuroFilter:
    # Smooths a value that changes over time: heavy smoothing while it moves slowly (no jitter),
    # light smoothing while it moves fast (no lag). See https://gery.casiez.net/1euro/
    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.x_prev, self.dx_prev, self.t_prev = None, 0.0, None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(t - self.t_prev, 1e-6)
        a_d = self._alpha(self.d_cutoff, dt)
        dx = a_d * (x - self.x_prev) / dt + (1 - a_d) * self.dx_prev
        a = self._alpha(self.min_cutoff + self.beta * abs(dx), dt)
        x = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x, dx, t
        return x


# --- GESTURE STATE ---
filter_x = OneEuroFilter(MIN_CUTOFF, BETA)
filter_y = OneEuroFilter(MIN_CUTOFF, BETA)
cursor = (0, 0)               # Where the cursor currently is
hand_pos = None               # Where the hand is steering the cursor (differs from cursor while locked)
palm_prev = None              # Smoothed palm position last frame, in webcam px
palm_prev_t = 0.0
raw_palm_prev = None          # Unsmoothed palm position last frame, for spotting bad detections
last_seen = 0.0               # When the hand was last tracked
frame_count = 0
face_result = None
fps = 0.0
prev_frame_time = 0.0
history = deque(maxlen=30)    # (time, hand_pos) for the last ~second
anchor = None                 # Position the cursor is locked to around a click, or None
anchor_until = 0.0            # After a click is released, the lock ends at this time

pose = "NORMAL"               # Current pose: NORMAL, TWO, THREE, OPEN or ZOOM
pose_candidate = "NORMAL"     # Pose seen this frame, waiting for POSE_TIME before taking over
candidate_since = 0.0
pose_since = 0.0              # When the current pose started
open_travel = 0.0             # How far the hand has moved since the open-hand pose started
gesture_fired = False         # True once this pose has fired a swipe or Mission Control
swipe_x = swipe_y = 0.0       # Distance covered by the current flick
swipe_ready_at = 0.0          # Swipes are ignored until this time (cooldown)
zoom_ref = None               # Thumb-index gap at the last zoom step
scroll_carry_x = scroll_carry_y = 0.0  # Leftover scroll smaller than one pixel

# --- MOUSE & KEYBOARD (macOS) ---
# Everything is sent through Quartz, macOS's own event system, so clicks, double clicks, drags,
# scrolling and shortcuts all behave like real hardware.
BUTTONS = {
    #         button id                    press                         release                     move while held
    "left":  (Quartz.kCGMouseButtonLeft,  Quartz.kCGEventLeftMouseDown,  Quartz.kCGEventLeftMouseUp,  Quartz.kCGEventLeftMouseDragged),
    "right": (Quartz.kCGMouseButtonRight, Quartz.kCGEventRightMouseDown, Quartz.kCGEventRightMouseUp, Quartz.kCGEventRightMouseDragged),
}
held = {"left": False, "right": False}
pending = {"left": None, "right": None}  # When a finger bent, while waiting to confirm it's a click
last_press = {"left": (0.0, (0, 0), 0), "right": (0.0, (0, 0), 0)}  # (time, position, click count)


def post(event_type, button, pos, clicks=1):
    event = Quartz.CGEventCreateMouseEvent(None, event_type, pos, button)
    Quartz.CGEventSetIntegerValueField(event, Quartz.kCGMouseEventClickState, clicks)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


def press(name, pos):
    button, down, _, _ = BUTTONS[name]
    last_time, last_pos, count = last_press[name]
    # Presses close together in time and space count as a double (or triple) click
    if (time.monotonic() - last_time < NSEvent.doubleClickInterval()
            and math.dist(pos, last_pos) < DOUBLE_CLICK_DISTANCE):
        count += 1
    else:
        count = 1
    last_press[name] = (time.monotonic(), pos, count)
    post(down, button, pos, count)
    held[name] = True


def release(name, pos):
    button, _, up, _ = BUTTONS[name]
    post(up, button, pos, last_press[name][2])
    held[name] = False


def release_all():
    for name, is_held in held.items():
        if is_held:
            release(name, cursor)
        pending[name] = None


def real_cursor():
    # Where the cursor actually is right now (so the hand picks up from wherever the real mouse left it)
    loc = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
    return (loc.x, loc.y)


def move(pos):
    # While a button is held, macOS expects "dragged" events instead of plain moves
    for name, is_held in held.items():
        if is_held:
            button, _, _, dragged = BUTTONS[name]
            post(dragged, button, pos)
            return
    post(Quartz.kCGEventMouseMoved, Quartz.kCGMouseButtonLeft, pos)


def scroll(dx, dy):
    # Scroll by pixels: positive dy = scroll up, positive dx = scroll left
    event = Quartz.CGEventCreateScrollWheelEvent(None, Quartz.kCGScrollEventUnitPixel, 2, int(dy), int(dx))
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


def shortcut(keycode, flags):
    # Press and release one key with modifier keys held, e.g. Ctrl + Right arrow
    for is_down in (True, False):
        event = Quartz.CGEventCreateKeyboardEvent(None, keycode, is_down)
        Quartz.CGEventSetFlags(event, flags)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


def mission_control():
    # Shows every open window; doing it again (or clicking a window) closes it
    subprocess.Popen(["open", "-a", "Mission Control"])


# --- INITIALIZING ---
cap = cv2.VideoCapture(0) # Opens the webcam (0 is the default cam)

# Only tracking 1 hand to avoid confusion.
# Kept fairly loose: stricter values drop the hand whenever a finger bends (which breaks clicks),
# and the face detector below already deals with faces.
options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
hands = vision.HandLandmarker.create_from_options(options)
HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS

# Face detector, used to throw away "hands" that are really a face
face_options = vision.FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path="blaze_face_short_range.tflite"),
    running_mode=vision.RunningMode.VIDEO,
    min_detection_confidence=0.6,
)
faces = vision.FaceDetector.create_from_options(face_options)

screen = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID()).size
screen_width, screen_height = screen.width, screen.height


def bend(world, knuckle, tip):
    # Angle (degrees) between the back of the hand (wrist -> knuckle) and the finger (knuckle -> tip).
    # Uses MediaPipe's 3D points, so it stays the same when the hand tilts toward or away from the camera.
    w, k, t = world[WRIST], world[knuckle], world[tip]
    palm_dir = np.array([k.x - w.x, k.y - w.y, k.z - w.z])
    finger_dir = np.array([t.x - k.x, t.y - k.y, t.z - k.z])
    cos = palm_dir.dot(finger_dir) / (np.linalg.norm(palm_dir) * np.linalg.norm(finger_dir))
    return math.degrees(math.acos(np.clip(cos, -1, 1)))


def world_dist(world, a, b):
    return math.dist((world[a].x, world[a].y, world[a].z), (world[b].x, world[b].y, world[b].z))


def classify(bends, spread, thumb_gap):
    # Which pose the hand is making this frame
    straight = {f: b < CLICK_RELEASE for f, b in bends.items()}
    curled = {f: b > CLICK_PRESS for f, b in bends.items()}
    if all(straight.values()) and spread > SPREAD_OPEN and thumb_gap > THUMB_OUT:
        return "OPEN"
    if straight["index"] and straight["middle"] and straight["ring"] and curled["pinky"]:
        return "THREE"
    if straight["index"] and straight["middle"] and curled["ring"] and curled["pinky"]:
        return "TWO"
    # Zoom: middle, ring and pinky curled. The index has to be out to start it, but may curl
    # while pinching once zooming has begun.
    if curled["middle"] and curled["ring"] and curled["pinky"] and (pose == "ZOOM" or not curled["index"]):
        return "ZOOM"
    return "NORMAL"


def change_pose(new_pose, now):
    global pose, pose_since, open_travel, gesture_fired, swipe_x, swipe_y, anchor, zoom_ref
    release_all()  # Finish any click or three-finger drag from the old pose
    anchor = None
    if new_pose == "THREE":
        # Three-finger drag: hold the left button down for as long as the pose lasts
        press("left", cursor)
    pose, pose_since = new_pose, now
    open_travel, gesture_fired, swipe_x, swipe_y = 0.0, False, 0.0, 0.0
    zoom_ref = None


def flick(dx, dy, dt, frame_width, now):
    # Returns "left", "right", "up" or "down" once a quick flick has travelled far enough, else None
    global swipe_x, swipe_y, swipe_ready_at
    speed = math.hypot(dx, dy) / frame_width / dt
    if speed > SWIPE_SPEED:
        swipe_x += dx / frame_width
        swipe_y += dy / frame_width
    else:
        swipe_x = swipe_y = 0.0
    if now < swipe_ready_at:
        return None
    direction = None
    if abs(swipe_x) > SWIPE_DISTANCE and abs(swipe_x) > abs(swipe_y):
        direction = "right" if swipe_x > 0 else "left"
    elif abs(swipe_y) > SWIPE_DISTANCE:
        direction = "down" if swipe_y > 0 else "up"
    if direction:
        swipe_x = swipe_y = 0.0
        swipe_ready_at = now + SWIPE_COOLDOWN
    return direction


def position_before(t):
    # Cursor position from the frame closest to time t
    for when, pos in reversed(history):
        if when <= t:
            return pos
    return history[0][1] if history else cursor


def on_a_face(pts, face_result):
    # True if the centre of the hand sits inside a detected face box
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    for face in face_result.detections:
        box = face.bounding_box
        if (box.origin_x <= cx <= box.origin_x + box.width
                and box.origin_y <= cy <= box.origin_y + box.height):
            return True
    return False


def pause_tracking():
    # Hand missing for a frame: keep buttons and cursor as they are, but don't let the
    # gap turn into a cursor jump when the hand comes back
    global palm_prev, raw_palm_prev
    filter_x.reset()
    filter_y.reset()
    palm_prev, raw_palm_prev = None, None


def let_go():
    # Hand gone for good: never leave a mouse button stuck down, and start fresh when it comes back
    global anchor, hand_pos, pose, pose_candidate
    release_all()
    pause_tracking()
    history.clear()
    anchor, hand_pos = None, None
    pose = pose_candidate = "NORMAL"


while True:
    # STEP 1: SEE
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1) # Mirrors image
    frame_height, frame_width, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # STEP 2: THINK
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    now = time.monotonic()
    timestamp = int(now * 1000)
    results = hands.detect_for_video(mp_image, timestamp)
    # Faces barely move between frames, so checking every few frames is enough and keeps the frame rate up
    if frame_count % FACE_EVERY == 0:
        face_result = faces.detect_for_video(mp_image, timestamp)
    frame_count += 1

    for face in face_result.detections:
        box = face.bounding_box
        cv2.rectangle(img, (box.origin_x, box.origin_y),
                      (box.origin_x + box.width, box.origin_y + box.height), (0, 0, 255), 1)

    mode = "NO HAND"
    hand_landmarks = results.hand_landmarks[0] if results.hand_landmarks else None
    pts = None
    if hand_landmarks:
        pts = [(lm.x * frame_width, lm.y * frame_height) for lm in hand_landmarks]
        raw_palm = (sum(pts[i][0] for i in PALM) / len(PALM), sum(pts[i][1] for i in PALM) / len(PALM))
        if on_a_face(pts, face_result):
            pts = None
            mode = "IGNORED (FACE)"
        elif raw_palm_prev is not None and math.dist(raw_palm, raw_palm_prev) > MAX_JUMP * frame_width:
            # A real hand can't teleport between frames: this is a bad detection, skip the frame.
            # (If the hand really did move, LOST_GRACE runs out and tracking restarts from the new spot.)
            pts = None
            mode = "GLITCH SKIPPED"
        else:
            raw_palm_prev = raw_palm

    if pts is None:
        # Short dropouts are common (especially mid-click), so wait a moment before giving up on the hand
        if now - last_seen > LOST_GRACE:
            let_go()
        elif mode != "GLITCH SKIPPED":
            pause_tracking()
            if mode == "NO HAND":
                mode = "LOST (HOLDING)"
    else:
        last_seen = now
        drawing_utils.draw_landmarks(img, hand_landmarks, HAND_CONNECTIONS)

        # Read the hand: how bent each finger is, how spread the hand is, where the thumb is
        world = results.hand_world_landmarks[0]
        bends = {f: bend(world, k, t) for f, (k, t) in FINGERS.items()}
        hand_size = world_dist(world, WRIST, MIDDLE_KNUCKLE)
        spread = world_dist(world, INDEX_TIP, PINKY_TIP) / hand_size
        thumb_gap = world_dist(world, THUMB_TIP, INDEX_KNUCKLE) / hand_size
        pinch = world_dist(world, THUMB_TIP, INDEX_TIP) / hand_size

        # Switch pose only once the new one has been held for POSE_TIME
        seen = classify(bends, spread, thumb_gap)
        if seen == pose:
            pose_candidate = pose
        elif seen != pose_candidate:
            pose_candidate, candidate_since = seen, now
        elif now - candidate_since >= POSE_TIME:
            change_pose(seen, now)

        if hand_pos is None:
            hand_pos = cursor = real_cursor()

        # How far the palm moved since last frame (smoothed, so it doesn't jitter)
        palm_x = filter_x(raw_palm[0], now)
        palm_y = filter_y(raw_palm[1], now)
        dx = dy = 0.0
        dt = max(now - palm_prev_t, 1e-6)
        if palm_prev is not None:
            dx, dy = palm_x - palm_prev[0], palm_y - palm_prev[1]
        palm_prev, palm_prev_t = (palm_x, palm_y), now

        # STEP 3: ACTION
        if pose == "TWO":
            # Two fingers: scroll, or flick left/right to go back/forward
            mode = "TWO: SCROLL"
            direction = flick(dx, 0.0, dt, frame_width, now)
            if direction == "right":
                shortcut(KEY_LEFT_BRACKET, CMD)   # Cmd+[ = back
                mode = "TWO: BACK"
            elif direction == "left":
                shortcut(KEY_RIGHT_BRACKET, CMD)  # Cmd+] = forward
                mode = "TWO: FORWARD"
            hand_px = math.dist(pts[WRIST], pts[MIDDLE_KNUCKLE])  # Hand size in webcam px
            direction_sign = 1 if NATURAL_SCROLL else -1
            scroll_carry_y += dy / hand_px * SCROLL_SPEED * direction_sign
            if abs(dx) / frame_width / dt < SWIPE_SPEED:  # Slow sideways movement scrolls; fast is a flick
                scroll_carry_x += dx / hand_px * SCROLL_SPEED * direction_sign
            step_x, step_y = int(scroll_carry_x), int(scroll_carry_y)
            if step_x or step_y:
                scroll(step_x, step_y)
                scroll_carry_x -= step_x
                scroll_carry_y -= step_y

        elif pose == "ZOOM":
            # Pinch: thumb and index apart = zoom in, together = zoom out (Cmd+= / Cmd+-, which most apps support)
            mode = "ZOOM"
            if seen != "ZOOM":
                pass  # Hand is changing pose: don't let the fingers moving read as a pinch
            elif zoom_ref is None:
                zoom_ref = pinch
            elif pinch > zoom_ref * (1 + ZOOM_STEP):
                shortcut(KEY_EQUAL, CMD)
                zoom_ref *= 1 + ZOOM_STEP
                mode = "ZOOM: IN"
            elif pinch < zoom_ref / (1 + ZOOM_STEP):
                shortcut(KEY_MINUS, CMD)
                zoom_ref /= 1 + ZOOM_STEP
                mode = "ZOOM: OUT"

        elif pose == "OPEN":
            # Open hand: hold still for Mission Control, flick to switch desktops or show the app's windows
            mode = "OPEN"
            open_travel += math.hypot(dx, dy) / frame_width
            direction = flick(dx, dy, dt, frame_width, now)
            if direction == "left":
                shortcut(KEY_RIGHT, CTRL | ARROW)  # Natural direction: hand left = desktop on the right slides in
                mode, gesture_fired = "OPEN: NEXT DESKTOP", True
            elif direction == "right":
                shortcut(KEY_LEFT, CTRL | ARROW)
                mode, gesture_fired = "OPEN: PREV DESKTOP", True
            elif direction == "down":
                shortcut(KEY_DOWN, CTRL | ARROW)   # App Exposé
                mode, gesture_fired = "OPEN: APP WINDOWS", True
            elif direction == "up":
                mission_control()
                mode, gesture_fired = "OPEN: MISSION CONTROL", True
            elif not gesture_fired and now - pose_since > MISSION_CONTROL_HOLD and open_travel < 0.05:
                mission_control()
                mode, gesture_fired = "OPEN: MISSION CONTROL", True

        else:
            # NORMAL and THREE both move the cursor (in THREE the left button is held, so it drags)
            if dx or dy:
                # Move the cursor by how far the palm moved, faster the quicker the hand moves
                speed = math.hypot(dx, dy) / frame_width / dt
                fast = min(max((speed - SLOW_SPEED) / (FAST_SPEED - SLOW_SPEED), 0), 1)
                gain = SLOW_GAIN + (FAST_GAIN - SLOW_GAIN) * fast
                scale = gain * screen_width / frame_width
                hand_pos = (min(max(hand_pos[0] + dx * scale, 0), screen_width - 1),
                            min(max(hand_pos[1] + dy * scale, 0), screen_height - 1))
            history.append((now, hand_pos))

            # A locked click lets go once the hand moves far away (a drag),
            # or, after the button is released, once CLICK_HOLD runs out
            if anchor is not None:
                if math.dist(hand_pos, anchor) > DRAG_START:
                    anchor = None
                elif not any(held.values()) and now > anchor_until:
                    hand_pos = anchor  # Plain click: drop the small drift from the finger bending
                    anchor = None

            if pose == "NORMAL":
                # Clicking: bend the index finger = left button, bend the middle finger = right button.
                # The button stays down while the finger is bent, so holding it and moving drags.
                # A bend only counts if the neighbouring fingers stay straight for CLICK_CONFIRM seconds:
                # if they curl too, the hand is making a fist or heading into another pose, not clicking.
                for name, finger, neighbours in (("left", "index", ("middle",)), ("right", "middle", ("index", "ring"))):
                    other_name = "right" if name == "left" else "left"
                    neighbour_curled = any(bends[n] > CLICK_PRESS for n in neighbours)
                    if held[name]:
                        if bends[finger] < CLICK_RELEASE:
                            release(name, cursor)
                            if anchor is not None:
                                anchor_until = now + CLICK_HOLD  # Was a click, not a drag: keep still for a double click
                    elif pending[name] is not None:
                        if neighbour_curled or held[other_name]:
                            pending[name] = None  # Not a click after all
                        elif bends[finger] < CLICK_RELEASE:
                            # Quick tap, over before the confirm time: click
                            press(name, cursor)
                            release(name, cursor)
                            pending[name] = None
                            anchor_until = now + CLICK_HOLD
                        elif now - pending[name] >= CLICK_CONFIRM:
                            press(name, cursor)  # Still bent and neighbours still straight: hold the button
                            pending[name] = None
                    elif (not held[other_name] and pending[other_name] is None
                            and bends[finger] > CLICK_PRESS and not neighbour_curled):
                        pending[name] = now
                        # Undo the drift from the finger bending: go back to where the cursor was before it started
                        if anchor is None:
                            anchor = position_before(now - CLICK_LOOKBACK)
                        cursor = anchor
                        move(cursor)

            new_cursor = anchor if anchor is not None else hand_pos
            if new_cursor != cursor:
                cursor = new_cursor
                move(cursor)

            if pose == "THREE":
                mode = "THREE: DRAG"
            elif held["left"]:
                mode = "LEFT DOWN"
            elif held["right"]:
                mode = "RIGHT DOWN"
            else:
                mode = "MOVE"
            if anchor is not None:
                mode += " (LOCKED)"

        # Live hand readings, for tuning the CONFIG values above
        cv2.putText(img, f"index {bends['index']:.0f}  middle {bends['middle']:.0f}  ring {bends['ring']:.0f}  "
                         f"pinky {bends['pinky']:.0f} deg  spread {spread:.2f}  thumb {thumb_gap:.2f}  pinch {pinch:.2f}",
                    (10, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.putText(img, mode, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Frame rate: tracking gets unreliable below ~20 fps (usually a sign of poor lighting)
    if frame_count > 1:
        fps = 0.9 * fps + 0.1 / max(now - prev_frame_time, 1e-6)
    prev_frame_time = now
    cv2.putText(img, f"{fps:.0f} fps", (frame_width - 110, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Shows webcam with overlay
    cv2.imshow("JAVIS PROTO", img)

    if cv2.waitKey(1) & 0xFF == ord('q'): # Quits when q is pressed
        break

# Cleaup
let_go()
cap.release()
hands.close()
faces.close()
cv2.destroyAllWindows()
