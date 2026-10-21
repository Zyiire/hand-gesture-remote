"""Turns timestamped hand observations into semantic actions.

Deterministic: all timing comes from frame timestamps and nothing here touches the camera or
the OS, so recorded sessions replay identically and every rule can be unit-tested.

States
    PAUSED     User paused control (keyboard or on-screen button). Frames are ignored.
    IDLE       No hand in control. Hold an open hand for `arm_time` to take control.
    ARMING     Open hand being held; `Status.progress` shows how close it is.
    POINTER    The hand moves the cursor.
    CLICKING   A finger bend is being confirmed, or a button is held with the cursor locked.
    DRAGGING   A button is held and the cursor is moving.
    PARKED     Fist: the cursor stays put so the hand can be repositioned (like lifting a mouse).
    SCROLLING  Two fingers up.
    SYSTEM     Spread hand, for opt-in system gestures (Mission Control, desktops...).
    ZOOMING    Two open hands.
    LOST       Hand not tracked. Movement freezes; held buttons are released after `lost_grace`;
               control switches off (back to IDLE) after `disarm_after`.
"""
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from . import geometry as g
from .actions import ButtonDown, ButtonUp, Command, MoveCursor, Scroll, ZoomStep
from .filters import OneEuroFilter
from .metrics import Metrics

STILL_SPEED = 0.03  # Hand speed (webcam widths per second) below which the hand counts as held still
HOLD_TRAVEL = 0.05  # Movement (webcam widths) that cancels a timed hold


class State(str, Enum):
    PAUSED = "PAUSED"
    IDLE = "IDLE"
    ARMING = "ARMING"
    POINTER = "POINTER"
    CLICKING = "CLICKING"
    DRAGGING = "DRAGGING"
    PARKED = "PARKED"
    SCROLLING = "SCROLLING"
    SYSTEM = "SYSTEM GESTURE"
    ZOOMING = "ZOOMING"
    LOST = "TRACKING LOST"


POINTER_STATES = {State.POINTER, State.CLICKING, State.DRAGGING}
ACTIVE_STATES = POINTER_STATES | {State.PARKED, State.SCROLLING, State.SYSTEM, State.ZOOMING}


@dataclass
class Status:
    """What the engine is doing, for on-screen feedback."""
    state: State
    detail: str = ""        # e.g. "LEFT DOWN", "ZOOM +3"
    progress: float = None  # 0-1 for timed holds (arming, spread hold)
    readings: dict = field(default_factory=dict)  # Live numbers for the debug view
    hands: list = field(default_factory=list)     # Hands in use (lists of px points), for drawing
    note: str = ""          # e.g. why a detection was ignored


class PoseTracker:
    """A pose only counts once it has been seen continuously for `hold` seconds (stops flicker)."""

    def __init__(self, hold):
        self.hold = hold
        self.reset()

    def reset(self, pose=None):
        self.stable, self.candidate, self.since = pose, None, 0.0

    def update(self, pose, t):
        if self.stable is None:
            self.stable = pose
        if pose == self.stable:
            self.candidate = None
        elif pose != self.candidate:
            self.candidate, self.since = pose, t
        elif t - self.since >= self.hold:
            self.stable, self.candidate = pose, None
        return self.stable


class GestureEngine:
    def __init__(self, settings, screen_size, metrics=None):
        self.s = settings
        self.screen_w, self.screen_h = screen_size
        self.metrics = metrics if metrics is not None else Metrics()
        self.state = State.IDLE
        self.poses = PoseTracker(settings.pose_time)
        self.filter_x = OneEuroFilter(settings.min_cutoff, settings.beta)
        self.filter_y = OneEuroFilter(settings.min_cutoff, settings.beta)
        self.palm_prev = None        # Smoothed palm position last frame (webcam px)
        self.palm_prev_t = 0.0
        self.raw_palm_prev = None    # Unsmoothed palm position last frame, for spotting bad detections
        self.lost_since = None
        self.resume_state = State.POINTER
        self.arm_start = None
        self.cursor = (self.screen_w / 2, self.screen_h / 2)  # Where the cursor is
        self.hand_pos = self.cursor  # Where the hand is steering it (differs while a click is locked)
        self.history = deque(maxlen=60)  # (time, hand_pos)
        self.anchor = None           # Position the cursor is locked to around a click
        self.anchor_until = 0.0
        self.held = {"left": False, "right": False}
        self.pending = {"left": None, "right": None}  # When a finger bent, while confirming it's a click
        self.last_press = {"left": (-math.inf, (0, 0), 0), "right": (-math.inf, (0, 0), 0)}
        self.three_drag = False
        self.swipe = [0.0, 0.0]
        self.swipe_ready_at = 0.0
        self.system_since = 0.0
        self.system_travel = 0.0
        self.system_fired = False
        self.scroll_carry = [0.0, 0.0]
        self.zoom_candidate_since = None
        self.zoom_missing_since = None
        self.zoom_base = None        # Hand gap that counts as zoom level 0
        self.zoom_raw = 0.0          # Zoom level the hands are showing (continuous)
        self.zoom_level = 0.0        # Zoom level already sent
        self.zoom_clutched = False
        self.zoom_carry = 0.0
        self._out = []
        self.status = Status(self.state)

    # --- Public API ---

    def update(self, frame):
        """Process one frame. Returns (actions, status)."""
        self._out = []
        self.status = Status(self.state)
        if self.state != State.PAUSED:
            self._step(frame)
        self.status.state = self.state
        self.metrics.track_frame(frame.t, self.state.value)
        return self._out, self.status

    def pause(self):
        """Stop control immediately, releasing anything held. Returns the release actions."""
        self._out = []
        self._release_all()
        self._reset_tracking()
        self.state = State.PAUSED
        self.metrics.count("pauses")
        return self._out

    def resume(self):
        if self.state == State.PAUSED:
            self.state, self.arm_start = State.IDLE, None

    def cancel(self):
        """Drop control (e.g. permission lost), releasing anything held. Returns the release actions."""
        self._out = []
        self._deactivate()
        return self._out

    # --- Frame processing ---

    def _step(self, frame):
        s, t, width, height = self.s, frame.t, frame.width, frame.height
        faces_px = [(x * width, y * height, w * width, h * height) for x, y, w, h in frame.faces]
        hands = []
        for hand in frame.hands:
            pts = g.to_pixels(hand, width, height)
            if hand.score < s.min_hand_score:
                self.status.note = "LOW CONFIDENCE HAND IGNORED"
                self.metrics.count("low_confidence_ignored")
            elif g.on_a_face(pts, faces_px):
                self.status.note = "HAND ON FACE IGNORED"
                self.metrics.count("face_rejections")
            else:
                hands.append((hand, pts))

        if s.two_hand_zoom and self._zoom(frame, hands):
            return

        picked = self._pick_pointer_hand(hands)
        if picked is None:
            self._missing(t, glitch=False)
            return
        hand, pts = picked

        raw_palm = g.palm_centre(pts)
        if self.raw_palm_prev is not None and math.dist(raw_palm, self.raw_palm_prev) > s.max_jump * width:
            # A real hand can't teleport between frames: this is a bad detection, skip it.
            # (If the hand really did move, disarm/recovery restarts tracking from the new spot.)
            self.metrics.count("glitches_skipped")
            self._missing(t, glitch=True)
            return
        self.raw_palm_prev = raw_palm
        self._found()
        self.status.hands = [pts]

        bends = g.finger_bends(hand.world)
        spread, thumb = g.spread(hand.world), g.thumb_gap(hand.world)
        pose = g.classify(bends, spread, thumb, s)
        stable = self.poses.update(pose, t)
        self.status.readings = {**{f: round(b) for f, b in bends.items()},
                                "spread": round(spread, 2), "thumb": round(thumb, 2), "pose": pose}

        # Palm movement since last frame (smoothed)
        palm = (self.filter_x(raw_palm[0], t), self.filter_y(raw_palm[1], t))
        dx = dy = 0.0
        dt = max(t - self.palm_prev_t, 1e-6)
        if self.palm_prev is not None:
            dx, dy = palm[0] - self.palm_prev[0], palm[1] - self.palm_prev[1]
        self.palm_prev, self.palm_prev_t = palm, t

        if self.state in (State.IDLE, State.ARMING):
            self._arming(pose, frame)
        elif self.state in POINTER_STATES:
            self._pointer(stable, bends, dx, dy, dt, width, t)
        elif self.state == State.PARKED:
            if stable != "fist":
                self.state = State.POINTER
        elif self.state == State.SCROLLING:
            self._scrolling(stable, dx, dy, dt, width, g.hand_px(pts), t)
        elif self.state == State.SYSTEM:
            self._system(stable, dx, dy, dt, width, t)
        # ZOOMING with one hand: waiting for _zoom to end the zoom session

    def _pick_pointer_hand(self, hands):
        if self.s.pointer_hand in ("left", "right"):
            hands = [h for h in hands if h[0].handedness.lower() == self.s.pointer_hand]
        if not hands:
            return None
        if self.raw_palm_prev is None or len(hands) == 1:
            return hands[0]
        # Two hands but no zoom: stay with the hand closest to where the pointer hand was
        return min(hands, key=lambda h: math.dist(g.palm_centre(h[1]), self.raw_palm_prev))

    # --- Tracking loss ---

    def _missing(self, t, glitch):
        self.status.note = "GLITCH SKIPPED" if glitch else "NO HAND"
        if self.state in (State.IDLE, State.ARMING):
            self.state, self.arm_start = State.IDLE, None
            self._reset_tracking()
            return
        if self.state != State.LOST:
            self.resume_state, self.state, self.lost_since = self.state, State.LOST, t
            self.metrics.count("tracking_lost")
            self._reset_motion()
        if not glitch:
            self.raw_palm_prev = None  # The hand may come back anywhere
        lost_for = t - self.lost_since
        if lost_for > self.s.lost_grace and (any(self.held.values()) or any(self.pending.values())):
            self._release_all()
            self.metrics.count("buttons_released_on_loss")
        if lost_for > self.s.disarm_after:
            self._deactivate()

    def _found(self):
        if self.state == State.LOST:
            back = self.resume_state
            # A scroll / system / zoom gesture is over once the hand was lost; so is a click whose
            # button was already released
            if back not in POINTER_STATES | {State.PARKED} or (back in POINTER_STATES and not any(self.held.values())):
                back = State.POINTER
            self.state = back
            self.poses.reset()
        self.lost_since = None

    # --- Activation ---

    def _arming(self, pose, frame):
        if pose not in ("open", "spread"):
            self.state, self.arm_start = State.IDLE, None
            return
        if self.arm_start is None:
            self.arm_start = frame.t
        progress = (frame.t - self.arm_start) / self.s.arm_time
        self.state = State.ARMING
        self.status.progress = min(progress, 1.0)
        if progress >= 1:
            self.state = State.POINTER
            self.cursor = self.hand_pos = frame.cursor or self.cursor
            self.history.clear()
            self.poses.reset(pose)
            self.metrics.count("armed")

    def _deactivate(self):
        self._release_all()
        self._reset_tracking()
        if self.state not in (State.IDLE, State.PAUSED):
            self.metrics.count("disarmed")
        self.state, self.arm_start, self.lost_since = State.IDLE, None, None

    # --- Pointer, clicks, drags ---

    def _pointer(self, stable, bends, dx, dy, dt, width, t):
        if self.three_drag and stable != "three":
            self._release_all()
        if stable == "fist":
            self._release_all()
            self.state = State.PARKED
            self.metrics.count("parked")
            return
        busy = any(self.held.values()) or any(p is not None for p in self.pending.values())
        if not busy and not self.three_drag:
            if stable == "two":
                self.state = State.SCROLLING
                self.scroll_carry, self.swipe = [0.0, 0.0], [0.0, 0.0]
                self.metrics.count("scroll_sessions")
                return
            if stable == "spread" and any(self.s.mappings.get(k) for k in self.s.mappings if k.startswith("spread_")):
                self.state = State.SYSTEM
                self.system_since, self.system_travel, self.system_fired = t, 0.0, False
                self.swipe = [0.0, 0.0]
                return
            if stable == "three" and self.s.three_finger_drag:
                self.anchor = None
                self._press("left", t)
                self.three_drag = True
                self.metrics.count("drags")

        self._move_hand(dx, dy, dt, width, t)
        if not self.three_drag:
            self._clicks(bends, t)
        self._update_cursor()

        if self.three_drag or (any(self.held.values()) and self.anchor is None):
            self.state = State.DRAGGING
        elif any(self.held.values()) or any(p is not None for p in self.pending.values()):
            self.state = State.CLICKING
        else:
            self.state = State.POINTER
        held = [b.upper() for b, h in self.held.items() if h]
        self.status.detail = ("THREE-FINGER DRAG" if self.three_drag
                              else f"{' + '.join(held)} DOWN" if held
                              else "CONFIRMING CLICK" if self.state == State.CLICKING else "")

    def _move_hand(self, dx, dy, dt, width, t):
        s = self.s
        if dx or dy:
            # Move by how far the palm moved, faster the quicker the hand moves (pointer acceleration)
            speed = math.hypot(dx, dy) / width / dt
            fast = min(max((speed - s.slow_speed) / (s.fast_speed - s.slow_speed), 0), 1)
            gain = (s.slow_gain + (s.fast_gain - s.slow_gain) * fast) * s.sensitivity
            scale = gain * self.screen_w / width
            if speed < STILL_SPEED and self.state == State.POINTER:
                self.metrics.add_jitter(math.hypot(dx, dy) * scale)
            self.hand_pos = (min(max(self.hand_pos[0] + dx * scale, 0), self.screen_w - 1),
                             min(max(self.hand_pos[1] + dy * scale, 0), self.screen_h - 1))
        self.history.append((t, self.hand_pos))

        # A locked click lets go once the hand moves far away (a drag),
        # or, after the button is released, once click_hold runs out
        if self.anchor is not None:
            if math.dist(self.hand_pos, self.anchor) > s.drag_start:
                if any(self.held.values()):
                    self.metrics.count("drags")
                self.anchor = None
            elif not any(self.held.values()) and not any(self.pending.values()) and t > self.anchor_until:
                self.hand_pos = self.anchor  # Plain click: drop the small drift from the finger bending
                self.anchor = None

    def _clicks(self, bends, t):
        # Bend the index finger = left button, bend the middle finger = right button.
        # The button stays down while the finger is bent, so holding it and moving drags.
        # A bend only counts if the neighbouring fingers stay straight for click_confirm seconds:
        # if they curl too, the hand is making a fist or another pose, not clicking.
        s = self.s
        for name, finger, neighbours, press_at, release_at in (
                ("left", "index", ("middle",), s.index_press, s.index_release),
                ("right", "middle", ("index", "pinky"), s.middle_press, s.middle_release)):
            other = "right" if name == "left" else "left"
            neighbour_curled = any(bends[n] > s.curled_bend for n in neighbours)
            if self.held[name]:
                if bends[finger] < release_at:
                    self._release(name)
                    if self.anchor is not None:
                        self.anchor_until = t + s.click_hold  # A click, not a drag: stay put for a double click
            elif self.pending[name] is not None:
                if neighbour_curled or self.held[other]:
                    self.pending[name] = None
                    self.metrics.count("clicks_cancelled")
                elif bends[finger] < release_at:
                    # Quick tap, over before the confirm time: click
                    self._press(name, t)
                    self._release(name)
                    self.pending[name] = None
                    self.anchor_until = t + s.click_hold
                elif t - self.pending[name] >= s.click_confirm:
                    self._press(name, t)  # Still bent, neighbours still straight: hold the button
                    self.pending[name] = None
            elif (not self.held[other] and self.pending[other] is None
                    and bends[finger] > press_at and not neighbour_curled):
                self.pending[name] = t
                # Undo the drift from the finger bending: go back to where the cursor was before it started
                if self.anchor is None:
                    self.anchor = self._position_before(t - s.click_lookback)

    def _position_before(self, t):
        for when, pos in reversed(self.history):
            if when <= t:
                return pos
        return self.history[0][1] if self.history else self.cursor

    def _update_cursor(self):
        target = self.anchor if self.anchor is not None else self.hand_pos
        if target != self.cursor:
            self.cursor = target
            self._out.append(MoveCursor(*target))

    def _press(self, name, t):
        last_t, last_pos, count = self.last_press[name]
        # Presses close together in time and space count as a double (or triple) click
        if t - last_t < self.s.double_click_time and math.dist(self.cursor, last_pos) < self.s.double_click_distance:
            count += 1
        else:
            count = 1
        self.last_press[name] = (t, self.cursor, count)
        self.held[name] = True
        self._out.append(ButtonDown(name, *self.cursor, count))
        self.metrics.count(f"{name}_presses")

    def _release(self, name):
        self.held[name] = False
        self._out.append(ButtonUp(name, *self.cursor, self.last_press[name][2]))

    def _release_all(self):
        for name in self.held:
            if self.held[name]:
                self._release(name)
            self.pending[name] = None
        self.three_drag = False

    # --- Scrolling and system gestures ---

    def _scrolling(self, stable, dx, dy, dt, width, hand_size, t):
        s = self.s
        if stable != "two":
            self.state = State.PARKED if stable == "fist" else State.POINTER
            return
        self.status.detail = "SCROLL"
        flicks_on = s.mappings.get("two_finger_flick_left") or s.mappings.get("two_finger_flick_right")
        if flicks_on:
            direction = self._flick(dx, 0.0, dt, width, t)
            if direction in ("left", "right"):
                self._command(f"two_finger_flick_{direction}")
        sign = 1 if s.natural_scroll else -1
        self.scroll_carry[1] += dy / hand_size * s.scroll_speed * sign
        if not (flicks_on and abs(dx) / width / dt >= s.swipe_speed):  # Fast sideways = flick, not scroll
            self.scroll_carry[0] += dx / hand_size * s.scroll_speed * sign
        step_x, step_y = int(self.scroll_carry[0]), int(self.scroll_carry[1])
        if step_x or step_y:
            self._out.append(Scroll(step_x, step_y))
            self.scroll_carry[0] -= step_x
            self.scroll_carry[1] -= step_y

    def _system(self, stable, dx, dy, dt, width, t):
        s = self.s
        if stable != "spread":
            self.state = State.PARKED if stable == "fist" else State.POINTER
            return
        self.system_travel += math.hypot(dx, dy) / width
        direction = self._flick(dx, dy, dt, width, t)
        if direction and self._command(f"spread_flick_{direction}"):
            self.system_fired = True
        elif not self.system_fired and s.mappings.get("spread_hold") and self.system_travel < HOLD_TRAVEL:
            progress = (t - self.system_since) / s.system_hold
            self.status.progress = min(progress, 1.0)
            if progress >= 1:
                self._command("spread_hold")
                self.system_fired = True

    def _flick(self, dx, dy, dt, width, t):
        # "left", "right", "up" or "down" once a quick flick has travelled far enough, else None
        s = self.s
        if math.hypot(dx, dy) / width / dt > s.swipe_speed:
            self.swipe[0] += dx / width
            self.swipe[1] += dy / width
        else:
            self.swipe = [0.0, 0.0]
        if t < self.swipe_ready_at:
            return None
        sx, sy = self.swipe
        direction = None
        if abs(sx) > s.swipe_distance and abs(sx) > abs(sy):
            direction = "right" if sx > 0 else "left"
        elif abs(sy) > s.swipe_distance:
            direction = "down" if sy > 0 else "up"
        if direction:
            self.swipe = [0.0, 0.0]
            self.swipe_ready_at = t + s.swipe_cooldown
        return direction

    def _command(self, gesture):
        name = self.s.mappings.get(gesture)
        if not name:
            return False
        self._out.append(Command(name))
        self.status.detail = name.replace("_", " ").upper()
        self.metrics.count(f"command:{name}")
        return True

    # --- Two-hand zoom ---

    def _zoom(self, frame, hands):
        """Two-hand zoom. Returns True when it handled this frame."""
        s, t = self.s, frame.t
        if self.state == State.ZOOMING:
            if len(hands) < 2:
                if self.zoom_missing_since is None:
                    self.zoom_missing_since = t
                if t - self.zoom_missing_since < s.pose_time:
                    self.status.detail = "ZOOM (WAITING FOR BOTH HANDS)"
                    return True
                self._end_zoom()
                return False
            self.zoom_missing_since = None
        else:
            # Start only from active control, with both hands open for pose_time and nothing held
            if (self.state not in ACTIVE_STATES or len(hands) < 2 or any(self.held.values())
                    or self.three_drag or not all(self._pose_of(h) in ("open", "spread") for h, _ in hands[:2])):
                self.zoom_candidate_since = None
                return False
            if self.zoom_candidate_since is None:
                self.zoom_candidate_since = t
            if t - self.zoom_candidate_since < s.pose_time:
                return False
            self._start_zoom()

        (hand_a, pts_a), (hand_b, pts_b) = hands[:2]
        self.lost_since = None
        self.status.hands = [pts_a, pts_b]
        palm_a, palm_b = g.palm_centre(pts_a), g.palm_centre(pts_b)
        gap = math.dist(palm_a, palm_b) / ((g.hand_px(pts_a) + g.hand_px(pts_b)) / 2)
        both_open = all(self._pose_of(h) in ("open", "spread") for h in (hand_a, hand_b))
        self.status.readings = {"hands gap": round(gap, 2), "zoom level": round(self.zoom_raw, 2)}

        if not both_open:
            # Clutch: hands closed, zoom holds still while they're repositioned
            self.zoom_clutched = True
            self.status.detail = f"ZOOM {self.zoom_level:+.0f} (CLUTCH - open hands to continue)"
            return True
        if self.zoom_base is None or self.zoom_clutched:
            # Start (or re-centre after a clutch) so the current hand gap means the current zoom level
            self.zoom_base = gap / (1 + s.zoom_step) ** self.zoom_raw
            self.zoom_clutched = False

        # Continuous ratio: current gap / starting gap, in zoom steps, within bounds
        limit = s.zoom_max_steps
        self.zoom_raw = min(max(math.log(gap / self.zoom_base) / math.log(1 + s.zoom_step), -limit), limit)
        if s.zoom_output == "ctrl_scroll":
            # Smooth zoom with a dead zone of half a step around the starting gap
            target = 0.0 if abs(self.zoom_raw) < 0.5 else self.zoom_raw - math.copysign(0.5, self.zoom_raw)
            self.zoom_carry += (target - self.zoom_level) * s.zoom_scroll_pixels
            self.zoom_level = target
            pixels = int(self.zoom_carry)
            if pixels:
                self._out.append(Scroll(0, pixels, zoom=True))
                self.zoom_carry -= pixels
        else:
            # One shortcut per step; the extra margin (hysteresis) stops flicker at a step boundary,
            # and the first step needs 0.5 + margin steps of change (the dead zone)
            margin = 0.5 + s.zoom_hysteresis
            while self.zoom_raw - self.zoom_level > margin and self.zoom_level < limit:
                self.zoom_level += 1
                self._out.append(ZoomStep(+1))
                self.metrics.count("zoom_steps")
            while self.zoom_level - self.zoom_raw > margin and self.zoom_level > -limit:
                self.zoom_level -= 1
                self._out.append(ZoomStep(-1))
                self.metrics.count("zoom_steps")
        self.status.detail = f"ZOOM {self.zoom_level:+.1f}" if s.zoom_output == "ctrl_scroll" else f"ZOOM {self.zoom_level:+.0f}"
        return True

    def _pose_of(self, hand):
        return g.classify(g.finger_bends(hand.world), g.spread(hand.world), g.thumb_gap(hand.world), self.s)

    def _start_zoom(self):
        self._release_all()
        self.anchor = None  # The cursor stays where it is: zoom is anchored on the pointer
        self.state = State.ZOOMING
        self.zoom_base, self.zoom_raw, self.zoom_level = None, 0.0, 0.0
        self.zoom_clutched, self.zoom_carry, self.zoom_missing_since = False, 0.0, None
        self.metrics.count("zoom_sessions")

    def _end_zoom(self):
        self.state = State.POINTER
        self.zoom_candidate_since = self.zoom_missing_since = None
        self._reset_motion()
        self.raw_palm_prev = None  # Either hand may carry on as the pointer
        self.poses.reset()

    # --- Housekeeping ---

    def _reset_motion(self):
        self.filter_x.reset()
        self.filter_y.reset()
        self.palm_prev = None

    def _reset_tracking(self):
        self._reset_motion()
        self.raw_palm_prev = None
        self.poses.reset()
        self.history.clear()
        self.anchor = None
