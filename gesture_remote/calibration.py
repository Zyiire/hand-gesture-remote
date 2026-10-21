"""Guided setup: measures your hand so click and pose thresholds fit it.

Each step waits until the hand is doing what's asked, then samples it for a moment.
No mouse events are sent while this runs.
"""
import statistics

from . import geometry as g

STEPS = [
    ("relaxed", "Hold one hand up, fingers relaxed and together", 1.5),
    ("index", "Bend your INDEX finger like a mouse click and hold it", 1.0),
    ("middle", "Straighten it, then bend your MIDDLE finger and hold it", 1.0),
    ("spread", "Spread all your fingers wide", 1.0),
]
MIN_SPAN = 15.0  # Degrees between straight and bent needed to trust a finger's measurement


class Calibrator:
    def __init__(self):
        self.index = 0
        self.samples = []
        self.since = None
        self.values = {}
        self.progress = 0.0
        self.message = ""

    @property
    def step_number(self):
        return min(self.index + 1, len(STEPS))

    @property
    def total_steps(self):
        return len(STEPS)

    @property
    def instruction(self):
        return STEPS[min(self.index, len(STEPS) - 1)][1]

    def update(self, frame):
        """Feed one frame. Returns True once every step is done."""
        if not frame.hands:
            self._restart_step("No hand in view: hold one hand up in front of the camera")
            return False
        hand = max(frame.hands, key=lambda h: h.score)
        bends = g.finger_bends(hand.world)
        spread = g.spread(hand.world)
        key, _, duration = STEPS[self.index]
        relaxed = self.values.get("relaxed")

        if key == "relaxed":
            ready = True
        elif key in ("index", "middle"):
            ready = bends[key] > relaxed["bends"][key] + MIN_SPAN
        else:
            ready = spread > relaxed["spread"] * 1.15
        if not ready:
            self._restart_step("Waiting for the pose...")
            return False

        if self.since is None:
            self.since = frame.t
        self.samples.append((bends, spread))
        self.progress = min((frame.t - self.since) / duration, 1.0)
        self.message = "Hold it..."
        if self.progress < 1:
            return False

        self.values[key] = {
            "bends": {f: statistics.median(s[0][f] for s in self.samples) for f in g.FINGERS},
            "spread": statistics.median(s[1] for s in self.samples),
        }
        self.index += 1
        self._restart_step("")
        return self.index >= len(STEPS)

    def _restart_step(self, message):
        self.samples, self.since, self.progress, self.message = [], None, 0.0, message

    def apply(self, settings):
        """Write the measured thresholds into settings. Returns notes about what changed."""
        notes = []
        relaxed = self.values["relaxed"]["bends"]
        for finger in ("index", "middle"):
            straight = relaxed[finger]
            bent = self.values[finger]["bends"][finger]
            span = bent - straight
            if span < MIN_SPAN:
                notes.append(f"{finger}: bend too small to measure, kept defaults")
                continue
            # Press 60% of the way to your bend, release 35% of the way: a clear gap between them
            setattr(settings, f"{finger}_press", round(straight + 0.6 * span, 1))
            setattr(settings, f"{finger}_release", round(straight + 0.35 * span, 1))
            notes.append(f"{finger}: straight {straight:.0f} deg, bent {bent:.0f} deg -> "
                         f"press {getattr(settings, finger + '_press')}, release {getattr(settings, finger + '_release')}")
        # Pose thresholds: "straight" must cover your relaxed hand, "curled" sits above it
        settings.straight_bend = round(max(35.0, max(relaxed.values()) + 10), 1)
        settings.curled_bend = round(max(settings.straight_bend + 10, 50.0), 1)
        relaxed_spread, wide = self.values["relaxed"]["spread"], self.values["spread"]["spread"]
        settings.spread_open = round((relaxed_spread + wide) / 2, 2)
        notes.append(f"spread: relaxed {relaxed_spread:.2f}, wide {wide:.2f} -> threshold {settings.spread_open}")
        settings.calibrated = True
        return notes
