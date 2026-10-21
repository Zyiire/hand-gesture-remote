"""Builds fake hand observations with chosen finger bends, for testing the engine without a camera."""
import math

from gesture_remote.observation import Frame, Hand

FPS = 30
KNUCKLE_X = {5: -0.3, 9: 0.0, 13: 0.25, 17: 0.45}  # Knuckle x positions (hand sizes)
FAN = (-0.35, 0.0, 0.3, 0.6)                        # Sideways finger angle when spread


def hand(x, y, bends=(5, 5, 5, 5), spread=False, thumb_out=False, handedness="Right", score=0.95):
    """A hand with its palm near (x, y) in normalised image coordinates.
    bends = (index, middle, ring, pinky) in degrees."""
    world = [(0.0, 0.0, 0.0)] * 21
    for (knuckle, kx), angle, fan in zip(KNUCKLE_X.items(), bends, FAN):
        a = math.radians(angle)
        d = [fan if spread else 0.0, -math.cos(a), math.sin(a)]
        n = math.sqrt(sum(c * c for c in d))
        d = [c / n * 0.8 for c in d]
        world[knuckle] = (kx, -1.0, 0.0)
        for i in (1, 2, 3):
            world[knuckle + i] = (kx + d[0] * i / 3, -1.0 + d[1] * i / 3, d[2] * i / 3)
    world[4] = (-1.1, -0.4, 0.0) if thumb_out else (-0.45, -0.75, 0.0)
    world[1], world[2], world[3] = (-0.3, -0.2, 0.0), (-0.5, -0.35, 0.0), (-0.6, -0.5, 0.0)
    points = [(x + wx * 0.06, y + 0.15 + wy * 0.08) for wx, wy, _ in world]
    return Hand(points=points, world=world, handedness=handedness, score=score)


OPEN = {}
CLICK = {"bends": (75, 5, 5, 5)}
RIGHT_CLICK = {"bends": (5, 75, 5, 5)}
FIST = {"bends": (90, 90, 90, 90)}
TWO = {"bends": (5, 5, 85, 85)}
THREE = {"bends": (5, 5, 5, 85)}
SPREAD = {"spread": True, "thumb_out": True}


class Session:
    """Feeds frames to an engine at a fixed frame rate and collects what comes out."""

    def __init__(self, engine, cursor=(700.0, 400.0)):
        self.engine = engine
        self.t = 100.0
        self.cursor = cursor
        self.actions = []
        self.status = None

    def frame(self, hands, faces=()):
        frame = Frame(t=self.t, width=640, height=480, hands=list(hands), faces=list(faces), cursor=self.cursor)
        actions, self.status = self.engine.update(frame)
        self.actions.extend(actions)
        self.t += 1 / FPS
        return actions

    def hold(self, seconds, x=0.5, y=0.5, **pose):
        out = []
        for _ in range(round(seconds * FPS)):
            out += self.frame([hand(x, y, **pose)])
        return out

    def glide(self, seconds, start, end, **pose):
        out = []
        n = round(seconds * FPS)
        for i in range(n):
            f = i / max(n - 1, 1)
            out += self.frame([hand(start[0] + (end[0] - start[0]) * f, start[1] + (end[1] - start[1]) * f, **pose)])
        return out

    def nothing(self, seconds):
        out = []
        for _ in range(round(seconds * FPS)):
            out += self.frame([])
        return out

    def two_hands(self, seconds, gap_start, gap_end, pose_a=None, pose_b=None):
        out = []
        n = round(seconds * FPS)
        for i in range(n):
            gap = gap_start + (gap_end - gap_start) * i / max(n - 1, 1)
            out += self.frame([hand(0.5 - gap / 2, 0.5, **(pose_a or {})),
                               hand(0.5 + gap / 2, 0.5, handedness="Left", **(pose_b or {}))])
        return out

    def arm(self):
        """Hold an open hand long enough to take control."""
        self.hold(self.engine.s.arm_time + 0.2)
