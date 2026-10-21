"""Pure measurements of a hand (no state, no timing)."""
import math

import numpy as np

# Hand landmark ids (21 points per hand)
WRIST, THUMB_TIP = 0, 4
INDEX_KNUCKLE, INDEX_TIP = 5, 8
MIDDLE_KNUCKLE, MIDDLE_TIP = 9, 12
RING_KNUCKLE, RING_TIP = 13, 16
PINKY_KNUCKLE, PINKY_TIP = 17, 20
PALM = (WRIST, INDEX_KNUCKLE, MIDDLE_KNUCKLE, RING_KNUCKLE, PINKY_KNUCKLE)
FINGERS = {"index": (INDEX_KNUCKLE, INDEX_TIP), "middle": (MIDDLE_KNUCKLE, MIDDLE_TIP),
           "ring": (RING_KNUCKLE, RING_TIP), "pinky": (PINKY_KNUCKLE, PINKY_TIP)}


def to_pixels(hand, width, height):
    return [(x * width, y * height) for x, y in hand.points]


def palm_centre(pts):
    # The palm moves less than any fingertip when fingers bend, so it's the steadiest point to track
    return (sum(pts[i][0] for i in PALM) / len(PALM), sum(pts[i][1] for i in PALM) / len(PALM))


def hand_px(pts):
    # Hand size in px (wrist to middle knuckle), for measurements that work at any distance from the camera
    return max(math.dist(pts[WRIST], pts[MIDDLE_KNUCKLE]), 1e-6)


def bend(world, knuckle, tip):
    # Angle (degrees) between the back of the hand (wrist -> knuckle) and the finger (knuckle -> tip).
    # Uses 3D points, so it stays the same when the hand tilts toward or away from the camera.
    w, k, t = (np.array(world[i]) for i in (WRIST, knuckle, tip))
    palm_dir, finger_dir = k - w, t - k
    cos = palm_dir.dot(finger_dir) / max(np.linalg.norm(palm_dir) * np.linalg.norm(finger_dir), 1e-9)
    return math.degrees(math.acos(np.clip(cos, -1, 1)))


def finger_bends(world):
    return {name: bend(world, k, t) for name, (k, t) in FINGERS.items()}


def _world_size(world):
    return max(math.dist(world[WRIST], world[MIDDLE_KNUCKLE]), 1e-9)


def spread(world):
    # Index-to-pinky fingertip gap, in hand sizes
    return math.dist(world[INDEX_TIP], world[PINKY_TIP]) / _world_size(world)


def thumb_gap(world):
    # Thumb tip to index knuckle, in hand sizes (small = thumb tucked in)
    return math.dist(world[THUMB_TIP], world[INDEX_KNUCKLE]) / _world_size(world)


def on_a_face(pts, faces_px):
    # True if the centre of the hand sits inside a face box (x, y, w, h in px)
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return any(x <= cx <= x + w and y <= cy <= y + h for x, y, w, h in faces_px)


def classify(bends, spread_value, thumb_value, settings):
    """The hand's pose: "spread", "open", "two", "three", "fist" or "other"."""
    straight = {f: b < settings.straight_bend for f, b in bends.items()}
    curled = {f: b > settings.curled_bend for f, b in bends.items()}
    if all(straight.values()):
        if spread_value > settings.spread_open and thumb_value > settings.thumb_out:
            return "spread"
        return "open"
    if all(curled.values()):
        return "fist"
    if straight["index"] and straight["middle"] and straight["ring"] and curled["pinky"]:
        return "three"
    if straight["index"] and straight["middle"] and curled["ring"] and curled["pinky"]:
        return "two"
    return "other"
