"""What the camera saw in one frame, in a form that can be saved and replayed.

The gesture engine only ever sees these objects, never the camera, so recorded sessions
can be replayed through it to test changes deterministically.
"""
import json
from dataclasses import asdict, dataclass, field


@dataclass
class Hand:
    points: list   # 21 (x, y) landmarks, normalised 0-1 in the mirrored webcam image
    world: list    # 21 (x, y, z) landmarks in metres, centred on the hand (MediaPipe world landmarks)
    handedness: str = "Unknown"  # "Left" or "Right" (from the user's point of view)
    score: float = 1.0           # Detection confidence, 0-1


@dataclass
class Frame:
    t: float      # Capture time in seconds (monotonic clock)
    width: int    # Webcam image size in px
    height: int
    hands: list = field(default_factory=list)  # Hand objects
    faces: list = field(default_factory=list)  # Face boxes as normalised (x, y, w, h)
    cursor: tuple = None  # Real cursor position (screen points) when the frame was captured

    def to_json(self):
        return json.dumps(asdict(self), separators=(",", ":"))

    @staticmethod
    def from_json(line):
        data = json.loads(line)
        data["hands"] = [Hand(**{**h, "points": [tuple(p) for p in h["points"]],
                                 "world": [tuple(p) for p in h["world"]]}) for h in data["hands"]]
        data["faces"] = [tuple(box) for box in data["faces"]]
        if data.get("cursor") is not None:
            data["cursor"] = tuple(data["cursor"])
        return Frame(**data)
