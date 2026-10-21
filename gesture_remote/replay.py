"""Replay a recorded session through the gesture engine, without a camera or any real events.

    python -m gesture_remote.replay recordings/<file>.jsonl [--actions] [--settings settings.json]

Use it to check a change to the gesture rules against real recordings, or to count false
activations: record yourself reading, talking and touching your face without meaning to
control anything, then replay it. Every action reported is a false activation.
"""
import argparse
from collections import Counter

from . import config
from .engine import GestureEngine
from .metrics import Metrics, format_summary
from .observation import Frame


def replay(path, settings, screen=(1440, 900), show_actions=False):
    metrics = Metrics()
    engine = GestureEngine(settings, screen, metrics)
    actions_seen = Counter()
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            frame = Frame.from_json(line)
            actions, status = engine.update(frame)
            for action in actions:
                actions_seen[type(action).__name__] += 1
                if show_actions:
                    print(f"{frame.t:10.3f}  {status.state.value:<15} {action}")
    return metrics, actions_seen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording")
    parser.add_argument("--settings", default=str(config.SETTINGS_PATH))
    parser.add_argument("--actions", action="store_true", help="print every action")
    parser.add_argument("--screen", default="1440x900", help="screen size, e.g. 1470x956")
    args = parser.parse_args()
    width, height = (int(v) for v in args.screen.lower().split("x"))
    metrics, actions_seen = replay(args.recording, config.load(args.settings), (width, height), args.actions)
    print(format_summary(metrics.summary()))
    print("Actions: " + (", ".join(f"{k} {v}" for k, v in actions_seen.items()) or "none"))


if __name__ == "__main__":
    main()
