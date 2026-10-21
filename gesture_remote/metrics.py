"""Session measurements: latency, cursor jitter, gesture counts, tracking losses, time per state."""
import json
import math
import time
from collections import Counter, deque


class Metrics:
    def __init__(self):
        self.counts = Counter()
        self.latency = deque(maxlen=20000)   # Seconds from frame capture to events posted
        self.jitter = deque(maxlen=20000)    # Cursor px moved per frame while the hand was held still
        self.state_time = Counter()          # Seconds spent in each state
        self.first_t = self.last_t = None
        self.last_state = None
        self.frames = 0

    def count(self, name, n=1):
        self.counts[name] += n

    def add_latency(self, seconds):
        self.latency.append(seconds)

    def add_jitter(self, px):
        self.jitter.append(px)

    def track_frame(self, t, state):
        if self.last_t is not None and self.last_state is not None:
            self.state_time[self.last_state] += t - self.last_t
        if self.first_t is None:
            self.first_t = t
        self.last_t, self.last_state = t, state
        self.frames += 1

    def summary(self):
        duration = (self.last_t - self.first_t) if self.frames > 1 else 0.0
        lat = sorted(self.latency)
        return {
            "duration_s": round(duration, 1),
            "frames": self.frames,
            "avg_fps": round(self.frames / duration, 1) if duration else None,
            "latency_ms_median": round(lat[len(lat) // 2] * 1000, 1) if lat else None,
            "latency_ms_p95": round(lat[int(len(lat) * 0.95)] * 1000, 1) if lat else None,
            "still_jitter_px_rms": (round(math.sqrt(sum(j * j for j in self.jitter) / len(self.jitter)), 2)
                                    if self.jitter else None),
            "counts": dict(sorted(self.counts.items())),
            "seconds_per_state": {k: round(v, 1) for k, v in sorted(self.state_time.items())},
        }

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2) + "\n")


def format_summary(summary):
    lines = [f"Session: {summary['duration_s']}s, {summary['frames']} frames, {summary['avg_fps']} fps avg"]
    if summary["latency_ms_median"] is not None:
        lines.append(f"Latency (capture -> event): median {summary['latency_ms_median']} ms, "
                     f"95th percentile {summary['latency_ms_p95']} ms")
    if summary["still_jitter_px_rms"] is not None:
        lines.append(f"Cursor jitter with hand still: {summary['still_jitter_px_rms']} px RMS per frame")
    lines.append("Counts: " + (", ".join(f"{k} {v}" for k, v in summary["counts"].items()) or "none"))
    lines.append("Time per state: " + ", ".join(f"{k} {v}s" for k, v in summary["seconds_per_state"].items()))
    return "\n".join(lines)


def session_stamp():
    return time.strftime("%Y%m%d-%H%M%S")
