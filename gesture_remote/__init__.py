"""Hand-gesture control for macOS.

Layers, kept separate so each can be changed or tested on its own:
    capture.py  - webcam + MediaPipe -> timestamped observations (observation.py)
    engine.py   - observations -> semantic actions, as a deterministic state machine (no OS calls)
    macos.py    - semantic actions -> real macOS events, plus permission checks
"""
