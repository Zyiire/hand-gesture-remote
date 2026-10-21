# hand-gesture-remote

Control your Mac with your hands instead of a mouse or trackpad. Think Tony Stark in his lab, minus the holograms (for now).

A webcam tracks your hands, and a small set of deliberate gestures moves the cursor, clicks, drags, scrolls and zooms. Control only starts when you ask for it, the window always shows what state it's in, and any held mouse button is released if tracking is lost, you pause, or the app quits.

## Setup

Requires macOS and Python 3.13 (MediaPipe doesn't support newer Python versions yet).

```sh
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python hand_mouse.py
```

The two model files (`hand_landmarker.task`, `blaze_face_short_range.tflite`) must be in the project folder.

### Permissions

macOS grants these to the app that runs the script (Terminal or VS Code), in **System Settings → Privacy & Security**:

| Permission | Needed for | Without it |
|---|---|---|
| **Camera** | Seeing your hands | The app explains and exits |
| **Accessibility** | Sending clicks, moves, scrolls and shortcuts | Gestures are tracked and shown, but nothing reaches your Mac. The window says so |
| **Input Monitoring** (optional) | The Ctrl+Option+P pause shortcut from any app | Pause with SPACE or the on-screen button instead |

Restart Terminal or VS Code after changing a permission. If Accessibility access is turned off while the app is running, it releases any held button and stops sending events.

### First run

The first run starts a short setup. You hold your hand relaxed, bend your index finger, bend your middle finger, then spread your fingers. It measures your hand and saves the click and pose thresholds to `settings.json`. Mouse control stays off during setup. Run it again with `python hand_mouse.py --calibrate` or the **C** key.

## Using it

1. **Take control:** hold an open hand up for 1 second. A progress bar fills, then the banner turns green (POINTER).
2. **Move:** move your hand. The cursor moves like a trackpad: slow movements are precise, quick ones cover the screen.
3. **Park:** make a fist. The cursor stays put while you reposition your hand, like lifting a mouse. Open your hand to carry on.
4. **Stop:** press SPACE, click PAUSE in the window, or press Ctrl+Option+P from any app.

### Gestures

Core gestures, always on:

| Gesture | Action |
|---|---|
| Open hand, held 1s | Take control |
| Move open hand | Move the cursor |
| Bend index finger | Left click (two quick bends = double click) |
| Hold index bent and move | Drag |
| Bend middle finger | Right click |
| Fist | Park the cursor |
| Index + middle up, ring + pinky curled | Scroll in any direction (content follows your hand) |
| Both hands open, apart / together | Zoom in / out (see below) |

System gestures are **opt-in**. Each one is off unless set in `settings.json` under `mappings`:

| Gesture (mapping name) | Suggested action |
|---|---|
| Two-finger flick right / left (`two_finger_flick_right` / `_left`) | `back` / `forward` |
| Spread hand held still (`spread_hold`) | `mission_control` |
| Spread hand flick left / right (`spread_flick_left` / `_right`) | `next_desktop` / `previous_desktop` |
| Spread hand flick up / down (`spread_flick_up` / `_down`) | `mission_control` / `app_expose` |
| Three fingers up, pinky curled (`"three_finger_drag": true`) | Hold the left button and drag (windows, files) |

Available actions: `mission_control`, `app_expose`, `next_desktop`, `previous_desktop`, `back`, `forward`, or any shortcut written like `"keys:cmd+shift+4"`. Set a mapping to `null` to turn it off.

### Keys (in the webcam window)

| Key | Action |
|---|---|
| SPACE | Pause / resume |
| C | Run setup (calibration) again |
| D | Show live readings (finger bends, spread, zoom level) |
| + / − | Cursor speed up / down |
| R, R | Reset settings to defaults (keeps your opt-in mappings) |
| Q or ESC | Quit |

### States

The banner colour and text always show the current state:

| State | Meaning |
|---|---|
| IDLE | No hand in control. Hold an open hand to start |
| ARMING | Open hand being held, progress bar filling |
| POINTER | Hand moves the cursor |
| CLICKING | A finger bend is being confirmed, or a button is held with the cursor locked in place |
| DRAGGING | Button held while moving |
| PARKED | Fist: cursor frozen |
| SCROLLING | Two-finger scroll |
| SYSTEM GESTURE | Spread hand, for mapped system actions |
| ZOOMING | Two-hand zoom |
| TRACKING LOST | Hand not seen. Movement freezes. Held buttons are released after 0.3s, and control switches off after 3s |
| PAUSED | Everything ignored until you resume |

## Zoom: what it can and can't do

macOS has no public way for an app to send a real trackpad pinch. Two-hand zoom therefore controls other apps through one of two documented inputs, set by `zoom_output` in `settings.json`:

- `"shortcut"` (default) sends Cmd+= / Cmd+−, one per zoom step. Works in most browsers, Preview, Finder and many other apps, in steps. Apps without those shortcuts won't zoom.
- `"ctrl_scroll"` sends smooth Ctrl+scroll. This zooms continuously in apps that support it (for example Chrome and Maps). It also zooms the whole screen if you've turned on *Use scroll gesture with modifier keys to zoom* in System Settings → Accessibility → Zoom.

How it behaves:
- **Ratio:** zoom follows the ratio of the current gap between your hands to the gap when zoom started, so it feels the same close to or far from the camera.
- **Dead zone:** small wobbles don't zoom.
- **Limit:** each session zooms at most 8 steps in or out.
- **Clutch:** close both hands to pause zooming and reposition them, then open them to carry on.
- **Cursor:** stays still throughout, so Ctrl+scroll zoom is anchored on the pointer.

## Settings

Every tunable value lives in `settings.json`, with a comment for each in [gesture_remote/config.py](gesture_remote/config.py). The ones you're most likely to change:

- **Speed:** `sensitivity`, `slow_gain`, `fast_gain`
- **Click thresholds:** `index_press`, `index_release`, `middle_press`, `middle_release` (setup measures these)
- **Pointer hand:** `pointer_hand`, to use only your left or right hand
- **Timing:** `arm_time`, `lost_grace`, `disarm_after`
- **Zoom:** `zoom_output`, `zoom_step`, `zoom_max_steps`

Delete `settings.json` to start over.

## How it's built

```
hand_mouse.py               app: permissions, main loop, keys, setup flow
gesture_remote/
  capture.py                webcam + MediaPipe -> timestamped observations
  observation.py            the observation format (also what recordings store)
  geometry.py               hand measurements: finger bends, spread, poses
  engine.py                 observations -> actions, as a deterministic state machine
  actions.py                the actions the engine can ask for
  macos.py                  actions -> real macOS events; permission checks
  overlay.py, calibration.py, metrics.py, replay.py, config.py, filters.py
tests/                      engine tests using synthetic hands
```

The engine never touches the camera or macOS. All its timing comes from frame timestamps, so the same frames always produce the same actions. That's what makes recorded sessions replayable and the gesture rules testable.

## Testing and measuring

Run the tests:

```sh
python -m pytest tests
```

Record a session, then replay it through the engine (no camera, no real events):

```sh
python hand_mouse.py --record
python -m gesture_remote.replay recordings/<file>.jsonl --actions
```

Each run also saves a metrics summary to `logs/`:
- latency from camera frame to event
- cursor jitter while your hand is still
- counts of clicks, drags, scrolls, zoom steps, commands, tracking losses and skipped glitches
- time spent in each state

**Checking for false activations:** record yourself reading, talking, touching your face and moving in and out of frame without meaning to control anything, then replay that recording. Every action it reports is a false activation. Do this across different lighting and camera positions before relying on the app.

Use `python hand_mouse.py --view-only` to try gestures without sending any events.

## Limitations

- Webcam tracking adds latency and jitter. Low light and bright backlighting make it worse (watch the fps counter; below about 20 fps, tracking gets unreliable).
- This is not trackpad emulation. System gestures and zoom are mapped to shortcuts and scroll events, so behaviour varies by app.
- It hasn't been evaluated across different users, hands, cameras or Macs, so it makes no accessibility-grade reliability claims.

## Built with

- [MediaPipe](https://developers.google.com/mediapipe): hand and finger landmark tracking, face detection
- [OpenCV](https://opencv.org/): webcam capture and the on-screen window
- [PyObjC](https://pyobjc.readthedocs.io/) (Quartz, AppKit, AVFoundation): macOS events and permissions
- [NumPy](https://numpy.org/): geometry
