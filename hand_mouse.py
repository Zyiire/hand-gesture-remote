"""Hand-gesture mouse control for macOS.

    python hand_mouse.py              run (first run starts the setup/calibration flow)
    python hand_mouse.py --calibrate  run setup again
    python hand_mouse.py --record     also save every frame to recordings/ for replay
    python hand_mouse.py --view-only  track and show gestures, but never send mouse/keyboard events

Keys (in the webcam window): SPACE pause, C calibrate, D debug readings, + / - speed,
R twice reset settings, Q or ESC quit. Ctrl+Option+P pauses from any app.
"""
import argparse
import signal
import sys
import time
from pathlib import Path

import cv2

from gesture_remote import config, macos, overlay
from gesture_remote.calibration import Calibrator
from gesture_remote.capture import CameraError, Capture
from gesture_remote.engine import GestureEngine, State
from gesture_remote.metrics import Metrics, format_summary, session_stamp

ROOT = Path(__file__).resolve().parent
WINDOW = "JAVIS PROTO"
PERMISSION_CHECK_EVERY = 2.0  # Seconds between checks that event access is still granted


def check_camera():
    status = macos.camera_permission()
    if status == "not_determined":
        print("Asking for camera access...")
        if macos.request_camera():
            return True
        status = macos.camera_permission()
    if status == "authorized":
        return True
    print("Camera access is off for this app.\n"
          "Turn it on in System Settings > Privacy & Security > Camera (for Terminal or VS Code,\n"
          "whichever runs this), then quit and reopen that app and try again.")
    return False


def event_warning():
    return ("MOUSE CONTROL OFF: gestures are tracked but clicks won't land. Allow this app in "
            "System Settings > Privacy & Security > Accessibility")


def main():
    parser = argparse.ArgumentParser(description="Hand-gesture mouse control for macOS")
    parser.add_argument("--calibrate", action="store_true", help="run the setup/calibration flow")
    parser.add_argument("--record", action="store_true", help="save frames to recordings/ for replay")
    parser.add_argument("--view-only", action="store_true", help="never send mouse/keyboard events")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    args = parser.parse_args()

    settings = config.load()
    settings.double_click_time = macos.double_click_time()
    problems = macos.check_mappings(settings.mappings)
    if problems:
        print("Fix these in settings.json:\n  " + "\n  ".join(problems))
        sys.exit(1)

    if not check_camera():
        sys.exit(1)

    events_ok = macos.events_allowed()
    if not events_ok and not args.view_only:
        macos.request_events()
        print(event_warning())

    metrics = Metrics()
    dispatcher = macos.Dispatcher(metrics, enabled=events_ok and not args.view_only)
    engine = GestureEngine(settings, macos.screen_size(), metrics)
    try:
        capture = Capture(args.camera, settings.face_every, macos.real_cursor)
    except CameraError as error:
        print(error)
        sys.exit(1)

    recording = None
    if args.record:
        path = ROOT / "recordings" / f"{session_stamp()}.jsonl"
        path.parent.mkdir(exist_ok=True)
        recording = open(path, "w")
        print(f"Recording to {path}")

    calibrator = Calibrator() if args.calibrate or not settings.calibrated else None
    hotkey = macos.PauseHotkey()
    debug = False
    message, message_until = "", 0.0
    reset_armed_until = 0.0
    last_permission_check = time.monotonic()
    hotkey_ok = macos.hotkey_allowed()
    fps, prev_t = 0.0, None
    clicked = {"button": False}
    pause_button = (0, 0, 0, 0)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            x1, y1, x2, y2 = pause_button
            if x1 <= x <= x2 and y1 <= y <= y2:
                clicked["button"] = True

    def stop(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    def toggle_pause():
        if engine.state == State.PAUSED:
            engine.resume()
            return "Resumed: hold an open hand to take control"
        dispatcher.run(engine.pause(), time.monotonic())
        dispatcher.release_all()
        return "Paused"

    try:
        while True:
            frame, img = capture.read()
            if frame is None:
                print("The camera stopped sending frames.")
                break
            now = frame.t
            fps = fps * 0.9 + 0.1 / max(now - prev_t, 1e-6) if prev_t else 0.0
            prev_t = now

            # Event access can be revoked while running: release everything and stop sending
            if now - last_permission_check > PERMISSION_CHECK_EVERY:
                last_permission_check = now
                hotkey_ok = macos.hotkey_allowed()
                allowed = macos.events_allowed()
                if dispatcher.enabled and not allowed:
                    dispatcher.run(engine.cancel(), now)
                    dispatcher.set_enabled(False)
                    message, message_until = "Accessibility access was turned off: mouse control stopped", now + 5
                elif not dispatcher.enabled and allowed and not args.view_only:
                    engine.cancel()
                    dispatcher.set_enabled(True)
                    message, message_until = "Accessibility access granted: mouse control on", now + 5

            warnings = []
            if not dispatcher.enabled:
                warnings.append("VIEW ONLY: no mouse/keyboard events are sent" if args.view_only else event_warning())
            if not hotkey_ok:
                warnings.append("Ctrl+Option+P needs Input Monitoring permission; use SPACE in this window")
            info = {"fps": fps, "sensitivity": settings.sensitivity, "recording": bool(recording),
                    "debug": debug, "warnings": warnings, "message": message if now < message_until else "",
                    "latency_ms": metrics.latency[-1] * 1000 if metrics.latency else None}

            if calibrator:
                if calibrator.update(frame):
                    notes = calibrator.apply(settings)
                    config.save(settings)
                    print("Calibration saved:\n  " + "\n  ".join(notes))
                    engine = GestureEngine(settings, macos.screen_size(), metrics)
                    calibrator = None
                    message, message_until = "Setup done. Hold an open hand up to take control", now + 6
                else:
                    overlay.draw_calibration(img, frame, calibrator, info)
            if not calibrator:
                if recording:  # Only frames the engine sees, so a replay matches what happened live
                    recording.write(frame.to_json() + "\n")
                actions, status = engine.update(frame)
                dispatcher.run(actions, frame.t)
                pause_button = overlay.draw(img, status, frame, info)

            cv2.imshow(WINDOW, img)
            key = cv2.waitKey(1) & 0xFF
            if hotkey.pressed() or clicked["button"] or key in (ord(" "), ord("p")):
                clicked["button"] = False
                if not calibrator:
                    message, message_until = toggle_pause(), now + 3
            elif key in (ord("q"), 27):
                break
            elif key == ord("s") and calibrator:
                settings.calibrated = True
                config.save(settings)
                calibrator = None
            elif key == ord("c") and not calibrator:
                dispatcher.run(engine.cancel(), now)
                calibrator = Calibrator()
            elif key == ord("d"):
                debug = not debug
            elif key in (ord("+"), ord("=")):
                settings.sensitivity = round(min(settings.sensitivity + 0.1, 5.0), 1)
                config.save(settings)
            elif key == ord("-"):
                settings.sensitivity = round(max(settings.sensitivity - 0.1, 0.2), 1)
                config.save(settings)
            elif key == ord("r"):
                if now < reset_armed_until:
                    dispatcher.run(engine.cancel(), now)
                    settings = config.reset()
                    settings.double_click_time = macos.double_click_time()
                    engine = GestureEngine(settings, macos.screen_size(), metrics)
                    calibrator = Calibrator()
                    message, message_until = "Settings reset to defaults", now + 4
                else:
                    reset_armed_until = now + 2
                    message, message_until = "Press R again to reset all settings to defaults", now + 2
    except KeyboardInterrupt:
        pass
    finally:
        # Whatever happened, never leave a mouse button held down
        dispatcher.release_all()
        capture.close()
        if recording:
            recording.close()
        cv2.destroyAllWindows()
        summary_path = ROOT / "logs" / f"session-{session_stamp()}.json"
        metrics.save(summary_path)
        print(format_summary(metrics.summary()))
        print(f"Session metrics saved to {summary_path}")


if __name__ == "__main__":
    main()
