"""Webcam + MediaPipe -> Frame observations. Knows nothing about gestures or macOS events."""
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from .observation import Frame, Hand

MODELS = Path(__file__).resolve().parent.parent
HAND_MODEL = MODELS / "hand_landmarker.task"
FACE_MODEL = MODELS / "blaze_face_short_range.tflite"


class CameraError(Exception):
    pass


class Capture:
    def __init__(self, camera_index=0, face_every=3, cursor_fn=None):
        for model in (HAND_MODEL, FACE_MODEL):
            if not model.exists():
                raise CameraError(f"Missing model file {model.name} in {model.parent}")
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise CameraError("Could not open the webcam. Is another app using it, "
                              "or is camera access turned off for this app?")
        # Loose confidence values: stricter ones drop the hand whenever a finger bends.
        # The engine filters low-confidence hands and faces itself.
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        ))
        self.faces = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(FACE_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=0.6,
        ))
        self.face_every = face_every
        self.cursor_fn = cursor_fn
        self.frame_count = 0
        self.last_faces = []
        self.last_ms = -1
        self.inference_ms = 0.0

    def read(self):
        """(Frame, mirrored BGR image), or (None, None) if the camera stopped."""
        ok, img = self.cap.read()
        t = time.monotonic()
        if not ok:
            return None, None
        img = cv2.flip(img, 1)  # Mirror, so moving your hand right moves things right
        height, width = img.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        timestamp = max(int(t * 1000), self.last_ms + 1)  # MediaPipe needs strictly increasing timestamps
        self.last_ms = timestamp

        started = time.monotonic()
        result = self.hands.detect_for_video(mp_image, timestamp)
        # Faces barely move between frames, so checking every few frames is enough
        if self.frame_count % self.face_every == 0:
            faces = self.faces.detect_for_video(mp_image, timestamp)
            self.last_faces = [(d.bounding_box.origin_x / width, d.bounding_box.origin_y / height,
                                d.bounding_box.width / width, d.bounding_box.height / height)
                               for d in faces.detections]
        self.frame_count += 1
        self.inference_ms = (time.monotonic() - started) * 1000

        hands = []
        for landmarks, world, handedness in zip(result.hand_landmarks, result.hand_world_landmarks, result.handedness):
            best = handedness[0] if handedness else None
            hands.append(Hand(
                points=[(lm.x, lm.y) for lm in landmarks],
                world=[(lm.x, lm.y, lm.z) for lm in world],
                handedness=best.category_name if best else "Unknown",
                score=best.score if best else 1.0,
            ))
        cursor = self.cursor_fn() if self.cursor_fn else None
        return Frame(t=t, width=width, height=height, hands=hands, faces=list(self.last_faces), cursor=cursor), img

    def close(self):
        self.cap.release()
        self.hands.close()
        self.faces.close()
