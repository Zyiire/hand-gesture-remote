# hand-gesture-remote

Control your computer with your hands instead of a mouse or trackpad. Think Tony Stark in his lab, minus the holograms (for now).

## Goal

This project aims to let a user control their computer with hand and finger gestures in place of a mouse and trackpad. A webcam tracks your hand in real time, and your movements and gestures become cursor actions:

- **Move the cursor** by moving your hand
- **Click** with a finger gesture, like a pinch
- **Scroll and drag** with other gestures, the way you would on a trackpad

## Built with

- [OpenCV](https://opencv.org/): webcam capture and image processing
- [MediaPipe](https://developers.google.com/mediapipe): hand and finger landmark tracking
- [PyAutoGUI](https://pyautogui.readthedocs.io/): controlling the mouse and keyboard
- [NumPy](https://numpy.org/): math for mapping hand positions to screen coordinates
