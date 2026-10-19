import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# --- CONFIG --- 
SMOOTHING = 5 
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# --- INITIALIZING --- 
cap = cv2.VideoCapture(0) # Opens the webcam (0 is the default cam)

mp_hands = mp.solutions.hands 
# Only tracking 1 hand to avoid confusion
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidance=0.7)
mp_draw = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size() 

while True:
    # STEP 1: SEE
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1) # Mirrors image
    frame_height, frame_width, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # STEP 2: THINK
    results =  hands.process(img_rgb)

    # If hand is found
    if results.multi_hand_landmark:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_finger = hand_landmarks.landmark[8]

            

