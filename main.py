import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.9)
mp_draw = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

# Gesture control variables
prev_gesture = None
gesture_delay = 1  # seconds
last_gesture_time = time.time()

def fingers_up(hand_landmarks):
    """
    Returns a list indicating which fingers are up (1 = up, 0 = down).
    Order: [Thumb, Index, Middle, Ring, Pinky]
    """
    fingers = []
    # Thumb
    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
        fingers.append(1)
    else:
        fingers.append(0)
    # Fingers
    for tip_id in [mp_hands.HandLandmark.INDEX_FINGER_TIP,
                   mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                   mp_hands.HandLandmark.RING_FINGER_TIP,
                   mp_hands.HandLandmark.PINKY_TIP]:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

def detect_gesture(hand_landmarks):
    """
    Detects gesture based on which fingers are up.
    """
    fingers = fingers_up(hand_landmarks)

    # 🖨 Print raw list of fingers
    # print(fingers)

    # Check gestures
    if fingers == [1, 0, 0, 0, 0]:  # Fist
        return 'roll'
    elif fingers == [1, 1, 1, 1, 1]:  # Open palm
        return 'jump'
    elif fingers == [1, 1, 0, 0, 0]:  # Index finger up
        return 'left'
    elif fingers == [1, 0, 0, 0, 1]:  # Pinky finger up
        return 'right'

    return None

def send_key_press(gesture):
    if gesture == 'left':
        print("Index Finger → Left Arrow")
        pyautogui.press('left')
    elif gesture == 'right':
        print("Pinky Finger → Right Arrow")
        pyautogui.press('right')
    elif gesture == 'jump':
        print("Open Palm → Jump (Up Arrow)")
        pyautogui.press('up')
    elif gesture == 'roll':
        print("Fist → Roll (Down Arrow)")
        pyautogui.press('down')

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Mirror the image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            gesture = detect_gesture(handLms)

            # Avoid repeated gestures too quickly
            if gesture and gesture != prev_gesture and (time.time() - last_gesture_time) > gesture_delay:
                send_key_press(gesture)
                prev_gesture = gesture
                last_gesture_time = time.time()

    else:
        prev_gesture = None  # Reset gesture when hand not detected

    # Display webcam feed
    cv2.imshow("Subway Surfers Gesture Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
