import cv2
from cloak import apply_invisibility_cloak
from gesture_control import GestureDetector
from mood_detect import detect_emotion
import numpy as np
import os

# HSV color range for red cloak
HSV_LOWER = np.array([0, 120, 70])
HSV_UPPER = np.array([10, 255, 255])

gesture_detector = GestureDetector()
cap = cv2.VideoCapture(0)

# Read background frame
print("Capturing background... Stay still for 3 seconds.")
for i in range(30):
    ret, background = cap.read()
    background = np.flip(background, axis=1)

mode = "normal"
emoji_dir = "assets/mood_emojis"
emotion = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = np.flip(frame, axis=1)

    gesture = gesture_detector.detect_gesture(frame)

    if gesture == "cloak":
        mode = "cloak"
    elif gesture == "mood":
        mode = "mood"

    if mode == "cloak":
        output = apply_invisibility_cloak(frame, background, HSV_LOWER, HSV_UPPER)
    elif mode == "mood":
        emotion = detect_emotion(frame)
        output = frame.copy()
        if emotion:
            emoji_path = os.path.join(emoji_dir, f"{emotion.lower()}.png")
            if os.path.exists(emoji_path):
                emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
                emoji = cv2.resize(emoji, (100, 100))
                y1, y2, x1, x2 = 20, 120, 20, 120
                alpha_emoji = emoji[:, :, 3] / 255.0
                for c in range(3):
                    output[y1:y2, x1:x2, c] = (alpha_emoji * emoji[:, :, c] +
                                               (1 - alpha_emoji) * output[y1:y2, x1:x2, c])
            cv2.putText(output, f"Emotion: {emotion}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    else:
        output = frame

    cv2.putText(output, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imshow("MoodCloak by Gauri Babu ❤️", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
