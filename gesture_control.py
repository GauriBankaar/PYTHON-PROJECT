import mediapipe as mp
import cv2

class GestureDetector:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(max_num_hands=1)
        self.mp_draw = mp.solutions.drawing_utils

    def detect_gesture(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, handLms, mp.solutions.hands.HAND_CONNECTIONS)
                # Example: index finger tip (landmark 8) above middle finger tip (landmark 12) = gesture "1"
                y_index = handLms.landmark[8].y
                y_middle = handLms.landmark[12].y
                if y_index < y_middle:
                    return "cloak"
                else:
                    return "mood"
        return "none"
