import cv2
import numpy as np

def apply_invisibility_cloak(frame, background, hsv_lower, hsv_upper):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    mask = cv2.medianBlur(mask, 5)

    cloak_area = cv2.bitwise_and(background, background, mask=mask)
    inverse_mask = cv2.bitwise_not(mask)
    rest = cv2.bitwise_and(frame, frame, mask=inverse_mask)

    final = cv2.addWeighted(cloak_area, 1, rest, 1, 0)
    return final
