import cv2
import numpy as np

LINE_MAX_WIDTH = 15
MIN_BRIGHTER = 20
MIN_BRIGHTNESS = 140
PALER_FACTOR = 0.75


def line_mask(frame):
    """True for pixels that look like court-line paint: thin, brighter and paler than nearby grass."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    brightness = hsv[:, :, 2]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (LINE_MAX_WIDTH, LINE_MAX_WIDTH))
    thin_and_bright = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel) > MIN_BRIGHTER
    surrounding = cv2.blur(saturation, (3 * LINE_MAX_WIDTH, 3 * LINE_MAX_WIDTH))
    paler = saturation < PALER_FACTOR * surrounding
    return thin_and_bright & paler & (brightness > MIN_BRIGHTNESS)