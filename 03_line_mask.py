import sys
import cv2
import numpy as np

path = sys.argv[1]
frame_number = int(sys.argv[2])

video = cv2.VideoCapture(path)
video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)   # jump straight to this frame
ok, frame = video.read()
if not ok:
    sys.exit("could not read that frame")

# Settings, in pixels for 1080p footage.
LINE_MAX_WIDTH = 15      # anything wider than this is treated as background (grass stripes)
MIN_BRIGHTER = 20        # how much brighter than its surroundings a line pixel must be (0-255)
MIN_BRIGHTNESS = 140     # line pixels are fairly bright overall (0-255)
PALER_FACTOR = 0.75      # line pixels have at most 75% of the surrounding grass's saturation

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
saturation = hsv[:, :, 1].astype(np.float32)
brightness = hsv[:, :, 2]

# 1. Thin and brighter than its surroundings ("top-hat" filter).
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (LINE_MAX_WIDTH, LINE_MAX_WIDTH))
brighter_than_surroundings = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
thin_and_bright = brighter_than_surroundings > MIN_BRIGHTER

# 2. Paler than the grass around it.
surrounding_saturation = cv2.blur(saturation, (3 * LINE_MAX_WIDTH, 3 * LINE_MAX_WIDTH))
paler = saturation < PALER_FACTOR * surrounding_saturation

# A line pixel passes all three tests.
mask = thin_and_bright & paler & (brightness > MIN_BRIGHTNESS)
print(f"{mask.mean() * 100:.2f}% of pixels marked as line")

# Save the mask (white = line) and an overlay (line pixels painted red on the frame).
cv2.imwrite("data/line_mask.png", mask.astype(np.uint8) * 255)
overlay = frame.copy()
overlay[mask] = (0, 0, 255)
cv2.imwrite("data/line_overlay.png", overlay)
print("Saved data/line_mask.png and data/line_overlay.png")