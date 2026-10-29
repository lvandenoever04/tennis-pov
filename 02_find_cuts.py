import sys
import cv2
import numpy as np

path = sys.argv[1]
video = cv2.VideoCapture(path)
fps = video.get(cv2.CAP_PROP_FPS)

CUT_THRESHOLD = 0.3   # colour difference above this between two frames = a cut

def colour_histogram(frame):
    """Count pixels in 30 hue bands x 32 saturation bands, normalised."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    return cv2.normalize(hist, None).flatten()

thumbnails = []   # a small copy of every frame, for the contact sheet
differences = []  # colour difference between each frame and the one before
previous = None

while True:
    ok, frame = video.read()
    if not ok:
        break                         # end of the video
    small = cv2.resize(frame, (320, 180))   # 1080p is far more detail than this needs
    thumbnails.append(small)
    hist = colour_histogram(small)
    if previous is None:
        differences.append(0.0)
    else:
        differences.append(cv2.compareHist(previous, hist, cv2.HISTCMP_BHATTACHARYYA))
    previous = hist

differences = np.array(differences)
cut_frames = [i for i, d in enumerate(differences) if d > CUT_THRESHOLD]

# A shot runs from one cut to the next.
starts = [0] + cut_frames
ends = cut_frames + [len(thumbnails)]
print(f"Found {len(starts)} shots:")
for n, (a, b) in enumerate(zip(starts, ends)):
    print(f"  shot {n}: frames {a}-{b - 1}  ({(b - a) / fps:.1f} s)")

# Show the biggest differences, so we can see how clearly cuts stand out from normal motion.
print("Largest frame-to-frame differences:")
for i in np.argsort(-differences)[:8]:
    print(f"  frame {i}: {differences[i]:.2f}")

# Contact sheet: the middle frame of each shot, labelled.
tiles = []
for n, (a, b) in enumerate(zip(starts, ends)):
    tile = thumbnails[(a + b) // 2].copy()
    cv2.putText(tile, f"shot {n}: {a}-{b - 1}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    tiles.append(tile)
while len(tiles) % 4:
    tiles.append(np.zeros_like(tiles[0]))   # pad the last row
rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
cv2.imwrite("data/shots.png", np.vstack(rows))
print("Saved data/shots.png")