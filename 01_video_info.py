import sys
import cv2

# The video file is given on the command line, e.g. python 01_video_info.py data/centre1.mp4
path = sys.argv[1]
video = cv2.VideoCapture(path)

if not video.isOpened():
    sys.exit(f"Could not open {path}")

# Ask OpenCV for the basic facts about the video.
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Resolution: {width} x {height}")
print(f"Frames per second: {fps:.2f}")
print(f"Number of frames: {frame_count}")
print(f"Length: {frame_count / fps:.1f} seconds")

# Read the very first frame and save it as an image so we can look at it.
ok, frame = video.read()
cv2.imwrite("data/first_frame.png", frame)
print("Saved data/first_frame.png")