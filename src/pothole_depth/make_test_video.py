import cv2
import os

IMAGE_DIR = "../../data/pothole_full/test/images"
OUTPUT_PATH = "../../data/test_road_video.mp4"
FPS = 2  # slow, so each image holds for ~0.5 sec
NUM_IMAGES = 10

images = sorted(os.listdir(IMAGE_DIR))[:NUM_IMAGES]
first_frame = cv2.imread(os.path.join(IMAGE_DIR, images[0]))
h, w = first_frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (w, h))

for img_name in images:
    frame = cv2.imread(os.path.join(IMAGE_DIR, img_name))
    frame = cv2.resize(frame, (w, h))
    writer.write(frame)

writer.release()
print(f"Test video created: {OUTPUT_PATH} ({len(images)} frames at {FPS} fps)")