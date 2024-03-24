import os
import sys
import time
from collections import deque

import cv2
import numpy as np
from velocities import BallTracker
import sort
from detect_ball import detect_ball

DO_CAPTURE = False
DO_PLAYBACK = True
DO_STEP_BY_STEP = False

if DO_PLAYBACK:
    assert not DO_CAPTURE
    cap = cv2.VideoCapture("video.mp4")
else:
    cap = cv2.VideoCapture(int(sys.argv[1]))
    cap.set(cv2.CAP_PROP_FPS, 60)

if DO_CAPTURE:
    writer = cv2.VideoWriter("video_tmp.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080))
else:
    writer = None

timestamps = deque(maxlen=10)
history = deque(maxlen=1)

bgr_ball = np.array([[[0, 127, 255]]], dtype=np.uint8)
hsv_ball = cv2.cvtColor(bgr_ball, cv2.COLOR_BGR2HSV)[0, 0] # type: ignore
hue_min = hsv_ball[0] - 10
hue_max = hsv_ball[0] + 10

circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)).astype(np.int8)
circle = (circle * 2 - 1) / sum(np.abs(circle))
# circle0 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)).astype(np.uint8)

# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'
skin_lower = np.array([0, 48, 80], dtype = "uint8")
skin_upper = np.array([20, 255, 255], dtype = "uint8")

previous_detection = None
previous_frame = None

min_x = 0.1
max_x = 0.9

event_handler = BallTracker(history_length=90, visualize=True)
# ball_filter = 

downsample = 4

roi_mask = (np.ones((1080//downsample, 1920//downsample)) * 255).astype(np.uint8)
roi_mask[:, :int(min_x * roi_mask.shape[1])] = 0
roi_mask[:, int(max_x * roi_mask.shape[1]):] = 0
roi_mask[int(0.8 * roi_mask.shape[0]):, :] = 0

frame_width = 1920 // downsample
frame_height = 1080 // downsample

recent_detections = deque(maxlen=4)

tracker = sort.Sort()

counter = 0
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        counter += 1
        if DO_PLAYBACK and counter < 200:
            continue

        if writer is not None:
            writer.write(frame)

        if DO_PLAYBACK:
            time.sleep(0.1)
        else:
            timestamps.append(time.time())
            if len(timestamps) > 1:
                fps = len(timestamps) / (timestamps[-1] - timestamps[0])
                # print(f"{fps:.3f}")

        # Downsample for faster processing.
        downsample = 4
        frame = np.ascontiguousarray(frame[::downsample, ::downsample, :])
        raw_frame = frame.copy()

        detection, ball_mask, ball_mask_color, frame = detect_ball(frame.copy(), previous_frame, roi_mask, frame_width, frame_height)
        previous_frame = raw_frame

        if detection is not None:
            (horizontal_bounce, vertical_bounce) = event_handler.handle_ball_detection(time.time(), detection[0], detection[1])

        cv2.imshow('ball_mask_color', ball_mask_color)

        frame[ball_mask > 0] = 255 # type: ignore
        cv2.imshow('frame', frame)

        key = cv2.waitKey(0 if DO_STEP_BY_STEP else 1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            continue

except KeyboardInterrupt:
    print("Interrupted.")

if writer is not None:
    writer.release()
    print("::: Correcting Video Format :::")
    os.system("ffmpeg -i video_tmp.mp4 video.mp4")
    os.system("rm video_tmp.mp4")
