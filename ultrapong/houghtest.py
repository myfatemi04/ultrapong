import os
import sys
import time
from collections import deque

import cv2
import numpy as np

DO_CAPTURE = False

# cap = cv2.VideoCapture("video2.mp4")
cap = cv2.VideoCapture(int(sys.argv[1]))

if DO_CAPTURE:
    writer = cv2.VideoWriter("video_tmp.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080))
else:
    writer = None

timestamps = deque(maxlen=10)
history = deque(maxlen=1)

bgr_ball = np.uint8([[[0, 127, 255]]])
hsv_ball = cv2.cvtColor(bgr_ball, cv2.COLOR_BGR2HSV)[0, 0]
hsv_ball_min = np.array([10, 100, 100])
hsv_ball_max = np.array([40, 255, 255])

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if writer is not None:
            writer.write(frame)

        timestamps.append(time.time())
        if len(timestamps) > 1:
            fps = len(timestamps) / (timestamps[-1] - timestamps[0])
            print(fps)

        # Downsample for faster processing.
        downsample = 4
        frame = np.ascontiguousarray(frame[::downsample, ::downsample, :])

        """
        We combine several filters to create final result, and hope that the ball is the only one that matches all of them.

        1. Color filter: We search for neon colors - colors with high saturation.
        """
        target_r = 0.6666
        target_g = 1 - target_r
        transform = np.array([
            [0.00, target_g, target_r],
            [0.00, -target_r, target_g],
            [1.00, 0.00, 0.00],
        ])
        frame_dot = frame @ transform
        # look for cases where other basis vectors are low
        frame_dot[frame_dot < 0] = 0
        high_intensity = frame_dot[..., 0] > ((frame_dot[..., 1] + frame_dot[..., 2]) * 4)


        frame_dot[~high_intensity] = 0
        cv2.imshow('frame_dot', frame_dot.astype(np.uint8))

        # frame_dot = (frame * np.array([0.00, 0.34, 0.66])).sum(axis=-1)
        # frame_dot = frame_dot.astype(np.uint8)

        # cv2.imshow('frame_dot', frame_dot.astype(np.uint8))

        ball_mask = (frame_dot > 0).any(axis=-1)
        height = ball_mask.shape[0]
        # Mask out anything below a certain height
        ball_mask[int(height * 0.9):, :] = 0
        ball_mask_u8 = (ball_mask * 255).astype(np.uint8)
        # Find ball contours
        # Erode and dilate
        erode_size = 3
        # kernel = np.ones((erode_size, erode_size), np.uint8)
        kernel = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ], np.uint8)
        ball_mask_u8 = cv2.erode(ball_mask_u8, kernel, iterations=1)
        ball_mask_u8 = cv2.dilate(ball_mask_u8, kernel, iterations=1)

        history.append(ball_mask_u8)

        for historical_mask in history:
            frame[historical_mask > 0, ...] = 255

        ball_contours, _ = cv2.findContours(ball_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_with_scores = []
        for contour in ball_contours:
            # find center
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # get area of contour
            area = cv2.contourArea(contour)
            # get aspect ratio
            x, y, w, h = cv2.boundingRect(contour)

            max_eccentricity = 1.2
            aspect_ratio = float(w) / h
            eccentricity = max(aspect_ratio, 1/aspect_ratio)

            solidity = area / cv2.contourArea(cv2.convexHull(contour))

            # Measure likelihood according to area
            min_expected_area = 25
            max_expected_area = 100
            # Use step function-like filtering
            below_penalty = 5.0
            above_penalty = 1.0
            area_penalty = (min_expected_area - min(area, min_expected_area)) * below_penalty + (max(area, max_expected_area) - max_expected_area) * above_penalty

            if area < 10:
                continue

            score = (solidity * 0.2 - eccentricity) - area_penalty * 0.1
            contours_with_scores.append((score, contour))

        contours_with_scores.sort(key=lambda x: x[0], reverse=True)
        if len(contours_with_scores) > 0:
            score, contour = contours_with_scores[0]
            # draw the contour and center of the shape on the image
            cv2.drawContours(frame, [contour], -1, (255, 0, 0), -1)

        cv2.imshow('frame', frame)
        # cv2.imshow('ball_mask', ball_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Interrupted.")

if writer is not None:
    writer.release()
    print("::: Correcting Video Format :::")
    os.system("ffmpeg -i video_tmp.mp4 video.mp4")
    os.system("rm video_tmp.mp4")
