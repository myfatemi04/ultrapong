import os
import sys
import time
from collections import deque

import cv2
import numpy as np

DO_CAPTURE = False
DO_PLAYBACK = True

if DO_PLAYBACK:
    assert not DO_CAPTURE
    cap = cv2.VideoCapture("video_0.mp4")
else:
    cap = cv2.VideoCapture(int(sys.argv[1]))
    cap.set(cv2.CAP_PROP_FPS, 60)

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

circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)).astype(np.int8)
circle = (circle * 2 - 1) / sum(np.abs(circle))
# circle0 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)).astype(np.uint8)

previous_detection = None

counter = 0
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        counter += 1
        if counter < 100:
            continue

        if writer is not None:
            writer.write(frame)

        if DO_PLAYBACK:
            time.sleep(0.1)
        else:
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
        target_g = 0.3333
        # target_r = 0.5555
        # target_g = 0.3555
        target_b = 1 - target_g - target_r
        A1 = np.array([target_b, target_g, target_r])
        A2 = np.array([target_b, -target_r, target_g])
        
        transform = np.array([
            A1,
            A2,
            np.cross(A1, A2),
        ])
        frame_dot = frame @ transform
        # look for cases where other basis vectors are low

        frame_dot[frame_dot < 0] = 0
        ratio = frame_dot[..., 0] / (((frame_dot[..., 1]) ** 2 + (frame_dot[..., 2]) ** 2) ** 0.5 + 1e-6)
        ratio_u8 = np.minimum(100 * ratio, 255).astype(np.uint8)
        # ratio_u8 = cv2.equalizeHist(ratio_u8)
        ratio_u8 = cv2.GaussianBlur(ratio_u8, (5, 5), 0)

        cv2.imshow('ratio_orig', ratio_u8)
        # calculate an adaptive threshold
        sanity_check = ratio_u8 > 100
        adaptive_check = cv2.adaptiveThreshold(ratio_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, -2)
        ratio_u8 = cv2.bitwise_and(sanity_check.astype(np.uint8) * 255, adaptive_check)

        cv2.imshow('ratio', ratio_u8)

        high_orangeness = ratio_u8 > 200
        dark = frame.mean(axis=-1) < 50

        # cv2.imshow('frame_dot_unfiltered', frame_dot.astype(np.uint8))
        frame_dot[~high_orangeness | dark] = 0
        # cv2.imshow('frame_dot', frame_dot.astype(np.uint8))
        # cv2.imshow('dark', dark.astype(np.uint8) * 255)

        ball_mask = (frame_dot > 0).any(axis=-1)
        height = ball_mask.shape[0]
        # Mask out anything below a certain height
        ball_mask[int(height * 0.9):, :] = 0
        ball_mask_u8 = (ball_mask * 255).astype(np.uint8)
        # Find ball contours
        # Erode and dilate
        erode_size = 3
        # kernel = np.ones((erode_size, erode_size), np.uint8)
        # kernel = np.array([
        #     [0, 1, 0],
        #     [1, 1, 1],
        #     [0, 1, 0],
        # ], np.uint8)
        # ball_mask_u8 = cv2.erode(ball_mask_u8, kernel, iterations=1)
        # ball_mask_u8 = cv2.dilate(ball_mask_u8, kernel, iterations=1)

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

            is_near_previous_detection = False
            if previous_detection is not None:
                is_near_previous_detection = np.linalg.norm(np.array([cX, cY]) - np.array(previous_detection)) < 50

            # https://docs.opencv.org/4.x/d1/d32/tutorial_py_contour_properties.html

            # get area of contour
            area = cv2.contourArea(contour)

            # get mask
            mask = np.zeros_like(ball_mask_u8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            # calculate color score
            color_score = ratio[mask > 0].mean()
            
            # get rotated bounding box
            ((_x, _y), (w, h), angle) = cv2.minAreaRect(contour)

            max_eccentricity = 1.2
            aspect_ratio = float(w) / h
            eccentricity = max(aspect_ratio, 1/aspect_ratio)

            solidity = area / cv2.contourArea(cv2.convexHull(contour))

            # Measure likelihood according to area
            min_expected_area = 25
            max_expected_area = 75
            # Use step function-like filtering
            below_penalty = 5.0
            above_penalty = 1.0
            area_penalty = (min_expected_area - min(area, min_expected_area)) * below_penalty + (max(area, max_expected_area) - max_expected_area) * above_penalty

            print(solidity, eccentricity, area, area_penalty)
            # input()

            # kill tail end detections
            # if not is_near_previous_detection:
            if area < 5:
                continue
            if area > 100:
                continue
            if eccentricity > 3:
                continue
            if solidity < 0.2:
                continue

            score = solidity * 0.2 - (eccentricity - 1) * 5 - area_penalty * 0.1 + color_score
            contours_with_scores.append((score, contour, solidity, eccentricity, area, area_penalty))

        ### Create detection through previous track of ball ###
        # if previous_detection is not None:
        #     # create a window around the previous detection
        #     x, y = previous_detection
        #     window_size = 50
        #     window = frame[max(0, y - window_size):min(frame.shape[0], y + window_size), max(0, x - window_size):min(frame.shape[1], x + window_size)]
        #     window_dot = window @ transform
        #     window_dot[window_dot < 0] = 0
        #     window_ratio = window_dot[..., 0] / ((abs(window_dot[..., 1]) + abs(window_dot[..., 2])) / 2 + 1)

        #     cv2.imshow("window_ratio", np.minimum(100 * window_ratio, 255).astype(np.uint8))

        #     positions = np.zeros((2, *window.shape[:-1]))
        #     positions[0] = np.expand_dims(np.arange(window.shape[0]), 1).repeat(window.shape[1], axis=1)
        #     positions[1] = np.expand_dims(np.arange(window.shape[1]), 0).repeat(window.shape[0], axis=0)
        #     x_frame, y_frame = (positions * window_ratio).sum(axis=(1, 2))/window_ratio.sum()
        #     x_frame = int(x_frame)
        #     y_frame = int(y_frame)

        #     pos = (x_frame + max(0, x - window_size), y_frame + max(0, y - window_size))
        #     # draw a circle around the maximum value
        #     cv2.circle(frame, pos, 20, (255, 255, 0), 5)
        #     # store the maximum value
        #     previous_detection = pos

        contours_with_scores.sort(key=lambda x: x[0], reverse=True)
        if len(contours_with_scores) > 0:
            score, contour, *other = contours_with_scores[0]

            solidity, eccentricity, area, area_penalty = other

            print(f"{score=:.4f} {solidity=:.4f} {eccentricity=:.4f} {area=:.4f} {area_penalty=:.4f}")

            # draw the contour and center of the shape on the image
            cv2.drawContours(frame, [contour], -1, (255, 0, 0), -1)

            # draw a bigass circle
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cX, cY), 20, (0, 255, 255), 5)
            previous_detection = (cX, cY)
        else:
            previous_detection = None

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
