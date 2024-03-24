import os
import sys
import time
from collections import deque

import cv2
import numpy as np

DO_CAPTURE = False
DO_PLAYBACK = False

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
                print(f"{fps:.3f}")

        # Downsample for faster processing.
        downsample = 4
        frame = np.ascontiguousarray(frame[::downsample, ::downsample, :])
        raw_frame = frame.copy()

        """
        We combine several filters to create final result, and hope that the ball is the only one that matches all of them.

        1. High Saturation OR Orange
        """

        frame_blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        HSV = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2HSV)
        # Calculate hue similarity.
        hue_difference = np.minimum(np.abs(HSV[..., 0].astype(np.int8) - hsv_ball[0]), np.abs((HSV[..., 0] - 255).astype(np.int8) - hsv_ball[0]))
        # Calculate the brightness.
        brightness = HSV[..., 2]

        hue_difference = hue_difference.astype(np.uint8)

        saturation_mask = HSV[..., 1].copy()
        saturation_mask = cv2.adaptiveThreshold(saturation_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, -2, saturation_mask)

        # value_mask = HSV[..., 2].copy()
        # value_mask = cv2.adaptiveThreshold(value_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, -2, value_mask)

        # skin_mask = cv2.inRange(HSV, skin_lower, skin_upper)

        # Create a motion mask.
        if previous_frame is not None:
            raw_frame_blurry = cv2.GaussianBlur(raw_frame, (11, 11), 0)
            previous_frame_blurry = cv2.GaussianBlur(previous_frame, (11, 11), 0)
            motion_magnitude = np.sqrt(((raw_frame_blurry.astype(int) - previous_frame_blurry.astype(int)) ** 2).sum(axis=-1)).astype(np.uint8)
            # Dilate.
            motion_mask_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            motion_magnitude = cv2.dilate(motion_magnitude, motion_mask_dilate_kernel, iterations=1)
            motion_mask_cutoff = 25
            motion_mask = (motion_magnitude > motion_mask_cutoff).astype(np.uint8) * 255
        else:
            motion_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        previous_frame = raw_frame

        # Check saturation and value masks.
        cv2.imshow('saturation_mask', saturation_mask)
        # cv2.imshow('skin_mask', skin_mask)
        cv2.imshow('motion_mask', motion_mask)

        ball_mask = cv2.bitwise_and(saturation_mask, motion_mask)
        # ball_mask = cv2.bitwise_and(ball_mask, 255 - skin_mask)
        ball_mask[:, :int(min_x * ball_mask.shape[1])] = 0
        ball_mask[:, int(max_x * ball_mask.shape[1]):] = 0
        # ball_mask = cv2.bitwise_and(cv2.bitwise_and(saturation_mask, value_mask), motion_mask)

        # Erode and dilate.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ball_mask = cv2.erode(ball_mask, kernel, iterations=1)
        ball_mask = cv2.dilate(ball_mask, kernel, iterations=1)

        cv2.imshow('ball_mask', ball_mask)

        #### Select region of interest ####

        history.append(ball_mask)

        for historical_mask in history:
            frame[historical_mask > 0, ...] = 255

        ball_contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
            mask = np.zeros_like(ball_mask)
            cv2.drawContours(mask, [contour], -1, 255, -1)

            # calculate saturation in mask
            # rank high-saturation detections higher
            average_saturation = HSV[..., 1][mask > 0].mean(axis=0)
            
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

            if DO_PLAYBACK:
                print(solidity, eccentricity, area, area_penalty)

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

            score = solidity * 0.2 - (eccentricity - 1) * 5 - area_penalty * 0.1 + average_saturation
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

            if DO_PLAYBACK:
                print(f"{score=:.4f} {solidity=:.4f} {eccentricity=:.4f} {area=:.4f} {area_penalty=:.4f} {cX=:.4f} {cY=:.4f}")
        else:
            previous_detection = None

        cv2.imshow('frame', frame)
        # cv2.imshow('ball_mask', ball_mask)

        if cv2.waitKey(0 if DO_PLAYBACK else 1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Interrupted.")

if writer is not None:
    writer.release()
    print("::: Correcting Video Format :::")
    os.system("ffmpeg -i video_tmp.mp4 video.mp4")
    os.system("rm video_tmp.mp4")
