import os
import sys
import time
from collections import deque

import cv2
import numpy as np
from velocities import VelocityClassifier

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

event_handler = VelocityClassifier(history_length=90, visualize=True)

roi_mask = (np.ones((1080//4, 1920//4)) * 255).astype(np.uint8)
roi_mask[:, :int(min_x * roi_mask.shape[1])] = 0
roi_mask[:, int(max_x * roi_mask.shape[1]):] = 0
roi_mask[int(0.8 * roi_mask.shape[0]):, :] = 0

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

        value_mask = HSV[..., 2].copy()
        value_mask = cv2.threshold(value_mask, 40, 255, cv2.THRESH_BINARY)[1]

        hue_mask = ((10 < HSV[..., 0]) & (HSV[..., 0] < 40)).astype(np.uint8) * 255

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
        # cv2.imshow('saturation_mask', saturation_mask)
        # cv2.imshow('skin_mask', skin_mask)
        # cv2.imshow('motion_mask', motion_mask)

        ball_mask = (saturation_mask > 0) & (motion_mask > 0) & (value_mask > 0) & (hue_mask > 0) & (roi_mask > 0)
        ball_mask = ball_mask.astype(np.uint8) * 255

        # Erode and dilate.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ball_mask = cv2.erode(ball_mask, kernel, iterations=1)
        ball_mask = cv2.dilate(ball_mask, kernel, iterations=1)

        cv2.imshow('ball_mask', ball_mask)

        #### Select region of interest ####

        history.append(ball_mask)

        for historical_mask in history:
            frame[historical_mask > 0, ...] = 255

        detection = None

        ball_contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_with_scores = []
        for contour in ball_contours:
            # find center
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

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

            score = 0
            trustworthy = (0.3 < cX/(1920//downsample) < 0.7)
            if trustworthy:
                score += 1000000
            
            # compare to previous detection
            if previous_detection is not None:
                distance = np.linalg.norm(np.array([cX, cY]) - np.array(previous_detection))
                score -= distance * 1000

            # kill tail end detections
            # if not is_near_previous_detection:
            kill = (area < 5 or area > 200 or eccentricity > 10 or solidity < 0.2)
            if kill:
                cv2.drawContours(frame, [contour], -1, (0, 0, 255), -1)
                continue

            target_x = (1920 // downsample) / 2
            laterality = ((target_x - int(cX)) / target_x) ** 2

            if DO_PLAYBACK:
                print(solidity, eccentricity, area, area_penalty, laterality)

            score += solidity * 0.2 - (eccentricity - 1) * 5 - area_penalty * 0.1 + average_saturation - laterality * 100000
            contours_with_scores.append((score, contour, solidity, eccentricity, area, area_penalty, laterality))

        contours_with_scores.sort(key=lambda x: x[0], reverse=True)
        if len(contours_with_scores) > 0:
            score, contour, *other = contours_with_scores[0]

            solidity, eccentricity, area, area_penalty, laterality = other

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
                print(f"{score=:.4f} {solidity=:.4f} {eccentricity=:.4f} {area=:.4f} {area_penalty=:.4f} {cX=:.4f} {cY=:.4f} {laterality=:.4f}")

            detection = (cX, cY)
        else:
            detection = None

        if detection is not None:
            event_handler.handle_ball_detection(time.time(), detection[0], detection[1])

        cv2.imshow('frame', frame)
        # cv2.imshow('ball_mask', ball_mask)

        if cv2.waitKey(0 if DO_STEP_BY_STEP else 1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Interrupted.")

if writer is not None:
    writer.release()
    print("::: Correcting Video Format :::")
    os.system("ffmpeg -i video_tmp.mp4 video.mp4")
    os.system("rm video_tmp.mp4")
