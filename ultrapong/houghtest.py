import os
import sys
import time
from collections import deque

import cv2
import numpy as np
from velocities import VelocityClassifier
import sort

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

        """
        We combine several filters to create final result, and hope that the ball is the only one that matches all of them.

        1. High Saturation OR Orange
        """

        frame_blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        HSV = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2HSV)

        # Calculate hue similarity.
        # hue_difference = np.minimum(np.abs(HSV[..., 0].astype(np.int8) - hsv_ball[0]), np.abs((HSV[..., 0] - 255).astype(np.int8) - hsv_ball[0]))
        # hue_difference = hue_difference.astype(np.uint8)
        # hue = HSV[..., 0].copy()
        # hue[hue < 20] = 0
        # hue *= 12
        # cv2.imshow('hue', hue)

        # calculate saturation mean
        # window_size = 11
        # padded = np.pad(HSV[..., 1], window_size // 2, mode='edge')
        # saturation_mean = np.zeros_like(HSV[..., 1], dtype=float)
        # saturation_variance = np.zeros_like(HSV[..., 1], dtype=float)
        # for i in range(window_size):
        #     for j in range(window_size):
        #         saturation_mean += padded[i:i+HSV.shape[0], j:j+HSV.shape[1]]
        #         saturation_variance += padded[i:i+HSV.shape[0], j:j+HSV.shape[1]] ** 2

        # saturation_mean /= float(window_size ** 2)
        # saturation_variance /= float(window_size ** 2)
        # # Visualize saturation [normalized]
        # saturation_normalized = (HSV[..., 1] - saturation_mean) / (saturation_variance ** 0.5 + 1e-6)
        # cv2.imshow('saturation_normalized', saturation_normalized)

        saturation_mask = HSV[..., 1].copy()
        saturation_mask = cv2.adaptiveThreshold(saturation_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, -2, saturation_mask)

        value_mask = HSV[..., 2].copy()
        value_mask = cv2.threshold(value_mask, 40, 255, cv2.THRESH_BINARY)[1]

        hue_mask = ((30 < HSV[..., 0]) & (HSV[..., 0] < 100)).astype(np.uint8) * 255
        # hue_mask = (frame_blurred[..., 1] > (frame_blurred[..., 0] + frame_blurred[..., 2])).astype(np.uint8) * 255

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
        cv2.imshow('motion_mask', motion_mask)
        cv2.imshow('hue_mask', hue_mask)

        ball_mask = (motion_mask > 0) & (hue_mask > 0) & (roi_mask > 0)
        # ball_mask = (saturation_mask > 0) & (motion_mask > 0) & (hue_mask > 0) & (roi_mask > 0)
        # ball_mask = (saturation_mask > 0) & (motion_mask > 0) & (value_mask > 0) & (hue_mask > 0) & (roi_mask > 0)
        ball_mask = ball_mask.astype(np.uint8) * 255

        # Erode and dilate.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        iters = 2
        ball_mask = cv2.erode(ball_mask, kernel, iterations=iters)
        ball_mask = cv2.dilate(ball_mask, kernel, iterations=iters)

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

            # get bbox
            x, y, w, h = cv2.boundingRect(contour)
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h

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

            """
            Comprehensive Scoring Framework:
             * Centrality: prior distribution for ball location represented as Gaussian
             * Solidity: represents convexity of the contour
             * Area: represented as a Gaussian
            """

            # compare to prior distribution
            cx_mean = 0.5 * frame_width
            cy_mean = 0.5 * frame_height
            cx_std = 0.1 * frame_width
            cy_std = 0.1 * frame_height
            cx_penalty = (cX - cx_mean) ** 2 / cx_std ** 2
            cy_penalty = (cY - cy_mean) ** 2 / cy_std ** 2
            prior_distribution_penalty = (cx_penalty + cy_penalty)

            # compare convexity
            convexity = area / cv2.contourArea(cv2.convexHull(contour))

            # compare area
            mean_area = 20
            std_area = 2
            area_penalty = (area - mean_area) ** 2 / std_area ** 2

            # compare to previous detections (to come soon)
            
            kill = (area < 10 or area > 200 or eccentricity > 10 or convexity < 0.2)
            if kill:
                cv2.drawContours(frame, [contour], -1, (0, 0, 255), -1)
                continue

            # if DO_PLAYBACK:
            #     print(solidity, eccentricity, area, area_penalty, laterality)

            score = 0
            score += (convexity - 0.8) - area_penalty - prior_distribution_penalty

            contours_with_scores.append((score, contour, (x1, x2, y1, y2), convexity, eccentricity, area, area_penalty))

        ball_mask_color = cv2.cvtColor(ball_mask, cv2.COLOR_GRAY2BGR)

        contours_with_scores.sort(key=lambda x: x[0], reverse=True)

        if len(contours_with_scores) > 0:
            # print("Providing", len(contours_with_scores), "detections.")
            # object_ids = tracker.update(np.array([
            #     # x1, y1, x2, y2, score
            #     np.array([*bbox, 1.0])
            #     for score, contour, bbox, *_ in contours_with_scores
            # ]))
            # print(object_ids)

            score, contour, *other = contours_with_scores[0]

            bbox, convexity, eccentricity, area, area_penalty = other

            # draw the contour and center of the shape on the image
            cv2.drawContours(frame, [contour], -1, (255, 0, 0), -1)
            cv2.drawContours(ball_mask_color, [contour], -1, (0, 255, 0), -1)

            # draw a bigass circle
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cX, cY), 20, (0, 255, 255), 5)
            previous_detection = (cX, cY)

            if DO_PLAYBACK:
                print(f"{score=:.4f} {convexity=:.4f} {eccentricity=:.4f} {area=:.4f} {area_penalty=:.4f} {cX=:.4f} {cY=:.4f}")

            detection = (cX, cY)
            recent_detections.append(detection)
        else:
            detection = None

        if detection is not None:
            event_handler.handle_ball_detection(time.time(), detection[0], detection[1])

        cv2.imshow('ball_mask_color', ball_mask_color)
        cv2.imshow('frame', frame)

        if cv2.waitKey(0 if DO_STEP_BY_STEP else 1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Interrupted.")

if writer is not None:
    writer.release()
    print("::: Correcting Video Format :::")
    os.system("ffmpeg -i video_tmp.mp4 video.mp4")
    os.system("rm video_tmp.mp4")
