import cv2
import numpy as np

def detect_ball(raw_frame, previous_frame, roi_mask, frame_width, frame_height):
    """
    We combine several filters to create final result, and hope that the ball is the only one that matches all of them.

    1. High Saturation OR Orange
    """

    frame = raw_frame.copy()
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
        motion_mask = (motion_magnitude > motion_mask_cutoff).astype(np.uint8) * 255 # type: ignore
    else:
        motion_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

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

    # history.append(ball_mask)
    # for historical_mask in history:
    #     frame[historical_mask > 0, ...] = 255

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
        cv2.drawContours(mask, [contour], -1, 255, -1) # type: ignore

        # calculate saturation in mask
        # rank high-saturation detections higher
        # average_saturation = HSV[..., 1][mask > 0].mean(axis=0)
        
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
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        cv2.circle(frame, (cX, cY), 20, (0, 255, 255), 5)
        
        # if DO_PLAYBACK:
        #     print(f"{score=:.4f} {convexity=:.4f} {eccentricity=:.4f} {area=:.4f} {area_penalty=:.4f} {cX=:.4f} {cY=:.4f}")

        res = (cX, cY)
    else:
        res = None

    return (res, ball_mask, ball_mask_color, frame)
