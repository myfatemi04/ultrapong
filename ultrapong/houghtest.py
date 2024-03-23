import time
import cv2
import numpy as np
from collections import deque

def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while len(non_working_ports) < 6: # if there are more than 5 non working ports stop the testing. 
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
    return available_ports,working_ports,non_working_ports

# print(list_ports())

cap = cv2.VideoCapture("video2.mp4")
# cap = cv2.VideoCapture(0)
# writer = cv2.VideoWriter("video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080))
writer = None

timestamps = deque(maxlen=10)
history = deque(maxlen=1)

bgr_ball = np.uint8([[[0, 127, 255]]])
hsv_ball = cv2.cvtColor(bgr_ball, cv2.COLOR_BGR2HSV)[0, 0]
hsv_ball_min = np.array([0, 100, 100])
hsv_ball_max = np.array([30, 255, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    time.sleep(0.05)

    if writer is not None:
        writer.write(frame)

    timestamps.append(time.time())
    if len(timestamps) > 1:
        fps = len(timestamps) / (timestamps[-1] - timestamps[0])
        print(fps)

    # combine several filters to create final result, and hope that the ball is the only one that matches all of them

    downsample = 4
    frame = np.ascontiguousarray(frame[::downsample, ::downsample, :])

    # Detect circles
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # kernel = np.array([0.0, 0.2, 0.8]) # bgr
    # gray = ((frame * kernel).sum(axis=-1) / frame.sum(axis=-1) * 255).astype(np.uint8)
    # gray = cv2.medianBlur(gray, 5)

    # frame_blurred = cv2.medianBlur(frame, 5)
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # frame_hsv[..., -1] = cv2.equalizeHist(frame_hsv[..., -1])

    # ball_mask = ((frame/255.0 >= (np.array([0.0, 0.2, 0.8]))).all(axis=-1) & (frame/255.0 <= (np.array([0.7, 1.0, 1.0]))).all(axis=-1)).astype(np.uint8) * 255

    ball_h_mask = (frame_hsv[..., 0] > hsv_ball_min[0]) & (frame_hsv[..., 0] < hsv_ball_max[0])
    ball_s_mask = (frame_hsv[..., 1] > hsv_ball_min[1]) & (frame_hsv[..., 1] < hsv_ball_max[1])
    ball_s_mask_2 = (frame_hsv[..., 1] > 160) & (frame_hsv[..., 1] < hsv_ball_max[1])
    ball_v_mask = (frame_hsv[..., 2] > hsv_ball_min[2]) & (frame_hsv[..., 2] < hsv_ball_max[2])
    ball_mask = (ball_h_mask | ball_s_mask_2) & ball_s_mask & ball_v_mask
    ball_mask = ball_mask.astype(np.uint8) * 255

    table_mask = (
        (frame/255.0 >= (np.array([0.2, 0.0, 0.0]))).all(axis=-1) & 
        (frame/255.0 <= (np.array([0.6, 0.4, 0.6]))).all(axis=-1)
        # (frame[..., 0] > frame[..., 1]) &
        # (frame_blurred[..., 0] > frame_blurred[..., 2] * 0.5)
    ).astype(np.uint8) * 255
    
    history.append(ball_mask)

    # for frame_2 in history:
    #     frame[frame_2 > 0, ...] = 255

    # Find ball contours
    ball_contours, _ = cv2.findContours(ball_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

        max_aspect_ratio = 1.2
        aspect_ratio = float(w) / h
        if (aspect_ratio > max_aspect_ratio) or (1/aspect_ratio > max_aspect_ratio):
            continue

        if area > 1000:
            continue

        # solidity = area / cv2.contourArea(cv2.convexHull(contour))
        # if solidity < 0.8:
        #     continue

        # (x,y), (MA,ma), angle = cv2.fitEllipse(contour)

        # draw the contour and center of the shape on the image
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        # cv2.circle(frame_hsv, (cX, cY), 7, (255, 255, 255), 2)

    find_table = False

    if find_table:
        # Find table contours
        contours, _ = cv2.findContours(table_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter for large rectangular contours
        for contour in contours:
            # Approximate the contour to a polygon
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            # Check if the polygon has 4 sides (potential rectangle/table)
            if len(approx) == 4:
                area = cv2.contourArea(contour)
                if area > 400:  # Assuming the table will have a significant area
                    # Draw the contour on the original image
                    cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
                    # break

    # find contour
    # ret, thresh = cv2.threshold(gray, 127, 255, 0)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # for contour in contours:
    #     area = cv2.contourArea(contour)
    #     if area < 100:
    #         continue

    #     cv2.drawContours(frame, [contour], 0, (0, 255, 0), 3)

    # scale = 0.5
    # scale = 1/downsample
    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, 1.0, minDist=50, param1=50, param2=0.1, minRadius=0, maxRadius=50)

    # edges = cv2.Canny(gray, 50, 100)

    # frame[:] = edges[..., None]

    # if circles is not None:
    #     circles = circles[0]
    #     for (x, y, r) in circles:
    #         x = int(x)
    #         y = int(y)
    #         r = int(r)

    #         # Check average color inside circle
    #         avg_color = gray[y-r:y+r, x-r:x+r].mean()
    #         if avg_color < 200:
    #             continue
    #         # Check if greater than 0.6
    #         # if not (avg_color[0] > 0.6 and avg_color[1] > 0.6 and avg_color[1] > 0.6):
    #         #     continue

    #         cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
    #         cv2.rectangle(frame, (x-5, y-5), (x+5, y+5), (0, 128, 255), -1)

    cv2.imshow('frame', frame)
    # cv2.imshow('frame_h', frame_hsv[..., 0])
    frame_s = frame_hsv[..., 1]
    frame_s[~ball_v_mask] = 0
    # cv2.imshow('frame_s', frame_s)
    # cv2.imshow('frame_v', frame_hsv[..., 2])
    cv2.imshow('ball_mask', ball_mask)
    # cv2.imshow('ball_h_mask', ball_h_mask.astype(np.uint8) * 255)
    # cv2.imshow('ball_s_mask', ball_s_mask.astype(np.uint8) * 255)
    # cv2.imshow('ball_v_mask', ball_v_mask.astype(np.uint8) * 255)
    # cv2.imshow('table_mask', table_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if writer is not None:
    writer.release()
