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

cap = cv2.VideoCapture(0)

timestamps = deque(maxlen=10)
history = deque(maxlen=10)

while True:
    ret, frame = cap.read()

    timestamps.append(time.time())
    if len(timestamps) > 1:
        fps = len(timestamps) / (timestamps[-1] - timestamps[0])
        print(fps)

    downsample = 4
    frame = np.ascontiguousarray(frame[::downsample, ::downsample, :])

    # Detect circles
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # kernel = np.array([0.0, 0.2, 0.8]) # bgr
    # gray = ((frame * kernel).sum(axis=-1) / frame.sum(axis=-1) * 255).astype(np.uint8)
    # gray = cv2.medianBlur(gray, 5)

    frame_blurred = cv2.medianBlur(frame, 5)
    ball_mask = ((frame/255.0 >= (np.array([0.0, 0.2, 0.8]))).all(axis=-1) & (frame/255.0 <= (np.array([0.7, 1.0, 1.0]))).all(axis=-1)).astype(np.uint8) * 255
    table_mask = (
        (frame/255.0 >= (np.array([0.15, 0.15, 0.15]))).all(axis=-1) & 
        (frame/255.0 <= (np.array([0.5, 0.4, 0.4]))).all(axis=-1) &
        (frame[..., 2].astype(np.short) - frame[..., 0].astype(np.short) < 15) &
        (frame[..., 2].astype(np.short) - frame[..., 1].astype(np.short) < 15)
        # (frame_blurred[..., 0] > frame_blurred[..., 2] * 0.5)
    ).astype(np.uint8) * 255
    
    history.append(ball_mask)

    # Find ball contours
    ball_contours, _ = cv2.findContours(ball_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    # Find table contours
    contours, _ = cv2.findContours(table_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter for large rectangular contours
    # for contour in contours:
    #     # Approximate the contour to a polygon
    #     perimeter = cv2.arcLength(contour, True)
    #     approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
    #     # Check if the polygon has 4 sides (potential rectangle/table)
    #     if len(approx) == 4:
    #         area = cv2.contourArea(contour)
    #         if area > 400:  # Assuming the table will have a significant area
    #             # Draw the contour on the original image
    #             cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
    #             # break

    # Get the two largest contours (likely two parts of the table)
    contours_by_perimeter = [(cv2.arcLength(contour, True), contour) for contour in contours]
    two_largest_contours = [contour for perimeter, contour in sorted(contours_by_perimeter, key=lambda x: x[0], reverse=True)[:2]]
    # cv2.drawContours(frame, two_largest_contours, -1, (0, 255, 0), 3)

    # Combine the two largest contours with a convex hull, and simplify using approxPolyDP
    # if len(two_largest_contours) > 1:
    #     cv2.drawContours(frame, cv2.convexHull(np.concatenate(two_largest_contours[0], two_largest_contours[1])), -1, (0, 255, 0), 3)
    

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

    for frame_2 in history:
        frame[frame_2 > 0, ...] = 255

    cv2.imshow('frame', frame)
    cv2.imshow('ball_mask', ball_mask)
    cv2.imshow('table_mask', table_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
