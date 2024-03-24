import time
import cv2
import numpy as np
from collections import deque

def detect_table(frame):
    # Detect circles
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # kernel = np.array([0.0, 0.2, 0.8]) # bgr
    # gray = ((frame * kernel).sum(axis=-1) / frame.sum(axis=-1) * 255).astype(np.uint8)
    # gray = cv2.medianBlur(gray, 5)

    frame_blurred = cv2.medianBlur(frame, 5)
    table_mask = (
        (frame/255.0 >= (np.array([0.15, 0.15, 0.15]))).all(axis=-1) & 
        (frame/255.0 <= (np.array([0.5, 0.4, 0.4]))).all(axis=-1) &
        (frame[..., 2].astype(np.short) - frame[..., 0].astype(np.short) < 15) &
        (frame[..., 2].astype(np.short) - frame[..., 1].astype(np.short) < 15)
        # (frame_blurred[..., 0] > frame_blurred[..., 2] * 0.5)
    ).astype(np.uint8) * 255
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise and enhance edges
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    # Apply adaptive thresholding to dynamically adjust the threshold
    table_mask_a = cv2.adaptiveThreshold(blurred_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # Find table contours

    contours, _ = cv2.findContours(table_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter for large rectangular contours
    max_area = -1
    potential_contours = []
    for contour in contours:
        # Approximate the contour to a polygon
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # Check if the polygon has 4 sides (potential rectangle/table)
        if len(approx) == 4:
            area = cv2.contourArea(contour)
            if area > 100:  # Assuming the table will have a significant area
                # Draw the contour on the original image
                proper_contour = contour
                max_area = area
                count = 0 
                for coord in table_mask:
                    if cv2.pointPolygonTest(contour,(coord[0],coord[1]),True) < 0:
                        count += 1
                if count/len(contour) >= 0.8:
                    potential_contours.append(approx)
                    # draw the approximation in green color
                    cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)

    potential_contours.sort(key=cv2.contourArea, reverse=True)

    if len(potential_contours) == 2:
        moment0 = cv2.moments(potential_contours[0])
        moment1 = cv2.moments(potential_contours[1])
        cx1, cy1 = int(moment0["m10"] / moment0["m00"]), int(moment0["m01"] / moment0["m00"])
        cx2, cy2 = int(moment1["m10"] / moment1["m00"]), int(moment1["m01"] / moment1["m00"])

        # returns left contour and then right contour
        if cx1 < cx2:
            C = potential_contours[0], potential_contours[1]
        else:
            C = potential_contours[1], potential_contours[0]
    else:
        C = None

    return C, table_mask

# test if you want
if __name__ == '__main__':
    cap = cv2.VideoCapture(1)
    isImage = True

    timestamps = deque(maxlen=10)

    while True:
        ret, frame = cap.read()

        downsample = 4
        frame = np.ascontiguousarray(frame[::downsample, ::downsample, :])
        C, table_mask = detect_table(frame)

        cv2.imshow('frame', frame)
        cv2.imshow('table_mask', table_mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
