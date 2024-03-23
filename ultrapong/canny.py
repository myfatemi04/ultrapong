import cv2
import numpy as np
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    downsample = 2
    frame = np.ascontiguousarray(frame[::downsample, ::downsample, :])
    # Load the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Preprocess the image
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter for large rectangular contours
    for contour in contours:
        # Approximate the contour to a polygon
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # Check if the polygon has 4 sides (potential rectangle/table)
        if len(approx) == 4:
            area = cv2.contourArea(contour)
            if area > 1000:  # Assuming the table will have a significant area
                # Draw the contour on the original image
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
                break

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break