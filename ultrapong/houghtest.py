import cv2
import numpy as np

cap = cv2.VideoCapture(0)

snapshot_counter = 1
take_snapshot = False

while True:
    ret, frame = cap.read()

    downsample = 2
    frame = np.ascontiguousarray(frame[::downsample, ::downsample, :])

    # Detect circles
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.medianBlur(gray, 3)
    # gray = cv2.Canny(gray, 100, 200)
    # scale = 0.5
    scale = 1/downsample
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=int(40 * scale), minRadius=0, maxRadius=int(100 * scale))

    if circles is not None:
        circles = circles[0]

        if cv2.waitKey(1) & 0xFF == ord('s'):
            take_snapshot = True

        for (x, y, r) in circles:
            x = int(x)
            y = int(y)
            r = int(r)

            # Check average color inside circle
            avg_color = frame[y-r:y+r, x-r:x+r].reshape(-1, 3).mean(axis=0) / 255.0
            # Check if greater than 0.6
            if not (avg_color[0] > 0.6 and avg_color[1] > 0.6 and avg_color[1] > 0.6):
                continue

            if take_snapshot:
                print(f"Circle for snapshot {snapshot_counter}:", x, y, r)

            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(frame, (x-5, y-5), (x+5, y+5), (0, 128, 255), -1)
        
        if take_snapshot:
            cv2.imwrite(f"snapshot-{snapshot_counter}.jpg", frame)
            snapshot_counter += 1
            take_snapshot = False

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
