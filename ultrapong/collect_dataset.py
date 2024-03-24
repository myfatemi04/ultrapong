import matplotlib.pyplot as plt
import cv2

indexes = []
labels = []
label = None

def label_click_callback(event, x, y, flags, param):
    global label
    if event == cv2.EVENT_LBUTTONDOWN:
        label = (x, y)
        print("Labelled", x, y)

cap = cv2.VideoCapture("video.mp4")
cv2.namedWindow("frame")
cv2.setMouseCallback("frame", label_click_callback)

counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    counter += 1
    if (counter % 10) != 0:
        continue
    
    cv2.imshow("frame", frame)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        print("skip")
        label = None
    if key == ord(' '):
        print('ok')
        
    indexes.append(counter - 1)
    labels.append(label)
    label = None

cap.release()
cv2.destroyAllWindows()

import json

with open("labels.json", "w") as f:
    json.dump({"labels": labels, "indexes": indexes}, f)
