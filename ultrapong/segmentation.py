import time
import cv2
import numpy as np
from collections import deque
import torch
from segment_anything import sam_model_registry
from segment_anything import SamPredictor
from segment_anything import SamAutomaticMaskGenerator
import supervision as sv

cap = cv2.VideoCapture(0)
isImage = True

timestamps = deque(maxlen=10)
def perform_segmentation(image):
    # Perform segmentation using the Sam model
    # Assume sam.segment is the method for segmentation in your Sam model
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"
    CHECKPOINT_PATH = "../sam_vit_h_4b8939.pth"

    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    mask_predictor = SamPredictor(sam)

    mask_generator = SamAutomaticMaskGenerator(sam)
    print("segmenting")
    image_bgr = cv2.imread(image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = mask_generator.generate(image_rgb)
    
    print("annotating")
    mask_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX)
    detections = sv.Detections.from_sam(result)
    annotated_image = mask_annotator.annotate(image_bgr, detections)
        
    # Display the segmented image
    cv2.imwrite("segmented.jpg", annotated_image)
    cv2.destroyAllWindows()
    exit

while True:
    ret, frame = cap.read()

    timestamps.append(time.time())
    if len(timestamps) > 1:
        fps = len(timestamps) / (timestamps[-1] - timestamps[0])
        #print(fps)

    downsample = 4
    frame = np.ascontiguousarray(frame[::downsample, ::downsample, :])

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
    table_mask_a = cv2.adaptiveThreshold(blurred_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    # Find table contours
    contours, _ = cv2.findContours(table_mask_a.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
                   potential_contours.append(contour)
    potential_contours.sort(reverse=True)
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
    cv2.imshow('table_mask', table_mask)
    key = cv2.waitKey(1)
    if key == ord('s'):
        # Save the captured frame as a still image
        cv2.drawContours(table_mask, [potential_contours[0]], 0, (255, 255, 255), thickness=cv2.FILLED)
        cv2.imwrite('mask.jpg', table_mask)
        cv2.imwrite('adaptivemask.jpg', table_mask_a)
        cv2.imwrite("snapshot.jpg", frame)
        
        print("Snapshot captured as snapshot.jpg")
        print("mask captured as mask.jpg")
        # Perform segmentation on the captured frame
        #perform_segmentation("snapshot.jpg")
    
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
