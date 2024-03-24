import time
import cv2
import numpy as np
from collections import deque

import torch
from segment_anything import sam_model_registry
from segment_anything import SamPredictor
from segment_anything import SamAutomaticMaskGenerator

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "../sam_vit_h_4b8939.pth"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)

def perform_segmentation(image):
    # Perform segmentation using the Sam model
    # Assume sam.segment is the method for segmentation in your Sam model
    mask_predictor = SamPredictor(sam)

    mask_generator = SamAutomaticMaskGenerator(sam)
    print("segmenting")
    image_bgr = cv2.imread(image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = mask_generator.generate(image_rgb)
    
    print("annotating")
    import supervision as sv
    mask_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX)
    detections = sv.Detections.from_sam(result)
    annotated_image = mask_annotator.annotate(image_bgr, detections)
        
    # Display the segmented image
    cv2.imwrite("segmented.jpg", annotated_image)
    cv2.destroyAllWindows()
    exit

# Function to capture a snapshot when 's' key is pressed
def capture_snapshot():
    camera = cv2.VideoCapture(0)  # Assuming you want to use the default camera (index 0)
    while True:
        ret, frame = camera.read()
        cv2.imshow("Camera", frame)
        
        key = cv2.waitKey(1)
        if key == ord('s'):
            # Save the captured frame as a still image
            cv2.imwrite("snapshot.jpg", frame)
            print("Snapshot captured as snapshot.jpg")
            
            # Perform segmentation on the captured frame
            perform_segmentation("snapshot.jpg")
        
        if key == ord('q'):
            break
    
    camera.release()
    cv2.destroyAllWindows()


capture_snapshot()