import os
import sys
import time
from collections import deque

import cv2
import numpy as np
from velocities import BallTracker
import sort
from detect_ball import detect_ball
from detect_table import detect_table
from check_table_side import check_table_side, get_table_points
from states import MatchState

import os

def speak_async(text: str):
    os.system(f"say '{text}' &")

def main():

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
        writer = cv2.VideoWriter("video_tmp.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080)) # type: ignore
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

    event_handler = BallTracker(history_length=90, visualize=False)
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

    table_detection = None

    match_state = MatchState()
    ball_lost_counter = 0

    counter = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            counter += 1
            if DO_PLAYBACK and counter < 200:
                continue

            if DO_PLAYBACK:
                time.sleep(0.1)
            else:
                timestamps.append(time.time())
                if len(timestamps) > 1:
                    fps = len(timestamps) / (timestamps[-1] - timestamps[0])
                    # print(f"{fps:.3f}")

            if writer is not None:
                writer.write(frame)

            # Downsample for faster processing.
            downsample = 4
            frame = np.ascontiguousarray(frame[::downsample, ::downsample, :])
            raw_frame = frame.copy()
            if table_detection is None:
                C, table_mask = detect_table(frame)
                if C is not None:
                    table_detection = C
            else:
                # Table has been found!
                detection, ball_mask, ball_mask_color, frame = detect_ball(frame.copy(), previous_frame, roi_mask, frame_width, frame_height)
                previous_frame = raw_frame

                middle_top, middle_bottom = get_table_points(table_detection)

                cv2.line(frame, middle_top.astype(int), middle_bottom.astype(int), (0, 200, 255), 5, cv2.LINE_AA) # type: ignore

                if detection is not None:
                    ball_side = check_table_side(middle_top, middle_bottom, detection[0], detection[1])

                    if ball_side: # left
                        frame[ball_mask > 0, :] = (255, 0, 255)
                    else:
                        frame[ball_mask > 0, :] = (0, 255, 0)

                if detection is not None:
                    (x_bounce_left, x_bounce_right, y_bounce) = event_handler.handle_ball_detection(time.time(), detection[0], detection[1])
                    x_bounce = x_bounce_left or x_bounce_right
                    result = None
                    PRINT_RESULTS = False
                    if y_bounce and ball_side == 0:
                        ball_lost_counter = 0
                        result = match_state.transition("ball_bounced_on_table_1")
                        if PRINT_RESULTS:
                            print("ball_bounced_on_table_1")
                            print("RESULT", result)
                    elif y_bounce and ball_side == 1:
                        ball_lost_counter = 0
                        result = match_state.transition("ball_bounced_on_table_2")
                        if PRINT_RESULTS:
                            print("ball_bounced_on_table_2")
                            print("RESULT", result)
                    elif x_bounce and ball_side == 0:
                        ball_lost_counter = 0
                        result = match_state.transition("ball_hit_by_player_1")
                        if PRINT_RESULTS:
                            print("ball_hit_by_player_1", "ball_side", ball_side, detection[0])
                            print("RESULT", result)
                    elif x_bounce and ball_side == 1:
                        ball_lost_counter = 0
                        result = match_state.transition("ball_hit_by_player_2")
                        if PRINT_RESULTS:
                            print("ball_hit_by_player_2", "ball_side", ball_side, detection[0])
                            print("RESULT", result)
                    elif detection[1] > middle_bottom[1]:
                        ball_lost_counter += 1
                        if ball_lost_counter > 90:
                            result = match_state("ball_lost")
                            if PRINT_RESULTS:
                                print("ball_lost")
                    

                cv2.imshow('ball_mask_color', ball_mask_color)

                # frame[ball_mask > 0] = 255 # type: ignore
            cv2.imshow('frame', frame)

            key = cv2.waitKey(0 if DO_STEP_BY_STEP else 1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                speak_async("new game")
                continue

    except KeyboardInterrupt:
        print("Interrupted.")

    if writer is not None:
        writer.release()
        print("::: Correcting Video Format :::")
        os.system("ffmpeg -i video_tmp.mp4 video.mp4")
        os.system("rm video_tmp.mp4")


if __name__ == "__main__":
    main()
