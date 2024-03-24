import os
import sys
import time
from collections import deque

import cv2
import numpy as np
from check_table_side import get_net_offset, get_table_points
from commentary import Commentary
from detect_ball import detect_ball
from detect_table import detect_table
from point_tracker import PointTracker
from states import MatchState
from velocities import BallTracker


def speak_sync(text: str):
    os.system(f"say '{text}'")

def speak_async(text: str):
    if '\'' in text:
        text = text.replace('\'', '')

    print(text)

    os.system(f"say '{text}' &")

def main():
    DO_CAPTURE = len(sys.argv) == 2
    DO_PLAYBACK = not DO_CAPTURE
    DO_STEP_BY_STEP = False

    if DO_PLAYBACK:
        cap = cv2.VideoCapture(sys.argv[2])
    else:
        cap = cv2.VideoCapture(int(sys.argv[1]))
        cap.set(cv2.CAP_PROP_FPS, 60)

    if DO_CAPTURE:
        writer = cv2.VideoWriter("video_tmp.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080)) # type: ignore
    else:
        writer = None

    timestamps = deque(maxlen=10)

    circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)).astype(np.int8)
    circle = (circle * 2 - 1) / sum(np.abs(circle))

    previous_frame = None

    min_x = 0.1
    max_x = 0.9

    ball_tracker = BallTracker(history_length=90, visualize=False)
    commentary = Commentary()

    downsample = 4

    roi_mask = (np.ones((1080//downsample, 1920//downsample)) * 255).astype(np.uint8)
    roi_mask[:, :int(min_x * roi_mask.shape[1])] = 0
    roi_mask[:, int(max_x * roi_mask.shape[1]):] = 0
    roi_mask[int(0.8 * roi_mask.shape[0]):, :] = 0

    frame_width = 1920 // downsample
    frame_height = 1080 // downsample

    table_detection = None
    valid_ball_bounce_hitbox = None

    match_state = MatchState()
    ball_lost_counter = 0
    point_tracker = PointTracker()

    # for PLAYBACK mode.
    artificial_time = 0

    previous_detection = None
    previous_detection_timestamp = None

    pos_history = deque(maxlen=5)

    num_hits_this_round = 0

    counter = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            artificial_time += 0.1

            counter += 1
            if DO_PLAYBACK:
                if counter >= 100:
                    input()
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
                    kernel = np.ones((5, 5), np.uint8)
                    valid_ball_bounce_hitbox = np.zeros_like(table_mask)
                    cv2.drawContours(valid_ball_bounce_hitbox, [C[0], C[1]], -1, 255, -1)
                    valid_ball_bounce_hitbox = cv2.dilate(valid_ball_bounce_hitbox, kernel, iterations=10)
                    table_detection = C

                    middle_top, middle_bottom = get_table_points(table_detection)
                    max_y = middle_bottom[1]
                    roi_mask[int(max_y):, :] = 0
                    
            pause = False
            if match_state.current_state() == 'standby':
                pass
            elif table_detection is not None:
                # Table has been found!
                # Visualize the table.
                frame[valid_ball_bounce_hitbox > 0, :] = (0, 0, 255)
                middle_top, middle_bottom = get_table_points(table_detection)
                
                current_time = time.time() if not DO_PLAYBACK else artificial_time
                time_since_previous_detection = (current_time - previous_detection_timestamp) if previous_detection_timestamp is not None else None
                detection, ball_mask, ball_mask_color, frame = detect_ball(raw_frame.copy(), previous_frame, roi_mask, frame_width, frame_height, previous_detection, time_since_previous_detection)
                previous_frame = raw_frame

                frame[ball_mask > 0] = 255

                pos_history.append(detection)
                if detection is not None:
                    for pos in pos_history:
                        if pos is not None:
                            cv2.circle(frame, (pos[0], pos[1]), 10, (0, 255, 0), 2)
                    # cv2.circle(frame, (detection[0], detection[1]), 20, (0, 255, 255), 5)
                    previous_detection = detection
                    previous_detection_timestamp = current_time

                cv2.line(frame, middle_top.astype(int), middle_bottom.astype(int), (0, 200, 255), 5, cv2.LINE_AA) # type: ignore
                cv2.drawContours(frame, [C[0], C[1]], -1, 255, 4)

                if detection is not None:
                    net_offset = get_net_offset(middle_top, middle_bottom, detection[0], detection[1])

                    ball_side = 0 if net_offset < 0 else 1
                    if ball_side == 0: # left
                        frame[ball_mask > 0, :] = (255, 0, 255)
                    else:
                        frame[ball_mask > 0, :] = (0, 255, 0)

                if detection is not None:
                    (x_, y_, x_bounce_left, x_bounce_right, y_bounce, bounce_location) = ball_tracker.handle_ball_detection(current_time, detection[0], detection[1], valid_ball_bounce_hitbox)
                    pause = x_bounce_left or x_bounce_right
                    
                    # if DO_PLAYBACK:
                    #     print(x_bounce_left, x_bounce_right, y_bounce, bounce_location, ball_side)
                    if bounce_location is not None:
                        bounce_net_offset = get_net_offset(middle_top, middle_bottom, bounce_location[0], bounce_location[1])

                    if y_bounce:
                        pause = True

                    result = None
                    PRINT_RESULTS = True
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
                    elif x_bounce_left and ball_side == 0:
                        if abs(bounce_net_offset) > 50:
                            ball_lost_counter = 0
                            result = match_state.transition("ball_hit_by_player_1")
                            num_hits_this_round += 1
                            if PRINT_RESULTS:
                                print("ball_hit_by_player_1", "ball_side", ball_side, detection[0])
                                print("RESULT", result)
                        else:
                            print("Player 1 hit detected, but too close to the net.")
                    elif x_bounce_right and ball_side == 0:
                        if abs(bounce_net_offset) < 20:
                            ball_lost_counter = 0
                            result = match_state.transition("net_hit_by_player_1")
                            if PRINT_RESULTS:
                                print("net_hit_by_player_1", "ball_side", ball_side, detection[0])
                                print("RESULT", result)
                        else:
                            print("Net hit by player 1, but too far from the net.")
                    elif x_bounce_right and ball_side == 1:
                        if abs(bounce_net_offset) > 50:
                            ball_lost_counter = 0
                            result = match_state.transition("ball_hit_by_player_2")
                            num_hits_this_round += 1
                            if PRINT_RESULTS:
                                print("ball_hit_by_player_2", "ball_side", ball_side, detection[0])
                                print("RESULT", result)
                        else:
                            print("Player 2 hit detected, but too close to the net.")
                    elif x_bounce_left and ball_side == 1:
                        if abs(bounce_net_offset) < 20:
                            ball_lost_counter = 0
                            result = match_state.transition("net_hit_by_player_2")
                            if PRINT_RESULTS:
                                print("net_hit_by_player_2", "ball_side", ball_side, detection[0])
                                print("RESULT", result)
                        else:
                            print("Net hit by player 2, but too far from the net.")

                    # draw the filtered detection
                    cv2.circle(frame, (int(x_), int(y_)), 10, (255, 0, 0), 3)
                else:
                    S = match_state.current_state()
                    if 'liable' in S:
                        print("increasing ball_lost_counter")
                        ball_lost_counter += 1
                        if ball_lost_counter > 10: # if no other action is detected for 3 seconds
                            result = match_state.transition("ball_lost")
                            print("ball_lost")

                s = match_state.current_state()
                if "_loses" in s:
                    pause = True
                    print("Someone lost!")
                    match_state._current_state = "start"

                    comms = ""
                    if s == 'p1_loses':
                        match_outcome = point_tracker.update(2)
                        # if not DO_PLAYBACK:
                        #     comms = commentary.get_commentary(2, num_hits_this_round)
                        # else:
                        comms = f"Player 2 scored."
                    elif s == 'p2_loses':
                        match_outcome = point_tracker.update(1)
                        # if not DO_PLAYBACK:
                        #     comms = commentary.get_commentary(1, num_hits_this_round)
                        # else:
                        comms = f"Player 1 scored."

                    num_hits_this_round = 0

                    if match_outcome is not None:
                        comms += f" Player {match_outcome} wins!"
                        match_state._current_state = "standby"
                        point_tracker.reset()
                        commentary.reset()
                        
                    speak_async(comms)

                # frame[ball_mask > 0] = 255 # type: ignore
            cv2.imshow('frame', frame)

            if pause and DO_PLAYBACK:
                input()

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
        c = 0
        while os.path.exists(f"video_{c}.mp4"):
            c += 1
        os.system(f"ffmpeg -i video_tmp.mp4 video_{c}.mp4")
        os.system("rm video_tmp.mp4")


if __name__ == "__main__":
    main()
