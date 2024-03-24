import os
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from kalman_filter import Filter

# from scipy.signal import butter, filtfilt


def speak_async(text: str):
    os.system(f"say '{text}' &")

class BallTracker:
    def __init__(self, history_length=128, visualize=False):
        self.history_length = history_length
        self.visualize = visualize
        self.buf = deque(maxlen=self.history_length)
        self.last_vertical_bounce = 0
        self.last_horizontal_bounce = 0
        self.counter = 0

        self.filter = Filter()

        if visualize:
            self.counter = 0
            self.fig = plt.figure(figsize=(8, 6))
            self.axes = {
                'x': self.fig.add_subplot(221),
                'y': self.fig.add_subplot(222),
                'vx': self.fig.add_subplot(223),
                'vy': self.fig.add_subplot(224)
            }
            self.axes['x'].set_title('x')
            self.axes['y'].set_title('y')
            self.axes['vx'].set_title('vx')
            self.axes['vy'].set_title('vy')

    def median_filter(self, x, length=5):
        """
        Applies a median filter to the input x.
        """
        if len(x) < length:
            return x

        left = length // 2
        right = length - left

        x_filtered = list(x[:left])
        for i in range(left, len(x) - right):
            x_filtered.append(np.median(x[i - left:i + right]))
        x_filtered.extend(x[-right:])

        return np.array(x_filtered)

    def calculate_velocity(self):
        """
        Calculates velocity.
        """

        # apply a filtfilt
        t, x, y = zip(*self.buf)

        x = np.array(x)
        y = np.array(y)
        t = np.array(t)

        x = self.median_filter(x)
        y = self.median_filter(y)

        vx = (x[1:] - x[:-1]) * (t[1:] - t[:-1])
        vy = (y[1:] - y[:-1]) * (t[1:] - t[:-1])

        return vx, vy

    def visualize_history(self):
        if not self.visualize:
            return

        t, x, y = zip(*self.buf)
        t = np.array(t) - t[0]
        self.axes['x'].cla()
        self.axes['y'].cla()
        self.axes['x'].set_title('x')
        self.axes['y'].set_title('y')
        self.axes['x'].set_ylim(0, 1920//4)
        self.axes['y'].set_ylim(0, 1080//4)
        self.axes['x'].plot(t, x)
        self.axes['y'].plot(t, y)
        if len(x) >= 10:
            vx, vy = self.calculate_velocity()
            self.axes['vx'].cla()
            self.axes['vy'].cla()
            self.axes['vx'].set_title('vx')
            self.axes['vy'].set_title('vy')
            self.axes['vx'].set_ylim(-1, 1)
            self.axes['vy'].set_ylim(-1, 1)
            self.axes['vx'].plot(t[1:], vx)
            self.axes['vy'].plot(t[1:], vy)

        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()
        plt.pause(0.1)

    def handle_ball_detection(self, t_, x_, y_):
        dt = t_ - self.buf[-1][0] if len(self.buf) > 0 else 0.1

        # check distance from previous point. if dt is small and the distance is large, ignore the point
        if len(self.buf) > 0:
            prev_x, prev_y = self.buf[-1][1], self.buf[-1][2]
            dist = np.sqrt((prev_x - x_) ** 2 + (prev_y - y_) ** 2)
            if dist > 100 and dt < 0.1:
                print('ignoring point', x_, y_, 'dt', dt)
                return x_, y_, False, False, False

        x_, y_ = self.filter(x_, y_, dt)

        self.buf.append((t_, x_, y_))
        self.counter += 1

        if self.visualize:
            show_every = 10
            if self.counter % show_every == 0:
                self.visualize_history()

        x_bounce_left = x_bounce_right = False
        y_bounce = False

        # bounce detection (vy)
        if len(self.buf) >= 10:
            # vx, vy = self.calculate_velocity()
            # if vy[-2] > 0.5 and vy[-1] < -0.1:\
            y = np.array([y for t, x, y in self.buf])
            x = np.array([x for t, x, y in self.buf])

            min_time_between_x_bounces = 0.7
            min_time_between_y_bounces = 0.7

            x_bounce_right = (x[-3] < x[-2] and x[-1] < x[-2])
            x_bounce_left = (x[-3] > x[-2] and x[-1] > x[-2])
            if x_bounce_left or x_bounce_right:
                curr_time = t_
                elapsed = curr_time - self.last_horizontal_bounce
                if elapsed > min_time_between_x_bounces:
                    self.last_horizontal_bounce = t_
                    print('x bounce detected', t_, 'on', 'left' if x_bounce_left else 'right', 'side')
                    # speak_async("Bounce detected")
                    x_bounce = True

            y_bounce_2 = (y[-3] < y[-2] and y[-1] < y[-2])
            x_continued = (x[-3] < x[-2] < x[-1]) or (x[-3] > x[-2] > x[-1])
            if y_bounce_2 and x_continued:
                curr_time = t_
                elapsed = curr_time - self.last_vertical_bounce
                if elapsed > min_time_between_y_bounces:
                    self.last_vertical_bounce = t_
                    print('y bounce detected', t_)
                    # speak_async("Bounce detected")
                    y_bounce = True

        return x_, y_, x_bounce_left, x_bounce_right, y_bounce
