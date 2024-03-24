from collections import deque

import matplotlib.pyplot as plt
import numpy as np

# from scipy.signal import butter, filtfilt


class VelocityClassifier:
    def __init__(self, history_length=128, visualize=False):
        self.history_length = history_length
        self.visualize = visualize
        self.buf = deque(maxlen=self.history_length)

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

        # Define the filter parameters
        # order = 4
        # fs = 1/((t[-1] - t[0]) / len(t))  # Sample rate, Hz
        # cutoff = 5.0  # Desired cutoff frequency of the filter, Hz

        # # Create a Butterworth low-pass filter
        # b, a = butter(order, cutoff / (fs / 2), btype='low')
        
        # # Apply the filter to x and y
        # x_filtered = filtfilt(b, a, x)
        # y_filtered = filtfilt(b, a, y)

        # Calculate the velocity
        # vx = (x_filtered[1:] - x_filtered[:-1]) * (t[1:] - t[:-1])
        # vy = (y_filtered[1:] - y_filtered[:-1]) * (t[1:] - t[:-1])
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
            self.axes['vx'].set_ylim(-40, 40)
            self.axes['vy'].set_ylim(-40, 40)
            self.axes['vx'].plot(t[1:], vx)
            self.axes['vy'].plot(t[1:], vy)

        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()
        plt.pause(0.01)

    def handle_ball_detection(self, t, x, y):
        self.buf.append((t, x, y))

        if self.visualize:
            self.counter += 1
            if self.counter % self.history_length == 0:
                self.visualize_history()
