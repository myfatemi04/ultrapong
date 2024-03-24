import filterpy.kalman as kf
from filterpy.common import Q_discrete_white_noise
import numpy as np

class Filter:
    def __init__(self):
        self.k = kf.KalmanFilter(dim_x=4, dim_z=2)
        self.initialized = False

    def __call__(self, x, y, dt):
        # x, vx, y, vy
        # x gets vx * dt, same for y
        self.k.F = np.array([[1, dt, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, dt],
                                [0, 0, 0, 1]])
        # Measurement function
        self.k.H = np.array([[1, 0, 0, 0],
                                [0, 0, 1, 0]])
        # Identity covariance matrix
        self.k.P = np.eye(4) * 500
        # Measurement noise
        self.k.R = np.array([[5., 0],
                                [0, 5.]])
        # Process noise
        self.k.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.13)

        if not self.initialized:
            self.k.x = np.array([x, 0, y, 0])
            return np.array([x, y])


        self.k.predict()
        self.k.update([x, y])

        return np.array([self.k.x[0], self.k.x[2]])
