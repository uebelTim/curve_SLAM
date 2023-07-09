from filterpy.kalman import KalmanFilter
import numpy as np

f = KalmanFilter (dim_x=4, dim_z=4)
#initial state
f.x = np.array([0., 0., 0., 0.])
#define state transition matrix F /[A]
f.F = np.array([1., 0., 1., 0.],)