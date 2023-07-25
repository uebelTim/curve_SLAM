from filterpy.kalman import KalmanFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('../Aufnahmen/data/kalmanVars.csv')

# Extract the state variables
state_vars = data[['lateral_offset', 'heading_angle', 'curvature', 'curvature_derivative']].values

# Compute the differences in the state variables from one time step to the next
state_diffs = np.diff(state_vars, axis=0)

# Calculate the sample covariance of these differences to estimate the process noise covariance
Q = np.cov(state_diffs, rowvar=False)

# Set up the system matrix based on the vehicle kinematic model
lr = lf = 0.18
dt = 0.01  # assuming a time step of 0.01 seconds; adjust as needed
F = np.array([[1, dt, 0, 0],
              [0, 1, dt, 0],
              [0, 0, 1, dt],
              [0, 0, 0, 1]])

# Set up the control input matrix
B = np.array([[dt * data['speed'].values[0] * lr / (lr + lf), 0, 0, 0],
              [0, dt * data['speed'].values[0], 0, 0],
              [0, 0, dt, 0],
              [0, 0, 0, 0]])

# Set up the control vector
U = data['steering_angle'].values

# Set up the observation matrix
H = np.eye(3)  # assuming the state variables are directly observed

# Compute the residuals between the observations and the initial state
residuals = state_vars - np.dot(F, state_vars.T).T

# Calculate the sample covariance of the residuals to estimate the observation noise covariance
R = np.cov(residuals, rowvar=False)

# Initialize the Kalman filter
kf = KalmanFilter(dim_x=4, dim_z=3)
kf.x = state_vars[0]  # initial state
kf.F = F  # transition matrix
kf.B = B  # control input matrix
kf.U = U  # control vector
kf.H = H  # observation matrix
kf.R = R  # observation noise covariance
kf.Q = Q  # process noise covariance
kf.P *= 1000  # initial state covariance

# Run the Kalman filter to get the state estimates
n = len(data)
states = np.zeros((n, 4))
curvatures = []
for i in range(n):
    kf.predict()
    kf.update(data[['lateral_offset', 'heading_angle', 'curvature']].values[i])
    states[i] = kf.x
    curvatures.append(kf.x[2])

plt.plot(data['curvature'], label='Measured')
plt.title('Curvature')
plt.plot(curvatures, label='Estimated')
plt.legend()
plt.show()