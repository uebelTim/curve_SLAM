# Import necessary libraries
#%%
import pandas as pd
import numpy as np
from scipy.linalg import inv
import matplotlib.pyplot as plt
from curvature_estimation import CurvatureEstimator
import os
import natsort 
import cv2

# Function to perform the Kalman filter
def kalman_filter(params):
    A = params['A']
    B = params['B']
    H = params['H']
    Q = params['Q']
    R = params['R']
    x = params['x0']
    P = params['P0']
    Z = params['Z']
    U = params['U']
    
    xs = [x]
    Ps = [P]
    
    for i, z in enumerate(Z[1:]):
        u = U[i]
        A[0, 1] = A[1, 2] = dt * u[1] # Update A matrix
        B[0, 0] = dt * lr/(lr+lf) * u[1] # Update B matrix
        
        # Prediction
        x = A @ x + B @ u
        P = A @ P @ A.T + Q
        # Update
        S = H @ P @ H.T + R
        K = P @ H.T @ inv(S)
        y = z - H @ x # Pre-fit residual
        x += K @ y
        P = (np.eye(len(x)) - K @ H) @ P
        xs.append(x)
        Ps.append(P)
        
    return np.array(xs), np.array(Ps)





#%%
#generate data
estimator = CurvatureEstimator(mode='otsu')
dir = '../Aufnahmen/data/debug'
#list files in directory
files = os.listdir(dir)
files = natsort.natsorted(files)
print('len files: ',len(files))
y,phi,k = [],[],[]
for i,file in enumerate(files):
        print('i: ',i)
        img = cv2.imread(os.path.join(dir,file),0)
        estimator.process_frame(img,debug=False)
        curvature =estimator.get_curvature()
        heading_angle = estimator.get_heading_angle()
        offset = estimator.get_lateral_offset()
        k.append(curvature)
        phi.append(heading_angle)
        y.append(offset)
        if i > 500:
            break
#%%
y = np.array(y)
phi = np.array(phi)
k= np.array(k)
y_nans = np.where(np.isnan(y))[0]
phi_nans = np.where(np.isnan(phi))[0]
k_nans = np.where(np.isnan(k))[0]
print('y_nans: ',y_nans)
print('phi_nans: ',phi_nans)
print('k_nans: ',k_nans)
for i in y_nans:
    y[i] = np.mean(y[i-3:i])
for i in phi_nans:
    phi[i] = np.mean(phi[i-3:i])
for i in k_nans:
    k[i] = np.mean(k[i-3:i])
        
#%%
# Define constants
lr = lf = 0.18
# Observation matrix
H = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0]
])
# Process noise covariance; represents the uncertainty in the model itself
#Increasing the values in Q will make the Kalman filter place less trust in the model's predictions and more trust in the actual measurements
Q = np.eye(4)*0.1
Q[0,0]=0.15
Q[1,1]=0.0005
# Measurement noise covariance; represents the uncertainty in the measurements
R = np.eye(3)*0.01
# Initial state
x0 = np.array([y[0], phi[0], k[0], 0])
# Initial state covariance
P0 = np.eye(4)

# Observations
Z = np.array([y, phi, k]).T

# Control inputs
df_control = pd.read_csv('../Aufnahmen/data/control.csv')
U = df_control[['steering_angle', 'speed']].values

#load timestamps of images
df_timestamps = pd.read_csv('../Aufnahmen/data/debug_timestamps.csv')
# # Calculate time step
dt = (pd.to_datetime(df_timestamps['datetime']).diff().mean()).total_seconds()
print('mean dt:', dt)

# Initialize A and B with the first speed value
A = np.array([
    [1, dt*df_control['speed'][0], 0, 0],
    [0, 1, dt*df_control['speed'][0], 0],
    [0, 0, 1, dt],
    [0, 0, 0, 1]
])
B = np.array([
    [dt*lr/(lr+lf)*df_control['speed'][0], 0],
    [0, 0],
    [0, 0],
    [0, 0]
])


# Store all parameters and initial values in a dictionary
params = {'A': A, 'B': B, 'H': H, 'Q': Q, 'R': R, 'x0': x0, 'P0': P0, 'Z': Z, 'U': U}

# Run the Kalman filter
xs, Ps = kalman_filter(params)

# Corrected curvature values
corrected_curvature = xs[:,2]
print('len corrected_curvature: ',len(corrected_curvature))

plt.plot(k, label='Measured curvature')
plt.plot(corrected_curvature, label='Corrected curvature')
plt.legend()
plt.show()

plt.plot(y, label='Measured lateral offset')
plt.plot(xs[:,0], label='Corrected lateral offset')
plt.legend()
plt.title('Lateral offset')
plt.show()

plt.plot(phi, label='Measured heading angle')
plt.plot(xs[:,1], label='Corrected heading angle')
plt.legend()
plt.title('Heading angle')
plt.show()
# %%
df = pd.DataFrame({'corrected_y': xs[:,0], 'corrected_phi': xs[:,1], 'corrected_k': xs[:,2],'measured_y': y, 'measured_phi': phi, 'measured_k': k})
df.to_csv('../Aufnahmen/data/kalmanVars.csv', index=False)
print('saved to ../Aufnahmen/data/kalmanVars.csv')
# %%
