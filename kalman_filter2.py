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
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
import seaborn as sns

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

def remove_outliers(data):
    plt.figure()
    plt.plot(data)
    plt.title('data')
    plt.show()
    # Remove outliers
    df = pd.DataFrame(data,columns=['data'])
    df.boxplot(column=['data'])
    plt.show()
    q75, q25 = np.percentile(df['data'], [80 ,20])
    min = q25 - (1.5*(q75-q25))
    max = q75 + (1.5*(q75-q25))
    #make outliers nan
    #get indices of outliers
    idx = df.loc[(df['data'] < min) | (df['data'] > max)].index
    #replace outliers with mean of non-outlier neighbors
    for i in idx:
        #get index of previous non-outlier
        idx_prev = df.loc[:i].loc[(df['data'] >= min) & (df['data'] <= max)].index[-1]
        idx_next = df.loc[i:].loc[(df['data'] >= min) & (df['data'] <= max)].index[0]
        #replace outlier with mean of previous and next non-outlier
        df.loc[i,'data'] = (df.loc[idx_prev,'data']+df.loc[idx_next,'data'])/2

    plt.plot(df['data'])
    plt.title('cleaned data')
    plt.show()
    return df['data'].values
    



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
        if i > 1000:
            break
#y,phi,k = np.array(y),np.array(phi),np.array(k)
print('max min y: ',max(y),min(y))

      
#%%
#handle nans
y_nans = np.where(np.isnan(y))[0]
phi_nans = np.where(np.isnan(phi))[0]
k_nans = np.where(np.isnan(k))[0]
print('y_nans: ',y_nans)
print('phi_nans: ',phi_nans)
print('k_nans: ',k_nans)
for i in y_nans:
    next_not_nan = np.where(~np.isnan(y[i:]))[0][0]
    y[i] = (y[i-1]+next_not_nan)/2
    print(f'y{i} replaced with {y[i]}')
for i in phi_nans:
    next_not_nan = np.where(~np.isnan(phi[i:]))[0][0]
    phi[i] = (phi[i-1]+next_not_nan)/2
for i in k_nans:
    next_not_nan = np.where(~np.isnan(k[i:]))[0][0]
    k[i] = (k[i-1]+next_not_nan)/2

#%%
#handle outliers
y = remove_outliers(y)

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
#Increasing the values in Q will make the Kalman filter place  more trust in the actual measurements and less trust in the model's predictions and
Q = np.eye(4)
Q[0,0]=0.5
Q[1,1]=0.0002
Q[2,2]=0.2
# Measurement noise covariance; represents the uncertainty in the measurements
R = np.eye(3)*0.01
# Initial state
x0 = np.array([y[0], phi[0], k[0], 0])
# Initial state covariance
P0 = np.eye(4)

# Observations
Z = np.array([y, phi, k]).T



#load timestamps of images
df_timestamps = pd.read_csv('../Aufnahmen/data/debug_timestamps.csv')
df_timestamps['datetime'] = pd.to_datetime(df_timestamps['datetime'])
# # Calculate time step
#dt = (pd.to_datetime(df_timestamps['datetime']).diff().mean()).total_seconds()
dt = 1/30
print('mean dt:', dt)


# Control inputs
df_control = pd.read_csv('../Aufnahmen/data/control.csv')
df_control['datetime'] = pd.to_datetime(df_control['datetime'])
df_timestamps = pd.merge_asof(df_timestamps, df_control, on='datetime', direction='nearest')
print(df_timestamps.head())
U = df_timestamps[['steering_angle', 'speed']].values

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
df =pd.DataFrame({ 'corrected_k': xs[:,2]})
df['speed'] = df_timestamps['speed']
df['datetime'] = df_timestamps['datetime']
df.to_csv('../Aufnahmen/data/correctedCurvature.csv', index=False)
print('saved to ../Aufnahmen/data/correctedCurvature.csv')
# %%
