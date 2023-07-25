from filterpy.kalman import KalmanFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def kalman_filter(zs, Q, R, v_xs, l_r, l_f, delta_fs):
    # Create Kalman filter object
    kf = KalmanFilter(dim_x=4, dim_z=4)

    # Define the initial state
    kf.x = np.array([[-0.06],[-0.8],[0.69],[3]])

    # Define the initial uncertainty
    kf.P *= 1000.

    # Define the process noise covariance
    kf.Q = Q

    # Define the measurement noise covariance
    kf.R = R
    
    kf.H =np.eye(4)

    # Initialize list to hold state estimates
    states = []

    # Loop over all measurements
    for z, v_x, delta_f in zip(zs, v_xs, delta_fs):
        #print('z:', z)
        # Update the state transition matrix
        kf.F = np.array([[0, v_x, 0, 0],
                         [0, 0, v_x, 0],
                         [0, 0, 0, 1],
                         [0, 0, 0, 0]])

        # Update the control input matrix
        kf.B = np.array([[l_r / (l_r + l_f) * v_x],
                         [0],
                         [0],
                         [0]])

        kf.predict(u=delta_f)
        kf.update(z)
        states.append(kf.x)

    return states


def load_data():
    df_measurements = pd.read_csv('../Aufnahmen/data/frame_measurements.csv')
    df_measurements['datetime'] = pd.to_datetime(df_measurements['datetime'])
    #heading to rad
    df_measurements['heading_angle'] = df_measurements['heading_angle'].apply(lambda x: x * np.pi/180 )
    df_speed = pd.read_csv('../Aufnahmen/data/speed.csv')
    df_speed['datetime'] = pd.to_datetime(df_speed['datetime'])
    df = pd.merge_asof(df_measurements, df_speed, on='datetime', direction='nearest')
    df_ackermann = pd.read_csv('../Aufnahmen/data/ackermann.csv')
    df_ackermann['datetime'] = pd.to_datetime(df_ackermann['datetime'])
    df_ackermann.drop(['speed'], axis=1, inplace=True)
    df = pd.merge_asof(df, df_ackermann, on='datetime', direction='nearest')
    #calculate derivtive of k
    df['time_seconds'] = (df['datetime'] - df['datetime'].iloc[0]).dt.total_seconds()
    # Calculate the differences in curvature and time
    df['curvature_diff'] = df['curvature'].diff()
    df['time_diff'] = df['time_seconds'].diff()
    # Calculate the derivative of curvature with respect to time
    df['curvature_derivative'] = df['curvature_diff'] / df['time_diff']
    df=df.drop(['time_seconds', 'curvature_diff', 'time_diff'], axis=1)
    #sort df in order 'datetime','lateral_offset','heading_angle','curvature','curvature_derivative','speed','steering_angle'
    df = df[['datetime','lateral_offset','heading_angle','curvature','curvature_derivative','speed','steering_angle']]
    df.fillna(0, inplace=True)
    df.to_csv('../Aufnahmen/data/kalmanVars.csv', index=False)
    print(df[:30])
    return df


# Define your measurements here
# Each row in zs should be a set of three measurements: [y, phi, k, dk/dt]
vars = load_data()



vars = vars[:1000].values
dt,y, phi, k, dkdt, speed, steering_angle = vars[:,0], vars[:,1], vars[:,2], vars[:,3], vars[:,4], vars[:,5], vars[:,6]
zs = [[[y], [phi], [k], [dkdt]] for y, phi, k, dkdt in zip(y, phi, k, dkdt)]
#to numpy array

print(np.shape(zs))
print(zs[:10])


# Define your process noise covariance here
Q = np.eye(4) * 0.001

# Define your measurement noise covariance here
R = np.eye(4) * 0.1

# Define your vehicle parameters here
v_x = speed  # longitudinal speed
l_r = 0.18  # length from the rear axle to the CG 18cm
l_f = 0.18  # length from the front axle to the CG
delta_f = steering_angle  # front steering angle

# # Run the Kalman filter
states = kalman_filter(zs, Q, R, v_x, l_r, l_f, delta_f)
# # Print the state estimates
corrected_curvature = []
corrected_offset = []
corrected_heading = []
for state in states:
    #print(state)
    corrected_offset.append(state[0])
    corrected_curvature.append(state[2])
    corrected_heading.append(state[1])

plt.plot(k, label='measured curvature', color='blue')
plt.plot(corrected_curvature, label='corrected curvature', color='red')
plt.title('Curvature')
plt.show()

plt.plot(corrected_curvature, label='corrected curvature')
plt.title('corected Curvature')
plt.show()

plt.plot(y, label='measured offset', color='blue')
plt.plot(corrected_offset, label='corrected offset', color='red')
plt.legend()
plt.title('Offset')
plt.show()

plt.plot(phi, label='measured heading', color='blue')
plt.plot(corrected_offset, label='corrected offset', color='red')
plt.title('corrected heading')
plt.show()