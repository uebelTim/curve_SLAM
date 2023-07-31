import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import correlate
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

def get_distance_column(df):
    # Convert the 'datetime' column to datetime format
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Calculate the time difference in seconds
    df['time_diff'] = df['datetime'].diff().dt.total_seconds()

    # Forward fill to handle the NaN value in the first row
    df['time_diff'].fillna(method='ffill', inplace=True)

    # Compute the distance traveled at each time step (speed * time)
    df['distance_traveled'] = df['speed'] * df['time_diff']

    # Calculate the cumulative distance traveled
    df['distance'] = df['distance_traveled'].cumsum()
    df.drop(['time_diff', 'distance_traveled'], axis=1, inplace=True)

    df.head()
    return df

def resample_to_distance(df,interv=0.1):
    plt.figure(figsize=(10, 5))
    plt.plot( df['curvature'])
    plt.title('original curvature per time')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    # Calculate time difference in seconds
    df['delta_time'] = df.index.to_series().diff().dt.total_seconds()
    df['delta_time'] = df['delta_time'].fillna(0)
    # Calculate distance in each time step (assuming speed is in m/s)
    df['distance'] = df['speed'] * df['delta_time']
    df['cumulative_distance'] = df['distance'].cumsum()
    new_distance_points = np.arange(df['cumulative_distance'].min(), df['cumulative_distance'].max(), interv)

    # Define an interpolation function based on the original data
    interp_func = interp1d(df['cumulative_distance'], df['curvature'])
    # Use the interpolation function to get the curvature values at the new distance points
    resampled_curvature = interp_func(new_distance_points)
    # Create a new DataFrame for the resampled data
    df_resampled = pd.DataFrame({
        'cumulative_distance': new_distance_points,
        'curvature': resampled_curvature
    })
    plt.figure(figsize=(10, 5))
    plt.plot(df_resampled['cumulative_distance'], df_resampled['curvature'])
    plt.title('curvature per distance')
    plt.xlabel('distance (m)')
    plt.show()
    
    return df_resampled,df
        
    

    
df = pd.read_csv('../Aufnahmen/data/correctedCurvature.csv')
df = get_distance_column(df)
curvature = df['corrected_k'].values

#smooth_curvature = gaussian_filter(curvature, 2)

df_distance,df_time = resample_to_distance(df.rename(columns={'corrected_k':'curvature'}))

# # Compute the cross-correlation of the smoothed data with itself
cross_corr = correlate(curvature, curvature, mode='full')
cross_corr = cross_corr[cross_corr.size // 2:]
time_delays = df_time['delta_time'].cumsum().values - df_time['delta_time'].values[0]
#get the index of the highest peak
loop_idx = np.argmax(cross_corr[10:])+10
loop_time_points =int(loop_idx)
loop_time =df_time['delta_time'].cumsum().values[loop_time_points]
print(f'loop length: {loop_time_points} timesteps, {loop_time} seconds ')

cross_corr_dist = correlate(df_distance['curvature'].values, df_distance['curvature'].values, mode='full')
cross_corr_dist = cross_corr_dist[cross_corr_dist.size // 2:]
distance_delays = df_distance['cumulative_distance'].values - df_distance['cumulative_distance'].values[0]
loop_dist_points = np.argmax(cross_corr_dist[10:])+10
loop_distance = df_distance['cumulative_distance'].values[loop_dist_points]
print(f'loop length: {loop_dist_points} distancesteps, {loop_distance} meters ')

fig, ax = plt.subplots(2, 1, figsize=(10, 10))
ax[0].plot(time_delays, cross_corr)
ax[0].set_title('Autocorrelation with respect to time')
ax[0].set_xlabel('Time delay (s)')
ax[0].set_ylabel('Autocorrelation')
ax[0].axvline(loop_time, color='k', linestyle='--')
ax[1].plot(distance_delays, cross_corr_dist)
ax[1].set_title('Autocorrelation with respect to distance')
ax[1].set_xlabel('Distance delay (m)')
ax[1].set_ylabel('Autocorrelation')
ax[1].axvline(loop_distance, color='k', linestyle='--')
plt.tight_layout()
plt.show()

curvature = df_distance['curvature'].values
smooth_curvature = savgol_filter(curvature, 5, 2)
distance = df_distance['cumulative_distance'].values

plt.figure(figsize=(10, 6))
plt.plot(df_distance['curvature'], label='curvature')
plt.plot(smooth_curvature, label='smooth curvature')
plt.legend()
plt.title('Curvature per distance')
plt.xticks(np.arange(0, len(curvature), 50))
plt.grid()
plt.yticks(np.arange(-2, 2.2, 0.2))	
plt.show()



threshold =0.2
segments = []
i=0
j=0
while i < (loop_dist_points+1):
    j=i+1
    # if j >= len(k):
    #     break
    while abs(smooth_curvature[j] - smooth_curvature[i]) < threshold:
        # print('i: ',i)
        # print('diff: ',k.loc[i,'diff'])
        j+=1
        # if j >= len(k):
        #     break
    if (j-i) > 10:
        segments.append([i,j,round(smooth_curvature[i:j].mean(),2),round(distance[j]-distance[i],2)])
        i=j
    else:
        i+=1
print('segments: ',segments[:20])

search_width = int(loop_dist_points * 0.1)
tolerance_k = 0.1
matches = []
for segment in segments:
    plt.figure(figsize=(10, 6))
    plt.plot(smooth_curvature)
    plt.title(f'matched segments for {segment}')
    plt.xticks(np.arange(0, len(smooth_curvature), 20), rotation=90)
    plt.grid()
    plt.axvline(segment[0], color='r', linestyle='--')
    plt.axvline(segment[1], color='r', linestyle='--')
    print('search for segment: ',segment)
    middle = segment[0] + (segment[1]-segment[0])//2
    mean_k = segment[2]
    search_idx = middle + loop_dist_points
    match =[ (segment[0],segment[1])]
    while search_idx < len(smooth_curvature):
        print('search at idx: ',search_idx)
        #same_k = np.full((search_width),-1)
        idxs = np.where(abs(smooth_curvature[search_idx-search_width:search_idx+search_width]-mean_k) < tolerance_k)[0]
        print('found idxs: ',idxs)
        if len(idxs) > 0:
            start = idxs[0] + search_idx-search_width
            end = idxs[-1] + search_idx-search_width
            match.append((start,end))
            plt.axvline(start, color='k', linestyle='--')
            plt.axvline(end, color='k', linestyle='--')
            plt.axvline(search_idx, color='g', linestyle='--')
        
        search_idx += loop_dist_points
    matches.append(match)
    plt.show()
    
print('matches: ',matches)	
    

