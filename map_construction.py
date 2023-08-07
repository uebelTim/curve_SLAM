import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def interpolate_group(group,target_length):
    entries = max(group['loop_idx'])+1
    print('entries: ',entries)
    print('target_length: ',target_length)
    indices = np.linspace(0,1,int(entries))#from 0 to max loop_idx with max loop_idx entries
    new_indices = np.linspace(0,1,int(target_length))#from 0 to max loop_idx with target_length entries
    interp_x_left = interp1d(indices,group['x_left'],kind='cubic')
    interp_y_left = interp1d(indices,group['y_left'],kind='cubic')
    interp_x_right = interp1d(indices,group['x_right'],kind='cubic')
    interp_y_right = interp1d(indices,group['y_right'],kind='cubic')
    interp_x_left = interp_x_left(new_indices)
    interp_y_left = interp_y_left(new_indices)
    interp_x_right = interp_x_right(new_indices)
    interp_y_right = interp_y_right(new_indices)
    df_borders = pd.DataFrame({'x_left':interp_x_left,'y_left':interp_y_left,'x_right':interp_x_right,'y_right':interp_y_right})
    df_borders['loop_nr'] = group['loop_nr'].iloc[0]
    df_borders['loop_idx'] = np.arange(len(df_borders))
    print('entries after interpolation: ',len(df_borders))
    return df_borders
    

def get_mean_borders(df):
    target_length = max(df['loop_idx'])+1
    print('target_length: ',target_length)
    df_interpolated = pd.DataFrame(columns=['x_left','y_left','x_right','y_right'])
    df_interpolated = pd.concat([interpolate_group(group, target_length) for _, group in df.groupby('loop_nr')])
    print(df_interpolated.head())
    print(df_interpolated.tail())
    print('len df_interpolated: ',len(df_interpolated))
    plt.figure(figsize=(10,10))
    plt.scatter(df_interpolated['x_left'],df_interpolated['y_left'],s=2,c='b',label='left_border')
    plt.scatter(df_interpolated['x_right'],df_interpolated['y_right'],s=2,c='g',label='right_border')
    plt.legend()
    plt.title('interpolated')
    plt.show()
    x_left_mean = df_interpolated.groupby('loop_idx')['x_left'].transform('mean')
    y_left_mean = df_interpolated.groupby('loop_idx')['y_left'].transform('mean')
    x_right_mean = df_interpolated.groupby('loop_idx')['x_right'].transform('mean')
    y_right_mean = df_interpolated.groupby('loop_idx')['y_right'].transform('mean')
    df_interpolated['x_left_mean'] = x_left_mean
    df_interpolated['y_left_mean'] = y_left_mean
    df_interpolated['x_right_mean'] = x_right_mean
    df_interpolated['y_right_mean'] = y_right_mean
    print(df_interpolated.head())
    print(df_interpolated.tail())
    plt.figure(figsize=(10,10))
    plt.scatter(x_left_mean,y_left_mean,s=2,c='b',label='left_border')
    plt.scatter(x_right_mean,y_right_mean,s=2,c='b',label='right_border')
    plt.title('mean')
    plt.show()
    
    return df_interpolated
    
def construct_track(df,first_loop_closures,df_frame_measurements=None):
    print('constructing track')
    #drop nans
    
    distance =0.375
    df['x_left'] = df['x'] + distance * np.sin(df['theta'])
    df['y_left'] = df['y'] - distance * np.cos(df['theta'])
    df['x_right'] = df['x'] - distance * np.sin(df['theta'])
    df['y_right'] = df['y'] + distance * np.cos(df['theta'])

    df['loop_nr'] = np.nan
    df['loop_idx'] = np.nan
    loops = first_loop_closures
    for i in range(1,len(loops)):
        df['loop_nr'].iloc[loops[i-1][0]:loops[i][0]] = i
        df['loop_idx'][loops[i-1][0]:loops[i][0]] = df.iloc[loops[i-1][0]:loops[i][0]].index- loops[i-1][0]
    df = df.dropna()
    print(df.head())
    
    plt.figure(figsize=(10,10))
    plt.scatter(df['x'],df['y'],s=5,c='g',label='poses')
    dx = np.cos(df['theta'][500]) # calculate the change in x
    dy = np.sin(df['theta'][500])  # calculate the change in y
    plt.plot([df['x'][500],df['x'][500]+dx],[df['y'][500],df['y'][500]+dy],c='r',label='theta')
    plt.scatter(df['x_left'],df['y_left'],s=2,c='b',label='left_border')
    plt.scatter(df['x_right'],df['y_right'],s=2,c='b',label='right_border')
    #plot mean borderlines
    plt.plot(df.groupby('loop_idx')['x_left'].mean().values,df.groupby('loop_idx')['y_left'].mean().values,c='r',label='left_border',linewidth=3)
    plt.plot(df.groupby('loop_idx')['x_right'].mean().values,df.groupby('loop_idx')['y_right'].mean().values,c='r',label='right_border',linewidth=3)
    plt.show
    print('*'*50)
    print('theta[500]: ',np.degrees(df['theta'][500]))
    
    #interpolate df_frame_measurements
    # df_frame_measurements['x'] = np.interp(df_frame_measurements['datetime'],df['datetime'],df['x'])

    get_mean_borders(df)

    
    
    if df_frame_measurements is not None:
        start = df['datetime'].iloc[0]
        end = df['datetime'].iloc[-1]
        df_frame_measurements = df_frame_measurements[(df_frame_measurements['datetime'] >= start) & (df_frame_measurements['datetime'] <= end)]
        indices = np.linspace(0,1,len(df_frame_measurements))
        new_indices = np.linspace(0,1,len(df))
        offset_interpolated = interp1d(indices,df_frame_measurements['lateral_offset'])
        heading_interpolated = interp1d(indices,df_frame_measurements['heading_angle'])
        offset_interpolated = offset_interpolated(new_indices)
        heading_interpolated = heading_interpolated(new_indices)
        heading_interpolated = np.radians(heading_interpolated)
        #left from tangent centerline is positive
        df_corrected = df.copy()
        print('len df_corrected: ',len(df_corrected))
        print('len offset_interpolated: ',len(offset_interpolated))
        print('len heading_interpolated: ',len(heading_interpolated))
        df_corrected['x_left'] = df['x'] - distance * np.sin(df['theta']-heading_interpolated)#+offset_interpolated*np.sin(df['theta'])
        df_corrected['y_left'] = df['y'] + distance * np.cos(df['theta']-heading_interpolated)#-offset_interpolated*np.cos(df['theta'])
        df_corrected['x_right'] = df['x'] + distance * np.sin(df['theta']-heading_interpolated)#-offset_interpolated*np.sin(df['theta'])
        df_corrected['y_right'] = df['y'] - distance * np.cos(df['theta']-heading_interpolated)#+offset_interpolated*np.cos(df['theta'])

        plt.figure(figsize=(10,10))
        plt.scatter(df['x_left'],df['y_left'],s=2,c='y',label='left_border')
        plt.scatter(df['x_right'],df['y_right'],s=2,c='y',label='right_border')
        plt.scatter(df_corrected['x_left'],df_corrected['y_left'],s=2,c='b',label='left_border_corrected')
        plt.scatter(df_corrected['x_right'],df_corrected['y_right'],s=2,c='r',label='right_border_corrected')
        plt.legend()
        plt.show()
        
        get_mean_borders(df_corrected)
    
    return df
        

corrected_poses = pd.read_csv('../Aufnahmen/data/corrected_poses.csv')
corrected_poses = corrected_poses[:-100]
corrected_poses.to_csv('../Aufnahmen/data/corrected_poses_test.csv',index=False)
plt.figure(figsize=(10,10))
plt.scatter(corrected_poses['x'],corrected_poses['y'],s=2,c='g')
#make cross at pose 2
plt.scatter(corrected_poses['x'].iloc[2],corrected_poses['y'].iloc[2],s=100,c='r',marker='x')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
print(corrected_poses.head())

#-150 because cut start of poses
loop_closures = [[(152, 211), (743, 783), (1331, 1389), (1916, 1966), (2483, 2539)], [(278, 391), (871, 965), (1460, 1550), (2049, 2138), (2614, 2711)]]
loop_closures = np.array(loop_closures) - 150
print('loop closures: ',loop_closures)

df_track = construct_track(corrected_poses,loop_closures[0])

#plt.plot(mean_line_x,mean_line_y,c='r',label='mean_line')


frame_measurements = pd.read_csv('../Aufnahmen/data/kalmanVars.csv')
frame_measurements.rename(columns={'corrected_y':'lateral_offset','corrected_phi':'heading_angle'},inplace=True)
print(frame_measurements.head())
start = df_track['datetime'].iloc[0]
end = df_track['datetime'].iloc[-1]
frame_measurements = frame_measurements[(frame_measurements['datetime'] >= start) & (frame_measurements['datetime'] <= end)]
print(frame_measurements.head())

df_track = construct_track(corrected_poses,loop_closures[0],frame_measurements)
