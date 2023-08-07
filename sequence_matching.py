import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import correlate
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

class SequenceMatcher:
    def __init__(self,df=None):
        if df is not None:
            self.find_matching_sequences(df)
    
    def find_matching_sequences(self,df):
        assert isinstance(df, pd.DataFrame)
        assert 'curvature' in df.columns
        assert 'speed' in df.columns
        assert 'datetime' in df.columns
        self.df_distance, self.df_time = self.resample_to_distance(df)
        self.loop_dist_points,self.loop_distance = self.get_loop_length()
        dist_matches = self.match_sequences()
        time_matches =  []
        for segments in dist_matches:
            segment_matches = []
            for match in segments:
                print('match: ',match)
                dist_start = self.df_distance.loc[match[0],'cumulative_distance']
                dist_end = self.df_distance.loc[match[1],'cumulative_distance']
                #find index of time df where cumulative distance is closest to cum_dist
                time_idx_start = (np.abs(self.df_time['cumulative_distance'] - dist_start)).idxmin()
                time_idx_end = (np.abs(self.df_time['cumulative_distance'] - dist_end)).idxmin()
                segment_matches.append((time_idx_start, time_idx_end))
            time_matches.append(segment_matches)
        self.matches = time_matches
        print('matches: ',self.matches)
        
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(len(self.df_time)),self.df_time['curvature'])
        for match in self.matches[1]:
            print('match: ',match)
            plt.axvline(match[0], color='k', linestyle='--')
            plt.axvline(match[1], color='k', linestyle='--')
        plt.title('curvature per time')
        plt.show()
        
    def get_matches(self):
        return self.matches
        
    def resample_to_distance(self, df,interv=0.1):
        plt.figure(figsize=(10, 5))
        plt.plot( df['curvature'])
        plt.title('original curvature per time')
        df['datetime'] = pd.to_datetime(df['datetime'])
        #df = df.set_index('datetime')
        df.rename(columns={'curvature': 'curvature'}, inplace=True)
        # Calculate time difference in seconds
        df['delta_time'] = df['datetime'].diff().dt.total_seconds()
        df['delta_time'] = df['delta_time'].fillna(0)
        # Calculate distance in each time step (assuming speed is in m/s)
        df['distance'] = df['speed'] * df['delta_time']
        df['cumulative_distance'] = df['distance'].cumsum()
        new_distance_points = np.arange(df['cumulative_distance'].min(), df['cumulative_distance'].max(), interv)

        # Define an interpolation function based on the original data
        interp_func_curvature = interp1d(df['cumulative_distance'], df['curvature'])
        df = df.set_index('datetime')
        interp_func_time = interp1d(df['cumulative_distance'], df.index.astype(int))
        df.reset_index(inplace=True)
        # Use the interpolation function to get the curvature values at the new distance points
        resampled_curvature = interp_func_curvature(new_distance_points)
        resampled_time = pd.to_datetime(interp_func_time(new_distance_points))
        #add index of original df to resampled df
        print('len original df: ',len(df),' len resampled df: ',len(resampled_curvature))
        ratio = len(df)/len(resampled_curvature)
        print('ratio: ',ratio)
        # Create a new DataFrame for the resampled data
        df_resampled = pd.DataFrame({
            
            'cumulative_distance': new_distance_points,
            'curvature': resampled_curvature
        })
        print(df_resampled[:30])
        plt.figure(figsize=(10, 5))
        plt.plot(df_resampled['cumulative_distance'], df_resampled['curvature'])
        plt.title('curvature per distance')
        plt.xlabel('distance (m)')
        plt.show()
        
        # df=pd.merge_asof(df, df_resampled, on='cumulative_distance', direction='nearest',tolerance=0.05)
        # print(df[:30])
        return df_resampled,df
    
    def get_loop_length(self):
        autocorr_dist = correlate(self.df_distance['curvature'].values, self.df_distance['curvature'].values, mode='full')
        autocorr_dist = autocorr_dist[autocorr_dist.size // 2:]
        distance_delays = self.df_distance['cumulative_distance'].values - self.df_distance['cumulative_distance'].values[0]
        loop_dist_points = np.argmax(autocorr_dist[10:])+10
        loop_distance = self.df_distance['cumulative_distance'].values[loop_dist_points]
        print(f'loop length: {loop_dist_points} distancesteps, {loop_distance} meters ')
        plt.figure(figsize=(10, 5))
        plt.plot(distance_delays, autocorr_dist)
        plt.title('Autocorrelation of curvature per distance')
        plt.axvline(distance_delays[loop_dist_points], color='k', linestyle='--')
        plt.xlabel('distance (m)')
        #add sexond x axis with index
        plt.tight_layout()
        plt.show()
        
        return loop_dist_points,loop_distance
    
    def match_sequences(self, thresh_segments=0.2):
        curvature = self.df_distance['curvature'].values
        smooth_curvature = savgol_filter(curvature, 5, 2)
        distance = self.df_distance['cumulative_distance'].values
        loop_dist_points = self.loop_dist_points
        threshold =thresh_segments
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
        print('segments: ',segments)
        
        search_width = int(loop_dist_points * 0.1)
        tolerance_k = 0.2
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
        
        return matches
                        

df = pd.read_csv('../Aufnahmen/data/correctedCurvature.csv')
df = df.rename(columns={'corrected_k':'curvature'})
print(df.head())
print('len df: ',len(df))
matcher =SequenceMatcher(df)
matches = matcher.get_matches()
print('matches: ',matches)
df['datetime'] = pd.to_datetime(df['datetime'])
match_times = []
matches_dict = {'sequence':[],'matches':{'start':[],'end':[]}}

for i,sequence in enumerate(matches):
    segment_times = []
    for j,match in enumerate(sequence):
        segment_times.append((df.loc[match[0],'datetime'],df.loc[match[1],'datetime']))
    match_times.append(segment_times)
print('match times: ',match_times)       

 
matches_dict =  {"sequence_" + str(i+1): {"match_" + str(j+1): {"start": df.loc[match[0],'datetime'], "end": df.loc[match[1],'datetime']} 
                                   for j, match in enumerate(sequence)} for i, sequence in enumerate(matches)}
print('matches dict: ',matches_dict)

#print all start times of matches
for i,sequence in enumerate(matches):
    print(f'sequence {i+1}: ')
    for j,match in enumerate(sequence):
        print(df.loc[match[0],'datetime'])
#sequence 1 starts:
#2023-06-29 13:55:47.659475
#2023-06-29 13:55:59.475904
#2023-06-29 13:56:11.238479
#2023-06-29 13:56:22.936184
#2023-06-29 13:56:34.268945