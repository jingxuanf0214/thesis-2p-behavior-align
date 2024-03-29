
import imaging_behavior_functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import scipy.io
import os
import mat73
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from scipy.stats import iqr
from PIL import Image
import cv2
from IPython.display import display, clear_output
import time
import seaborn as sns
from scipy.stats import iqr
from scipy.ndimage import gaussian_filter1d
import glob
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

base_path = "//research.files.med.harvard.edu/neurobio/wilsonlab/Jingxuan/processed/hDeltaB_imaging/qualified_sessions/bar_jump/"
example_path_data = base_path+"20230518-8_hDeltaB_EPG_syntGCAMP7f_fly2_long_jump_15/data/"
example_path_results = base_path+"20230518-8_hDeltaB_EPG_syntGCAMP7f_fly2_long_jump_15/results/"
trial_num = 1
odor_threshold = 5

is_mat73, roi_df, dff_raw, kinematics_raw, preprocessed_vars_ds, preprocessed_vars_odor = imaging_behavior_functions.load_intermediate_mat(example_path_data,trial_num)
behav_df = imaging_behavior_functions.make_df_behavior(dff_raw, preprocessed_vars_ds, preprocessed_vars_odor,trial_num,ball_d = 9)
xPos, yPos = imaging_behavior_functions.reconstruct_path(behav_df, ball_d = 9)
roi_names, hdeltab_index, epg_index, hdeltab_sequence, epg_sequence = imaging_behavior_functions.get_roi_seq(roi_df)
dff_all_rois, dff_time = imaging_behavior_functions.load_dff_raw(is_mat73, dff_raw)
neural_df = imaging_behavior_functions.make_df_neural(dff_all_rois, dff_time, roi_names, hdeltab_index, epg_index, hdeltab_sequence, epg_sequence)
#combined_df = imaging_behavior_functions.combine_df(behav_df, neural_df)

def store_odor_on_off_times(behav_df):
    # Initialize an empty list to store (on_time, off_time) tuples
    on_off_times = []
    
    # Track whether we are currently in an "on" state and its start time
    on_state = False
    on_time = None
    
    for i in range(len(behav_df) - 1):
        # Check if current state is "on"
        if behav_df.loc[i, 'odor'] > 5 and not on_state:
            on_state = True
            on_time = behav_df.loc[i, 'time']
        # Check for transition from "on" to "off"
        elif behav_df.loc[i, 'odor'] <= 5 and on_state:
            on_state = False
            off_time = behav_df.loc[i, 'time']
            # Store the on-off tuple
            on_off_times.append((on_time, off_time))
    
    # Check if the last state was "on" and has no "off"; if so, do nothing
    # (The logic to discard the last "on" event if it doesn't have an "off" event is inherently handled by the loop)
    
    return on_off_times

#odor_events = store_odor_on_off_times(behav_df)
#print(odor_events)
def cluster_odor_events_temporal(events, time_interval_threshold):
    """
    Clusters odor events based on temporal closeness.
    
    Parameters:
    - events: List of (on_time, off_time) tuples for each odor event.
    - time_interval_threshold: Maximum allowed gap between events to be in the same cluster.
    
    Returns:
    - List of odor event clusters, each represented as a list of events.
    """
    clusters = []
    current_cluster = [events[0]]
    
    for i in range(1, len(events)):
        if events[i][0] - current_cluster[-1][1] <= time_interval_threshold:
            current_cluster.append(events[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [events[i]]
    clusters.append(current_cluster)  # Add the last cluster
    
    return clusters

def label_df_with_clusters(df, clusters):
    """
    Labels the DataFrame with cluster IDs based on identified clusters.
    
    Parameters:
    - df: DataFrame with 'time' column.
    - clusters: List of odor event clusters.
    
    Returns:
    - Modified DataFrame with a 'cluster_label' column.
    """
    df['cluster_label_temporal'] = 0  # Default label for non-odor zones
    for i, cluster in enumerate(clusters, start=1):
        for event in cluster:
            df.loc[(df['time'] >= event[0]) & (df['time'] <= event[1]), 'cluster_label_temporal'] = i
    return df


#clusters = cluster_odor_events_temporal(odor_events, time_interval_threshold)
#print(clusters)
#behav_df = label_df_with_clusters(behav_df, clusters)
#print(behav_df.head(5))
#print("Odor clusters (start, end):", clusters)

def label_df_with_soft_clusters(df, clusters, time_interval_threshold):
    """
    Labels the DataFrame with soft cluster IDs based on identified clusters
    and extends the labeling period by half the time_interval_threshold.

    Parameters:
    - df: DataFrame with 'time' column.
    - clusters: List of odor event clusters.
    - time_interval_threshold: Time interval used to determine the soft cluster extension.

    Returns:
    - Modified DataFrame with a 'soft_cluster_label' column.
    """
    df['soft_cluster_label_temporal'] = 0  # Default label for non-odor zones
    extension = time_interval_threshold / 2  # Calculate the extension period for each cluster

    for i, cluster in enumerate(clusters, start=1):
        # Calculate extended start and end times for the soft cluster
        extended_start = cluster[0][0] - extension
        extended_end = cluster[-1][1] + extension

        # Ensure extended_start is not negative
        extended_start = max(extended_start, df['time'].min())

        # Label the DataFrame rows within the extended cluster period
        df.loc[(df['time'] >= extended_start) & (df['time'] <= extended_end), 'soft_cluster_label_temporal'] = i

    return df
#behav_df = label_df_with_soft_clusters(behav_df, clusters, time_interval_threshold)

def plot_odor_cluster_tp(behav_df, label, example_path_results):
    xPos = behav_df.xPos
    yPos = behav_df.yPos
    x_range = max(xPos) - min(xPos)
    y_range = max(yPos) - min(yPos)
    aspect_ratio = y_range / x_range

    # Set figure dimensions based on data range while keeping unit scale the same
    fig_width = 10  # Width of figure in inches
    fig_height = fig_width * aspect_ratio  # Height is scaled according to the aspect ratio of the data

    plt.figure(figsize=(fig_width, fig_height))

    plt.scatter(xPos, yPos, c=behav_df[label], cmap = 'tab20c',s=3)
    plt.scatter(0, 0, color='red')  # Mark the origin

    # Enforce equal aspect ratio so that one unit in x is the same as one unit in y
    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.title('Fly Trajectory')

    # Save the plot
    plt.savefig(example_path_results+'odor_cluster_'+label+'.png')
    plt.close()  # Close the plot explicitly after saving to free resources

#plot_odor_cluster_tp(behav_df, 'soft_cluster_label_temporal', example_path_results)
#plot_odor_cluster_tp(behav_df, 'cluster_label_temporal', example_path_results)

def calculate_centroids(df):
    """
    Calculates centroids for each odor on segment.
    """
    centroids = []
    segment_indices = []
    is_odor_on = df['odor'] > 5
    start_idx = None
    
    for i in range(len(df)):
        if is_odor_on[i] and start_idx is None:
            start_idx = i  # Start of a new odor segment
        elif not is_odor_on[i] and start_idx is not None:
            # Calculate centroid of the segment
            centroid_x = df.loc[start_idx:i-1, 'xPos'].mean()
            centroid_y = df.loc[start_idx:i-1, 'yPos'].mean()
            centroids.append((centroid_x, centroid_y))
            segment_indices.append((start_idx, i-1))
            start_idx = None
    
    # Handle case where the last segment goes till the end
    if start_idx is not None:
        centroid_x = df.loc[start_idx:, 'xPos'].mean()
        centroid_y = df.loc[start_idx:, 'yPos'].mean()
        centroids.append((centroid_x, centroid_y))
        segment_indices.append((start_idx, len(df)-1))
    
    return centroids, segment_indices
#centroids, segment_indices = calculate_centroids(behav_df)
#print(centroids)

def cluster_centroids_kmeans(centroids, k):
    """
    Clusters centroids using K-Means given a cluster number k.
    
    Parameters:
    - centroids: List of tuples representing the (x, y) coordinates of each centroid.
    - k: The number of clusters to form.
    
    Returns:
    - Array of cluster labels for each centroid, labels start from 1.
    """
    # Convert centroids to a numpy array for clustering
    centroid_array = np.array(centroids)
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=0,n_init="auto").fit(centroid_array)
    
    # Get cluster labels from K-Means (add 1 to start labels from 1 instead of 0)
    cluster_labels = kmeans.labels_ + 1
    
    return cluster_labels

#cluster_labels = cluster_centroids_kmeans(centroids, k)

def assign_spatial_cluster_labels(df, segment_indices, cluster_labels):
    """
    Assigns spatial cluster labels to the DataFrame.
    """
    df['cluster_label_spatial'] = 0  # Default to non-odor zone
    
    for label, (start_idx, end_idx) in zip(cluster_labels, segment_indices):
        df.loc[start_idx:end_idx+1, 'cluster_label_spatial'] = label
    
    return df

#behav_df = assign_spatial_cluster_labels(behav_df, segment_indices, cluster_labels)


def plot_odor_cluster_sp(behav_df, centroids, cluster_labels, example_path_results):
    xPos = behav_df.xPos
    yPos = behav_df.yPos
    x_coords = [centroid[0] for centroid in centroids]
    y_coords = [centroid[1] for centroid in centroids]
    x_range = max(xPos) - min(xPos)
    y_range = max(yPos) - min(yPos)
    aspect_ratio = y_range / x_range

    # Set figure dimensions based on data range while keeping unit scale the same
    fig_width = 10  # Width of figure in inches
    fig_height = fig_width * aspect_ratio  # Height is scaled according to the aspect ratio of the data

    plt.figure(figsize=(fig_width, fig_height))

    plt.scatter(xPos, yPos, c=behav_df.cluster_label_spatial, cmap = 'tab20c',s=3)
    plt.scatter(x_coords, y_coords, c=cluster_labels, cmap='tab20c', s=100)
    plt.scatter(0, 0, color='red')  # Mark the origin

    # Enforce equal aspect ratio so that one unit in x is the same as one unit in y
    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.title('Fly Trajectory')

    # Save the plot
    plt.savefig(example_path_results+'odor_cluster_spatial.png')
    plt.close()  # Close the plot explicitly after saving to free resources

#plot_odor_cluster_sp(behav_df, centroids, cluster_labels, example_path_results)

def smooth_circular_variable(series, sigma=2):
    """Smooths a circular variable by smoothing its sine and cosine components."""
    sin_series = np.sin(series)
    cos_series = np.cos(series)
    sin_smoothed = gaussian_filter1d(sin_series, sigma=sigma)
    cos_smoothed = gaussian_filter1d(cos_series, sigma=sigma)
    smoothed_series = np.arctan2(sin_smoothed, cos_smoothed)
    return (smoothed_series + 2 * np.pi) % (2 * np.pi)

def calculate_straightness(df, window_size):
    # Calculate displacement and path length for each window
    straightness_series = []
    for i in range(len(df) - window_size + 1):
        window = df.iloc[i:i+window_size]
        displacement = np.sqrt((window.iloc[-1]['xPos'] - window.iloc[0]['xPos'])**2 + 
                               (window.iloc[-1]['yPos'] - window.iloc[0]['yPos'])**2)
        path_length = np.sum(np.sqrt(np.diff(window['xPos'])**2 + np.diff(window['yPos'])**2))
        
        straightness = displacement / path_length if path_length != 0 else np.nan
        straightness_series.append(straightness)
    
    # Return as a pandas Series
    return pd.Series(straightness_series)

def vectorized_angle_diff(angles):
    """
    Calculates the difference between consecutive angles in a series,
    accounting for the circular nature of angles.

    Parameters:
    - angles: A Series or array of angles in radians.

    Returns:
    - A Series or array of angle differences, adjusted for circularity.
    """
    # Calculate differences between consecutive angles
    diff = np.diff(angles)
    
    # Adjust differences for circularity
    diff_adjusted = (diff + np.pi) % (2 * np.pi) - np.pi
    
    return diff_adjusted

def circular_variance(angles):
    """
    Calculates the circular variance of a set of heading angles.

    Parameters:
    - angles: A numpy array or pandas Series of angles in radians.

    Returns:
    - Circular variance as a float.
    """
    # Convert angles to cartesian coordinates
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    
    # Compute the mean vector components
    C_mean = np.mean(cos_angles)
    S_mean = np.mean(sin_angles)
    
    # Calculate the length of the mean vector
    R = np.sqrt(C_mean**2 + S_mean**2)
    
    # Calculate circular variance
    variance = 1 - R
    
    return variance

def calculate_curvature(x, y,sigma=2):
    """
    Calculates the curvature for a window of points in a trajectory.

    Parameters:
    - x: A numpy array of x positions.
    - y: A numpy array of y positions.

    Returns:
    - A numpy array of curvature values for each point in the window, except
      the first and last points where curvature cannot be calculated.
    """
    # Ensure inputs are numpy arrays for element-wise operations
    x = imaging_behavior_functions.apply_gaussian_smoothing(x,sigma=2)
    y = imaging_behavior_functions.apply_gaussian_smoothing(y,sigma=2)
    x = np.array(x)
    y = np.array(y)

    # Calculate first derivatives (velocities)
    dx = np.gradient(x)
    dy = np.gradient(y)
    
    # Calculate second derivatives (accelerations)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # Calculate curvature using the formula
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = np.power(dx**2 + dy**2, 1.5)
    curvature = numerator / denominator
    
    # Handle division by zero for stationary points
    curvature[denominator == 0] = np.nan
    
    return curvature

def calculate_movement_metrics(df, window_size):
    # Initialize lists to store the results
    curvature_means = []
    curvature_stds = []
    turning_angles_distribution = {'tight': [], 'moderate': [], 'wide': []}
    heading_variance = []
    forward_side_ratio_means = []
    yaw_velocity_means = []
    yaw_velocity_stds = []
    
    # Define angle thresholds for turning angle categorization
    tight_turn_threshold = np.pi / 6  # 30 degrees
    wide_turn_threshold = np.pi / 3  # 60 degrees
    
    for i in range(0, len(df) - window_size + 1):
        window = df.iloc[i:i + window_size]
        
        # Calculate curvature using a simplified approximation (for demonstration)
        # In practice, more precise methods may be needed
        #dx = np.diff(window['xPos'])
        #dy = np.diff(window['yPos'])
        curvature = calculate_curvature(window['xPos'], window['yPos'],sigma=2)
        #curvature = np.abs(np.diff(dx) / dy[:-1] - dx[:-1] / np.diff(dy)) / (dx[:-1]**2 + dy[:-1]**2)**1.5
        curvature_means.append(np.nanmean(curvature))
        curvature_stds.append(np.nanstd(curvature))
        
        # Calculate turning angle for the window
        # This is a simplified estimation, assuming small angles
        #turning_angles = np.arctan2(dy, dx)
        #turning_angle_diff = np.diff(turning_angles)
        turning_angle_diff = vectorized_angle_diff(window['heading'])
        turning_angle_categories = np.digitize(np.abs(turning_angle_diff), [tight_turn_threshold, wide_turn_threshold])
        for category in [0, 1, 2]:  # tight, moderate, wide
            turning_angles_distribution[['tight', 'moderate', 'wide'][category]].append(np.sum(turning_angle_categories == category)/len(turning_angle_categories))
        
        heading_variance.append(circular_variance(window['heading']))

        # Calculate mean ratio of forward to side velocity
        mean_ratio_forward_side = np.mean(window['fwV'] / window['sideV'].replace(0, np.nan))
        forward_side_ratio_means.append(mean_ratio_forward_side)
        
        # Calculate mean and std of yaw velocity
        yaw_velocity_means.append(np.mean(window['yawV']))
        yaw_velocity_stds.append(np.std(window['yawV']))
    
    # Compile results into a DataFrame
    results_df = pd.DataFrame({
        'Curvature_Mean': curvature_means,
        'Curvature_Std': curvature_stds,
        'Tight_Turns':turning_angles_distribution['tight'],
        'Moderate_Turns': turning_angles_distribution['moderate'],
        'Wide_Turns': turning_angles_distribution['wide'],
        'Heading_Variance':heading_variance,
        'Mean_fwV_sideV_Ratio': forward_side_ratio_means,
        'Yaw_Velocity_Mean': yaw_velocity_means,
        'Yaw_Velocity_Std': yaw_velocity_stds
    })
    
    return results_df

#window_size = 30
#results_df = calculate_movement_metrics(behav_df, window_size)
#print(results_df.head(5))

def pad_nan(df, num_front_pad, num_end_pad):
    """
    Pads the DataFrame with NaNs at the beginning and end of all columns.
    
    Parameters:
    - df: The DataFrame to pad.
    - num_front_pad: Number of NaN rows to add at the beginning of the DataFrame.
    - num_end_pad: Number of NaN rows to add at the end of the DataFrame.
    
    Returns:
    - DataFrame with NaN rows added at the beginning and end.
    """
    # Create a DataFrame with NaNs for padding at the beginning
    front_pad = pd.DataFrame({col: np.nan for col in df.columns}, index=[np.nan]*num_front_pad)
    
    # Create a DataFrame with NaNs for padding at the end
    end_pad = pd.DataFrame({col: np.nan for col in df.columns}, index=[np.nan]*num_end_pad)
    
    # Concatenate the padding DataFrames and the original DataFrame
    padded_df = pd.concat([front_pad, df, end_pad], ignore_index=True)
    
    return padded_df

def extract_features(df, window_size=25, sigma=2):
    """
    Extracts features and calculates descriptive statistics for each segment.

    Parameters:
    - df: DataFrame with behavioral variables and 'soft_cluster_label'.
    - sigma: Smoothing parameter for Gaussian smoothing.
    
    Returns:
    - DataFrame with summary statistics for each segment.
    """
    smoothed_df = df.copy()
    # Assuming gaussian smoothing function for velocity variables exists as gaussian_smooth
    for col in ['fwV', 'sideV', 'yawV']:
        smoothed_df[col + '_smoothed'] = imaging_behavior_functions.apply_gaussian_smoothing(df[col], sigma=sigma)
    
    # Smooth heading direction
    smoothed_df['heading_smoothed'] = smooth_circular_variable(df['heading'], sigma=sigma)
    smoothed_df['straightness_smoothed'] = calculate_straightness(df, window_size)
    # Initialize list to store summary statistics
    summaries = []
    
    # Calculate statistics for each cluster and non-odor zones (label 0)
    cluster_labels = smoothed_df['soft_cluster_label_temporal'].unique()
    for label in cluster_labels:
        segment_df = smoothed_df[smoothed_df['soft_cluster_label_temporal'] == label]
        duration = segment_df['time'].iloc[-1] - segment_df['time'].iloc[0]
        
        summary = {
            'cluster_label': label,
            'mean_fwV_smoothed': segment_df['fwV_smoothed'].mean(),
            'std_fwV_smoothed': segment_df['fwV_smoothed'].std(),
            'iqr_fwV_smoothed': segment_df['fwV_smoothed'].quantile(0.75) - segment_df['fwV_smoothed'].quantile(0.25),
            'mean_sideV_smoothed': segment_df['sideV_smoothed'].mean(),
            'mean_yawV_smoothed': segment_df['yawV_smoothed'].mean(),
            'straightness_smoothed': segment_df['straightness_smoothed'].mean(),
            'time_spent': duration,
            # Additional metrics can be added here based on smoothed variables
        }
        summaries.append(summary)
    
    # Convert summaries to DataFrame for easy analysis
    summary_df = pd.DataFrame(summaries)
    
    return smoothed_df, summary_df

#smoothed_df, summary_df = extract_features(behav_df, sigma=2)
#print(smoothed_df.head())  # To see the original DataFrame with added smoothed columns
#print(summary_df)          # To see the summary statistics

def create_violin_plots(smoothed_df, example_path_results):
    """
    Creates violin plots for each smoothed variable across clusters from summary_df.

    Parameters:
    - summary_df: DataFrame containing summary statistics for each cluster.
    """
    # Filter out columns to plot (exclude non-smoothed variables and non-statistical columns)
    smoothed_vars = [col for col in smoothed_df.columns if '_smoothed' in col]
    
    for var in smoothed_vars:
        plt.figure(figsize=(10, 6))  # Set figure size for each plot
        sns.violinplot(x='soft_cluster_label_temporal', y=var, data=smoothed_df)
        plt.title(var)
        plt.xlabel('cluster label')  # Remove x-axis label
        plt.ylabel('value')  # Remove y-axis label
        plt.savefig(example_path_results+f'{var}.png')
        plt.close()

#create_violin_plots(smoothed_df, example_path_results)

# summary function for acquiring preprocessed dfs
def analysis_dfs(behav_df, time_interval_threshold, k, window_size):
    odor_events = store_odor_on_off_times(behav_df)
    # cluster odor encounter by time proximity
    clusters = cluster_odor_events_temporal(odor_events, time_interval_threshold)
    # update behavioral df -> add temporal cluster labels
    behav_df = label_df_with_clusters(behav_df, clusters)
    # update behavioral df -> add soft temporal cluster labels
    behav_df = label_df_with_soft_clusters(behav_df, clusters, time_interval_threshold)
    # plotting and save
    plot_odor_cluster_tp(behav_df, 'soft_cluster_label_temporal', example_path_results)
    plot_odor_cluster_tp(behav_df, 'cluster_label_temporal', example_path_results)
    # spatial clustering
    centroids, segment_indices = calculate_centroids(behav_df)
    cluster_labels = cluster_centroids_kmeans(centroids, k)
    behav_df = assign_spatial_cluster_labels(behav_df, segment_indices, cluster_labels)
    plot_odor_cluster_sp(behav_df, centroids, cluster_labels, example_path_results)
    results_df = calculate_movement_metrics(behav_df, window_size)
    num_front_pad = window_size // 2  # For even window sizes; adjust as needed
    num_end_pad = window_size-num_front_pad-1
    # Pad the result DataFrame
    padded_result_df = pad_nan(results_df, num_front_pad, num_end_pad)
    smoothed_df, summary_df = extract_features(behav_df, sigma=2)
    create_violin_plots(smoothed_df, example_path_results)
    return behav_df, padded_result_df, smoothed_df
    #imaging_behavior_functions.plot_fly_traj(behav_df.xPos, behav_df.yPos, padded_result_df, 'Heading_Variance', example_path_results)

#time_interval_threshold = 16  # Assuming time is in seconds or an equivalent unit
#k = 8
#window_size = 30
#behav_df, padded_result_df, smoothed_df = analysis_dfs(behav_df, time_interval_threshold, k, window_size)
#combined_df = imaging_behavior_functions.combine_df(behav_df, neural_df)
#imaging_behavior_functions.calc_nonpara(combined_df)
#imaging_behavior_functions.nonpara_plot_bybehav(nonpara_summ_df, behavior_var, example_path_results, trial_num)
# notes to give Sat
# connectome data, functions to do analysis, a readme for explaining the dataframe fields, a notebook for visualization 