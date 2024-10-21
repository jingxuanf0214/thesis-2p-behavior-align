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
import re
import shutil
from scipy.stats import circmean, circstd
from scipy.stats import sem
import json
from matplotlib.widgets import Button
from abc import ABC, abstractmethod
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

def is_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # Kernel not running
            return False
    except ImportError:
        return False  # IPython is not installed
    except AttributeError:
        return False  # get_ipython() didn't return Configurable, not in notebook

    return True

# load mat data new

def extract_struct_fields(struct, prefix=""):
    """
    Recursively extract fields from a MATLAB struct, including nested structs.

    Args:
    struct (dict): The MATLAB struct or sub-struct to extract.
    prefix (str): The prefix for the field names (used for nested fields).
    
    Returns:
    dict: A flat dictionary with keys as 'prefix_field' and values as the field data.
    """
    data = {}
    
    for field_name, field_value in struct.items():
        # Create a new key name for this field
        new_key = f"{prefix}_{field_name}" if prefix else field_name

        if isinstance(field_value, dict):
            # If the field is a struct (dict), recursively extract its fields
            nested_data = extract_struct_fields(field_value, new_key)
            data.update(nested_data)
        else:
            # Otherwise, it's a regular field, so add it to the data dictionary
            data[new_key] = field_value
    
    return data

def search_for_neural_data(field_data):
    """
    Recursively search for 'rawf_f_f_n' and fields containing 'dff' in the nested structure.
    
    Args:
    field_data (dict): A dictionary-like structure representing a MATLAB struct.
    
    Returns:
    tuple: Two dictionaries, one for 'rawf_f_f_n' data and one for 'dff' data, or None if not found.
    """
    rawf_data = None
    dff_data = {}
    
    # Base case: If rawf_f_f_n is in the current struct
    if 'rawf_f_f_n' in field_data:
        rawf_data = field_data['rawf_f_f_n']
    
    # Search for fields containing 'dff'
    for key, value in field_data.items():
        if isinstance(value, np.ndarray) and 'dff' in key.lower():
            dff_data[key] = value
    
    # If we found both rawf and dff data, return them
    if rawf_data is not None or dff_data:
        return rawf_data, dff_data
    
    # Recursively search in nested structs
    for key, value in field_data.items():
        if isinstance(value, dict):
            result = search_for_neural_data(value)
            if result is not None:
                return result
    
    return None

def process_resp_to_neural_df(ts_resp):
    """
    Process the ts.resp structure to extract raw fluorescence data (rawf_f_f_n) from nested structs.
    
    Args:
    ts_resp (dict): The ts.resp structure as a nested dictionary.
    
    Returns:
    pd.DataFrame: A DataFrame containing the extracted neural data with appropriate column names.
    """
    neural_data = {}
    
    # Loop through the first-level fields in ts_resp (e.g., 'mbon09')
    for field_name, field_data in ts_resp.items():
        # Search for the 'rawf_f_f_n' field in nested structs
        result = search_for_neural_data(field_data)
        
        # If data is found, process it
        if result is not None:
            raw_f_data, dff_data = result
            
            # Process rawf_f_f_n data
            if raw_f_data is not None:
                if raw_f_data.ndim == 1:
                    # If raw_f_data is 1-dimensional
                    neural_data[f"{field_name}_rawf1"] = raw_f_data
                else:
                    for i in range(raw_f_data.shape[0]):
                        column_name = f"{field_name}_rawf{i+1}"
                        neural_data[column_name] = raw_f_data[i, :]
            
            # Process dff data
            for dff_key, dff_value in dff_data.items():
                if dff_value.ndim == 1:
                    neural_data[f"{field_name}_dff1"] = dff_value
                else:
                    for i in range(dff_value.shape[0]):
                        column_name = f"{field_name}_dff{i+1}"
                        neural_data[column_name] = dff_value[i, :]
    
    # Convert the dictionary to a pandas DataFrame
    neural_df = pd.DataFrame(neural_data)
    
    return neural_df


def load_matfile_to_df(example_path_data, trial_num):
    """
    Load a MATLAB v7.3 .mat file using the mat73 library and extract struct variables into a pandas DataFrame.
    
    Args:
    matfile_path (str): Path to the .mat file.
    
    Returns:
    pd.DataFrame: A DataFrame with extracted fields from the MATLAB struct.
    """
    # Load the .mat file
    mat_data = mat73.loadmat(example_path_data+'ts_trial'+str(trial_num)+'.mat')

    # Extract the 'ts' struct from the loaded data
    ts = mat_data['ts']
    
    # Dictionary to store all extracted data
    data = {}

    # List of relevant sub-structs, including 'resp'
    sub_structs = ['flypos', 'ball', 'vis', 'resp']
    
    # Loop over each sub-struct and extract its fields
    for sub_struct_name in sub_structs:
        if sub_struct_name in ts:
            sub_struct = ts[sub_struct_name]
            #print(sub_struct)
            # Special case for 'resp' sub-struct
            if sub_struct_name == 'resp':
                neural_df = process_resp_to_neural_df(sub_struct)
                #print(neural_df)
                # Add the neural data columns to the main data dictionary
                #for col in neural_df.columns:
                    #data[col] = neural_df[col]
            else:
                # Use the recursive function to extract all other fields
                sub_struct_data = extract_struct_fields(sub_struct, sub_struct_name)
                # Update the main data dictionary
                data.update(sub_struct_data)

    # Convert the dictionary into a pandas DataFrame
    behav_df = pd.DataFrame(data)
    # Add 'ti' (time) to both df and neural_df
    if 'ti' in ts:
        time = ts['ti']
        behav_df['time'] = time  # Add time to df
        neural_df['time'] = time  # Add time to neural_df as well
    
    return behav_df, neural_df


def load_roiData_struct(path_to_folder):
    # Construct the search pattern for files containing 'roiData_struct'
    search_pattern = path_to_folder + '/*roiData_struct*.mat'
    
    # Use glob to find files matching the pattern
    matching_files = glob.glob(search_pattern)
    
    # Assuming you want to load the first matching file
    if matching_files:
        # Load the first matching file found
        try:
            roi_data = scipy.io.loadmat(matching_files[0])
        except NotImplementedError as e:
        # If scipy.io.loadmat fails due to version incompatibility, try with mat73.loadmat
            print(f"Loading with scipy.io failed: {e}. Trying with mat73.")
            roi_data = mat73.loadmat(matching_files[0])
        return roi_data
    else:
        print("No matching files found.")
        return None

def find_complete_trials(path_to_folder):
    # List all .mat files in the folder
    all_files = [f for f in os.listdir(path_to_folder) if f.endswith('.mat')]
    
    # Extract trial numbers from file names
    trial_numbers = set()
    for file_name in all_files:
        match = re.search(r'trial(\d+)', file_name)
        if match:
            trial_numbers.add(int(match.group(1)))
    
    # Check for complete set of files for each trial
    complete_trials = []
    for trial_num in trial_numbers:
        dff_path = os.path.join(path_to_folder, f'dff raw trial{trial_num}.mat')
        preprocessed_path = os.path.join(path_to_folder, f'preprocessed_vars_ds trial{trial_num}.mat')
        # Check if both required files exist
        if os.path.exists(dff_path) and os.path.exists(preprocessed_path):
            complete_trials.append(trial_num)
    
    # Return the list of complete trial numbers
    return complete_trials

def load_intermediate_mat(path_to_folder,trial_num):
    is_mat73 = 0
    try:
        dff_raw = scipy.io.loadmat(path_to_folder + f'dff raw trial{trial_num}.mat')
        kinematics_raw = scipy.io.loadmat(path_to_folder + f'kinematics raw trial{trial_num}.mat')
        preprocessed_vars_ds = scipy.io.loadmat(path_to_folder + f'preprocessed_vars_ds trial{trial_num}.mat')
        odor_path = os.path.join(path_to_folder, f'preprocessed_vars_odor trial{trial_num}.mat')
        if os.path.exists(odor_path):
            preprocessed_vars_odor = scipy.io.loadmat(odor_path)
        else:
            preprocessed_vars_odor = None  # or any other placeholder value you find appropriate
        #preprocessed_vars_odor = scipy.io.loadmat(path_to_folder + f'preprocessed_vars_odor trial{trial_num}.mat')
    except NotImplementedError as e:
        # If scipy.io.loadmat fails due to version incompatibility, try with mat73.loadmat
        print(f"Loading with scipy.io failed: {e}. Trying with mat73.")
        is_mat73 = 1
        dff_raw = mat73.loadmat(path_to_folder + f'dff raw trial{trial_num}.mat')
        kinematics_raw = mat73.loadmat(path_to_folder + f'kinematics raw trial{trial_num}.mat')
        preprocessed_vars_ds = mat73.loadmat(path_to_folder + f'preprocessed_vars_ds trial{trial_num}.mat')
        odor_path = os.path.join(path_to_folder, f'preprocessed_vars_odor trial{trial_num}.mat')
        if os.path.exists(odor_path):
            preprocessed_vars_odor = mat73.loadmat(odor_path)
        else:
            preprocessed_vars_odor = None  # or any other placeholder value you find appropriate
    roi_data = load_roiData_struct(path_to_folder)
    roi_df = pd.DataFrame.from_dict(np.squeeze(roi_data['convertedStruct'],axis=1))
    return is_mat73, roi_df, dff_raw, kinematics_raw, preprocessed_vars_ds, preprocessed_vars_odor

def load_intermediate_mat_new(path_to_folder,trial_num):
    is_mat73 = 0
    try:
        dff_raw = scipy.io.loadmat(path_to_folder + f'dff_raw_trial{trial_num}.mat')
    except NotImplementedError as e:
        # If scipy.io.loadmat fails due to version incompatibility, try with mat73.loadmat
        print(f"Loading with scipy.io failed: {e}. Trying with mat73.")
        is_mat73 = 1
        dff_raw = mat73.loadmat(path_to_folder + f'dff_raw_trial{trial_num}.mat')
    roi_data = load_roiData_struct(path_to_folder)
    roi_df = pd.DataFrame.from_dict(np.squeeze(roi_data['convertedStruct'],axis=1))
    return is_mat73, roi_df, dff_raw

def plot_interactive_histogram(series):
    fig, ax = plt.subplots()
    counts, bins, patches = ax.hist(series, bins=30, color='skyblue', edgecolor='gray')
    
    threshold_line = ax.axvline(color='r', linestyle='--')
    threshold_value = [None]  # Use a list to store the threshold value
    
    def onclick(event):
        # Event handler that draws a vertical line where the user clicks and updates threshold_value
        threshold_line.set_xdata([event.xdata, event.xdata])
        threshold_value[0] = event.xdata  # Update the threshold value
        fig.canvas.draw()
    
    # Connect the click event to the handler
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Interactive Histogram')
    plt.show()
    
    return threshold_value[0]  # Return the updated list containing the threshold value


def calculate_deltaF_over_F_df(df, fluorescence_columns, method='percentile', window_size=100, percentile=8):
    """
    Calculate deltaF/F for specified columns in a pandas DataFrame and add new columns with '_dff' appended to the names.

    Args:
    df (pd.DataFrame): The input DataFrame containing raw fluorescence time series.
    fluorescence_columns (list): List of column names in the DataFrame representing raw fluorescence signals.
    method (str): Method for calculating F0 ('median', 'mean', 'sliding_window', 'percentile', 'smooth'). Default is 'percentile'.
    window_size (int): The size of the sliding window (used for 'sliding_window' and 'percentile' methods). Default is 100.
    percentile (int): The percentile to use for the baseline calculation in 'percentile' method. Default is 8.

    Returns:
    pd.DataFrame: The input DataFrame with new columns appended containing deltaF/F values.
    """
    df = df.copy()  # To avoid modifying the original DataFrame

    for col in fluorescence_columns:
        fluorescence_series = df[col].values
        
        # Baseline calculation
        if method == 'median':
            F0 = np.median(fluorescence_series)
        elif method == 'mean':
            F0 = np.mean(fluorescence_series)
        elif method == 'sliding_window':
            F0 = np.array([np.mean(fluorescence_series[max(0, i-window_size):i]) for i in range(len(fluorescence_series))])
        elif method == 'percentile':
            F0 = np.array([np.percentile(fluorescence_series[max(0, i-window_size):i], percentile) for i in range(len(fluorescence_series))])
        elif method == 'smooth':
            F0 = gaussian_filter1d(fluorescence_series, sigma=window_size)
        elif method == 'whole_session_percentile':
            F0 = np.percentile(fluorescence_series, percentile)
        else:
            raise ValueError(f"Method {method} not recognized.")
        
        # Calculate deltaF/F
        deltaF_over_F = (fluorescence_series - F0) / F0
        
        # Add new column to the DataFrame with '_dff' appended to the original column name
        df[col + '_dff'] = deltaF_over_F

    return df

# Example Usage:
# raw_fluorescence = np.array([...])  # your raw fluorescence time series
# deltaF_F = calculate_deltaF_over_F(raw_fluorescence, method='percentile', window_size=100, percentile=10)
# print(deltaF_F)


def make_df_behavior(dff_raw, preprocessed_vars_ds, preprocessed_vars_odor,trial_num,ball_d = 9):
    circum = ball_d * np.pi
    df = pd.DataFrame()
    # add dff_raw
    dff_time = dff_raw['roiTime']
    df['time'] = np.squeeze(dff_time)
    df['fwV'] = np.squeeze(preprocessed_vars_ds['ftT_fwSpeedDown2']) # unit in mm/s
    df['sideV'] = circum*np.squeeze(preprocessed_vars_ds['ftT_sideSpeedDown2'])/(2*np.pi)
    df['yawV'] = circum*np.squeeze(preprocessed_vars_ds['ftT_yawSpeedDown2'])/(2*np.pi) # unit in mm/s
    df['heading'] = np.squeeze(preprocessed_vars_ds['ftT_intHDDown2'])
    df['abssideV'] = circum*np.abs(np.squeeze(preprocessed_vars_ds['ftT_sideSpeedDown2']))/(2*np.pi)
    df['absyawV'] = circum*np.abs(np.squeeze(preprocessed_vars_ds['ftT_yawSpeedDown2']))/(2*np.pi)
    df['net_motion'] = df['abssideV']+df['absyawV']+np.abs(df['fwV'])
    in_notebook = is_notebook()
    if in_notebook:
        threshold = np.percentile(df.net_motion,5)
    else:
        threshold = plot_interactive_histogram(df.net_motion)
    df['net_motion_state'] = (df['net_motion']>threshold).astype(int)
    df['heading_adj'] = np.unwrap(df['heading'])
    if preprocessed_vars_odor != None:
        odor_all = preprocessed_vars_odor['odorDown']
        if len(odor_all) == 1:
            df['odor'] = np.squeeze(odor_all)
        elif len(odor_all) == len(df.time) and len(odor_all.shape) == 1:
            df['odor'] = odor_all
        else:
            df['odor'] = odor_all[:,trial_num-1]
    return df 

#roi_df, dff_raw, kinematics_raw, preprocessed_vars_ds, preprocessed_vars_odor = load_intermediate_mat(example_path_data,1)
#print(roi_df.head(5))
#behav_df = make_df_behavior(dff_raw, preprocessed_vars_ds, preprocessed_vars_odor,1,ball_d = 9)
#print(behav_df.head(5))

def reconstruct_path(df, ball_d = 9):
    circum = ball_d * np.pi #circumference of ball, in mm
    mmPerDeg = circum / 360 # mm per degree of ball
    fwdAngVel = df.fwV/mmPerDeg 
    # zero heading
    zeroedH = df.heading - df.heading[0]
    time_bin = np.diff(df.time)
    # movement in x (in degrees) at each time point
    xChangePos = (fwdAngVel[0:-1]*time_bin)*np.sin(zeroedH[0:-1]) + (df.sideV[0:-1]*time_bin)*np.sin(zeroedH[0:-1]+np.pi/4)
    xPos = (np.cumsum(xChangePos) - xChangePos[0])*mmPerDeg
    yChangePos = (fwdAngVel[0:-1]*time_bin)*np.cos(zeroedH[0:-1]) + (df.sideV[0:-1]*time_bin)*np.cos(zeroedH[0:-1]+np.pi/4)
    yPos = (np.cumsum(yChangePos) - yChangePos[0])*mmPerDeg
    xPos_padded = pd.concat([xPos, pd.Series(xPos.iloc[-1])], ignore_index=True) 
    yPos_padded = pd.concat([yPos, pd.Series(yPos.iloc[-1])], ignore_index=True) 
    df['xPos'] = xPos_padded
    df['yPos'] = yPos_padded
    return xPos_padded, yPos_padded

#xPos, yPos = reconstruct_path(behav_df, ball_d = 9)

def plot_fly_traj(xPos, yPos, behav_df, label, example_path_results,trial_num):
    x_range = max(xPos) - min(xPos)
    y_range = max(yPos) - min(yPos)
    aspect_ratio = y_range / x_range

    # Set figure dimensions based on data range while keeping unit scale the same
    fig_width = 10  # Width of figure in inches
    fig_height = fig_width * aspect_ratio  # Height is scaled according to the aspect ratio of the data

    plt.figure(figsize=(fig_width, fig_height))

    if label in behav_df.columns:
        # If the label exists, color the scatter plot based on the label values
        plt.scatter(xPos, yPos, c=behav_df[label], s=3)
        plt.colorbar()  # Optionally, add a color bar to indicate the mapping of color to label values
    else:
        # If the label does not exist, plot a normal scatter plot without coloring
        plt.scatter(xPos, yPos, s=3)
        label = "nothing"
    plt.scatter(0, 0, color='red')  # Mark the origin

    # Enforce equal aspect ratio so that one unit in x is the same as one unit in y
    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.title('Fly Trajectory')

    # Save the plot
    # Construct the full file path
    file_path = os.path.join(example_path_results, f'fly_trajectory_colored_by_{label}_trial_{str(trial_num)}.png')

    # Extract the directory path
    dir_path = os.path.dirname(file_path)

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig(file_path)
    plt.close()  # Close the plot explicitly after saving to free resources

def plot_fly_traj_interactive(xPos, yPos, behav_df, label, example_path_results, trial_num):
    x_range = max(xPos) - min(xPos)
    y_range = max(yPos) - min(yPos)
    aspect_ratio = y_range / x_range

    fig_width = 10
    fig_height = fig_width * aspect_ratio

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    if label in behav_df.columns:
        scatter = ax.scatter(xPos, yPos, c=behav_df[label], s=3)
        plt.colorbar(scatter)
    else:
        ax.scatter(xPos, yPos, s=3)
        label = "nothing"
    ax.scatter(0, 0, color='red')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x position')
    ax.set_ylabel('y position')
    ax.set_title('Fly Trajectory')

    selected_points = []
    line, = ax.plot([], [], 'ro-', linewidth=2, markersize=8)

    def onclick(event):
        if event.inaxes != ax:
            return
        selected_points.append((event.xdata, event.ydata))
        x = [p[0] for p in selected_points]
        y = [p[1] for p in selected_points]
        line.set_data(x, y)
        fig.canvas.draw()

    def on_finish(event):
        plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', onclick)

    finish_ax = plt.axes([0.81, 0.05, 0.1, 0.075])
    finish_button = Button(finish_ax, 'Finish')
    finish_button.on_clicked(on_finish)

    plt.show()

    # Find indices of clicked points
    clicked_indices = []
    for point in selected_points:
        distances = np.sqrt((xPos - point[0])**2 + (yPos - point[1])**2)
        closest_index = np.argmin(distances)
        clicked_indices.append(closest_index)

    # Save the plot
    file_path = os.path.join(example_path_results, f'fly_trajectory_colored_by_{label}_trial_{str(trial_num)}.png')
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.figure(figsize=(fig_width, fig_height))
    if label in behav_df.columns:
        plt.scatter(xPos, yPos, c=behav_df[label], s=3)
        plt.colorbar()
    else:
        plt.scatter(xPos, yPos, s=3)
    plt.scatter(0, 0, color='red')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.title('Fly Trajectory')
    for i, idx in enumerate(clicked_indices):
        plt.plot(xPos[idx], yPos[idx], 'ro', markersize=8)
        plt.text(xPos[idx], yPos[idx], f' {i+1}', fontsize=12, verticalalignment='bottom')
    plt.savefig(file_path)
    plt.close()

    return clicked_indices

#plot_fly_traj(xPos, yPos, behav_df,'odor', example_path_results)

#clicked_indices = plot_fly_traj_interactive(xPos, yPos, behav_df, 'odor', example_path_results, trial_num)
#print("Indices of clicked points:", clicked_indices)

def get_roi_seq(roi_df):
    roi_df['trialNum'] = roi_df['trialNum'].apply(lambda x: x[0][0])
    # Find the trial number with the maximum number of rows
    trial_num_max_rows = roi_df['trialNum'].value_counts().idxmax()

    # Filter the DataFrame to only include rows with that trial number
    roi_df = roi_df[roi_df['trialNum'] == trial_num_max_rows].reset_index(drop=True)
    roi_names = roi_df['roiName'].apply(lambda x: x[0])
    roi_hdeltab = roi_names[roi_names.str.contains('hDeltaB',case=False)]
    hdeltab_index = roi_hdeltab.index
    roi_epg = roi_names[roi_names.str.contains('EPG')]
    epg_index = roi_epg.index
    roi_fr1 = roi_names[roi_names.str.contains('FR1') & ~roi_names.str.contains('CRE')]
    fr1_index = roi_fr1.index
    hdeltab_seq = roi_hdeltab.str.extract(r'_(\d+)')[0].astype(int)
    hdeltab_seq = hdeltab_seq.to_numpy()
    if epg_index.size>0:
        epg_seq = roi_epg.str.extract(r'_(\d+)')[0].astype(int)
        epg_seq = epg_seq.to_numpy()
    else:
        epg_seq =  None 
    if fr1_index.size>0:
        fr1_seq = roi_fr1.str.extract(r'_(\d+)')[0].astype(int)
        fr1_seq = fr1_seq.to_numpy()
    else:
        fr1_seq =  None 
    return np.array(roi_names), hdeltab_index, epg_index, fr1_index,hdeltab_seq, epg_seq,fr1_seq

#roi_names, hdeltab_index, epg_index, hdeltab_sequence, epg_sequence = get_roi_seq(roi_df)
#print(hdeltab_sequence)

def sort_rois(dff_tosort, roi_names, query_idx, query_seq):
    sorting_indices = np.argsort(query_seq)
    segment_to_sort = dff_tosort[query_idx]
    sorted_dff_rois = segment_to_sort[sorting_indices]
    dff_tosort[query_idx] = sorted_dff_rois
    roi_names_sort = roi_names[query_idx]
    roi_names_sort = roi_names_sort[sorting_indices]
    roi_names[query_idx] = roi_names_sort

def load_dff_raw(is_mat73, dff_raw):
    dff_rois = dff_raw['flDataC']
    dff_time = dff_raw['roiTime']
    # Sort dff_rois according to roi_sequence
    # Ensure roi_sequence is a list of integers that corresponds to the order you want
    if is_mat73:
        dff_all_rois = np.array(dff_rois)
    else:
        dff_all_rois = dff_rois[0]
    return dff_all_rois, dff_time

def make_df_neural(dff_all_rois, dff_time, roi_names, hdeltab_index, epg_index, fr1_index, hdeltab_sequence, epg_sequence,fr1_sequence):
    #TODO
    sort_rois(dff_all_rois, roi_names, hdeltab_index, hdeltab_sequence)
    if epg_index.size > 0:
        sort_rois(dff_all_rois, roi_names, epg_index, epg_sequence)
    else:
        pass
    if fr1_index.size > 0:
        sort_rois(dff_all_rois, roi_names, fr1_index, fr1_sequence)
    else:
        pass
    # Create a new DataFrame for the reordered data
    neural_df = pd.DataFrame()
    neural_df['time'] = np.squeeze(dff_time)
    # Add each sorted ROI data to the DataFrame with the specified naming convention
    for i, roi_data in enumerate(dff_all_rois):
        column_name =  roi_names[i] # Generate column name starting from hDeltaB1
        neural_df[column_name] = np.squeeze(roi_data)
    # Identify columns where all values are 0
    cols_to_drop = [col for col in neural_df.columns if neural_df[col].eq(0).all()]
    
    # Drop these columns from the DataFrame
    neural_df.drop(columns=cols_to_drop, inplace=True)
    return neural_df
#neural_df = make_df_neural(dff_raw,roi_names, hdeltab_index, epg_index, hdeltab_sequence, epg_sequence)
#print(neural_df)


def combine_df(behav_df, neural_df):
    return pd.merge(behav_df,neural_df,on="time")

#combined_df = combine_df(behav_df, neural_df)
#print(combined_df.head(5))

def apply_gaussian_smoothing(series, sigma):
    """
    Applies Gaussian smoothing to a pandas Series.
    
    Parameters:
    - series: The pandas Series to smooth.
    - sigma: Standard deviation for Gaussian kernel, controlling the smoothing degree.
    
    Returns:
    - A pandas Series containing the smoothed data.
    """
    # Ensure the series is a numpy array for processing
    series_array = series.values
    
    # Apply Gaussian smoothing
    smoothed_array = gaussian_filter1d(series_array, sigma=sigma)
    
    # Convert back to pandas Series
    smoothed_series = pd.Series(smoothed_array, index=series.index)
    
    return smoothed_series


def calc_nonpara(combined_df, roi_kw,roi_kw2, roi_mtx=None, do_truncate=False):
    if do_truncate:
        combined_df = combined_df[(combined_df["fwV"]>0.2) | (combined_df["fwV"]<-0.2)]
    sigma = 5
    smooth_fwV = apply_gaussian_smoothing(combined_df.fwV, sigma)
    smooth_sideV = apply_gaussian_smoothing(combined_df.sideV, sigma)
    translational_speed = np.sqrt(smooth_fwV**2+smooth_sideV**2)
    forward_speed = np.abs(smooth_fwV)
    if roi_mtx is None:
        if roi_kw2:
            filtered_columns = [col for col in combined_df.columns if roi_kw in col and roi_kw2 not in col]
            neural_df_rois = combined_df[filtered_columns]
        else:
            neural_df_rois = combined_df[[col for col in combined_df.columns if roi_kw.lower() in col.lower()]]
        row_means = neural_df_rois.apply(np.mean, axis=1)
    
        row_iqrs = neural_df_rois.apply(lambda x: iqr(x, interpolation='midpoint'), axis=1)
    else:
        row_means = np.mean(roi_mtx, axis=1)
        row_iqrs = iqr(roi_mtx,interpolation='midpoint',axis=1)
    
    # Combine mean and IQR into a new DataFrame
    stats_df = pd.DataFrame({'Mean': row_means, 'IQR': row_iqrs, 'translationalV': translational_speed, 'fwV':forward_speed})
    
    return combined_df, stats_df

#nonpara_summ_df = calc_nonpara(combined_df)
#print(nonpara_summ_df)

def nonpara_plot_bybehav(nonpara_summ_df, combined_df, behavior_var, example_path_results, trial_num):
    plt.figure(figsize=(6, 6))
    plt.scatter(nonpara_summ_df['Mean'],nonpara_summ_df['IQR'],c= combined_df[behavior_var])
    plt.colorbar()
    plt.xlabel('mean')
    plt.ylabel('iqr')
    plt.title('mean vs. amplitude')

    # Save the plot
    plt.savefig(example_path_results+'mean_amp_' + behavior_var + str(trial_num) +'.png')
    plt.close()  # Close the plot explicitly after saving to free resources

#behavior_var = 'translationalV'
#nonpara_plot_bybehav(nonpara_summ_df, behavior_var)

def extract_heatmap(df, roi_kw, roi_kw2, do_normalize, example_path_results, trial_num):
    if roi_kw2:
        filtered_columns = [col for col in df.columns if roi_kw in col and roi_kw2 not in col]
        roi_mtx = df[filtered_columns]
    else:
        roi_mtx = df[[col for col in df.columns if roi_kw.lower() in col.lower()]]
    if roi_mtx.empty:
        return None
    if do_normalize:
        scaler = StandardScaler()
        roi_mtx = scaler.fit_transform(roi_mtx)
        plt.figure(figsize=(10, 6))
        sns.heatmap(np.transpose(roi_mtx))
        plt.savefig(example_path_results+'heatmap_norm' + str(trial_num)+ '.png')
        plt.close()  # Close the plot explicitly after saving to free resources
    elif not do_normalize:
        plt.figure(figsize=(10, 6))
        sns.heatmap(np.transpose(roi_mtx))
        plt.savefig(example_path_results+'heatmap_nonnorm' + str(trial_num)+ '.png')
        plt.close()  # Close the plot explicitly after saving to free resources
    return roi_mtx

#roi_mtx = extract_heatmap(combined_df, "hDeltaB", True)

# for EPG type imaging only 
def calculate_pva_epg(activity_matrix):
    num_neurons, time_steps = activity_matrix.shape
    directions = np.linspace(0, 2*np.pi, num_neurons//2, endpoint=False)
    
    # Repeat directions for both halves of the neuron population
    directions = np.tile(directions, 2)
    
    # Calculate vector components for each neuron's activity
    x_components = np.cos(directions)[:, np.newaxis] * activity_matrix
    y_components = np.sin(directions)[:, np.newaxis] * activity_matrix
    
    # Sum components across neurons for each time step
    sum_x = np.sum(x_components, axis=0)
    sum_y = np.sum(y_components, axis=0)
    
    # Calculate PVA for each time step
    pva_phase = np.arctan2(sum_y, sum_x)  # Phase in radians
    pva_amplitude = np.sqrt(sum_x**2 + sum_y**2)  # Magnitude of the vector
    return pva_phase, pva_amplitude

def calculate_pva_hdeltab(activity_matrix):
    num_neurons, time_steps = activity_matrix.shape
    directions = np.linspace(0, 2*np.pi, num_neurons, endpoint=False)
    
    # Repeat directions for both halves of the neuron population
    #directions = np.tile(directions, 2)
    
    # Calculate vector components for each neuron's activity
    x_components = np.cos(directions)[:, np.newaxis] * activity_matrix
    y_components = np.sin(directions)[:, np.newaxis] * activity_matrix
    
    # Sum components across neurons for each time step
    sum_x = np.sum(x_components, axis=0)
    sum_y = np.sum(y_components, axis=0)
    
    # Calculate PVA for each time step
    pva_phase = np.arctan2(sum_y, sum_x)  # Phase in radians
    pva_amplitude = np.sqrt(sum_x**2 + sum_y**2)  # Magnitude of the vector
    return pva_phase, pva_amplitude

def fit_sinusoid(neural_df, roi_mtx):
    def test_func(x, dist, amp, phi):
        return dist + amp * np.cos(x + phi)
    timestamp, num_roi = np.shape(roi_mtx)
    x_p = np.linspace(0, 2*np.pi, num=num_roi)
    trial_len = timestamp
    phase_sinfit = np.zeros(trial_len)
    base_sinfit = np.zeros(trial_len)
    amp_sinfit = np.zeros(trial_len)
    phase_perr = np.zeros(trial_len)
    base_perr = np.zeros(trial_len)
    amp_perr = np.zeros(trial_len)
    for i in range(trial_len):
        params, params_covariance = optimize.curve_fit(test_func, x_p, roi_mtx[i,:],maxfev = 5000)
        phase_sinfit[i] = x_p[np.argmax(test_func(x_p, params[0], params[1], params[2]))]
        amp_sinfit[i] = np.abs(params[1])
        base_sinfit[i] = params[0]
        perr = np.sqrt(np.diag(params_covariance))
        phase_perr[i] = perr[2]
        base_perr[i] = perr[0]
        amp_perr[i] = perr[1]
    time = neural_df.time
    paramfit_df = pd.DataFrame({'time': time, 'phase': phase_sinfit, 'baseline': base_sinfit, 'amplitude': amp_sinfit, 'phase_error':phase_perr, "baseline_error": base_perr, "amplitude_error":amp_perr})
    return paramfit_df

#paramfit_df = fit_sinusoid(neural_df, roi_mtx)
#print(paramfit_df.head(5))

def plot_with_error_shading(df, example_path_results,trial_num):
    # Set up the figure and subplots
    fig, axs = plt.subplots(3, 1, figsize=(15, 10))
    
    # Plot phase
    axs[0].plot(df['time'], df['phase'], label='Phase', color = 'orange')
    phase_error_threshold = np.percentile(df['phase_error'], 90)
    high_error_times = df['time'][df['phase_error'] > phase_error_threshold]
    for time in high_error_times:
        axs[0].axvline(x=time, color='gray', alpha=0.1)  # Adjust alpha for desired faintness
    axs[0].set_title('Phase')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Phase')
    
    # Plot amplitude
    axs[1].plot(df['time'], df['amplitude'], label='Amplitude',color = 'red')
    amp_error_threshold = np.percentile(df['amplitude_error'], 90)
    high_error_times = df['time'][df['amplitude_error'] > amp_error_threshold]
    for time in high_error_times:
        axs[1].axvline(x=time, color='gray', alpha=0.1)  # Adjust alpha for desired faintness
    axs[1].set_title('Amplitude')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Amplitude')

    # Plot baseline
    axs[2].plot(df['time'], df['baseline'], label='Baseline', color = 'green')
    base_error_threshold = np.percentile(df['baseline_error'], 90)
    high_error_times = df['time'][df['baseline_error'] > base_error_threshold]
    for time in high_error_times:
        axs[2].axvline(x=time, color='gray', alpha=0.1)  # Adjust alpha for desired faintness
    axs[2].set_title('Baseline')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Baseline')

    plt.tight_layout()
    plt.savefig(example_path_results+'parametric_fit' + str(trial_num) +'.png')
    plt.close()  # Close the plot explicitly after saving to free resources

#plot_with_error_shading(paramfit_df)

def fit_nonparametric(roi_mtx):
    #use iqr as a proxy of height
    bump_height = iqr(roi_mtx, axis=1)
    #bump phase TODO 
    #bump_phase = 
    # level of flourescence at 50% 
    threshold = np.percentile(roi_mtx, 50, axis = 1)
    mean_hDeltaB = np.mean(roi_mtx, axis=1)
    # width TODO
    return bump_height, threshold, mean_hDeltaB

# post scopa analysis 

def load_tif_images(tif_path):
    img = Image.open(tif_path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        images.append(img.copy())
    return images

def generate_plots(df, variable_name, plot_width, plot_height, window_size=50, dpi=100):
    plots = []
    for i in range(len(df)):
        # Create a figure with specified size and DPI
        fig, ax = plt.subplots(figsize=(plot_width, plot_height), dpi=dpi)
        
        # Determine the range for the moving window
        start_range = max(0, i - window_size)
        end_range = i

        # Plot the data within the moving window
        ax.plot(df[variable_name][start_range:end_range])
        ax.set_xlim(start_range, end_range)
        
        # Set other plot properties as needed
        fig.canvas.draw()

        # Convert plot to grayscale image
        img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        plots.append(img_gray)
        plt.close(fig)
    return plots


# plottings 
def pairwise_corr(df,height,width):
    fig, ax = plt.subplots(figsize = (height, width))
    sns.pairplot(df, ax = ax)

def phase_plot(df,height,width,ind_l, ind_h):
    fig, ax = plt.subplots(figsize = (height, width))
    ax.scatter(df.time[ind_l:ind_h], df.heading[ind_l:ind_h],color='blue', alpha = 0.5)
    ax.plot(df.time,2*np.pi-df['phase_sinfit'], color = 'orange')


def plot_time_series(neural_df, behav_df,example_path_results,trial_num):
    neural_columns = len(neural_df.columns.drop('time'))
    behav_columns = len(['fwV', 'yawV', 'sideV', 'heading'])
    total_plots = neural_columns + behav_columns
    
    # Create a figure with subplots
    fig, axs = plt.subplots(total_plots, 1, figsize=(12, 3 * total_plots), sharex=True)
    
    # Plot each column from neural_df as a subplot
    for i, column in enumerate(neural_df.columns.drop('time')):
        axs[i].plot(neural_df['time'], neural_df[column], label=column)
        axs[i].set_ylabel(column)
        axs[i].legend(loc='upper right')
    
    # Plot specified columns from behav_df as subplots
    behav_columns = ['fwV', 'yawV', 'sideV', 'heading']
    for j, column in enumerate(behav_columns, start=neural_columns):
        if column in behav_df.columns:
            axs[j].plot(behav_df['time'], behav_df[column], label=column, linestyle='--')
            axs[j].set_ylabel(column)
            axs[j].legend(loc='upper right')
    
    # Check if 'odor' column exists and shade where odor > 5
    if 'odor' in behav_df.columns:
        odor_mask = behav_df['odor'] > 5
        # Apply shading to all subplots
        for ax in axs:
            ax.fill_between(behav_df['time'], ax.get_ylim()[0], ax.get_ylim()[1], where=odor_mask, color='red', alpha=0.3, transform=ax.get_xaxis_transform())
    
    # Set common labels
    plt.xlabel('Time')
    fig.suptitle('Neural and Behavioral Data Over Time', fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the suptitle
    
    plt.savefig(example_path_results+'time_series_plotting' + str(trial_num) +'.png')
    plt.close()  # Close the plot explicitly after saving to free resources

#########################################################################
# tuning curves 

# Creating a circular histogram
def calc_circu_stats(circu_var,num_bins,example_path_results,trial_num):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    # Histogram with 30 bins
    n, bins, patches = ax.hist(circu_var, bins=num_bins, density=True)

    # Set the circumference labels
    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax.set_xticklabels(['0', '$\pi/4$', '$\pi/2$', '$3\pi/4$', '$\pi$', '$5\pi/4$', '$3\pi/2$', '$7\pi/4$'])

    # Display the plot
    plt.savefig(example_path_results+ circu_var.name + str(trial_num) +'.png')
    plt.close()  # Close the plot explicitly after saving to free resources

    # Circular mean using scipy.stats.circmean for accuracy
    mean_angle = circmean(circu_var, high=2*np.pi)

    # Circular median: Sort the angles and find the middle value
    sorted_angles = np.sort(circu_var)
    median_angle = np.pi if len(sorted_angles) % 2 == 0 else sorted_angles[len(sorted_angles) // 2]

    # Circular mode: Use histogram bins, find the bin with the highest frequency
    n, bins = np.histogram(circu_var, bins=num_bins, range=(0, 2*np.pi))
    max_bin_index = np.argmax(n)
    mode_angle = (bins[max_bin_index] + bins[max_bin_index + 1]) / 2

    return mean_angle, median_angle, mode_angle
#mean_angle, median_angle, mode_angle = calc_circu_stats(behav_df.heading,30)
def plot_scatter(behavior_variable, filtered_columns, neural_activity, neurons_to_plot, num_bins=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    for j in neurons_to_plot:
        #print(len(behavior_variable))
        ax.scatter(behavior_variable,neural_activity[j,:])
        ax.set_ylabel(filtered_columns[j])
    ax.set_xlabel(behavior_variable.name)
    #ax[j,i].legend()
    #plt.tight_layout()

def tuning_curve_1d(behavior_variable, filtered_columns, neural_activity, neurons_to_plot, num_bins, ax=None):
    """
    Plot tuning curves on the given matplotlib Axes.

    Parameters:
    - behavior_variable: array-like, the behavioral variable to bin.
    - neural_activity: 2D array-like, neural activity data with shape (neurons, observations).
    - neurons_to_plot: list of int, indices of neurons to plot.
    - num_bins: int, number of bins to divide the behavior variable into.
    - ax: matplotlib.axes.Axes, the axes object to plot on. If None, a new figure is created.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bin the behavioral variable
    bins = np.linspace(np.min(behavior_variable), np.max(behavior_variable), num_bins+1)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # Plot tuning curves with SEM for selected neurons
    for neuron_index in neurons_to_plot:
        binned_activity = np.empty(num_bins)
        binned_sem = np.empty(num_bins)
        
        for i in range(num_bins):
            indices = np.where((behavior_variable >= bins[i]) & (behavior_variable < bins[i+1]))[0]
            bin_data = neural_activity[neuron_index, indices]
            binned_activity[i] = np.mean(bin_data)
            binned_sem[i] = sem(bin_data)

        ax.plot(bin_centers, binned_activity, label=f'{filtered_columns[neuron_index]}')
        ax.fill_between(bin_centers, binned_activity - binned_sem, binned_activity + binned_sem, alpha=0.3)
        ax.set_ylabel(filtered_columns[neuron_index])

    # Setting labels and title
    ax.set_xlabel(behavior_variable.name)
    #ax.set_ylabel('Average Neural Activity')
    #ax.legend()
    #plt.savefig(example_path_results+'1d_tuning_curve_' + str(trial_num) +'.png')
    #plt.close()  # Close the plot explicitly after saving to free resources

# Example usage:
# fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
# for i, ax in enumerate(axs):
#     tuning_curve_1d(behavior_variable, neural_activity, [i], 10, ax=ax)
# plt.show()

def tuning_heatmap_2d(behavior_var1, behavior_var2, filtered_columns, neural_activity, neurons_to_plot, num_bins, example_path_results, trial_num, ax=None):
    """
    Plot a 2D tuning heatmap on the given matplotlib Axes.

    Parameters:
    - behavior_var1: array-like, the first behavioral variable to bin (x-axis).
    - behavior_var2: array-like, the second behavioral variable to bin (y-axis).
    - filtered_columns: list of str, names of the neurons.
    - neural_activity: 2D array-like, neural activity data with shape (neurons, observations).
    - neuron_to_plot: int, index of the neuron to plot.
    - num_bins: int, number of bins to divide each behavior variable into.
    - ax: matplotlib.axes.Axes, the axes object to plot on. If None, a new figure is created.

    Returns:
    - fig: matplotlib.figure.Figure, the figure object containing the plot.
    - ax: matplotlib.axes.Axes, the axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    # Bin the behavioral variables
    bins1 = np.linspace(np.min(behavior_var1), np.max(behavior_var1), num_bins+1)
    bins2 = np.linspace(np.min(behavior_var2), np.max(behavior_var2), num_bins+1)
    
    # Create a 2D array to store binned activity
    binned_activity = np.zeros((num_bins, num_bins))
    
    # Bin the data and calculate mean activity
    for i in range(num_bins):
        for j in range(num_bins):
            indices = np.where(
                (behavior_var1 >= bins1[i]) & (behavior_var1 < bins1[i+1]) &
                (behavior_var2 >= bins2[j]) & (behavior_var2 < bins2[j+1])
            )[0]
            binned_activity[j, i] = np.mean(neural_activity[neurons_to_plot, indices])
    
    # Create the heatmap
    im = ax.imshow(binned_activity, origin='lower', aspect='auto', cmap='viridis')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f'Mean activity of {filtered_columns[neurons_to_plot]}')
    
    # Set labels and title
    ax.set_xlabel(behavior_var1.name)
    ax.set_ylabel(behavior_var2.name)
    #ax.set_title(f'2D Tuning Heatmap for {filtered_columns[neuron_to_plot]}')
    
    # Set tick labels
    ax.set_xticks(np.linspace(0, num_bins-1, 5))
    ax.set_yticks(np.linspace(0, num_bins-1, 5))
    ax.set_xticklabels([f'{x:.2f}' for x in np.linspace(np.min(behavior_var1), np.max(behavior_var1), 5)])
    ax.set_yticklabels([f'{y:.2f}' for y in np.linspace(np.min(behavior_var2), np.max(behavior_var2), 5)])
    save_path = f"{example_path_results}fwv_heading_heatmap_{trial_num}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #print(f"Figure saved to {save_path}")
    return fig, ax

def filter_based_on_histogram(behavior_variable, min_freq_threshold):
    """
    Filters the behavioral variable data to remove points distant from the main mode,
    isolated by at least two consecutive bins with very low frequency. Automatically adjusts
    bins based on the Freedman-Diaconis rule. Skips filtering if the variable name is 'heading'.
    
    Parameters:
    - behavior_variable: Pandas Series, the behavioral variable data.
    - min_freq_threshold: The minimum frequency (as a proportion of total) to consider a bin as non-negligible.
    
    Returns:
    - Pandas Series, filtered behavioral variable data (or original data if it's 'heading').
    """
    # Check if the variable is 'heading'
    if behavior_variable.name == 'heading':
        return behavior_variable  # Return the original data without filtering

    # Calculate bin width using the Freedman-Diaconis rule
    bin_width = 1
    range_ = np.max(behavior_variable) - np.min(behavior_variable)
    bins = int(np.round(range_ / bin_width))
    print(bins)
    
    # Compute the histogram with the calculated number of bins
    counts, bin_edges = np.histogram(behavior_variable, bins=bins)
    
    # Identify bins below the frequency threshold
    low_freq_bins_mask = counts < min_freq_threshold
    
    # Find indices where two consecutive bins are below the threshold
    two_consecutive_low_bins = np.where(np.convolve(low_freq_bins_mask, [1,1], mode='valid') == 2)[0]
    
    if two_consecutive_low_bins.size == 0:
        return behavior_variable  # No such consecutive low bins, return original series
    
    # Determine cutoffs
    mode_bin_index = np.argmax(counts)
    lower_cutoff_bins = two_consecutive_low_bins[two_consecutive_low_bins < mode_bin_index]
    upper_cutoff_bins = two_consecutive_low_bins[two_consecutive_low_bins > mode_bin_index]
    lower_cutoff = bin_edges[0] if lower_cutoff_bins.size == 0 else bin_edges[lower_cutoff_bins.max() + 2]  # +2 to include the low bin
    upper_cutoff = bin_edges[-1] if upper_cutoff_bins.size == 0 else bin_edges[upper_cutoff_bins.min()]
    
    # Filter the behavioral variable
    filtered_variable = behavior_variable[(behavior_variable >= lower_cutoff) & (behavior_variable <= upper_cutoff)]
    
    return filtered_variable

def plot_tuning_curve_and_scatter(neural_activity, filtered_columns, neurons_to_plot, behavioral_variables, filtered_behavior_variables, num_behavioral_variables, mean_angle, mode_angle, num_bins, example_path_results, trial_num, tuning_whole_session, segment_id = None):
    def generate_plots(neurons_to_plot, neural_activity, filtered_columns, behavior_variables, plot_func, ax, is_filtered=False):
        for j in neurons_to_plot:
            for i, behavior_variable in enumerate(behavior_variables):
                plot_func(behavior_variable, filtered_columns, neural_activity, [j], num_bins if is_filtered else None, ax=ax[j, i])
                if is_filtered and i == 3:
                    ax[j, i].axvline(mean_angle, color='red', label='mean heading')
                    ax[j, i].axvline(mode_angle, linestyle='--', color='red', label='mode heading')
                    ax[j, i].legend()

    neurons = neurons_to_plot if neural_activity.shape[0] > 1 else [0]

    # First set of plots (tuning curves)
    fig, ax = plt.subplots(len(neurons), num_behavioral_variables, figsize=(num_behavioral_variables * 5, len(neurons) * 5))
    if len(ax.shape) == 1:
        ax = ax[np.newaxis, :]
    generate_plots(neurons, neural_activity, filtered_columns, filtered_behavior_variables, tuning_curve_1d, ax, is_filtered=True)
    plt.tight_layout()
    if tuning_whole_session:
        plt.savefig(f"{example_path_results}1d_tuning_curve_{trial_num}.png")
    else: 
        plt.savefig(f"{example_path_results}1d_tuning_curve_{trial_num}_segment_{segment_id}.png")
    plt.close()

    # Second set of plots (scatter plots)
    fig, ax = plt.subplots(len(neurons), num_behavioral_variables, figsize=(num_behavioral_variables * 5, len(neurons) * 5))
    if len(ax.shape) == 1:
        ax = ax[np.newaxis, :]
    generate_plots(neurons, neural_activity, filtered_columns, behavioral_variables, plot_scatter, ax)
    plt.tight_layout()
    if tuning_whole_session:
        plt.savefig(f"{example_path_results}scatterplot_{trial_num}.png")
    else: 
        plt.savefig(f"{example_path_results}scatterplot_{trial_num}_segment_{segment_id}.png")
    plt.close()

    # forwardV vs.heading heatmap


    # plot forwardV vs. heading as sanity check
    fig, ax = plt.subplots(figsize=(5,5))
    generate_plots([0], np.array([behavioral_variables[0]]), [filtered_behavior_variables[0].name], [filtered_behavior_variables[3]], tuning_curve_1d, np.array([[ax]]), is_filtered=True)
    plt.tight_layout()
    plt.savefig(f"{example_path_results}1d_tuning_curve_ctrl_{trial_num}.png")
    plt.close()


# Example usage:
#tuning_curve_1d(behavior_variable, neural_activity,neurons_to_plot,num_bins)


def calculate_theta_g_rho(df, window_size=30, speed_threshold=0.67, cue_jump_time=5):
    """
    Calculate the goal direction (theta_g) and consistency of head direction (rho_t) for each time point.
    
    Parameters:
    - df: DataFrame containing 'forward', 'side', 'yaw', and 'heading' columns (time series of a trial).
    - window_size: The size of the window in seconds to compute the rolling statistics.
    - speed_threshold: Speed threshold below which the fly is considered standing still.
    - cue_jump_time: Time after the cue jump (in seconds) to exclude data points.
    
    Returns:
    - A DataFrame with 'theta_g' and 'rho_t' columns added (same length as input).
    """
    # Calculate cumulative speed (forward + side + yaw)
    df['speed'] = np.sqrt(df['fwV']**2 + df['sideV']**2 + df['yawV']**2)
    
    # Initialize columns for theta_g and rho_t with NaN values
    df['theta_g'] = np.nan
    df['rho_t'] = np.nan
    
    # Filter data points where speed is above the threshold
    valid_indices = df[df['speed'] > speed_threshold].index
    
    # Sliding window calculation
    half_window = window_size // 2
    for i in valid_indices:
        # Define window boundaries, ensuring we don't exceed dataframe limits
        start_idx = max(i - half_window, 0)
        end_idx = min(i + half_window, len(df) - 1)
        
        # Extract window of head directions
        head_window = df['heading'].iloc[start_idx:end_idx]
        
        # Calculate theta_g (goal direction)
        sum_sin = np.sum(np.sin(head_window))
        sum_cos = np.sum(np.cos(head_window))
        theta_g = np.arctan2(sum_sin, sum_cos)

        # Ensure theta_g is between 0 and 2*pi
        theta_g = np.mod(theta_g, 2 * np.pi)
        
        # Calculate rho_t (consistency of head direction)
        N_w = len(head_window)
        rho_t = np.sqrt((sum_cos / N_w)**2 + (sum_sin / N_w)**2)
        
        # Store in DataFrame at index i
        df.at[i, 'theta_g'] = theta_g
        df.at[i, 'rho_t'] = rho_t
    
    return df


def circular_mode(data, bins=360):
    """Calculate the circular mode of the given data."""
    #data_rad = np.deg2rad(data)
    density = gaussian_kde(data)
    x = np.linspace(0, 2*np.pi, bins)
    y = density(x)
    peaks, _ = find_peaks(y)
    mode_idx = peaks[np.argmax(y[peaks])]
    return x[mode_idx]

def plot_heading_tuning_circular(behav_df, neural_df, unique_seg, filtered_columns, example_path_results, trial_num, unique_mode_headings=None):
    """
    Plot heading tuning for different segments of a time series for multiple neurons.
    
    Parameters:
    - behav_df: DataFrame containing behavioral data including 'heading' and 'segment'
    - neural_df: DataFrame containing neural data
    - unique_seg: List of unique segment numbers to plot
    - filtered_columns: List of neuron column names to plot
    - unique_mode_headings: List of mode headings to plot as radial lines (optional)
    """
    n_neurons = len(filtered_columns)
    fig, axes = plt.subplots(1, n_neurons, figsize=(5*n_neurons, 5), subplot_kw={'projection': 'polar'})
    
    # If there's only one neuron, axes will not be an array, so we convert it to a single-element list
    if n_neurons == 1:
        axes = [axes]
    
    # Define a color cycle for different segments
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_seg)))

    # Calculate circular mode headings if not provided
    if unique_mode_headings is None:
        unique_mode_headings = []
        for seg in unique_seg:
            mask = behav_df['segment'] == seg
            headings = behav_df.loc[mask, 'heading'].values
            mode_heading = circular_mode(headings)
            unique_mode_headings.append(mode_heading)

    for ax_idx, (ax, neuron) in enumerate(zip(axes, filtered_columns)):
        # Plot each segment
        for i, seg in enumerate(unique_seg):
            mask = behav_df['segment'] == seg
            ax.scatter(behav_df.loc[mask, 'heading'], neural_df.loc[mask, neuron], 
                       color=colors[i], alpha=0.1, label=f'Segment {seg}')
        
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        
        ax.set_title(f"{neuron}")
        if ax_idx == 0:  # Only set ylabel for the first subplot
            ax.set_ylabel("Neural Activity")
        
        #Plot mode headings
        for i, mode_hd in enumerate(unique_mode_headings):
            ax.plot([mode_hd, mode_hd], [0, ax.get_ylim()[1]], color=colors[i % len(colors)], 
                    linewidth=2, linestyle='--', label=f'Mode {i+1}')
        
        if ax_idx == n_neurons - 1:  # Only add legend to the last subplot
            ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"{example_path_results}scatter_segment_circular_{trial_num}.png", dpi=300, bbox_inches='tight')
    
    #plt.show()

# Example usage:
# filtered_columns = ['MBON09L', 'MBON10L', 'MBON11L']
# unique_seg = [0, 1, 2]
# plot_heading_tuning(behav_df, neural_df, unique_seg, filtered_columns, unique_mode_headings=[mode_hd, mode_hd2])



def binned_stats(headings, values, bins):
    """
    Calculate binned averages and standard errors.
    
    Parameters:
    - headings: array of angular data (in radians).
    - values: corresponding values (e.g., MBON09L) to be averaged.
    - bins: number of bins or a list of bin edges for angular data.
    
    Returns:
    - bin_centers: array of bin centers (in radians).
    - bin_means: array of mean values for each bin.
    - bin_errors: array of standard errors for each bin.
    """
    binned, bin_edges = pd.cut(headings, bins=bins, labels=False, retbins=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_means = []
    bin_errors = []
    for i in range(len(bin_edges) - 1):
        bin_values = values[binned == i]
        bin_means.append(np.mean(bin_values))
        bin_errors.append(np.std(bin_values) / np.sqrt(len(bin_values)))

    return bin_centers, np.array(bin_means), np.array(bin_errors)

def plot_binned_heading_tuning(behav_df, neural_df, unique_seg, filtered_columns, example_path_results, trial_num, unique_mode_headings=None, n_bins=16):
    """
    Plot binned heading tuning for different segments of a time series for multiple neurons.
    
    Parameters:
    - behav_df: DataFrame containing behavioral data including 'heading' and 'segment'
    - neural_df: DataFrame containing neural data
    - unique_seg: List of unique segment numbers to plot
    - filtered_columns: List of neuron column names to plot
    - unique_mode_headings: List of mode headings to plot as radial lines (optional)
    - n_bins: Number of bins for circular data (default 16)
    """
    n_neurons = len(filtered_columns)
    fig, axes = plt.subplots(1, n_neurons, figsize=(5*n_neurons, 5), subplot_kw={'projection': 'polar'})
    
    if n_neurons == 1:
        axes = [axes]
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_seg)))
    bins = np.linspace(0, 2 * np.pi, n_bins + 1)
    if unique_mode_headings is None:
        unique_mode_headings = []
        for seg in unique_seg:
            mask = behav_df['segment'] == seg
            headings = behav_df.loc[mask, 'heading'].values
            mode_heading = circular_mode(headings)
            unique_mode_headings.append(mode_heading)

    for ax_idx, (ax, neuron) in enumerate(zip(axes, filtered_columns)):
        for i, seg in enumerate(unique_seg):
            mask = behav_df['segment'] == seg
            headings = behav_df.loc[mask, 'heading']
            values = neural_df.loc[mask, neuron]
            
            centers, means, errors = binned_stats(headings, values, bins)
            
            ax.errorbar(centers, means, yerr=errors, fmt='o-', 
                        label=f'Segment {seg}', color=colors[i])
        
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        
        ax.set_title(f"{neuron}")
        if ax_idx == 0:
            ax.set_ylabel("Neural Activity")
        
        if unique_mode_headings is not None:
            for i, mode_hd in enumerate(unique_mode_headings):
                ax.plot([mode_hd, mode_hd], [0, ax.get_ylim()[1]], 
                        color=colors[i], linewidth=2, linestyle='--', 
                        label=f'Mode {i+1}: {mode_hd:.1f}°')
        
        if ax_idx == n_neurons - 1:
            ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
    
    plt.tight_layout()
    
    plt.savefig(f"{example_path_results}binned_heading_tuning_{trial_num}.png", dpi=300, bbox_inches='tight')
    
    plt.show()


class PathSegmenter(ABC):
    @abstractmethod
    def segment(self, df):
        pass

class RhoThresholdSegmenter(PathSegmenter):
    def __init__(self, rho_threshold=0.88, speed_threshold=0.67, min_duration_below_rho=1.0, min_inactivity_duration=2, time_column='time'):
        self.rho_threshold = rho_threshold
        self.speed_threshold = speed_threshold
        self.min_duration_below_rho = min_duration_below_rho
        self.min_inactivity_duration = min_inactivity_duration
        self.time_column = time_column

    def segment(self, df):
        # Implementation of the segment_path function you provided
        result_df = df.copy()
        result_df['segment'] = np.nan
        segment_id = 0
        
        in_segment = False
        segment_start_idx = None
        below_rho_start_idx = None
        
        for i in range(1, len(result_df)):
            if result_df['rho_t'].iloc[i] > self.rho_threshold:
                if not in_segment and result_df[self.time_column].iloc[i] - result_df[self.time_column].iloc[segment_start_idx or 0] >= self.min_duration_below_rho:
                    in_segment = True
                    segment_start_idx = i
            else:
                if in_segment:
                    if below_rho_start_idx is None:
                        below_rho_start_idx = i
                    elif result_df[self.time_column].iloc[i] - result_df[self.time_column].iloc[below_rho_start_idx] >= self.min_duration_below_rho:
                        in_segment = False
                        result_df.loc[segment_start_idx:i, 'segment'] = segment_id
                        segment_id += 1
                        below_rho_start_idx = None

            if result_df['rho_t'].iloc[i] == 1:
                result_df.loc[segment_start_idx:i, 'segment'] = np.nan

            inactive_duration = result_df[(result_df['speed'] < self.speed_threshold) & (result_df['segment'] == segment_id)]
            if len(inactive_duration) > 0 and inactive_duration[self.time_column].iloc[-1] - inactive_duration[self.time_column].iloc[0] >= self.min_inactivity_duration:
                result_df.loc[segment_start_idx:i, 'segment'] = np.nan

        if in_segment:
            result_df.loc[segment_start_idx:, 'segment'] = segment_id

        return result_df

class ManualSegmenter(PathSegmenter):
    def __init__(self, indices):
        self.indices = indices

    def segment(self, df):
        result_df = df.copy()
        result_df['segment'] = np.nan
        for i, idx in enumerate(self.indices[:-1]):
            result_df.loc[idx:self.indices[i+1], 'segment'] = i
        return result_df

class PathSegmentationFactory:
    @staticmethod
    def create_segmenter(method, **kwargs):
        if method == 'rho_threshold':
            return RhoThresholdSegmenter(**kwargs)
        elif method == 'manual':
            return ManualSegmenter(**kwargs)
        else:
            raise ValueError(f"Unknown segmentation method: {method}")

# Usage example:
# factory = PathSegmentationFactory()
# rho_segmenter = factory.create_segmenter('rho_threshold', rho_threshold=0.9)
# manual_segmenter = factory.create_segmenter('manual', indices=[0, 100, 200, 300])
# 
# segmented_df_rho = rho_segmenter.segment(df)
# segmented_df_manual = manual_segmenter.segment(df)

def process_behavioral_variables(behav_df, example_path_results, trial_num):
    """
    Processes the behavioral variables from the behavior dataframe.
    
    Parameters:
    - behav_df: DataFrame containing the behavioral data (with columns 'fwV', 'sideV', 'yawV', and 'heading').
    - example_path_results: Path where results are saved.
    - trial_num: Trial number for saving results.
    
    Returns:
    - mean_angle: Mean head direction angle.
    - median_angle: Median head direction angle.
    - mode_angle: Mode head direction angle.
    - behavioral_variables: List of raw behavioral variables.
    - filtered_behavior_variables: List of filtered behavioral variables.
    """
    # Calculate circular statistics (mean, median, and mode angles)
    mean_angle, median_angle, mode_angle = calc_circu_stats(behav_df.heading, 30, example_path_results, trial_num)
    
    # Define the behavioral variables
    behavioral_variables = [behav_df.fwV, behav_df.sideV, behav_df.yawV, behav_df.heading]
    
    # Filter behavioral variables based on histogram
    filtered_behavior_variables = [filter_based_on_histogram(var, 0.5) for var in behavioral_variables]
    num_behavioral_variables = len(filtered_behavior_variables)
    
    return mean_angle, median_angle, mode_angle, behavioral_variables, filtered_behavior_variables, num_behavioral_variables

# Example usage:
# mean_angle, median_angle, mode_angle, behavioral_vars, filtered_vars = process_behavioral_variables(behav_df, example_path_results, trial_num)



# encoding, decoding models 
# calcium imaging GLM 

base_path = "//research.files.med.harvard.edu/neurobio/wilsonlab/Jingxuan/processed/MBON_imaging/MBON21/"
example_path_data = base_path+"20230831-4_MBON09_GCAMP7f_apple_prob_sigmoid/data/"
example_path_results = base_path+"20230831-4_MBON09_GCAMP7f_apple_prob_sigmoid/results/"
#trial_num = 1
#qualified_trials = find_complete_trials(example_path_data)
#print(qualified_trials)
#is_mat73, roi_df, dff_raw, kinematics_raw, preprocessed_vars_ds, preprocessed_vars_odor = load_intermediate_mat(example_path_data,trial_num)
#print(roi_df.head(10))
    

def main(example_path_data, example_path_results, trial_num, tuning_whole_session=False, segment_method='manual'):
    # Define key variables
    behavior_var1, behavior_var2 = 'translationalV', 'heading'
    roi_kw, roi_kw2 = 'MBON', 'FB'
    #tuning_whole_session = False

    # Load data and preprocess
    is_mat73, roi_df, dff_raw, kinematics_raw, preprocessed_vars_ds, preprocessed_vars_odor = load_intermediate_mat(example_path_data, trial_num)
    behav_df = make_df_behavior(dff_raw, preprocessed_vars_ds, preprocessed_vars_odor, trial_num, ball_d=9)
    xPos, yPos = reconstruct_path(behav_df, ball_d=9)
    #plot_fly_traj(xPos, yPos, behav_df, 'odor', example_path_results, trial_num)
    clicked_indices = plot_fly_traj_interactive(xPos, yPos, behav_df, 'odor', example_path_results, trial_num)
    print("Indices of clicked points:", clicked_indices)
    # Load and validate ROI data
    roi_names, hdeltab_index, epg_index, fr1_index, hdeltab_sequence, epg_sequence, fr1_sequence = get_roi_seq(roi_df)
    dff_all_rois, dff_time = load_dff_raw(is_mat73, dff_raw)
    
    if len(dff_all_rois) < len(hdeltab_index):
        print("raw df/f matrix dimension doesn't align with ROI names")
        return
    
    # Handle special cases
    special_cases = {
        "//research.files.med.harvard.edu/neurobio/wilsonlab/Jingxuan/processed/dan_imaging/20220824-5_MB196B_GCAMP7f_long/data/": [0, 2, 3],
        "//research.files.med.harvard.edu/neurobio/wilsonlab/Jingxuan/processed/FR1_imaging/20230719-3_FR1_GCAMP7f_odor_odor_apple_width10_fly2/data/": 'hDeltaB',
        "//research.files.med.harvard.edu/neurobio/wilsonlab/Jingxuan/processed/hDeltaB_imaging/qualified_sessions/20220720-3_hDeltaB_GCAMP7f_long/data/": 'hDealtaB'
    }
    if example_path_data in special_cases:
        case = special_cases[example_path_data]
        if isinstance(case, list):
            dff_all_rois = dff_all_rois[case, :]
        else:
            roi_kw = case

    # Create neural dataframe and combine with behavioral data
    neural_df = make_df_neural(dff_all_rois, dff_time, roi_names, hdeltab_index, epg_index, fr1_index, hdeltab_sequence, epg_sequence, fr1_sequence)
    behav_df = calculate_theta_g_rho(behav_df)
    if not tuning_whole_session:
        factory = PathSegmentationFactory()
        manual_segmenter = factory.create_segmenter(segment_method, indices=clicked_indices)
        behav_df = manual_segmenter.segment(behav_df)
    combined_df = combine_df(behav_df, neural_df)
    
    # Extract and plot data
    do_normalize = True
    roi_mtx = extract_heatmap(combined_df, roi_kw, roi_kw2, do_normalize, example_path_results, trial_num)
    combined_df_trun, nonpara_summ_df = calc_nonpara(combined_df, roi_kw, roi_kw2, roi_mtx, False)
    
    nonpara_plot_bybehav(nonpara_summ_df, nonpara_summ_df, behavior_var1, example_path_results, trial_num)
    nonpara_plot_bybehav(nonpara_summ_df, combined_df_trun, behavior_var2, example_path_results, trial_num)
    
    if roi_kw == 'hDeltaB' and do_normalize and roi_mtx is not None:
        paramfit_df = fit_sinusoid(neural_df, roi_mtx)
        plot_with_error_shading(paramfit_df, example_path_results, trial_num)
    else:
        plot_time_series(neural_df, behav_df, example_path_results, trial_num)

    
    # Select neural activity based on ROI identity
    if roi_kw == 'FR1':
        roi_kw3 = 'CRE'
        filtered_columns = [col for col in combined_df.columns if roi_kw in col and roi_kw3 in col]
        fb_columns = [col for col in combined_df.columns if roi_kw in col and roi_kw2 in col]
        neural_df['fb_mean'] = neural_df[fb_columns].mean(axis=1)
        neural_activity = np.concatenate((neural_df[filtered_columns].T, neural_df[['fb_mean']].T), axis=0)
    else:
        filtered_columns = [col for col in combined_df.columns if roi_kw in col]
        #neural_activity = np.array(neural_df[filtered_columns].T)
        neural_activity = neural_df[filtered_columns]

    # Plot tuning curve and scatter plots
    num_bins = 20
    neurons_to_plot = range(neural_activity.shape[1])
    # not segment
    if tuning_whole_session:
        # Calculate and plot statistics
        mean_angle, median_angle, mode_angle, behavioral_variables, filtered_behavior_variables, num_behavioral_variables = process_behavioral_variables(behav_df, example_path_results, trial_num)
        fig, ax = tuning_heatmap_2d(behavioral_variables[3], behavioral_variables[0], filtered_columns, np.array(neural_activity.T), neurons_to_plot[0], num_bins, example_path_results, trial_num, ax=None)
        #plot_tuning_curve_and_scatter(np.array(neural_activity.T), filtered_columns, neurons_to_plot, behavioral_variables, filtered_behavior_variables, num_behavioral_variables, mean_angle, mode_angle, num_bins, example_path_results, trial_num, tuning_whole_session)
    else:
        seg_threshold = 50
        unique_seg = combined_df['segment'].unique()
        unique_seg = unique_seg[~np.isnan(unique_seg)]
        plot_heading_tuning_circular(behav_df, neural_df, unique_seg, filtered_columns, example_path_results, trial_num, unique_mode_headings=None)
        plot_binned_heading_tuning(behav_df, neural_df, unique_seg, filtered_columns, example_path_results, trial_num, unique_mode_headings=None)
        for i in range(len(unique_seg)):
            if np.sum(behav_df['segment'] == unique_seg[i]) > seg_threshold:
                neural_activity_i = neural_activity[behav_df['segment'] == unique_seg[i]]
                behav_df_i = behav_df[behav_df['segment'] == unique_seg[i]]
                mean_angle, median_angle, mode_angle, behavioral_variables, filtered_behavior_variables, num_behavioral_variables = process_behavioral_variables(behav_df_i, example_path_results, trial_num)
                plot_tuning_curve_and_scatter(np.array(neural_activity_i.T), filtered_columns, neurons_to_plot, behavioral_variables, filtered_behavior_variables, num_behavioral_variables, mean_angle, mode_angle, num_bins, example_path_results, trial_num, tuning_whole_session, unique_seg[i])


#main(example_path_data, example_path_results,1,False)

def calc_peak_correlation_full(series1, series2, max_lag):
    # Ensure series are zero-mean for meaningful correlation results
    series1 = series1 - np.mean(series1)
    series2 = series2 - np.mean(series2)
    
    # Calculate cross-correlation using 'full' mode
    cross_corr = np.correlate(series1, series2, mode='full')
    
    # Cross-correlation results include negative to positive lags; find the index of the zero lag
    zero_lag_index = len(series1) - 1
    
    # Focus on lags from -max_lag to +max_lag
    start_index = zero_lag_index - max_lag
    end_index = zero_lag_index + max_lag + 1
    valid_corrs = cross_corr[start_index:end_index]
    
    # Normalize correlation coefficients
    normalization = np.sqrt(np.sum(series1**2) * np.sum(series2**2))
    valid_corrs = valid_corrs / normalization
    
    # Get index of maximum correlation from the valid correlations
    max_index = np.argmax(valid_corrs)
    peak_lag = max_index - max_lag  # Adjust index to account for negative lags
    peak_correlation = valid_corrs[max_index]
    
    return peak_correlation, peak_lag

# https://jdblischak.github.io/fucci-seq/circ-simulation-correlation.html

def calc_circular_correlation(circular_series, series2):
    circular_series_cos = np.cos(circular_series)
    circular_series_sin = np.sin(circular_series)
    correlation_cos = series2.corr(pd.Series(circular_series_cos))
    correlation_sin = series2.corr(pd.Series(circular_series_sin))
    correlation_cs = pd.Series(circular_series_cos).corr(pd.Series(circular_series_sin))
    numerator = correlation_cos**2 + correlation_sin**2-2*correlation_cos*correlation_sin*correlation_cs
    denominator = 1- correlation_cs**2
    combined_correlation = np.sqrt(numerator/denominator)
    return combined_correlation


def calc_correlation_batch(example_path_data, example_path_results,trial_num,corr_dict):
    is_mat73, roi_df, dff_raw, kinematics_raw, preprocessed_vars_ds, preprocessed_vars_odor = load_intermediate_mat(example_path_data,trial_num)
    behav_df = make_df_behavior(dff_raw, preprocessed_vars_ds, preprocessed_vars_odor,trial_num,ball_d = 9)
    xPos, yPos = reconstruct_path(behav_df, ball_d = 9)
    #plot_fly_traj(xPos, yPos, behav_df, 'odor', example_path_results,trial_num)
    roi_names, hdeltab_index, epg_index, fr1_index, hdeltab_sequence, epg_sequence, fr1_sequence = get_roi_seq(roi_df)
    dff_all_rois, dff_time = load_dff_raw(is_mat73, dff_raw)
    if len(dff_all_rois) < len(hdeltab_index):
        print("raw df/f matrix dimension doesn't align with ROI names")
        return
    if example_path_data == "//research.files.med.harvard.edu/neurobio/wilsonlab/Jingxuan/processed/dan_imaging/20220824-5_MB196B_GCAMP7f_long/data/":
        dff_all_rois = dff_all_rois[[0,2,3],:]
    neural_df = make_df_neural(dff_all_rois, dff_time, roi_names, hdeltab_index, epg_index, fr1_index, hdeltab_sequence, epg_sequence, fr1_sequence)
    if not any('MBON09' in col for col in neural_df.columns):
        print('no MBON09 columns')
        return
    combined_df = combine_df(behav_df, neural_df)
    roi_kw = 'hDeltaB'
    if example_path_data == "//research.files.med.harvard.edu/neurobio/wilsonlab/Jingxuan/processed/FR1_imaging/20230719-3_FR1_GCAMP7f_odor_odor_apple_width10_fly2/data/":
        roi_kw = 'hDeltaB'
    if example_path_data == "//research.files.med.harvard.edu/neurobio/wilsonlab/Jingxuan/processed/hDeltaB_imaging/qualified_sessions/20220720-3_hDeltaB_GCAMP7f_long/data/":
        roi_kw = 'hDealtaB'
    roi_kw2 = None
    do_normalize = True
    roi_mtx = extract_heatmap(combined_df, roi_kw, roi_kw2, do_normalize, example_path_results, trial_num)
    if not (roi_mtx is None) and do_normalize:
        paramfit_df = fit_sinusoid(neural_df, roi_mtx)
        plot_with_error_shading(paramfit_df, example_path_results, trial_num)
    else:
        plot_time_series(neural_df, behav_df,example_path_results,trial_num)
    columns_to_loop = ['phase','amplitude', 'baseline']
    for col in columns_to_loop:
        # First series from behav_df
        series1 = paramfit_df[col]
        
        # Filter neural_df for columns containing 'MBON21', average them to get the second series
        series2 = neural_df.filter(regex='MBON09').mean(axis=1)
        
        # Calculate peak correlation with a max lag of 5
        corr, lag_with_peak_corr = calc_peak_correlation_full(series1, series2, 5)

        #print(f"Peak Correlation for {col}: {peak_corr}, Lag: {lag_with_peak_corr}")
        
        # Additional circular correlation calculation for 'heading'
        if col == 'phase':
            corr = calc_circular_correlation(series1, series2)
            #print(f"Circular Correlation for {col}: {circular_corr}")
        corr_dict[col].append(corr)

def save_dfs(example_path_data, example_path_results,trial_num,hdf5_file_path,folder_name):
    #folder_key = example_path_data + '../'
    hdf_key = f"{folder_name}_trial{trial_num}"
    is_mat73, roi_df, dff_raw, kinematics_raw, preprocessed_vars_ds, preprocessed_vars_odor = load_intermediate_mat(example_path_data,trial_num)
    behav_df = make_df_behavior(dff_raw, preprocessed_vars_ds, preprocessed_vars_odor,trial_num,ball_d = 9)
    xPos, yPos = reconstruct_path(behav_df, ball_d = 9)
    #plot_fly_traj(xPos, yPos, behav_df, 'odor', example_path_results,trial_num)
    roi_names, hdeltab_index, epg_index, fr1_index, hdeltab_sequence, epg_sequence, fr1_sequence = get_roi_seq(roi_df)
    dff_all_rois, dff_time = load_dff_raw(is_mat73, dff_raw)
    if len(dff_all_rois) < len(hdeltab_index):
        print("raw df/f matrix dimension doesn't align with ROI names")
        return
    if example_path_data == "//research.files.med.harvard.edu/neurobio/wilsonlab/Jingxuan/processed/dan_imaging/20220824-5_MB196B_GCAMP7f_long/data/":
        dff_all_rois = dff_all_rois[[0,2,3],:]
    neural_df = make_df_neural(dff_all_rois, dff_time, roi_names, hdeltab_index, epg_index, fr1_index, hdeltab_sequence, epg_sequence, fr1_sequence)
    combined_df = combine_df(behav_df, neural_df)
    roi_kw = 'hDeltaB'
    roi_kw2 = None
    do_normalize = True
    roi_mtx = extract_heatmap(combined_df, roi_kw, roi_kw2, do_normalize, example_path_results, trial_num)
    if not (roi_mtx is None) and do_normalize:
        paramfit_df = fit_sinusoid(neural_df, roi_mtx)
    combined_df = combine_df(combined_df, paramfit_df)
    combined_df.to_hdf(hdf5_file_path, key=hdf_key, mode='a')
        

'''def loop_trial(example_path_data, example_path_results,corr_dict = None,hdf5_file_path=None, folder_name=None):
    qualified_trials = find_complete_trials(example_path_data)
    if qualified_trials == []:
        return
    print(f"qualified trial numbers are {qualified_trials}")
    for trial_num in qualified_trials:
        #save_dfs(example_path_data, example_path_results,trial_num,hdf5_file_path,folder_name)
        main(example_path_data, example_path_results,trial_num)
        #calc_correlation_batch(example_path_data, example_path_results,trial_num, corr_dict)

#loop_trial(example_path_data, example_path_results)

def loop_folder(base_path, calc_corr):
    # Ensure base_path is a directory
    if not os.path.isdir(base_path):
        print(f"{base_path} is not a directory.")
        return
    if calc_corr:
        hdf5_file_path = base_path + 'pds_all_flies.h5'
        corr_dict = {}
        corr_dict['phase'] = []
        corr_dict['amplitude'] = []
        corr_dict['baseline'] = []
    #corr_dict['heading'] = []
    # Loop through all items in base_path
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        
        # Check if the item is a directory
        if os.path.isdir(folder_path):
            # Construct pathnames for the 'data' and 'results' subfolders
            example_path_data = folder_path + '/data/'
            example_path_results = folder_path + '/results/'
            if os.path.exists(example_path_data) and os.path.exists(example_path_results):
            # Check if the results folder is empty
                #if os.path.exists(example_path_results) and os.listdir(example_path_results):
                    #print(f"Results folder is not empty, skipping: {example_path_results}")
                    #continue  # Skip to the next folder

                # Print or process the constructed pathnames
                print(f"Data Path: {example_path_data}")
                print(f"Results Path: {example_path_results}")
                loop_trial(example_path_data, example_path_results,corr_dict,hdf5_file_path,folder_name)
    file_path = base_path+'summary_stats.json'
    with open(file_path, 'w') as file:
        json.dump(corr_dict, file)'''

def loop_trial(example_path_data, example_path_results, corr_dict=None, hdf5_file_path=None, folder_name=None, calc_corr=False):
    """
    Loop through qualified trials, run the main function, and optionally calculate correlation.
    """
    qualified_trials = find_complete_trials(example_path_data)
    
    if not qualified_trials:
        return

    print(f"Qualified trial numbers are {qualified_trials}")
    
    for trial_num in qualified_trials:
        main(example_path_data, example_path_results, trial_num, True)
        
        # Optionally calculate correlation
        if calc_corr and corr_dict is not None:
            calc_correlation_batch(example_path_data, example_path_results, trial_num, corr_dict)

#loop_trial(example_path_data, example_path_results)

def loop_folder(base_path, calc_corr=False):
    """
    Loop through all folders in the base path, process each trial, and optionally calculate correlation.
    """
    # Ensure base_path is a valid directory
    if not os.path.isdir(base_path):
        print(f"{base_path} is not a directory.")
        return
    
    # Initialize correlation dictionary if calculating correlation
    corr_dict, hdf5_file_path = None, None
    if calc_corr:
        hdf5_file_path = os.path.join(base_path, 'pds_all_flies.h5')
        corr_dict = {'phase': [], 'amplitude': [], 'baseline': []}

    # Loop through all subfolders in base_path
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        
        if os.path.isdir(folder_path):
            example_path_data = folder_path + '/data/'
            example_path_results = folder_path + '/results/'
            
            # Check if both 'data' and 'results' subfolders exist
            if os.path.exists(example_path_data) and os.path.exists(example_path_results):
                print(f"Processing Data Path: {example_path_data}")
                print(f"Processing Results Path: {example_path_results}")
                
                # Process the trials in the current folder
                loop_trial(example_path_data, example_path_results, corr_dict, hdf5_file_path, folder_name, calc_corr)

    # Save correlation results if applicable
    if calc_corr and corr_dict:
        file_path = os.path.join(base_path, 'summary_stats.json')
        with open(file_path, 'w') as file:
            json.dump(corr_dict, file)

    
#loop_folder(base_path)
#main(example_path_data, example_path_results, 1,True)

