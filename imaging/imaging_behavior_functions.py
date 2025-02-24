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
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from abc import ABC, abstractmethod
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks


################################# Loading #################################

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


def load_matfile_to_df(example_path_data, folder_name, trial_num, is_odor_trial=False):
    """
    Load a MATLAB v7.3 .mat file using the mat73 library and extract struct variables into a pandas DataFrame.
    
    Args:
        example_path_data (str): Path to the .mat file directory.
        folder_name (str): Folder name containing date and fly number.
        trial_num (int or str): Trial number identifier.
        is_odor_trial (bool): Whether to include odor-related processing.

    Returns:
        pd.DataFrame: A DataFrame with extracted behavioral and neural data.
    """
    # Extract relevant parts
    date_part = folder_name.split('-')[0]  # Extract the date (e.g., "20250105")
    fly_num = folder_name.split('-')[1].split('_')[0]  # Extract the fly number (e.g., "5")

    try:
        # Attempt to load the .mat file using mat73
        mat_data = mat73.loadmat(example_path_data + f"{date_part}_{fly_num}_{trial_num}_ts_.mat")
        print("Loaded using mat73.")
    except Exception as e:
        print(f"mat73 failed with error: {e}. Trying scipy.io.loadmat...")
        try:
            # Fall back to scipy.io.loadmat
            mat_data = scipy.io.loadmat(example_path_data + f"{date_part}_{fly_num}_{trial_num}_ts_.mat")
            print("Loaded using scipy.io.loadmat.")
        except Exception as e2:
            print(f"scipy.io.loadmat also failed with error: {e2}.")
            mat_data = None
    
    # Extract the 'ts' struct from the loaded data
    ts = mat_data['ts']
    
    # Dictionary to store all extracted data
    data = {}

    # List of relevant sub-structs, including 'resp'
    sub_structs = ['flypos', 'ball', 'vis', 'resp']
    neural_df = pd.DataFrame()

    # Loop over each sub-struct and extract its fields
    for sub_struct_name in sub_structs:
        if sub_struct_name in ts:
            sub_struct = ts[sub_struct_name]
            
            # Special case for 'resp' sub-struct
            if sub_struct_name == 'resp':
                neural_df = process_resp_to_neural_df(sub_struct)  # Process neural data
            else:
                # Use the recursive function to extract all other fields
                sub_struct_data = extract_struct_fields(sub_struct, sub_struct_name)
                data.update(sub_struct_data)

    # Convert the dictionary into a pandas DataFrame
    behav_df = pd.DataFrame(data)

    # Handle time column
    time_column = ts.get('ti', ts.get('t', None))
    if time_column is not None:
        behav_df['time'] = time_column
        neural_df['time'] = time_column

    # **Only add odor data if is_odor_trial is True**
    if is_odor_trial:
        if 'odor' in ts:
            behav_df['odor'] = ts['odor']
            behav_df['odor_state'] = behav_df['odor'] > -1.5
        else:
            behav_df.rename(columns={'ball_yaw': 'odor'}, inplace=True)
            behav_df['odor_state'] = behav_df['odor'] > -1.5

    return behav_df, neural_df

def rename_dataframe(df):
    # Drop columns containing 'int'
    df = df.drop(columns=[col for col in df.columns if 'int' in col])

    # Rename specified columns
    rename_dict = {
        'ball_forvel': 'fwV',
        'ball_sidevel': 'sideV',
        'ball_yawvel': 'yawV',
        'vis_yaw': 'heading',
        'ball_yaw': 'heading_unjumped',  # 'ball_yaw' may or may not exist
        'vis_yawvel': 'heading_diff',
        'flypos_x': 'xPos',
        'flypos_y': 'yPos',
    }
    
    # Only keep columns from rename_dict that exist in the DataFrame
    rename_dict = {k: v for k, v in rename_dict.items() if k in df.columns}
    
    # Rename columns
    df = df.rename(columns=rename_dict)
    
    # Adjust heading and heading_unjumped if they exist
    df['heading'] = df['heading'] + np.pi
    if 'heading_unjumped' in df.columns:
        df['heading_unjumped'] = df['heading_unjumped'] + np.pi
    
    return df

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


def get_roi_seq_2(roi_df, trial_number):
    roi_df['trialNum'] = roi_df['trialNum'].apply(lambda x: x[0][0])
    
    # Filter the DataFrame to only include rows with the specified trial number
    roi_df = roi_df[roi_df['trialNum'] == trial_number].reset_index(drop=True)
    
    roi_names = roi_df['roiName'].apply(lambda x: x[0])
    roi_hdeltab = roi_names[roi_names.str.contains('hDeltaB', case=False)]
    hdeltab_index = roi_hdeltab.index
    roi_epg = roi_names[roi_names.str.contains('EPG')]
    epg_index = roi_epg.index
    roi_fr1 = roi_names[roi_names.str.contains('FR1') & ~roi_names.str.contains('CRE')]
    fr1_index = roi_fr1.index
    
    hdeltab_seq = roi_hdeltab.str.extract(r'_(\d+)')[0].astype(int).to_numpy()
    
    if epg_index.size > 0:
        epg_seq = roi_epg.str.extract(r'_(\d+)')[0].astype(int).to_numpy()
    else:
        epg_seq = None 
    
    if fr1_index.size > 0:
        fr1_seq = roi_fr1.str.extract(r'_(\d+)')[0].astype(int).to_numpy()
    else:
        fr1_seq = None 
    
    return np.array(roi_names), hdeltab_index, epg_index, fr1_index, hdeltab_seq, epg_seq, fr1_seq


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

def merge_dataframes(behav_df, neural_df, method='interp', tolerance=0.05):
    """
    Merges neural data onto behavioral data using either linear interpolation or nearest time matching.

    Parameters:
    -----------
    behav_df : pd.DataFrame
        DataFrame containing 'time' column and behavioral data.
    neural_df : pd.DataFrame
        DataFrame containing 'time' column and neural data.
    method : str, optional
        Method for merging:
        - 'interp' (default): Uses linear interpolation to estimate neural data at exact behav_df['time'] points.
        - 'nearest': Uses nearest available timestamp in neural_df within the given tolerance.
    tolerance : float, optional
        Maximum allowable time difference for nearest neighbor matching (only used for 'nearest' method).
        
    Returns:
    --------
    pd.DataFrame
        Merged DataFrame with neural data aligned to behav_df['time'].
    """
    if method not in ['interp', 'nearest']:
        raise ValueError("Invalid method. Choose 'interp' (interpolation) or 'nearest' (nearest matching).")

    merged_df = behav_df.copy()

    if method == 'interp':
        # Linear interpolation method
        for col in neural_df.columns:
            if col != 'time':  # Skip the 'time' column
                merged_df[col] = np.interp(behav_df['time'], neural_df['time'], neural_df[col])

    elif method == 'nearest':
        # Ensure time columns are float for accurate merging
        behav_df['time'] = behav_df['time'].astype(float)
        neural_df['time'] = neural_df['time'].astype(float)
        
        # Use merge_asof to align based on nearest time match
        merged_df = pd.merge_asof(behav_df, neural_df, on='time', tolerance=tolerance, direction='nearest')

    return merged_df


################################# Feature processing #################################

# add basic behavioral neural features
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
        df['odor_state'] = (df['odor'] > 0.5).astype(int)
    return df 

def make_df_behavior_new(df):
    df['abssideV'] = np.abs(df['sideV'])
    df['absyawV'] = np.abs(df['yawV'])
    df['net_motion'] = df['abssideV']+df['absyawV']+np.abs(df['fwV'])
    in_notebook = is_notebook()
    if in_notebook:
        threshold = np.percentile(df.net_motion,5)
    else:
        threshold = plot_interactive_histogram(df.net_motion)
    df['net_motion_state'] = (df['net_motion']>threshold).astype(int)
    #df['heading_adj'] = np.unwrap(df['heading'])
    return df 

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

def apply_gaussian_smoothing_to_df(df, columns, sigma):
    """
    Applies Gaussian smoothing to specified columns in a pandas DataFrame.

    Parameters:
    - df: The pandas DataFrame containing the data.
    - columns: List of column names to apply Gaussian smoothing to.
    - sigma: Standard deviation for Gaussian kernel, controlling the smoothing degree.
    
    Returns:
    - A pandas DataFrame with the smoothed columns added as new columns with suffix '_smoothed'.
    """
    for column in columns:
        if column in df.columns:
            smoothed_column = apply_gaussian_smoothing(df[column], sigma)
            df[f'{column}_smoothed'] = smoothed_column
    return df

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

def reconstruct_path(df, ball_d=9):
    """
    Reconstructs the movement path of an object based on forward and side velocities.

    Args:
        df (pd.DataFrame): DataFrame containing 'fwV' (forward velocity), 'sideV' (side velocity),
                           'heading' (direction in radians), and 'time'.
        ball_d (float): Diameter of the ball (in mm), used for movement calculations.

    Returns:
        pd.DataFrame: Updated DataFrame with 'xPos' and 'yPos' columns.
    """
    circum = ball_d * np.pi  # Circumference of ball in mm
    mmPerDeg = circum / 360  # mm per degree of ball movement
    
    fwdAngVel = df.fwV / mmPerDeg  # Convert forward velocity to angular velocity
    zeroedH = df.heading - df.heading.iloc[0]  # Zero out heading relative to the first entry
    time_bin = np.diff(df.time)  # Time differences between consecutive frames

    # Compute x and y position changes
    xChangePos = (fwdAngVel[:-1] * time_bin) * np.sin(zeroedH[:-1]) + (df.sideV[:-1] * time_bin) * np.sin(zeroedH[:-1] + np.pi/4)
    yChangePos = (fwdAngVel[:-1] * time_bin) * np.cos(zeroedH[:-1]) + (df.sideV[:-1] * time_bin) * np.cos(zeroedH[:-1] + np.pi/4)

    # Integrate positions
    xPos = (np.cumsum(xChangePos) - xChangePos[0]) * mmPerDeg
    yPos = (np.cumsum(yChangePos) - yChangePos[0]) * mmPerDeg

    # Pad to maintain DataFrame length
    xPos_padded = pd.concat([xPos, pd.Series(xPos.iloc[-1])], ignore_index=True) 
    yPos_padded = pd.concat([yPos, pd.Series(yPos.iloc[-1])], ignore_index=True) 

    # Store in DataFrame
    df['xPos'] = xPos_padded
    df['yPos'] = yPos_padded

    return df  # Return updated DataFrame


# jump related features

def detect_local_peaks(df, init_t, prominence=0.1, min_time_gap=60):
    """
    Detects local peaks in the 'absolute_circular_diff' column after init_t, ensuring they are at least 60s apart.

    Parameters:
        df (DataFrame): Input DataFrame with 'time' and 'absolute_circular_diff' columns.
        init_t (float): Initial time threshold, only detect peaks after this time.
        prominence (float): Minimum prominence of peaks.
        min_time_gap (float): Minimum time gap between detected peaks in seconds.

    Returns:
        DataFrame: Updated DataFrame with a new binary column 'jump_detected' marking peak locations.
    """
    # Filter data after init_t
    df_filtered = df[df["time"] > init_t]

    # Extract time and data column
    time_values = df_filtered["time"].values
    signal_values = df_filtered["absolute_circular_diff"].values

    # Compute time-based distance in number of samples
    avg_sampling_interval = np.mean(np.diff(time_values))  # Estimate the sampling interval
    min_samples_gap = int(min_time_gap / avg_sampling_interval)  # Convert time to sample count

    # Find peaks with prominence and minimum sample gap
    peaks, properties = find_peaks(signal_values, prominence=prominence, distance=min_samples_gap)

    # Map peak indices back to original DataFrame indices
    peak_indices = df_filtered.index[peaks]

    # Create a new column for peak detection, initialized to 0
    df["jump_detected"] = 0
    df.loc[peak_indices, "jump_detected"] = 1

    return df

def compute_absolute_circular_diff(df, heading_col='heading'):
    """
    Computes absolute circular differences in the specified heading column.

    Parameters:
        df (pd.DataFrame): DataFrame containing the heading column.
        heading_col (str): Name of the column containing the circular variable in radians.

    Returns:
        pd.DataFrame: Modified DataFrame with 'absolute_circular_diff' column.
    """
    # Ensure the heading column exists
    if heading_col not in df.columns:
        raise ValueError(f"Column '{heading_col}' not found in DataFrame.")

    # Compute absolute circular difference
    circular_diff = np.abs(np.arctan2(
        np.sin(np.diff(df[heading_col], prepend=df[heading_col].iloc[0])),
        np.cos(np.diff(df[heading_col], prepend=df[heading_col].iloc[0]))
    ))

    # Add results to DataFrame
    df['absolute_circular_diff'] = circular_diff
    
    return df

# FB population imaging related processing

def calc_nonpara(combined_df, roi_kw, roi_kw2, roi_mtx=None, do_truncate=False):
    if do_truncate:
        combined_df = combined_df[(combined_df["fwV"] > 0.2) | (combined_df["fwV"] < -0.2)]
    
    sigma = 5
    
    # Assuming smoothed columns already exist in the dataframe
    smooth_fwV = combined_df.get('fwV_smoothed', apply_gaussian_smoothing(combined_df.fwV, sigma))
    smooth_sideV = combined_df.get('sideV_smoothed', apply_gaussian_smoothing(combined_df.sideV, sigma))
    
    translational_speed = np.sqrt(smooth_fwV**2 + smooth_sideV**2)
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
        row_iqrs = iqr(roi_mtx, interpolation='midpoint', axis=1)
    
    # Combine mean and IQR into a new DataFrame
    stats_df = pd.DataFrame({'Mean': row_means, 'IQR': row_iqrs, 'translationalV': translational_speed, 'fwV': forward_speed})
    
    # Save stats to the original dataframe with unique column names
    combined_df['mean_stat'] = row_means
    combined_df['iqr_stat'] = row_iqrs
    combined_df['translational_speed'] = translational_speed
    combined_df['forward_speed'] = forward_speed
    
    return combined_df, stats_df


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


def extract_heatmap_2(df, roi_kw, roi_kw2):
    if roi_kw2:
        filtered_columns = [col for col in df.columns if roi_kw in col and roi_kw2 not in col]
        roi_mtx = df[filtered_columns]
    else:
        roi_mtx = df[[col for col in df.columns if roi_kw.lower() in col.lower()]]
    if roi_mtx.empty:
        return None
    scaler = StandardScaler()
    roi_mtx = scaler.fit_transform(roi_mtx)
    return roi_mtx

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
    x_p = np.linspace(0, 2 * np.pi, num=num_roi)
    trial_len = timestamp

    phase_sinfit = np.zeros(trial_len)
    base_sinfit = np.zeros(trial_len)
    amp_sinfit = np.zeros(trial_len)
    phase_perr = np.zeros(trial_len)
    base_perr = np.zeros(trial_len)
    amp_perr = np.zeros(trial_len)

    for i in range(trial_len):
        params, params_covariance = optimize.curve_fit(test_func, x_p, roi_mtx[i, :], maxfev=5000)
        phase_sinfit[i] = x_p[np.argmax(test_func(x_p, params[0], params[1], params[2]))]
        amp_sinfit[i] = np.abs(params[1])
        base_sinfit[i] = params[0]
        perr = np.sqrt(np.diag(params_covariance))
        phase_perr[i] = perr[2]
        base_perr[i] = perr[0]
        amp_perr[i] = perr[1]

    paramfit_df = pd.DataFrame({
        'time': neural_df['time'], 
        'phase': phase_sinfit, 
        'baseline': base_sinfit, 
        'amplitude': amp_sinfit, 
        'phase_error': phase_perr, 
        'baseline_error': base_perr, 
        'amplitude_error': amp_perr
    })

    # Merge paramfit_df onto neural_df based on 'time'
    merged_df = neural_df.merge(paramfit_df, on='time', how='left')

    return merged_df, paramfit_df

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

def circular_mode(circular_data, method='kde', bins=360, num_bins=30):
    """
    Calculate the mode angle of circular data using either Kernel Density Estimation (KDE) or histogram binning.

    Parameters:
    -----------
    circular_data : array-like
        Array of angles in radians, expected to be in range [0, 2π].
    method : str, optional
        Method to use for calculating the mode. Options:
        - 'kde' (default) : Uses Kernel Density Estimation (KDE).
        - 'histogram' : Uses histogram binning.
    bins : int, optional
        Number of bins for KDE estimation (default: 360).
    num_bins : int, optional
        Number of bins for histogram binning (default: 30).

    Returns:
    --------
    float
        Mode angle in radians.
    """
    if method == 'kde':
        # Kernel Density Estimation method
        density = gaussian_kde(circular_data)
        x = np.linspace(0, 2*np.pi, bins)  # Range from 0 to 2π
        y = density(x)
        peaks, _ = find_peaks(y)
        if len(peaks) == 0:
            return None  # No peaks found
        mode_idx = peaks[np.argmax(y[peaks])]
        return x[mode_idx]

    elif method == 'histogram':
        # Histogram binning method
        n, bin_edges = np.histogram(circular_data, bins=num_bins, range=(0, 2*np.pi))
        max_bin_index = np.argmax(n)
        mode_angle = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2
        return mode_angle

    else:
        raise ValueError("Invalid method. Choose 'kde' or 'histogram'.")


def calc_segment_modes_filtered_v2(df, heading_col='heading', segment_col='block', 
                                motion_col='net_motion', motion_threshold=1, num_bins=30):
    """
    Calculate the mode heading for each segment in the trajectory after filtering by net motion threshold.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing heading angles, segment IDs, and net motion values.
    heading_col : str, optional
        Name of the column containing heading angles (default: 'heading').
    segment_col : str, optional
        Name of the column containing block IDs (default: 'block').
    motion_col : str, optional
        Name of the column containing net motion values (default: 'net_motion').
    motion_threshold : float, optional
        Minimum net motion required to include the row in calculations (default: 0.5).
    num_bins : int, optional
        Number of bins to use for circular histogram (default: 30).
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with a new column 'block_modal_heading' containing the mode heading for each block.
    """

    # Filter the DataFrame based on net motion threshold
    df_filtered = df[df[motion_col] > motion_threshold]
    
    # Get unique segment IDs after filtering
    segments = df_filtered[segment_col].unique()

    # Compute mode heading for each segment and store it in a new column
    mode_headings = {}
    for segment in segments:
        segment_headings = df_filtered[df_filtered[segment_col] == segment][heading_col]
        mode_headings[segment] = circular_mode(segment_headings, num_bins, method='histogram')
    
    # Assign mode headings to the full dataframe
    df['block_modal_heading'] = df[segment_col].map(mode_headings)
    
    return df


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


def label_blocks_3(df, init_t, start_dir_1=np.pi/3, end_dir_1=2*np.pi/3, 
    start_dir_2=4*np.pi/3, end_dir_2=5*np.pi/3):
    # Create a mask for rows after the initial time
    mask_after_init = df['time'] >= init_t

    # Create masks for the conditions
    mask_odor_on = df['odor_state']
    mask_heading_1 = (df['heading'] >= start_dir_1) & (df['heading'] <= end_dir_1)
    mask_heading_2 = (df['heading'] >= start_dir_2) & (df['heading'] <= end_dir_2)

    # Combine masks for block transitions
    mask_block_1_start = mask_after_init & mask_odor_on & mask_heading_1
    mask_block_1_end = mask_after_init & ~mask_odor_on & mask_heading_1
    mask_block_2_start = mask_after_init & mask_odor_on & mask_heading_2

    # Initialize the block column with zeros
    df['block'] = 0

    # Identify the indices for block transitions
    block_1_indices = mask_block_1_start.cumsum()
    block_1_persist = (block_1_indices > 0) & ~(mask_block_1_end & (block_1_indices > 0))
    
    block_2_indices = (mask_block_2_start & block_1_persist).cumsum()
    block_2_persist = block_2_indices > 0

    # Assign block values
    df.loc[block_1_persist, 'block'] = 1
    df.loc[block_2_persist, 'block'] = 2

    # Find block boundaries
    block_boundaries = df[df['block'].diff().fillna(0) != 0].index.tolist()

    return df, block_boundaries

def label_blocks_3_v2(df, init_t, time_lengths, start_dir_1=np.pi/3, end_dir_1=2*np.pi/3, 
                    start_dir_2=4*np.pi/3, end_dir_2=5*np.pi/3, detection_threshold=0.01, required_consecutive=1):
    """
    Labels 3 blocks in the dataframe based on odor and heading contingency.
    
    Parameters:
        df (DataFrame): The input dataframe containing 'time', 'odor_state', and 'heading'.
        init_t (float): Initial time threshold for block transitions.
        time_lengths (list): List containing the time durations for block 1 and 2.
        start_dir_1, end_dir_1 (float): Direction range for block 1.
        start_dir_2, end_dir_2 (float): Direction range for block 2.
    
    Returns:
        DataFrame: Updated dataframe with labeled blocks.
        list: Indices of block transitions.
    """
    df = df.copy()
    df['block'] = 0
    block_boundaries = []
    block = 0
    
    odor_detected = False
    odor_lost = False
    consecutive_count = 0
    odor_lost_count = 0
    
    for i, row in df.iterrows():
        if row['time'] < init_t:
            continue
        
        if block == 0:
            if row['odor_state'] and start_dir_1 <= row['heading'] <= end_dir_1:
                block = 1
                block_boundaries.append(i)
                block_start_time = row['time']
        
        elif block == 1:
            if (not row['odor_state'] and start_dir_1 + detection_threshold < row['heading'] < end_dir_1 - detection_threshold) or \
               (row['odor_state'] and start_dir_2 <= row['heading'] <= end_dir_2) or \
               (row['time'] - block_start_time >= time_lengths[0]):
                block = 2
                block_boundaries.append(i)
                block_start_time = row['time']
        
        elif block == 2:
            if row['odor_state'] and start_dir_2 + detection_threshold <= row['heading'] <= end_dir_2 - detection_threshold:
                consecutive_count += 1
                if consecutive_count >= required_consecutive:
                    odor_detected = True
            else:
                consecutive_count = 0
            
            if odor_detected or row['time'] - block_start_time >= time_lengths[1]:
                block_boundaries.append(i)
                block_start_time = row['time']
                odor_detected = False
                
        df.at[i, 'block'] = block
    
    return df, block_boundaries


def label_blocks_5(df, init_t, time_lengths, start_dir_1=np.pi/3, end_dir_1=2*np.pi/3, 
                   start_dir_2=4*np.pi/3, end_dir_2=5*np.pi/3):
    """
    Labels 5 blocks in the dataframe.
    
    Parameters:
        df (DataFrame): The input dataframe containing 'time', 'odor_state', and 'heading'.
        init_t (float): Initial time threshold for block transitions.
        time_lengths (list): List containing the time durations for block 1, 2, and 3.
        start_dir_1, end_dir_1 (float): Direction range for block 1.
        start_dir_2, end_dir_2 (float): Direction range for block 2.
    
    Returns:
        DataFrame: Updated dataframe with labeled blocks.
        list: Indices of block transitions.
    """
    # Create a mask for rows after the initial time
    mask_after_init = df['time'] >= init_t

    # Create masks for the conditions
    mask_odor_on = df['odor_state']
    mask_heading_1 = (df['heading'] >= start_dir_1) & (df['heading'] <= end_dir_1)
    mask_heading_2 = (df['heading'] >= start_dir_2) & (df['heading'] <= end_dir_2)

    # Combine masks for block transitions
    mask_block_1_start = mask_after_init & mask_odor_on & mask_heading_1
    mask_block_1_end = mask_after_init & ~mask_odor_on & mask_heading_1
    mask_block_2_start = mask_after_init & mask_odor_on & mask_heading_2

    # Initialize the block column with zeros
    df['block'] = 0

    # Identify the indices for block transitions
    block_1_indices = mask_block_1_start.cumsum()
    block_1_persist = (block_1_indices > 0) & ~(mask_block_1_end & (block_1_indices > 0))
    df.loc[block_1_persist, 'block'] = 1

    # Find start time for block 1
    block_1_start_time = df.loc[block_1_persist, 'time'].min()
    
    # Compute transition times for subsequent blocks
    block_2_start_time = block_1_start_time + time_lengths[0]
    block_3_start_time = block_2_start_time + time_lengths[1]
    block_4_start_time = block_3_start_time + time_lengths[2]

    # Assign blocks based on time transitions
    df.loc[(df['time'] >= block_2_start_time) & (df['time'] < block_3_start_time), 'block'] = 2
    df.loc[(df['time'] >= block_3_start_time) & (df['time'] < block_4_start_time), 'block'] = 3
    df.loc[df['time'] >= block_4_start_time, 'block'] = 4

    # Find block boundaries
    block_boundaries = df[df['block'].diff().fillna(0) != 0].index.tolist()
    
    return df, block_boundaries


def label_blocks_5_v2(df, init_t, time_lengths, start_dir_1=np.pi/3, end_dir_1=2*np.pi/3, 
                   start_dir_2=4*np.pi/3, end_dir_2=5*np.pi/3,detection_threshold=0.01,required_consecutive=1):
    """
    Labels 5 blocks in the dataframe based on odor and heading contingency.
    
    Parameters:
        df (DataFrame): The input dataframe containing 'time', 'odor_state', and 'heading'.
        init_t (float): Initial time threshold for block transitions.
        time_lengths (list): List containing the time durations for block 1, 2, and 3.
        start_dir_1, end_dir_1 (float): Direction range for block 1.
        start_dir_2, end_dir_2 (float): Direction range for block 2.
    
    Returns:
        DataFrame: Updated dataframe with labeled blocks.
        list: Indices of block transitions.
    """
    df = df.copy()
    df['block'] = 0
    mask_after_init = df['time'] >= init_t
    
    block_boundaries = []
    block = 0
    #block_start_time = init_t
    odor_detected = False
    odor_lost = False
    consecutive_count = 0
    #required_consecutive = 1
    for i, row in df.iterrows():
        if row['time'] < init_t:
            continue
        
        if block == 0:
            if row['odor_state'] and start_dir_1 <= row['heading'] <= end_dir_1:
                block = 1
                block_boundaries.append(i)
                block_start_time = row['time']
            
        elif block == 1:
            if not row['odor_state'] and start_dir_1+detection_threshold < row['heading'] < end_dir_1-detection_threshold:
                odor_lost_count += 1
                if odor_lost_count >= required_consecutive:
                    odor_lost = True
                    #print(f"transition 1 {row}")
            else:
                odor_lost_count = 0
            if odor_lost or row['time'] - block_start_time >= time_lengths[0]:
                block = 2
                block_boundaries.append(i)
                block_start_time = row['time']
                odor_lost = False
            
        elif block == 2:
            if row['odor_state'] and start_dir_2+detection_threshold <= row['heading'] <= end_dir_2-detection_threshold:
                consecutive_count += 1
                if consecutive_count >= required_consecutive:
                    odor_detected = True
                    #print(f"transition 2 {row}")
            else:
                consecutive_count = 0
            if odor_detected or row['time'] - block_start_time >= time_lengths[1]:
                #print(f"transition 2 {row}")
                block = 3
                block_boundaries.append(i)
                block_start_time = row['time']
                odor_detected = False
            
        elif block == 3:
            if not row['odor_state'] and start_dir_2+detection_threshold < row['heading'] < end_dir_2-detection_threshold:
                odor_lost_count += 1
                if odor_lost_count >= required_consecutive:
                    odor_lost = True
                    #print(f"transition 3 {row}")
            else:
                odor_lost_count = 0
            if odor_lost or row['time'] - block_start_time >= time_lengths[2]:
                block = 4
                block_boundaries.append(i)
                block_start_time = row['time']
                odor_lost = False
                
        df.at[i, 'block'] = block
    
    return df, block_boundaries

def compute_odor_metrics(df):
    """
    Adds 'odor_duration', 'past_interval', and 'odor_heading_avg' columns to the DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing 'odor_state', 'time', and 'heading' columns.

    Returns:
    pd.DataFrame: Updated DataFrame with new computed columns.
    """
    onset_indices = df.index[(df["odor_state"].shift(1, fill_value=0) == 0) & (df["odor_state"] == 1)]
    offset_indices = df.index[(df["odor_state"].shift(1, fill_value=0) == 1) & (df["odor_state"] == 0)]

    # Initialize new columns
    df["odor_duration"] = np.nan
    df["past_interval"] = np.nan
    df["odor_heading_avg"] = np.nan

    for onset_idx in onset_indices:
        # Find the first offset after the onset
        offset_idx = offset_indices[offset_indices > onset_idx].min()

        if pd.notna(offset_idx):
            # Calculate odor duration
            df.loc[onset_idx, "odor_duration"] = df.loc[offset_idx, "time"] - df.loc[onset_idx, "time"]

        # Calculate past interval
        last_offset_idx = offset_indices[offset_indices < onset_idx].max()
        if pd.notna(last_offset_idx):
            past_interval = df.loc[onset_idx, "time"] - df.loc[last_offset_idx, "time"]
        else:
            past_interval = df.loc[onset_idx, "time"] - df["time"].iloc[0]
        df.loc[onset_idx, "past_interval"] = past_interval

        # Compute average heading during each odor-on period
        if pd.notna(offset_idx):
            avg_heading = circmean(df.loc[onset_idx:offset_idx, "heading"])
            df.loc[onset_idx, "odor_heading_avg"] = avg_heading

    return df

def compute_odor_metrics(df, heading_window_size=5):
    """
    Adds 'odor_duration', 'past_interval', 'odor_heading_avg', and 'prior_odor_duration' 
    columns to the DataFrame. The 'odor_heading_avg' now calculates the average heading 
    over a defined window size before each odor onset.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing 'odor_state', 'time', and 'heading' columns.
    heading_window_size (float): Time window (in same units as 'time' column) before onset to average heading.

    Returns:
    pd.DataFrame: Updated DataFrame with new computed columns.
    """
    onset_indices = df.index[(df["odor_state"].shift(1, fill_value=0) == 0) & (df["odor_state"] == 1)]
    offset_indices = df.index[(df["odor_state"].shift(1, fill_value=0) == 1) & (df["odor_state"] == 0)]

    # Initialize new columns
    df["odor_duration"] = np.nan
    df["past_interval"] = np.nan
    df["odor_heading_avg"] = np.nan
    df["prior_odor_duration"] = np.nan

    last_odor_duration = 0  # Default for the first odor encounter

    for onset_idx in onset_indices:
        # Find the first offset after the onset
        offset_idx = offset_indices[offset_indices > onset_idx].min()

        if pd.notna(offset_idx):
            # Calculate odor duration
            odor_duration = df.loc[offset_idx, "time"] - df.loc[onset_idx, "time"]
            df.loc[onset_idx, "odor_duration"] = odor_duration

            # Store current duration as the prior duration for the next encounter
            df.loc[onset_idx, "prior_odor_duration"] = last_odor_duration
            last_odor_duration = odor_duration  # Update for the next iteration
        else:
            df.loc[onset_idx, "prior_odor_duration"] = last_odor_duration  # Maintain previous duration if no offset found

        # Calculate past interval
        last_offset_idx = offset_indices[offset_indices < onset_idx].max()
        if pd.notna(last_offset_idx):
            past_interval = df.loc[onset_idx, "time"] - df.loc[last_offset_idx, "time"]
        else:
            past_interval = df.loc[onset_idx, "time"] - df["time"].iloc[0]
        df.loc[onset_idx, "past_interval"] = past_interval

        # Compute average heading over a defined window before odor onset
        onset_time = df.loc[onset_idx, "time"]
        window_start_time = onset_time - heading_window_size
        heading_window = df[(df["time"] >= window_start_time) & (df["time"] < onset_time)]["heading"]

        if not heading_window.empty:
            df.loc[onset_idx, "odor_heading_avg"] = circmean(heading_window)

    # Set first encounter's prior odor duration to 0
    first_onset_idx = onset_indices.min()
    if pd.notna(first_onset_idx):
        df.loc[first_onset_idx, "prior_odor_duration"] = 0

    return df


def compute_event_metrics(df, odor_col="odor_state", time_col="time", window_size=5, w=0.5):
    """
    Computes event-specific metrics for each odor encounter and updates the original DataFrame.
    
    Parameters:
    - df (pd.DataFrame): Time series with 'time' and binary 'odor_state'.
    - odor_col (str): Column indicating odor state (1 = ON, 0 = OFF).
    - time_col (str): Column with timestamps.
    - window_size (int): Number of past events to use for burstiness score.
    - w (float): Weighting factor for Novelty Score.

    Returns:
    - pd.DataFrame with additional columns for event-specific metrics.
    """
    df = df.sort_values(by=time_col).reset_index(drop=True)

    # Identify rows where odor ON events occur
    event_rows = df[odor_col] == 1
    event_times = df.loc[event_rows, time_col].values
    event_indices = df.index[event_rows].values

    if len(event_times) < 2:
        return df  # Not enough events to compute metrics

    # Compute prior inter-event intervals
    prior_intervals = np.diff(event_times, prepend=event_times[0])

    # Compute prior durations
    prior_durations = np.zeros_like(prior_intervals)
    for i, idx in enumerate(event_indices):
        off_idx = df.index[(df.index > idx) & (df[odor_col] == 0)]
        if len(off_idx) > 0:
            prior_durations[i] = df.loc[off_idx[0], time_col] - df.loc[idx, time_col]

    # Compute event-specific metrics
    S_n = prior_intervals  # Surprise Index
    A_n = prior_durations / prior_intervals  # Adaptation Index
    N_n = prior_intervals - w * prior_durations  # Novelty Score

    # Compute Burstiness Score (using last K prior intervals)
    B_n = np.array([
        np.std(prior_intervals[max(0, i - window_size):i]) / np.mean(prior_intervals[max(0, i - window_size):i]) 
        if i > 0 and np.mean(prior_intervals[max(0, i - window_size):i]) > 0 else np.nan
        for i in range(len(prior_intervals))
    ])

    # Initialize new columns in the DataFrame with NaN
    df["surprise_index"] = np.nan
    df["adaptation_index"] = np.nan
    df["novelty_score"] = np.nan
    df["burstiness_score"] = np.nan

    # Assign values only to rows where odor ON events occur
    df.loc[event_indices, "surprise_index"] = S_n
    df.loc[event_indices, "adaptation_index"] = A_n
    df.loc[event_indices, "novelty_score"] = N_n
    df.loc[event_indices, "burstiness_score"] = B_n

    return df


def filter_by_motion(behav_df, motion_threshold=0.0, motion_col='net_motion', return_mask=False):
    """
    Filter behavioral DataFrame based on net_motion threshold.
    
    Parameters:
    -----------
    behav_df : pandas.DataFrame
        Behavioral DataFrame containing motion data
    motion_threshold : float, default=0.0
        Minimum threshold for net_motion. Rows with values below this are filtered out
    motion_col : str, default='net_motion'
        Name of the column containing motion data
    return_mask : bool, default=False
        If True, also return the boolean mask used for filtering
        
    Returns:
    --------
    pandas.DataFrame or tuple
        Filtered DataFrame, or (filtered DataFrame, mask) if return_mask=True
    
    Example:
    --------
    # Basic filtering
    filtered_df = filter_by_motion(behav_df, motion_threshold=0.1)
    
    # Get both filtered df and mask
    filtered_df, mask = filter_by_motion(behav_df, motion_threshold=0.1, return_mask=True)
    print(f"Kept {mask.sum()} out of {len(mask)} points ({mask.mean()*100:.1f}%)")
    """
    # Check if motion column exists
    if motion_col not in behav_df.columns:
        raise ValueError(f"Column '{motion_col}' not found in DataFrame")
    
    # Create motion mask
    motion_mask = behav_df[motion_col] >= motion_threshold
    
    # Apply filter
    filtered_df = behav_df[motion_mask].copy()
    
    # Print summary of filtering
    n_original = len(behav_df)
    n_kept = len(filtered_df)
    percent_kept = (n_kept / n_original) * 100
    
    print(f"Motion filtering summary:")
    print(f"Original points: {n_original}")
    print(f"Points kept: {n_kept} ({percent_kept:.1f}%)")
    print(f"Points removed: {n_original - n_kept} ({100 - percent_kept:.1f}%)")
    
    if return_mask:
        return filtered_df, motion_mask
    return filtered_df

################################# Plotting #################################

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

# binary variable on/offset aligned plot

def plot_aligned_traces(
    df, binary_col, query_col, time_col, color_col=None, align_to="on", window=(-1, 1), bins=None
):
    """
    Plots query variable traces aligned to binary state ON or OFF transitions.
    Colors traces by a third variable (if provided) and optionally bins it.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data.
    binary_col : str
        Column name of the binary state variable (0 or 1).
    query_col : str
        Column name of the query variable to be plotted.
    time_col : str
        Column name of the time variable.
    color_col : str, optional
        Column name of the variable used for coloring traces. If None, defaults to blue.
    align_to : str, optional
        Whether to align traces to binary state "on" (1) or "off" (0). Default is "on".
    window : tuple, optional
        Time window around the transition (before, after) in the same units as `time_col`. Default is (-1, 1).
    bins : int, optional
        Number of bins for the coloring variable. If None, uses continuous values.

    Returns:
    --------
    None
    """
    # Detect transitions (state changes)
    if align_to == "on":
        transition_idxs = df.index[(df[binary_col] == 1) & (df[binary_col].shift(1) == 0)]
    elif align_to == "off":
        transition_idxs = df.index[(df[binary_col] == 0) & (df[binary_col].shift(1) == 1)]
    else:
        raise ValueError("align_to must be 'on' or 'off'.")

    # Store all extracted traces
    all_traces = []
    time_shifts = []
    colors = []

    # Iterate over transitions
    for idx in transition_idxs:
        t0 = df.loc[idx, time_col]  # Get transition time
        start_time, end_time = t0 + window[0], t0 + window[1]  # Define window range
        subset = df[(df[time_col] >= start_time) & (df[time_col] <= end_time)]  # Extract data

        if not subset.empty:
            aligned_time = subset[time_col] - t0  # Align time to transition (t=0)
            all_traces.append(subset[query_col].values)  # Store query variable values
            time_shifts.append(aligned_time.values)  # Store aligned time values
            colors.append(df.loc[idx, color_col] if color_col else None)  # Store color variable

    # Convert to arrays
    all_traces = np.array(all_traces, dtype=object)
    time_shifts = np.array(time_shifts, dtype=object)

    # Default to blue if no color column is provided
    if color_col is None:
        colors = None
    else:
        colors = np.array(colors)

    # Handle binning of the color variable
    if color_col and bins is not None:
        bin_edges = np.linspace(colors.min(), colors.max(), bins + 1)
        color_bins = np.digitize(colors, bin_edges) - 1  # Bin indices
        unique_bins = np.unique(color_bins)
    else:
        color_bins = colors if colors is not None else None
        unique_bins = np.unique(colors) if colors is not None else None

    # Define colormap
    cmap = cm.get_cmap("viridis", len(unique_bins)) if color_col else None
    norm = mcolors.Normalize(vmin=colors.min(), vmax=colors.max()) if color_col else None

    plt.figure(figsize=(8, 5))

    # Plot individual traces
    for i, (time_trace, query_trace) in enumerate(zip(time_shifts, all_traces)):
        if color_col:
            bin_idx = color_bins[i] if bins is not None else colors[i]
            plt.plot(time_trace, query_trace, alpha=0.3, color=cmap(norm(bin_idx)))
        else:
            plt.plot(time_trace, query_trace, alpha=0.3, color="blue")

    # Plot averaged traces within bins
    if color_col and bins is not None:
        for bin_idx in unique_bins:
            bin_mask = color_bins == bin_idx
            if np.sum(bin_mask) > 0:
                avg_time = np.mean([time_shifts[i] for i in range(len(all_traces)) if bin_mask[i]], axis=0)
                avg_trace = np.mean([all_traces[i] for i in range(len(all_traces)) if bin_mask[i]], axis=0)
                plt.plot(avg_time, avg_trace, linewidth=2, color=cmap(norm(bin_idx)), label=f"Bin {bin_idx+1}")

    # Labels and legend
    plt.xlabel("Time (s, aligned to transition)")
    plt.ylabel(query_col)
    plt.axvline(0, color="k", linestyle="--", linewidth=1)  # Mark transition point
    plt.title(f"{query_col} aligned to {binary_col} {align_to.upper()} transitions")
    
    if color_col:
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), label=color_col)
    if bins is not None and color_col:
        plt.legend()
    
    plt.grid(True)
    plt.show()


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

# tuning curves 

# Creating a circular histogram
def plot_scatter(behavior_variable, filtered_columns, neural_activity, neurons_to_plot, num_bins=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    for j in neurons_to_plot:
        #print(len(behavior_variable))
        ax.scatter(behavior_variable,neural_activity[j,:])
        ax.set_ylabel(filtered_columns[j])
    ax.set_xlabel(behavior_variable.name)

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
    save_path = f"{example_path_results}_{behavior_var1.name}_{behavior_var2.name}_heatmap_{trial_num}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #print(f"Figure saved to {save_path}")
    return fig, ax


def plot_tuning_curve_and_scatter(neural_activity, filtered_columns, neurons_to_plot, behavioral_variables, filtered_behavior_variables, num_behavioral_variables, mean_angle, mode_angle, num_bins, example_path_results, trial_num, segment_by_dir, segment_by_block, segment_id = None):
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
    if segment_by_dir:
        plt.savefig(f"{example_path_results}1d_tuning_curve_{trial_num}_segment_{segment_id}.png")
    elif segment_by_block: 
        plt.savefig(f"{example_path_results}1d_tuning_curve_{trial_num}_block_{segment_id}.png")
    else: 
        plt.savefig(f"{example_path_results}1d_tuning_curve_{trial_num}.png")
    plt.close()

    # Second set of plots (scatter plots)
    fig, ax = plt.subplots(len(neurons), num_behavioral_variables, figsize=(num_behavioral_variables * 5, len(neurons) * 5))
    if len(ax.shape) == 1:
        ax = ax[np.newaxis, :]
    generate_plots(neurons, neural_activity, filtered_columns, behavioral_variables, plot_scatter, ax)
    plt.tight_layout()
    if segment_by_dir:
        plt.savefig(f"{example_path_results}scatterplot_{trial_num}_segment_{segment_id}.png")
    elif segment_by_block: 
        plt.savefig(f"{example_path_results}scatterplot_{trial_num}_block_{segment_id}.png")
    else: 
        plt.savefig(f"{example_path_results}scatterplot_{trial_num}.png")
    plt.close()
    # plot forwardV vs. heading as sanity check
    fig, ax = plt.subplots(figsize=(5,5))
    generate_plots([0], np.array([behavioral_variables[0]]), [filtered_behavior_variables[0].name], [filtered_behavior_variables[3]], tuning_curve_1d, np.array([[ax]]), is_filtered=True)
    plt.tight_layout()
    plt.savefig(f"{example_path_results}1d_tuning_curve_ctrl_{trial_num}.png")
    plt.close()

def plot_heading_tuning_circular(behav_df, neural_df, unique_seg, filtered_columns, example_path_results, trial_num, unique_mode_headings=None, segment_column='block'):
    """
    Plot heading tuning for different segments of a time series for multiple neurons.
    
    Parameters:
    - behav_df: DataFrame containing behavioral data including 'heading' and 'segment' (or 'block')
    - neural_df: DataFrame containing neural data
    - unique_seg: List of unique segment numbers to plot
    - filtered_columns: List of neuron column names to plot
    - unique_mode_headings: List of mode headings to plot as radial lines (optional)
    - segment_column: Column name in behav_df to use for segmenting the data ('segment' or 'block')
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
            mask = behav_df[segment_column] == seg
            headings = behav_df.loc[mask, 'heading'].values
            mode_heading = circular_mode(headings)
            unique_mode_headings.append(mode_heading)

    for ax_idx, (ax, neuron) in enumerate(zip(axes, filtered_columns)):
        # Plot each segment
        for i, seg in enumerate(unique_seg):
            mask = behav_df[segment_column] == seg  # Use the specified column ('segment' or 'block') for masking
            ax.scatter(behav_df.loc[mask, 'heading'], neural_df.loc[mask, neuron], 
                       color=colors[i], alpha=0.1, label=f'Segment {seg}')
        
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        
        ax.set_title(f"{neuron}")
        if ax_idx == 0:  # Only set ylabel for the first subplot
            ax.set_ylabel("Neural Activity")
        
        # Plot mode headings
        for i, mode_hd in enumerate(unique_mode_headings):
            ax.plot([mode_hd, mode_hd], [0, ax.get_ylim()[1]], color=colors[i % len(colors)], 
                    linewidth=2, linestyle='--', label=f'Mode {i+1}')
        
        if ax_idx == n_neurons - 1:  # Only add legend to the last subplot
            ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"{example_path_results}scatter_segment_circular_{trial_num}.png", dpi=300, bbox_inches='tight')

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

def plot_binned_heading_tuning(behav_df, neural_df, unique_seg, filtered_columns, example_path_results, trial_num, unique_mode_headings=None, n_bins=16, segment_column='block'):
    """
    Plot binned heading tuning for different segments of a time series for multiple neurons.
    
    Parameters:
    - behav_df: DataFrame containing behavioral data including 'heading' and 'segment' (or 'block')
    - neural_df: DataFrame containing neural data
    - unique_seg: List of unique segment numbers to plot
    - filtered_columns: List of neuron column names to plot
    - unique_mode_headings: List of mode headings to plot as radial lines (optional)
    - n_bins: Number of bins for circular data (default 16)
    - segment_column: Column name in behav_df to use for segmenting the data ('segment' or 'block')
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
            mask = behav_df[segment_column] == seg
            headings = behav_df.loc[mask, 'heading'].values
            mode_heading = circular_mode(headings)
            unique_mode_headings.append(mode_heading)

    for ax_idx, (ax, neuron) in enumerate(zip(axes, filtered_columns)):
        for i, seg in enumerate(unique_seg):
            mask = behav_df[segment_column] == seg  # Use the specified column ('segment' or 'block')
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



############################################################################################################
# Putting things together

base_path = "//research.files.med.harvard.edu/neurobio/wilsonlab/Jingxuan/standby/"
folder_name = "20250212-3_FB4R-odor"
example_path_data = base_path+f"{folder_name}/data/"
example_path_results = base_path+f"{folder_name}/results/"

def make_behav_df(example_path_data, example_path_results, trial_num, segment_by_dir, segment_by_block, block_num, segment_method='manual', is_old=True, bar_jump=False):
    # Load data and preprocess
    if is_old:
        is_mat73, roi_df, dff_raw, kinematics_raw, preprocessed_vars_ds, preprocessed_vars_odor = load_intermediate_mat(example_path_data, trial_num)
        behav_df = make_df_behavior(dff_raw, preprocessed_vars_ds, preprocessed_vars_odor, trial_num, ball_d=9)
        behav_df = reconstruct_path(behav_df, ball_d=9)
    else:
        behav_df, neural_df_new = load_matfile_to_df(example_path_data, folder_name, trial_num, is_odor_trial=True)
        behav_df = rename_dataframe(behav_df)
    # fly path
    clicked_indices = plot_fly_traj_interactive(behav_df.xPos, behav_df.yPos, behav_df, 'odor', example_path_results, trial_num)
    print("Indices of clicked points:", clicked_indices)
    # label segment structure by different criteria
    if segment_by_dir:
        factory = PathSegmentationFactory()
        manual_segmenter = factory.create_segmenter(segment_method, indices=clicked_indices)
        behav_df = manual_segmenter.segment(behav_df)
        behav_df = calc_segment_modes_filtered_v2(behav_df,segment_col='segment')
    elif segment_by_block:
        if block_num == 3:
            # TODO: still need to test block_3_v2
            behav_df, block_boundaries = label_blocks_3_v2(behav_df, 0, [60, 60], detection_threshold=0.05,required_consecutive=5)
        else:
            behav_df, block_boundaries = label_blocks_5_v2(behav_df,50,time_lengths=[300,120,300],detection_threshold=0.05,required_consecutive=5)
        behav_df = calc_segment_modes_filtered_v2(behav_df)
    # calculate straightness
    behav_df = calculate_theta_g_rho(behav_df)
    # bar jump state detection
    if bar_jump:
        behav_df = compute_absolute_circular_diff(behav_df)
        behav_df = detect_local_peaks(behav_df, init_t=10, prominence=0.1, min_time_gap=60)
    # odor related variable processing
    if 'odor_state' in behav_df.columns:
        behav_df = compute_odor_metrics(behav_df)
    return behav_df

def main(example_path_data, example_path_results, trial_num, segment_by_dir, segment_by_block, block_num, segment_method='manual'):
    # Define key variables
    behavior_var1, behavior_var2 = 'translationalV', 'heading'
    roi_kw, roi_kw2 = 'FB4R', 'MB'
    #tuning_whole_session = False

    # Load data and preprocess
    is_mat73, roi_df, dff_raw, kinematics_raw, preprocessed_vars_ds, preprocessed_vars_odor = load_intermediate_mat(example_path_data, trial_num)
    behav_df = make_df_behavior(dff_raw, preprocessed_vars_ds, preprocessed_vars_odor, trial_num, ball_d=9)
    behav_df = reconstruct_path(behav_df, ball_d=9)
    #plot_fly_traj(xPos, yPos, behav_df, 'odor', example_path_results, trial_num)
    clicked_indices = plot_fly_traj_interactive(behav_df.xPos, behav_df.yPos, behav_df, 'odor', example_path_results, trial_num)
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
    if segment_by_dir:
        factory = PathSegmentationFactory()
        manual_segmenter = factory.create_segmenter(segment_method, indices=clicked_indices)
        behav_df = manual_segmenter.segment(behav_df)
    elif segment_by_block:
        if block_num == 3:
            # TODO: still need to test block_3_v2
            behav_df, block_boundaries = label_blocks_3_v2(behav_df, 0, [60, 60], detection_threshold=0.05,required_consecutive=5)
        else:
            behav_df, block_boundaries = label_blocks_5_v2(behav_df,50,time_lengths=[300,120,300],detection_threshold=0.05,required_consecutive=5)
    combined_df = merge_dataframes(behav_df, neural_df)
    
    # Extract and plot data
    do_normalize = True
    roi_mtx = extract_heatmap(combined_df, roi_kw, roi_kw2, do_normalize, example_path_results, trial_num)
    combined_df, nonpara_summ_df = calc_nonpara(combined_df, roi_kw, roi_kw2, roi_mtx, False)
    
    nonpara_plot_bybehav(nonpara_summ_df, nonpara_summ_df, behavior_var1, example_path_results, trial_num)
    nonpara_plot_bybehav(nonpara_summ_df, combined_df, behavior_var2, example_path_results, trial_num)
    
    if roi_kw == 'hDeltaB' and do_normalize and roi_mtx is not None:
        combined_df, paramfit_df = fit_sinusoid(combined_df, roi_mtx)
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
    if segment_by_dir:
        seg_threshold = 50
        unique_seg = combined_df['segment'].unique()
        unique_seg = unique_seg[~np.isnan(unique_seg)]
        plot_heading_tuning_circular(behav_df, neural_df, unique_seg, filtered_columns, example_path_results, trial_num, unique_mode_headings=None, segment_column='segment')
        plot_binned_heading_tuning(behav_df, neural_df, unique_seg, filtered_columns, example_path_results, trial_num, unique_mode_headings=None, segment_column='segment')
        for i in range(len(unique_seg)):
            if np.sum(behav_df['segment'] == unique_seg[i]) > seg_threshold:
                neural_activity_i = neural_activity[behav_df['segment'] == unique_seg[i]]
                behav_df_i = behav_df[behav_df['segment'] == unique_seg[i]]
                mean_angle, median_angle, mode_angle, behavioral_variables, filtered_behavior_variables, num_behavioral_variables = process_behavioral_variables(behav_df_i, example_path_results, trial_num)
                plot_tuning_curve_and_scatter(np.array(neural_activity_i.T), filtered_columns, neurons_to_plot, behavioral_variables, filtered_behavior_variables, num_behavioral_variables, mean_angle, mode_angle, num_bins, example_path_results, trial_num, segment_by_dir, segment_by_block, unique_seg[i])

    elif segment_by_block:
        clicked_indices = plot_fly_traj_interactive(xPos, yPos, behav_df, 'block', example_path_results, trial_num)
        unique_block = combined_df['block'].unique()
        unique_block = unique_block[~np.isnan(unique_block)]
        plot_heading_tuning_circular(behav_df, neural_df, unique_block, filtered_columns, example_path_results, trial_num)
        plot_binned_heading_tuning(behav_df, neural_df, unique_block, filtered_columns, example_path_results, trial_num)
        for i in range(len(unique_block)):
            neural_activity_i = neural_activity[behav_df['block'] == unique_block[i]]
            behav_df_i = behav_df[behav_df['block'] == unique_block[i]]
            mean_angle, median_angle, mode_angle, behavioral_variables, filtered_behavior_variables, num_behavioral_variables = process_behavioral_variables(behav_df_i, example_path_results, trial_num)
            plot_tuning_curve_and_scatter(np.array(neural_activity_i.T), filtered_columns, neurons_to_plot, behavioral_variables, filtered_behavior_variables, num_behavioral_variables, mean_angle, mode_angle, num_bins, example_path_results, trial_num,segment_by_dir, segment_by_block, unique_block[i])

    else:
        # Calculate and plot statistics
        mean_angle, median_angle, mode_angle, behavioral_variables, filtered_behavior_variables, num_behavioral_variables = process_behavioral_variables(behav_df, example_path_results, trial_num)
        fig, ax = tuning_heatmap_2d(behavioral_variables[3], behavioral_variables[0], filtered_columns, np.array(neural_activity.T), neurons_to_plot[0], num_bins, example_path_results, trial_num, ax=None)
        plot_tuning_curve_and_scatter(np.array(neural_activity.T), filtered_columns, neurons_to_plot, behavioral_variables, filtered_behavior_variables, num_behavioral_variables, mean_angle, mode_angle, num_bins, example_path_results, trial_num, segment_by_dir, segment_by_block)

#main(example_path_data, example_path_results, 1,False,False)


def main_new(example_path_data, example_path_results, trial_num, segment_by_dir, segment_by_block, block_num, caiman_only, segment_method='manual',bar_jump=True):
    # Define key variables
    behavior_var1, behavior_var2 = 'translationalV', 'heading'
    roi_kw, roi_kw2 = 'FB4R', 'MB'
    #tuning_whole_session = False

    # Load data and preprocess
    #is_mat73, roi_df, dff_raw, kinematics_raw, preprocessed_vars_ds, preprocessed_vars_odor = load_intermediate_mat(example_path_data, trial_num)
    behav_df, neural_df_new = load_matfile_to_df(example_path_data, folder_name, trial_num, is_odor_trial=True)
    behav_df = rename_dataframe(behav_df)
    if bar_jump:
        behav_df = compute_absolute_circular_diff(behav_df)
        behav_df = detect_local_peaks(behav_df, init_t=10, prominence=0.1, min_time_gap=60)
    behav_df = make_df_behavior_new(behav_df)
    behav_df = reconstruct_path(behav_df, ball_d=9)
    #plot_fly_traj(xPos, yPos, behav_df, 'odor', example_path_results, trial_num)
    clicked_indices = plot_fly_traj_interactive(behav_df.xPos, behav_df.yPos, behav_df, 'odor_state', example_path_results, trial_num)
    print("Indices of clicked points:", clicked_indices)
    # Load and validate ROI data
    if caiman_only:
        neural_df = neural_df_new
    else:
        is_mat73, roi_df, dff_raw = load_intermediate_mat_new(example_path_data,trial_num)
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
    if segment_by_dir:
        factory = PathSegmentationFactory()
        manual_segmenter = factory.create_segmenter(segment_method, indices=clicked_indices)
        behav_df = manual_segmenter.segment(behav_df)
    elif segment_by_block:
        if block_num == 3:
            # TODO: still need to test block_3_v2
            behav_df, block_boundaries = label_blocks_3_v2(behav_df, 0, [60, 60], detection_threshold=0.05,required_consecutive=5)
        else:
            behav_df, block_boundaries = label_blocks_5_v2(behav_df,50,time_lengths=[300,120,300],detection_threshold=0.05,required_consecutive=5)
    
    # TODO: whether we need/where we need filter by motion
    # Get both filtered data and mask
    '''motion_threshold=1
    behav_df, motion_mask = filter_by_motion(behav_df, 
                                                motion_threshold, 
                                                return_mask=True)

    # Apply same mask to neural_df if needed
    neural_df = neural_df[motion_mask]'''
    combined_df = merge_dataframes(behav_df, neural_df)
    
    # Extract and plot data
    do_normalize = True
    roi_mtx = extract_heatmap(combined_df, roi_kw, roi_kw2, do_normalize, example_path_results, trial_num)
    combined_df, nonpara_summ_df = calc_nonpara(combined_df, roi_kw, roi_kw2, roi_mtx, False)
    
    nonpara_plot_bybehav(nonpara_summ_df, nonpara_summ_df, behavior_var1, example_path_results, trial_num)
    nonpara_plot_bybehav(nonpara_summ_df, combined_df, behavior_var2, example_path_results, trial_num)
    
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
    if segment_by_dir:
        seg_threshold = 50
        unique_seg = combined_df['segment'].unique()
        unique_seg = unique_seg[~np.isnan(unique_seg)]
        plot_heading_tuning_circular(behav_df, neural_df, unique_seg, filtered_columns, example_path_results, trial_num, unique_mode_headings=None, segment_column='segment')
        plot_binned_heading_tuning(behav_df, neural_df, unique_seg, filtered_columns, example_path_results, trial_num, unique_mode_headings=None, segment_column='segment')
        for i in range(len(unique_seg)):
            if np.sum(behav_df['segment'] == unique_seg[i]) > seg_threshold:
                neural_activity_i = neural_activity[behav_df['segment'] == unique_seg[i]]
                behav_df_i = behav_df[behav_df['segment'] == unique_seg[i]]
                mean_angle, median_angle, mode_angle, behavioral_variables, filtered_behavior_variables, num_behavioral_variables = process_behavioral_variables(behav_df_i, example_path_results, trial_num)
                plot_tuning_curve_and_scatter(np.array(neural_activity_i.T), filtered_columns, neurons_to_plot, behavioral_variables, filtered_behavior_variables, num_behavioral_variables, mean_angle, mode_angle, num_bins, example_path_results, trial_num, segment_by_dir, segment_by_block, unique_seg[i])

    elif segment_by_block:
        clicked_indices = plot_fly_traj_interactive(behav_df.xPos, behav_df.yPos, behav_df, 'block', example_path_results, trial_num)
        unique_block = combined_df['block'].unique()
        unique_block = unique_block[~np.isnan(unique_block)]
        plot_heading_tuning_circular(behav_df, neural_df, unique_block, filtered_columns, example_path_results, trial_num)
        plot_binned_heading_tuning(behav_df, neural_df, unique_block, filtered_columns, example_path_results, trial_num)
        for i in range(len(unique_block)):
            neural_activity_i = neural_activity[behav_df['block'] == unique_block[i]]
            behav_df_i = behav_df[behav_df['block'] == unique_block[i]]
            mean_angle, median_angle, mode_angle, behavioral_variables, filtered_behavior_variables, num_behavioral_variables = process_behavioral_variables(behav_df_i, example_path_results, trial_num)
            plot_tuning_curve_and_scatter(np.array(neural_activity_i.T), filtered_columns, neurons_to_plot, behavioral_variables, filtered_behavior_variables, num_behavioral_variables, mean_angle, mode_angle, num_bins, example_path_results, trial_num,segment_by_dir, segment_by_block, unique_block[i])

    else:
        # Calculate and plot statistics
        mean_angle, median_angle, mode_angle, behavioral_variables, filtered_behavior_variables, num_behavioral_variables = process_behavioral_variables(behav_df, example_path_results, trial_num)
        fig, ax = tuning_heatmap_2d(behavioral_variables[3], behavioral_variables[0], filtered_columns, np.array(neural_activity.T), neurons_to_plot[0], num_bins, example_path_results, trial_num, ax=None)
        plot_tuning_curve_and_scatter(np.array(neural_activity.T), filtered_columns, neurons_to_plot, behavioral_variables, filtered_behavior_variables, num_behavioral_variables, mean_angle, mode_angle, num_bins, example_path_results, trial_num, segment_by_dir, segment_by_block)



#main_new(example_path_data, example_path_results,1,False, False,False)

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
    combined_df = merge_dataframes(behav_df, neural_df)
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
    combined_df = merge_dataframes(behav_df, neural_df)
    roi_kw = 'hDeltaB'
    roi_kw2 = None
    do_normalize = True
    roi_mtx = extract_heatmap(combined_df, roi_kw, roi_kw2, do_normalize, example_path_results, trial_num)
    if not (roi_mtx is None) and do_normalize:
        paramfit_df = fit_sinusoid(neural_df, roi_mtx)
    combined_df = merge_dataframes(combined_df, paramfit_df)
    combined_df.to_hdf(hdf5_file_path, key=hdf_key, mode='a')
        

def loop_trial(example_path_data, example_path_results, corr_dict=None, hdf5_file_path=None, folder_name=None, calc_corr=False):
    """
    Loop through qualified trials, run the main function, and optionally calculate correlation.
    """
    qualified_trials = find_complete_trials(example_path_data)
    
    if not qualified_trials:
        return

    print(f"Qualified trial numbers are {qualified_trials}")
    
    for trial_num in qualified_trials:
        main(example_path_data, example_path_results, trial_num, False, False)
        
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

