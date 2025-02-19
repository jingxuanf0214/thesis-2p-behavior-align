import numpy as np
import pandas as pd
import scipy.io
import os
import mat73
from scipy.ndimage import gaussian_filter1d
import glob
import re



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
        df['odor_state'] = df['odor'] > 5
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

