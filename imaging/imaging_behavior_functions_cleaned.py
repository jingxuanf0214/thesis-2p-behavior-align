import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import sem, circmean
from scipy.signal import find_peaks
import scipy.io
import mat73
import os
import glob
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde

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
    roi_fb4r = roi_names[roi_names.str.contains('FB4R') & ~roi_names.str.contains('CRE')]
    fb4r_index = roi_fb4r.index
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
    if fb4r_index.size > 0:
        fb4r_seq = roi_fb4r.str.extract(r'_(\d+)')[0]
        # Only convert to int if there are valid numbers, otherwise return None
        if not fb4r_seq.isna().all():
            fb4r_seq = fb4r_seq.astype(int).to_numpy()
        else:
            fb4r_seq = None
    else:
        fb4r_seq = None  
    return np.array(roi_names), hdeltab_index, epg_index, fr1_index, fb4r_index, hdeltab_seq, epg_seq, fr1_seq,fb4r_seq


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
    roi_fb4r = roi_names[roi_names.str.contains('FB4R') & ~roi_names.str.contains('CRE')]
    fb4r_index = roi_fb4r.index
    
    hdeltab_seq = roi_hdeltab.str.extract(r'_(\d+)')[0].astype(int).to_numpy()
    
    if epg_index.size > 0:
        epg_seq = roi_epg.str.extract(r'_(\d+)')[0].astype(int).to_numpy()
    else:
        epg_seq = None 
    
    if fr1_index.size > 0:
        fr1_seq = roi_fr1.str.extract(r'_(\d+)')[0].astype(int).to_numpy()
    else:
        fr1_seq = None 
    if fb4r_index.size > 0:
        fb4r_seq = roi_fb4r.str.extract(r'_(\d+)')[0].astype(int).to_numpy()
        # Only convert to int if there are valid numbers, otherwise return None
        if not fb4r_seq.isna().all():
            fb4r_seq = fb4r_seq.astype(int).to_numpy()
        else:
            fb4r_seq = None
    else:
        fb4r_seq = None 
    return np.array(roi_names), hdeltab_index, epg_index, fr1_index, fb4r_index, hdeltab_seq, epg_seq, fr1_seq,fb4r_seq


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

def make_df_neural(dff_all_rois, dff_time, roi_names, hdeltab_index, epg_index, fr1_index, fb4r_index, hdeltab_sequence, epg_sequence,fr1_sequence,fb4r_sequence):
    #TODO
    if hdeltab_index.size > 0:
        #print(1)
        sort_rois(dff_all_rois, roi_names, hdeltab_index, hdeltab_sequence)
    else:
        pass
    if epg_index.size > 0:
        sort_rois(dff_all_rois, roi_names, epg_index, epg_sequence)
    else:
        pass
    if fr1_index.size > 0:
        sort_rois(dff_all_rois, roi_names, fr1_index, fr1_sequence)
    else:
        pass
    if fb4r_index.size > 0:
        sort_rois(dff_all_rois, roi_names, fb4r_index, fb4r_sequence)
    else:
        pass
    # Create a new DataFrame for the reordered data
    neural_df = pd.DataFrame()
    neural_df['time'] = np.squeeze(dff_time)
    #print(len(dff_all_rois))
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
    #in_notebook = is_notebook()
    #if in_notebook:
    threshold = np.percentile(df.net_motion,2)
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
    #in_notebook = is_notebook()
    #if in_notebook:
    threshold = np.percentile(df.net_motion,2)
   
    df['net_motion_state'] = (df['net_motion']>threshold).astype(int)
    #df['heading_adj'] = np.unwrap(df['heading'])
    return df 

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

def calculate_theta_g_rho(df, heading_column='heading', window_size=30, speed_threshold=1):
    """
    Calculate the goal direction (theta_g) and consistency of direction (rho_t) for each time point.
    
    Parameters:
    - df: DataFrame containing 'forward', 'side', 'yaw', and the specified heading column.
    - heading_column: Name of the column containing the circular variable to analyze.
    - window_size: The size of the window in seconds to compute the rolling statistics.
    - speed_threshold: Speed threshold below which the fly is considered standing still.
    
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
        
        # Extract window of directions
        direction_window = df[heading_column].iloc[start_idx:end_idx]
        
        # Calculate theta_g (goal direction)
        sum_sin = np.sum(np.sin(direction_window))
        sum_cos = np.sum(np.cos(direction_window))
        theta_g = np.arctan2(sum_sin, sum_cos)

        # Ensure theta_g is between 0 and 2*pi
        theta_g = np.mod(theta_g, 2 * np.pi)
        
        # Calculate rho_t (consistency of direction)
        N_w = len(direction_window)
        rho_t = np.sqrt((sum_cos / N_w)**2 + (sum_sin / N_w)**2)
        
        # Store in DataFrame at index i
        df.at[i, 'theta_g'] = theta_g
        df.at[i, 'rho_t'] = rho_t
    
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


def detect_bar_sweep(df, theta, t2, noise_threshold=0.05, duration_tol=0.5):
    """
    Detects sweep periods in a time series DataFrame with a 'heading' column and a 'time' column.
    
    Parameters:
      df            : pandas DataFrame with 'heading' (radians, 0 to 2π) and 'time' columns.
                      The 'time' column can be datetime-like or numeric.
      theta         : Expected constant heading (in radians) during the non-sweep period.
      t2            : Expected duration of the sweep (should cover roughly a 2π change).
      noise_threshold: Tolerance (in radians) to decide if heading is "close" to theta.
      duration_tol  : Tolerance factor for duration matching (e.g. 0.5 means ±50% of t2).
      
    Returns:
      df            : A copy of the original DataFrame with an added 'bar_sweep' column (1 indicates a sweep).
    """
    df = df.copy()
    
    # Ensure time column is in datetime format if it is not numeric.
    if not np.issubdtype(df['time'].dtype, np.number):
        df['time'] = pd.to_datetime(df['time'])
    
    # Compute the circular difference from theta.
    df['delta'] = np.angle(np.exp(1j*(df['heading'] - theta)))
    # Mark points as "constant" if the circular difference is below the noise threshold.
    df['is_constant'] = df['delta'].abs() < noise_threshold
    
    # Create segments based on changes in the constant flag.
    df['group'] = (df['is_constant'] != df['is_constant'].shift()).cumsum()
    
    # Initialize the bar_sweep indicator column.
    df['bar_sweep'] = 0
    
    # Process each contiguous segment.
    for _, group in df.groupby('group'):
        # Compute the duration of the segment.
        if len(group) > 1:
            if np.issubdtype(df['time'].dtype, np.datetime64):
                start_time = group['time'].iloc[0]
                end_time = group['time'].iloc[-1]
                duration = (end_time - start_time).total_seconds()
            else:
                duration = group['time'].iloc[-1] - group['time'].iloc[0]
        else:
            duration = 0
        
        # If the segment is non-constant, it is a candidate sweep.
        if not group['is_constant'].iloc[0]:
            # Check if the duration is near t2.
            if abs(duration - t2) <= duration_tol * t2:
                # Unwrap the heading to handle circular discontinuities.
                unwrapped = np.unwrap(group['heading'])
                total_change = unwrapped[-1] - unwrapped[0]  # Use array indexing here
                # If the total change is close to 2π, mark as sweep.
                if abs(total_change - 2*np.pi) < (np.pi/2):
                    df.loc[group.index, 'bar_sweep'] = 1
            else:
                # If the duration is off, check the mean rate of change.
                unwrapped = np.unwrap(group['heading'])
                total_change = unwrapped[-1] - unwrapped[0]  # Use array indexing here
                mean_rate = total_change / duration if duration > 0 else 0
                expected_rate = 2*np.pi / t2
                if mean_rate > expected_rate / 2 and total_change > np.pi:
                    df.loc[group.index, 'bar_sweep'] = 1
        else:
            # For constant segments, leave bar_sweep as 0.
            df.loc[group.index, 'bar_sweep'] = 0

    # Clean up temporary columns.
    df.drop(['delta', 'is_constant', 'group'], axis=1, inplace=True)
    
    return df



def extract_roi_mtx(df, roi_kw, roi_kw2):
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

def calculate_pva_hdeltab(activity_matrix, neural_df):
    num_neurons, time_steps = activity_matrix.shape
    directions = np.linspace(0, 2*np.pi, num_neurons, endpoint=False)
    
    # Calculate vector components for each neuron's activity
    x_components = np.cos(directions)[:, np.newaxis] * activity_matrix
    y_components = np.sin(directions)[:, np.newaxis] * activity_matrix
    
    # Sum components across neurons for each time step
    sum_x = np.sum(x_components, axis=0)
    sum_y = np.sum(y_components, axis=0)
    
    # Calculate PVA for each time step
    pva_phase = np.arctan2(sum_y, sum_x)  # Phase in radians
    pva_phase_mod = np.mod(pva_phase, 2*np.pi)
    pva_amplitude = np.sqrt(sum_x**2 + sum_y**2)  # Magnitude of the vector
    # Compute the normalization factor: the sum of all neuronal activity at each time step
    #normalization_factor = np.sum(activity_matrix, axis=0)
    
    # To avoid division by zero, you could add a small epsilon where normalization_factor is zero
    #epsilon = 1e-10
    #normalization_factor = np.where(normalization_factor == 0, epsilon, normalization_factor)
    # Normalize the amplitude
    #pva_amplitude_normalized = pva_amplitude / normalization_factor
    # Compute baseline: mean activity across neurons for each time step
    baseline = np.mean(activity_matrix, axis=0)
    
    # Save computed values into neural_df
    neural_df['phase'] = pva_phase_mod
    #neural_df['amplitude'] = pva_amplitude
    #neural_df['baseline'] = baseline
    
    return neural_df

def calculate_pva_hdeltab_v2(activity_matrix, neural_df):
    num_neurons, time_steps = activity_matrix.shape
    directions = np.linspace(0, 2*np.pi, num_neurons, endpoint=False)
    
    # Calculate vector components for each neuron's activity
    x_components = np.cos(directions)[:, np.newaxis] * activity_matrix
    y_components = np.sin(directions)[:, np.newaxis] * activity_matrix
    
    # Compute the complex vector for each time point
    # This multiplies each neuron's activity by its complex weight (e^(j*theta))
    # and sums over all neurons. We use np.dot to perform the summation over neurons.
    pva = np.dot(np.exp(1j * directions), activity_matrix)
    # Normalize by the total activity at each time point
    norm_factor = np.sum(activity_matrix, axis=0)
    pva_normalized = pva / norm_factor

    # Compute amplitude and phase from the normalized PVA
    pva_amplitude = np.abs(pva_normalized)
    pva_phase = np.angle(pva_normalized)
    
    # Compute baseline: mean activity across neurons for each time step
    baseline = np.mean(activity_matrix, axis=0)
    
    # Save computed values into neural_df
    neural_df['phase'] = pva_phase
    neural_df['amplitude'] = pva_amplitude
    neural_df['baseline'] = baseline
    
    return neural_df

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

# other processing

def compute_mean_neural_columns(neural_df, prefixes=["MBON09", "MBON21"]):
    """
    For each prefix, computes a new column 'mean_{prefix}'.
    - If both a right and a left column exist (in either format, e.g., '{prefix}R' or '{prefix}_R'),
      their mean is computed.
    - If only one exists, that column's values are used.
    - If neither exists, a message is printed.
    
    Parameters:
        neural_df (pd.DataFrame): DataFrame containing neural measurement columns.
        prefixes (list): List of prefixes for which to compute the mean columns.
        
    Returns:
        pd.DataFrame: The modified DataFrame with new mean columns.
    """
    for prefix in prefixes:
        # Define the possible column names for right and left sides
        possible_right = [f"{prefix}R", f"{prefix}_R"]
        possible_left = [f"{prefix}L", f"{prefix}_L"]
        
        # Check if any of the possible column names exist in the DataFrame
        right_col = next((col for col in possible_right if col in neural_df.columns), None)
        left_col = next((col for col in possible_left if col in neural_df.columns), None)
        mean_col = f"mean_{prefix}"
        
        if right_col and left_col:
            neural_df[mean_col] = (neural_df[right_col] + neural_df[left_col]) / 2
        elif right_col:
            neural_df[mean_col] = neural_df[right_col]
        elif left_col:
            neural_df[mean_col] = neural_df[left_col]
        else:
            print(f"Neither {possible_right} nor {possible_left} found in DataFrame.")
    
    return neural_df


def add_dan_mean_columns(df):
    # Identify columns containing "PAM0708"
    pam0708_cols = [col for col in df.columns if "PAM0708" in col]
    if pam0708_cols:
        df["mean_PAM0708"] = df[pam0708_cols].mean(axis=1)
    else:
        df["mean_PAM0708"] = np.nan  # or handle empty case as desired

    # Identify columns containing "MB441" or "PAM12"
    pam12_cols = [col for col in df.columns if ("MB441" in col) or ("PAM12" in col)]
    if pam12_cols:
        df["mean_PAM12"] = df[pam12_cols].mean(axis=1)
    else:
        df["mean_PAM12"] = np.nan

    return df

def mean_FB4R(df):
    # Identify columns that contain 'FB4R'
    fb4r_cols = df.columns[df.columns.str.contains('FB4R')]
    # Calculate the row-wise mean for these columns and assign to a new column
    df['mean_FB4R'] = df[fb4r_cols].mean(axis=1)
    return df

def mean_FB5V(df):
    # Identify columns that contain 'FB5V'
    fb5v_cols = df.columns[df.columns.str.contains('FB5V')]
    # Calculate the row-wise mean for these columns and assign to a new column
    df['mean_FB5V'] = df[fb5v_cols].mean(axis=1)
    return df

def add_min_max_normalized_columns(neural_df):
    """
    For each column in neural_df that does not contain 'hDeltaB' or 'EPG', 
    apply min-max normalization and save the result in a new column 
    with the suffix '_minmax'.
    
    Parameters:
        neural_df (pd.DataFrame): Input DataFrame containing neural data.
        
    Returns:
        pd.DataFrame: A new DataFrame with additional min-max normalized columns.
    """
    # Create a copy of the DataFrame to avoid modifying the original data
    neural_df_norm = neural_df.copy()
    
    # Identify columns that do not include 'hDeltaB' or 'EPG'
    columns_to_transform = [col for col in neural_df_norm.columns if 'hDeltaB' not in col and 'EPG' not in col and 'time' not in col]
    
    # Loop through each column to transform
    for col in columns_to_transform:
        min_val = neural_df_norm[col].min()
        max_val = neural_df_norm[col].max()
        # Avoid division by zero; if max equals min, set the normalized value to 0.0
        if max_val != min_val:
            neural_df_norm[col + '_minmax'] = (neural_df_norm[col] - min_val) / (max_val - min_val)
        else:
            neural_df_norm[col + '_minmax'] = 0.0
    return neural_df_norm

def filter_based_on_histogram(behavior_variable, min_freq_threshold):
    """
    Filters behavioral variable data to remove points distant from the main mode.
    
    Args:
        behavior_variable (pd.Series): Behavioral variable data
        min_freq_threshold (float): Minimum frequency threshold for bins
    
    Returns:
        pd.Series: Filtered behavioral variable data
    """
    if behavior_variable.name == 'heading':
        return behavior_variable
    
    bin_width = 1
    range_ = np.max(behavior_variable) - np.min(behavior_variable)
    bins = int(np.round(range_ / bin_width))
    
    counts, bin_edges = np.histogram(behavior_variable, bins=bins)
    low_freq_bins_mask = counts < min_freq_threshold
    two_consecutive_low_bins = np.where(np.convolve(low_freq_bins_mask, [1,1], mode='valid') == 2)[0]
    
    if two_consecutive_low_bins.size == 0:
        return behavior_variable
    
    mode_bin_index = np.argmax(counts)
    lower_cutoff_bins = two_consecutive_low_bins[two_consecutive_low_bins < mode_bin_index]
    upper_cutoff_bins = two_consecutive_low_bins[two_consecutive_low_bins > mode_bin_index]
    
    lower_cutoff = bin_edges[0] if lower_cutoff_bins.size == 0 else bin_edges[lower_cutoff_bins.max() + 2]
    upper_cutoff = bin_edges[-1] if upper_cutoff_bins.size == 0 else bin_edges[upper_cutoff_bins.min()]
    
    filtered_variable = behavior_variable[(behavior_variable >= lower_cutoff) & 
                                        (behavior_variable <= upper_cutoff)]
    return filtered_variable

def calc_circu_stats(circu_var, num_bins):
    """
    Calculate circular statistics for a variable.
    
    Args:
        circu_var (pd.Series): Circular variable data
        num_bins (int): Number of bins for histogram
    
    Returns:
        tuple: (mean_angle, median_angle, mode_angle)
    """
    mean_angle = circmean(circu_var, high=2*np.pi)
    sorted_angles = np.sort(circu_var)
    median_angle = np.pi if len(sorted_angles) % 2 == 0 else sorted_angles[len(sorted_angles) // 2]
    
    n, bins = np.histogram(circu_var, bins=num_bins, range=(0, 2*np.pi))
    max_bin_index = np.argmax(n)
    mode_angle = (bins[max_bin_index] + bins[max_bin_index + 1]) / 2
    
    return mean_angle, median_angle, mode_angle

def process_behavioral_variables(behav_df, num_bins):
    """
    Process behavioral variables and calculate statistics.
    
    Args:
        behav_df (pd.DataFrame): Behavioral data
        num_bins (int): Number of bins for circular statistics
    
    Returns:
        tuple: (mean_angle, median_angle, mode_angle, behavioral_variables, 
               filtered_behavior_variables, num_behavioral_variables)
    """
    mean_angle, median_angle, mode_angle = calc_circu_stats(behav_df.heading, num_bins)
    
    behavioral_variables = [behav_df.fwV, behav_df.sideV, behav_df.yawV, behav_df.heading]
    filtered_behavior_variables = [filter_based_on_histogram(var, 0.5) 
                                 for var in behavioral_variables]
    num_behavioral_variables = len(filtered_behavior_variables)
    
    return (mean_angle, median_angle, mode_angle, behavioral_variables, 
            filtered_behavior_variables, num_behavioral_variables)

def compute_odor_metrics(df, heading_window_size=5):
    """
    Compute odor-related metrics for each odor encounter.
    
    Args:
        df (pd.DataFrame): DataFrame with odor and behavioral data
        heading_window_size (float): Time window for heading average
    
    Returns:
        pd.DataFrame: Updated DataFrame with odor metrics
    """
    onset_indices = df.index[(df["odor_state"].shift(1, fill_value=0) == 0) & 
                            (df["odor_state"] == 1)]
    offset_indices = df.index[(df["odor_state"].shift(1, fill_value=0) == 1) & 
                             (df["odor_state"] == 0)]
    
    df["odor_duration"] = np.nan
    df["past_interval"] = np.nan
    df["odor_heading_avg"] = np.nan
    df["prior_odor_duration"] = np.nan
    
    last_odor_duration = 0
    
    for onset_idx in onset_indices:
        offset_idx = offset_indices[offset_indices > onset_idx].min()
        
        if pd.notna(offset_idx):
            odor_duration = df.loc[offset_idx, "time"] - df.loc[onset_idx, "time"]
            df.loc[onset_idx, "odor_duration"] = odor_duration
            df.loc[onset_idx, "prior_odor_duration"] = last_odor_duration
            last_odor_duration = odor_duration
        else:
            df.loc[onset_idx, "prior_odor_duration"] = last_odor_duration
        
        last_offset_idx = offset_indices[offset_indices < onset_idx].max()
        past_interval = (df.loc[onset_idx, "time"] - 
                        (df.loc[last_offset_idx, "time"] if pd.notna(last_offset_idx) 
                         else df["time"].iloc[0]))
        df.loc[onset_idx, "past_interval"] = past_interval
        
        onset_time = df.loc[onset_idx, "time"]
        window_start_time = onset_time - heading_window_size
        heading_window = df[(df["time"] >= window_start_time) & 
                          (df["time"] < onset_time)]["heading"]
        
        if not heading_window.empty:
            df.loc[onset_idx, "odor_heading_avg"] = circmean(heading_window)
    
    first_onset_idx = onset_indices.min()
    if pd.notna(first_onset_idx):
        df.loc[first_onset_idx, "prior_odor_duration"] = 0
    
    return df

def compute_event_metrics(df, odor_col="odor_state", time_col="time", 
                         window_size=5, w=0.5):
    """
    Compute event-specific metrics for odor encounters.
    
    Args:
        df (pd.DataFrame): Time series data
        odor_col (str): Column name for odor state
        time_col (str): Column name for time
        window_size (int): Window size for burstiness score
        w (float): Weighting factor for Novelty Score
    
    Returns:
        pd.DataFrame: DataFrame with added event metrics
    """
    df = df.sort_values(by=time_col).reset_index(drop=True)
    event_rows = df[odor_col] == 1
    event_times = df.loc[event_rows, time_col].values
    event_indices = df.index[event_rows].values
    
    if len(event_times) < 2:
        return df
    
    prior_intervals = np.diff(event_times, prepend=event_times[0])
    prior_durations = np.zeros_like(prior_intervals)
    
    for i, idx in enumerate(event_indices):
        off_idx = df.index[(df.index > idx) & (df[odor_col] == 0)]
        if len(off_idx) > 0:
            prior_durations[i] = df.loc[off_idx[0], time_col] - df.loc[idx, time_col]
    
    S_n = prior_intervals
    A_n = prior_durations / prior_intervals
    N_n = prior_intervals - w * prior_durations
    
    B_n = np.array([
        np.std(prior_intervals[max(0, i - window_size):i]) / 
        np.mean(prior_intervals[max(0, i - window_size):i]) 
        if i > 0 and np.mean(prior_intervals[max(0, i - window_size):i]) > 0 
        else np.nan
        for i in range(len(prior_intervals))
    ])
    
    df["surprise_index"] = np.nan
    df["adaptation_index"] = np.nan
    df["novelty_score"] = np.nan
    df["burstiness_score"] = np.nan
    
    df.loc[event_indices, "surprise_index"] = S_n
    df.loc[event_indices, "adaptation_index"] = A_n
    df.loc[event_indices, "novelty_score"] = N_n
    df.loc[event_indices, "burstiness_score"] = B_n
    
    return df

def compute_event_metrics_3(df, state_col="odor_state", time_col="time", heading_col="heading",  
                            heading_window_size=5, burstiness_window=5, novelty_weight=0.5):
    """
    Computes event-based metrics for both ON and OFF transitions of a binary state variable.
    
    Parameters:
    - df (pd.DataFrame): DataFrame with binary state variable (e.g., odor presence).
    - state_col (str): Name of the binary column indicating event presence (0 or 1).
    - time_col (str): Name of the time column.
    - heading_col (str): Name of the heading column.
    - heading_window_size (float): Time window before an event to average heading.
    - burstiness_window (int): Number of prior intervals to use for burstiness score.
    - novelty_weight (float): Weight factor for novelty score calculation.

    Returns:
    - pd.DataFrame with additional computed columns at both ON and OFF transitions.
    """

    onset_indices = df.index[(df[state_col].shift(1, fill_value=0) == 0) & (df[state_col] == 1)]
    offset_indices = df.index[(df[state_col].shift(1, fill_value=0) == 1) & (df[state_col] == 0)]

    # Initialize new columns
    df["event_duration"] = np.nan
    df["past_interval"] = np.nan
    df["prior_event_duration"] = np.nan
    df["heading_avg"] = np.nan

    # Additional sensory processing metrics
    df["surprise_index"] = np.nan
    df["adaptation_index"] = np.nan
    df["novelty_score"] = np.nan
    df["burstiness_score"] = np.nan

    # Track the last duration separately for ON and OFF events.
    last_on_duration = 0  
    last_off_duration = 0  
    prior_intervals = []  # For burstiness calculation

    all_indices = np.sort(np.concatenate([onset_indices, offset_indices]))  # Ordered events

    for i, event_idx in enumerate(all_indices):
        is_onset = event_idx in onset_indices

        # For an ON event, duration is time until next OFF; for an OFF event, it's until next ON.
        if is_onset:
            matching_offset_idx = offset_indices[offset_indices > event_idx].min()
        else:
            matching_onset_idx = onset_indices[onset_indices > event_idx].min()
        
        # Compute event duration with corrected order for OFF events
        if is_onset and pd.notna(matching_offset_idx):
            event_duration = df.loc[matching_offset_idx, time_col] - df.loc[event_idx, time_col]
        elif (not is_onset) and pd.notna(matching_onset_idx):
            event_duration = df.loc[matching_onset_idx, time_col] - df.loc[event_idx, time_col]
        else:
            event_duration = np.nan

        df.loc[event_idx, "event_duration"] = event_duration

        # Use the last duration from the same event type
        if is_onset:
            df.loc[event_idx, "prior_event_duration"] = last_on_duration
        else:
            df.loc[event_idx, "prior_event_duration"] = last_off_duration

        # Compute past interval (ISI)
        if i > 0:
            prev_event_idx = all_indices[all_indices < event_idx].max()
            past_interval = df.loc[event_idx, time_col] - df.loc[prev_event_idx, time_col]
        else:
            past_interval = df.loc[event_idx, time_col] - df[time_col].iloc[0]
        df.loc[event_idx, "past_interval"] = past_interval
        prior_intervals.append(past_interval)

        # Compute Surprise Index (using past interval)
        df.loc[event_idx, "surprise_index"] = past_interval

        # Use the saved prior event duration for adaptation and novelty
        prior_duration = df.loc[event_idx, "prior_event_duration"]

        # Compute Adaptation Index (ratio of prior event duration to past interval)
        df.loc[event_idx, "adaptation_index"] = np.nan if past_interval == 0 else prior_duration / past_interval

        # Compute Novelty Score (penalizes long prior event durations)
        df.loc[event_idx, "novelty_score"] = past_interval - (novelty_weight * prior_duration)

        # Compute Burstiness Score (variability of past K intervals)
        if len(prior_intervals) > 1:
            windowed_intervals = prior_intervals[max(0, i - burstiness_window):i]
            if len(windowed_intervals) > 1 and np.mean(windowed_intervals) > 0:
                burstiness = np.std(windowed_intervals) / np.mean(windowed_intervals)
            else:
                burstiness = np.nan
        else:
            burstiness = np.nan
        df.loc[event_idx, "burstiness_score"] = burstiness

        # Compute average heading before the event (if available)
        event_time = df.loc[event_idx, time_col]
        window_start_time = event_time - heading_window_size
        heading_window = df[(df[time_col] >= window_start_time) & (df[time_col] < event_time)][heading_col]
        if not heading_window.empty:
            df.loc[event_idx, "heading_avg"] = circmean(heading_window)

        # Update the corresponding last duration based on event type.
        if is_onset:
            last_on_duration = event_duration
        else:
            last_off_duration = event_duration

    # Ensure the first encounter's prior duration is set to 0.
    first_event_idx = all_indices.min()
    if pd.notna(first_event_idx):
        df.loc[first_event_idx, "prior_event_duration"] = 0

    return df

def filter_by_motion(behav_df, motion_threshold=0.0, motion_col='net_motion', 
                    return_mask=False):
    """
    Filter behavioral DataFrame based on motion threshold.
    
    Args:
        behav_df (pd.DataFrame): Behavioral DataFrame
        motion_threshold (float): Minimum threshold for motion
        motion_col (str): Column name for motion data
        return_mask (bool): Whether to return the filter mask
    
    Returns:
        pd.DataFrame or tuple: Filtered DataFrame, or (filtered DataFrame, mask)
    """
    if motion_col not in behav_df.columns:
        raise ValueError(f"Column '{motion_col}' not found in DataFrame")
    
    motion_mask = behav_df[motion_col] >= motion_threshold
    filtered_df = behav_df[motion_mask].copy()
    
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