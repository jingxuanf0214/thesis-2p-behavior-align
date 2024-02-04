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

def load_intermediate_mat(path_to_folder,trial_num):
    dff_raw = scipy.io.loadmat(path_to_folder + f'dff raw trial{trial_num}.mat')
    kinematics_raw = scipy.io.loadmat(path_to_folder + f'kinematics raw trial{trial_num}.mat')
    preprocessed_vars_ds = scipy.io.loadmat(path_to_folder + f'preprocessed_vars_ds trial{trial_num}.mat')
    odor_path = os.path.join(path_to_folder, f'preprocessed_vars_odor trial{trial_num}.mat')
    if os.path.exists(odor_path):
        preprocessed_vars_odor = scipy.io.loadmat(odor_path)
    else:
        preprocessed_vars_odor = None  # or any other placeholder value you find appropriate
    preprocessed_vars_odor = scipy.io.loadmat(path_to_folder + f'preprocessed_vars_odor trial{trial_num}.mat')
    roi_data = scipy.io.loadmat(path_to_folder + f'roiData_struct trial{trial_num}.mat')
    roi_df = pd.DataFrame.from_dict(np.squeeze(roi_data['s']))
    return roi_df, dff_raw, kinematics_raw, preprocessed_vars_ds, preprocessed_vars_odor

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
    
    return threshold_value  # Return the updated list containing the threshold value

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
    threshold = plot_interactive_histogram(df.net_motion)
    df['net_motion_state'] = (df['net_motion']>threshold[0]).astype(int)
    df['heading_adj'] = np.unwrap(df['heading'])
    if preprocessed_vars_odor != None:
        odor_all = preprocessed_vars_odor['odorDown']
        if len(odor_all) == 1:
            df['odor'] = np.squeeze(odor_all)
        else:
            df['odor'] = odor_all[:,trial_num]
    return df 

roi_df, dff_raw, kinematics_raw, preprocessed_vars_ds, preprocessed_vars_odor = load_intermediate_mat('data/',1)
#print(roi_df.head(5))
behav_df = make_df_behavior(dff_raw, preprocessed_vars_ds, preprocessed_vars_odor,1,ball_d = 9)
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
    return xPos, yPos

xPos, yPos = reconstruct_path(behav_df, ball_d = 9)
x_range = max(xPos) - min(xPos)
y_range = max(yPos) - min(yPos)
aspect_ratio = y_range / x_range

# Set figure dimensions based on data range while keeping unit scale the same
fig_width = 10  # Width of figure in inches
fig_height = fig_width * aspect_ratio  # Height is scaled according to the aspect ratio of the data

plt.figure(figsize=(fig_width, fig_height))

plt.scatter(xPos, yPos, c=behav_df.odor[1:], s=3)
plt.scatter(0, 0, color='red')  # Mark the origin

# Enforce equal aspect ratio so that one unit in x is the same as one unit in y
plt.gca().set_aspect('equal', adjustable='box')

plt.xlabel('x position')
plt.ylabel('y position')
plt.title('Fly Trajectory')

# Save the plot
plt.savefig('results/fly_trajectory.png')
plt.close()  # Close the plot explicitly after saving to free resources

def get_roi_seq(roi_df):
    roi_names = roi_df['roiName'].apply(lambda x: x[0])
    roi_hdeltab = roi_names[roi_names.str.contains('hDeltaB')]
    hdeltab_index = roi_hdeltab.index
    roi_epg = roi_names[roi_names.str.contains('EPG')]
    epg_index = roi_epg.index
    hdeltab_seq = roi_hdeltab.str.extract(r'_(\d+)')[0].astype(int)
    hdeltab_seq = hdeltab_seq.to_numpy()
    if epg_index != None:
        epg_seq = roi_epg.str.extract(r'_(\d+)')[0].astype(int)
        epg_seq = epg_seq.to_numpy()
    else:
        epg_seq =  None 
    return np.array(roi_names), hdeltab_index, epg_index, hdeltab_seq, epg_seq

roi_names, hdeltab_index, epg_index, hdeltab_sequence, epg_sequence = get_roi_seq(roi_df)
print(hdeltab_sequence)

def make_df_neural(dff_raw,roi_names, hdeltab_index, epg_index, hdeltab_sequence, epg_sequence):
    #TODO
    dff_rois = dff_raw['flDataC']
    dff_time = dff_raw['roiTime']
    # Sort dff_rois according to roi_sequence
    # Ensure roi_sequence is a list of integers that corresponds to the order you want
    sorting_indices = np.argsort(hdeltab_sequence)
    dff_all = dff_rois[0]
    segment_to_sort = dff_rois[0][hdeltab_index]
    sorted_dff_rois = segment_to_sort[sorting_indices]
    dff_all[hdeltab_index] = sorted_dff_rois
    roi_names_sort = roi_names[hdeltab_index]
    roi_names_sort = roi_names_sort[sorting_indices]
    roi_names[hdeltab_index] = roi_names_sort
    # Create a new DataFrame for the reordered data
    neural_df = pd.DataFrame()
    neural_df['time'] = np.squeeze(dff_time)
    # Add each sorted ROI data to the DataFrame with the specified naming convention
    for i, roi_data in enumerate(dff_all):
        column_name =  roi_names[i] # Generate column name starting from hDeltaB1
        neural_df[column_name] = np.squeeze(roi_data)
    
    return neural_df
neural_df = make_df_neural(dff_raw,roi_names, hdeltab_index, epg_index, hdeltab_sequence, epg_sequence)
print(neural_df)


def combine_df(behav_df, neural_df):
    return pd.merge(behav_df,neural_df,on="time")

combined_df = combine_df(behav_df, neural_df)
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

def calc_nonpara(combined_df):
    combined_df = combined_df[(combined_df["fwV"]>0.2) | (combined_df["fwV"]<-0.2)]
    sigma = 5
    smooth_fwV = apply_gaussian_smoothing(combined_df.fwV, sigma)
    smooth_sideV = apply_gaussian_smoothing(combined_df.sideV, sigma)
    translational_speed = np.sqrt(smooth_fwV**2+smooth_sideV**2)
    forward_speed = np.abs(smooth_fwV)
    neural_df_rois = combined_df[[col for col in combined_df.columns if 'hDeltaB' in col]]
    
    row_means = neural_df_rois.apply(np.mean, axis=1)
    
    # Calculate IQR for each row
    row_iqrs = neural_df_rois.apply(lambda x: iqr(x, interpolation='midpoint'), axis=1)
    
    # Combine mean and IQR into a new DataFrame
    stats_df = pd.DataFrame({'Mean': row_means, 'IQR': row_iqrs, 'translationalV': translational_speed, 'fwV':forward_speed})
    
    return stats_df

nonpara_summ_df = calc_nonpara(combined_df)
print(nonpara_summ_df)

plt.figure(figsize=(6, 6))
plt.scatter(nonpara_summ_df['Mean'],nonpara_summ_df['IQR'],c= nonpara_summ_df['translationalV'])
plt.colorbar()
plt.xlabel('mean')
plt.ylabel('iqr')
plt.title('mean vs. amplitude')

# Save the plot
plt.savefig('results/mean_amp.png')
plt.close()  # Close the plot explicitly after saving to free resources


def extract_heatmap(df, roi_kw, do_normalize):
    roi_mtx = df[[col for col in df.columns if roi_kw in col]]
    if do_normalize:
        scaler = StandardScaler()
        roi_mtx = scaler.fit_transform(roi_mtx)
    return roi_mtx

roi_mtx = extract_heatmap(combined_df, "hDeltaB", True)
plt.figure(figsize=(10, 6))
sns.heatmap(np.transpose(roi_mtx))
plt.savefig('results/heatmap_norm.png')
plt.close()  # Close the plot explicitly after saving to free resources


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
        params, params_covariance = optimize.curve_fit(test_func, x_p, roi_mtx[i,0:27],maxfev = 5000)
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

paramfit_df = fit_sinusoid(neural_df, roi_mtx)
#print(paramfit_df.head(5))

def plot_with_error_shading(df):
    # Set up the figure and subplots
    fig, axs = plt.subplots(3, 1, figsize=(15, 10))
    
    # Plot phase
    axs[0].plot(df['time'], df['phase'], label='Phase', color = 'orange')
    axs[0].fill_between(df['time'], 
                        df['phase'] - df['phase_error'], 
                        df['phase'] + df['phase_error'], color = 'orange', alpha=0.3)
    axs[0].set_title('Phase')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Phase')
    
    # Plot amplitude
    axs[1].plot(df['time'], df['amplitude'], label='Amplitude',color = 'red')
    axs[1].fill_between(df['time'], 
                        df['amplitude'] - df['amplitude_error'], 
                        df['amplitude'] + df['amplitude_error'], color = 'red', alpha=0.3)
    axs[1].set_title('Amplitude')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Amplitude')

    # Plot baseline
    axs[2].plot(df['time'], df['baseline'], label='Baseline', color = 'green')
    axs[2].fill_between(df['time'], 
                        df['baseline'] - df['baseline_error'], 
                        df['baseline'] + df['baseline_error'], color = 'green', alpha=0.3)
    axs[2].set_title('Baseline')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Baseline')

    plt.tight_layout()
    plt.show()

plot_with_error_shading(paramfit_df)

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


def time_series_plot(df):
    pass

# encoding, decoding models 
# calcium imaging GLM 


