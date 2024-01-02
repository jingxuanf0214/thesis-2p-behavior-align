import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import scipy.io
import mat73
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from scipy.stats import iqr
from PIL import Image
import cv2
from IPython.display import display, clear_output
import time
import seaborn as sns

def load_intermediate_mat(path_to_folder,trial_num):
    dff_raw = scipy.io.loadmat(f'{path_to_folder}dff raw trial{trial_num}.mat')
    kinematics_raw = scipy.io.loadmat(f'{path_to_folder}kinematics raw trial{trial_num}.mat')
    preprocessed_vars_ds = scipy.io.loadmat(f'{path_to_folder}preprocessed_vars_ds trial{trial_num}.mat')
    preprocessed_vars_odor = scipy.io.loadmat(f'{path_to_folder}preprocessed_vars_odor trial{trial_num}.mat')
    return dff_raw, kinematics_raw, preprocessed_vars_ds, preprocessed_vars_odor

def make_df_behavior(dff_raw, preprocessed_vars_ds, preprocessed_vars_odor,trial_num,ball_d = 9):
    circum = ball_d * np.pi
    df = pd.DataFrame()
    # add dff_raw
    dff_time = dff_raw['roiTime']
    df['time'] = np.squeeze(dff_time)
    df['fwV'] = np.squeeze(preprocessed_vars_ds['ftT_fwSpeedDown2']) # unit in mm/s
    df['sideV'] = np.squeeze(preprocessed_vars_ds['ftT_sideSpeedDown2']) 
    df['yawV'] = circum*np.squeeze(preprocessed_vars_ds['ftT_yawSpeedDown2'])/(2*np.pi) # unit in mm/s
    df['heading'] = np.squeeze(preprocessed_vars_ds['ftT_intHDDown2'])
    df['abssideV'] = np.abs(np.squeeze(preprocessed_vars_ds['ftT_sideSpeedDown2']))
    df['absyawV'] = circum*np.abs(np.squeeze(preprocessed_vars_ds['ftT_yawSpeedDown2']))/(2*np.pi)
    df['net_motion'] = df['abssideV']+df['absyawV']+np.abs(df['fwV'])
    df['net_motion_state'] = (df['net_motion']>2.5).astype(int)
    df['heading_adj'] = np.unwrap(df['heading'])
    odor_all = preprocessed_vars_odor['odorDown']
    if len(odor_all) == 1:
        df['odor'] = np.squeeze(odor_all)
    else:
        df['odor'] = odor_all[:,trial_num]
    return df 


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

def extract_heatmap(df, roi_kw, do_normalize):
    roi_mtx = df.columns[df.columns.str.contains(roi_kw)]
    if do_normalize:
        scaler = StandardScaler()
        roi_mtx = scaler.fit_transform(roi_mtx)
    return roi_mtx

def fit_sinusoid(roi_mtx):
    def test_func(x, dist, amp, phi):
        return dist + amp * np.cos(x + phi)
    timestamp, num_roi = np.shape(roi_mtx)
    x_p = np.linspace(0, 2*np.pi, num=num_roi)
    trial_len = timestamp
    phase_sinfit = np.zeros(trial_len)
    base_sinfit = np.zeros(trial_len)
    amp_sinfit = np.zeros(trial_len)
    for i in range(trial_len):
        params, params_covariance = optimize.curve_fit(test_func, x_p, roi_mtx[i,0:27],maxfev = 5000)
        phase_sinfit[i] = x_p[np.argmax(test_func(x_p, params[0], params[1], params[2]))]
        amp_sinfit[i] = np.abs(params[1])
        base_sinfit[i] = params[0]
    return phase_sinfit, base_sinfit, amp_sinfit, params_covariance

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

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

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


