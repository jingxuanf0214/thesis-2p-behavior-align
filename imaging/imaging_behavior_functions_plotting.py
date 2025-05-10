import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem, circmean
import cv2
from matplotlib.widgets import Button
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d
import matplotlib as mpl

# Basic Interactive Plotting Functions
def plot_interactive_histogram(series):
    """
    Creates an interactive histogram where user can click to set threshold.
    
    Args:
        series (pd.Series): Data to plot histogram
    
    Returns:
        float: Selected threshold value
    """
    fig, ax = plt.subplots()
    counts, bins, patches = ax.hist(series, bins=30, color='skyblue', edgecolor='gray')
    threshold_line = ax.axvline(color='r', linestyle='--')
    threshold_value = [None]
    
    def onclick(event):
        threshold_line.set_xdata([event.xdata, event.xdata])
        threshold_value[0] = event.xdata
        fig.canvas.draw()
    
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Interactive Histogram')
    plt.show()
    
    return threshold_value[0]

# Trajectory Plotting Functions
def plot_fly_traj(xPos, yPos, behav_df, label, colormap):
    """
    Plots fly trajectory with optional coloring based on behavioral variable.
    
    Args:
        xPos (array): X positions
        yPos (array): Y positions
        behav_df (pd.DataFrame): Behavioral data
        label (str): Column name for coloring
        example_path_results (str): Path to save results
        trial_num (int): Trial number
    """
    x_range = max(xPos) - min(xPos)
    y_range = max(yPos) - min(yPos)
    aspect_ratio = y_range / x_range
    fig_width = 10
    fig_height = fig_width * aspect_ratio
    
    plt.figure(figsize=(fig_width, fig_height))
    if label in behav_df.columns:
        plt.scatter(behav_df.xPos[behav_df[label]==0], behav_df.yPos[behav_df[label]==0], color = 'gray', s=1,alpha=0.6)
        plt.scatter(behav_df.xPos[behav_df[label]==1], behav_df.yPos[behav_df[label]==1], color = 'red', s=1,alpha=0.5)
        plt.colorbar()
    else:
        plt.scatter(xPos, yPos, s=1,color='black')
        label = "nothing"
    plt.scatter(0, 0, color='black')
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.title('Fly Trajectory')
    
    #plt.savefig(f"{example_path_results}fly_trajectory_colored_by_{label}_trial_{trial_num}.png")
    #plt.close()

def plot_fly_traj_interactive(xPos, yPos, behav_df, label, example_path_results, trial_num):
    """
    Interactive version of fly trajectory plotting with point selection.
    """
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
    
    clicked_indices = []
    for point in selected_points:
        distances = np.sqrt((xPos - point[0])**2 + (yPos - point[1])**2)
        closest_index = np.argmin(distances)
        clicked_indices.append(closest_index)
    
    return clicked_indices

# Time Series Plotting Functions
def plot_time_series(neural_df, behav_df, example_path_results, trial_num):
    """
    Plots neural and behavioral time series data.
    """
    neural_columns = len(neural_df.columns.drop('time'))
    behav_columns = len(['fwV', 'yawV', 'sideV', 'heading'])
    total_plots = neural_columns + behav_columns
    
    fig, axs = plt.subplots(total_plots, 1, figsize=(12, 3 * total_plots), sharex=True)
    
    for i, column in enumerate(neural_df.columns.drop('time')):
        axs[i].plot(neural_df['time'], neural_df[column], label=column)
        axs[i].set_ylabel(column)
        axs[i].legend(loc='upper right')
    
    behav_columns = ['fwV', 'yawV', 'sideV', 'heading']
    for j, column in enumerate(behav_columns, start=neural_columns):
        if column in behav_df.columns:
            axs[j].plot(behav_df['time'], behav_df[column], label=column, linestyle='--')
            axs[j].set_ylabel(column)
            axs[j].legend(loc='upper right')
    
    if 'odor' in behav_df.columns:
        odor_mask = behav_df['odor'] > 5
        for ax in axs:
            ax.fill_between(behav_df['time'], ax.get_ylim()[0], ax.get_ylim()[1], 
                          where=odor_mask, color='red', alpha=0.3, 
                          transform=ax.get_xaxis_transform())
    
    plt.xlabel('Time')
    fig.suptitle('Neural and Behavioral Data Over Time', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{example_path_results}time_series_plotting{trial_num}.png")
    plt.close()

def plot_aligned_traces(df, binary_col, query_col, time_col, color_col=None, 
                       align_to="on", window=(-1, 1), bins=None):
    """
    Plots query variable traces aligned to binary state transitions.
    """
    if align_to == "on":
        transition_idxs = df.index[(df[binary_col] == 1) & (df[binary_col].shift(1) == 0)]
    elif align_to == "off":
        transition_idxs = df.index[(df[binary_col] == 0) & (df[binary_col].shift(1) == 1)]
    else:
        raise ValueError("align_to must be 'on' or 'off'.")
    
    all_traces = []
    time_shifts = []
    colors = []
    
    for idx in transition_idxs:
        t0 = df.loc[idx, time_col]
        start_time, end_time = t0 + window[0], t0 + window[1]
        subset = df[(df[time_col] >= start_time) & (df[time_col] <= end_time)]
        
        if not subset.empty:
            aligned_time = subset[time_col] - t0
            all_traces.append(subset[query_col].values)
            time_shifts.append(aligned_time.values)
            colors.append(df.loc[idx, color_col] if color_col else None)
    
    all_traces = np.array(all_traces, dtype=object)
    time_shifts = np.array(time_shifts, dtype=object)
    colors = np.array(colors) if color_col else None
    
    if color_col and bins is not None:
        bin_edges = np.linspace(colors.min(), colors.max(), bins + 1)
        color_bins = np.digitize(colors, bin_edges) - 1
        unique_bins = np.unique(color_bins)
    else:
        color_bins = colors
        unique_bins = np.unique(colors) if colors is not None else None
    
    cmap = cm.get_cmap("viridis", len(unique_bins)) if color_col else None
    norm = mcolors.Normalize(vmin=colors.min(), vmax=colors.max()) if color_col else None
    
    plt.figure(figsize=(8, 5))
    for i, (time_trace, query_trace) in enumerate(zip(time_shifts, all_traces)):
        if color_col:
            bin_idx = color_bins[i] if bins is not None else colors[i]
            plt.plot(time_trace, query_trace, alpha=0.3, color=cmap(norm(bin_idx)))
        else:
            plt.plot(time_trace, query_trace, alpha=0.3, color="blue")
    
    if color_col and bins is not None:
        for bin_idx in unique_bins:
            bin_mask = color_bins == bin_idx
            if np.sum(bin_mask) > 0:
                avg_time = np.mean([time_shifts[i] for i in range(len(all_traces)) if bin_mask[i]], axis=0)
                avg_trace = np.mean([all_traces[i] for i in range(len(all_traces)) if bin_mask[i]], axis=0)
                plt.plot(avg_time, avg_trace, linewidth=2, color=cmap(norm(bin_idx)), 
                        label=f"Bin {bin_idx+1}")
    
    plt.xlabel("Time (s, aligned to transition)")
    plt.ylabel(query_col)
    plt.axvline(0, color="k", linestyle="--", linewidth=1)
    plt.title(f"{query_col} aligned to {binary_col} {align_to.upper()} transitions")
    
    if color_col:
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), label=color_col)
    if bins is not None and color_col:
        plt.legend()
    
    plt.grid(True)
    plt.show()

# Statistical and Tuning Curve Plotting Functions
def nonpara_plot_bybehav(nonpara_summ_df, combined_df, behavior_var, example_path_results, trial_num):
    """
    Plots mean vs amplitude colored by behavioral variable.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(nonpara_summ_df['Mean'], nonpara_summ_df['IQR'], c=combined_df[behavior_var])
    plt.colorbar()
    plt.xlabel('mean')
    plt.ylabel('iqr')
    plt.title('mean vs. amplitude')
    plt.savefig(f"{example_path_results}mean_amp_{behavior_var}{trial_num}.png")
    plt.close()

def plot_with_error_shading(df, example_path_results, trial_num):
    """
    Plots phase/amplitude/baseline with error shading.
    """
    fig, axs = plt.subplots(3, 1, figsize=(15, 10))
    
    # Phase plot
    axs[0].plot(df['time'], df['phase'], label='Phase', color='orange')
    phase_error_threshold = np.percentile(df['phase_error'], 90)
    high_error_times = df['time'][df['phase_error'] > phase_error_threshold]
    for time in high_error_times:
        axs[0].axvline(x=time, color='gray', alpha=0.1)
    axs[0].set_title('Phase')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Phase')
    
    # Amplitude plot
    axs[1].plot(df['time'], df['amplitude'], label='Amplitude', color='red')
    amp_error_threshold = np.percentile(df['amplitude_error'], 90)
    high_error_times = df['time'][df['amplitude_error'] > amp_error_threshold]
    for time in high_error_times:
        axs[1].axvline(x=time, color='gray', alpha=0.1)
    axs[1].set_title('Amplitude')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Amplitude')
    
    # Baseline plot
    axs[2].plot(df['time'], df['baseline'], label='Baseline', color='green')
    base_error_threshold = np.percentile(df['baseline_error'], 90)
    high_error_times = df['time'][df['baseline_error'] > base_error_threshold]
    for time in high_error_times:
        axs[2].axvline(x=time, color='gray', alpha=0.1)
    axs[2].set_title('Baseline')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Baseline')
    
    plt.tight_layout()
    plt.savefig(f"{example_path_results}parametric_fit{trial_num}.png")
    plt.close()

# Tuning Curve Functions
def plot_scatter(merged_df, behavior_col, neural_col, ax=None, return_fig=False):
    """
    Plots scatter of neural vs behavioral data from a merged DataFrame.

    Parameters
    ----------
    merged_df : pandas.DataFrame
        DataFrame containing both behavioral and neural data
    behavior_col : str
        Name of the behavioral variable column
    neural_col : str
        Name of the neural activity column
    ax : matplotlib.axes.Axes, optional
        Existing axes for plotting. If None, creates new figure
    return_fig : bool, optional
        If True, returns the figure and axes objects

    Returns
    -------
    tuple or None
        If return_fig is True, returns (fig, ax), otherwise None
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    ax.scatter(merged_df[behavior_col], merged_df[neural_col], alpha=0.5)
    ax.set_xlabel(behavior_col)
    ax.set_ylabel(neural_col)
    ax.set_title(f'{neural_col} vs {behavior_col}')
    
    if return_fig:
        return fig, ax

def tuning_curve_1d(merged_df, behavior_col, neural_cols, num_bins=20, ax=None, return_fig=False):
    """
    Plots 1D tuning curves from a merged DataFrame, showing how neural activity varies with a behavioral variable.

    Parameters
    ----------
    merged_df : pandas.DataFrame
        DataFrame containing both behavioral and neural data
    behavior_col : str
        Name of the behavioral variable column
    neural_cols : str or list of str
        Name(s) of neural activity column(s) to plot
    num_bins : int, optional
        Number of bins for the behavioral variable (default=20)
    ax : matplotlib.axes.Axes, optional
        Existing axes for plotting. If None, creates new figure
    return_fig : bool, optional
        If True, returns the figure and axes objects

    Returns
    -------
    tuple or None
        If return_fig is True, returns (fig, ax), otherwise None
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    # Convert neural_cols to list if it's a single string
    if isinstance(neural_cols, str):
        neural_cols = [neural_cols]
    
    # Create bins
    bins = np.linspace(merged_df[behavior_col].min(), merged_df[behavior_col].max(), num_bins + 1)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    
    # Plot tuning curve for each neural column
    for neural_col in neural_cols:
        binned_activity = np.empty(num_bins)
        binned_sem = np.empty(num_bins)
        
        for i in range(num_bins):
            mask = (merged_df[behavior_col] >= bins[i]) & (merged_df[behavior_col] < bins[i+1])
            bin_data = merged_df.loc[mask, neural_col]
            binned_activity[i] = bin_data.mean()
            binned_sem[i] = sem(bin_data) if len(bin_data) > 1 else 0
        
        # Plot mean line and error shading
        ax.plot(bin_centers, binned_activity, label=neural_col,color='black')
        ax.fill_between(bin_centers, 
                       binned_activity - binned_sem,
                       binned_activity + binned_sem,
                       alpha=0.3,color='black')
    
    ax.set_xlabel(behavior_col)
    ax.set_ylabel('Neural Activity')
    if len(neural_cols) > 1:
        ax.legend()
    ax.set_title(f'Neural Tuning to {behavior_col}')
    
    if return_fig:
        return fig, ax

def plot_neural_activity_heatmap(df, neural_col, fwV_col, heading_col, modal_heading=None, num_bins_fwV=20, num_bins_heading=20):
    """
    Plots a 2D heatmap of neural activity with an optional red dotted line indicating the modal heading.

    Parameters:
    - df: DataFrame containing the data.
    - neural_col: Column name for neural activity.
    - fwV_col: Column name for forward velocity (fwV).
    - heading_col: Column name for heading.
    - modal_heading: Value of the modal heading to be highlighted (optional).
    - num_bins_fwV: Number of bins for fwV axis (default=20).
    - num_bins_heading: Number of bins for heading axis (default=20).

    Returns:
    - None. Displays the heatmap.
    """
    # Binning the data
    fwV_bins = np.linspace(df[fwV_col].min(), df[fwV_col].max(), num_bins_fwV + 1)
    heading_bins = np.linspace(df[heading_col].min(), df[heading_col].max(), num_bins_heading + 1)

    # Assign bin labels
    df['fwV_bin'] = pd.cut(df[fwV_col], bins=fwV_bins, labels=False, include_lowest=True)
    df['heading_bin'] = pd.cut(df[heading_col], bins=heading_bins, labels=False, include_lowest=True)

    # Group by bins and calculate mean neural activity
    heatmap_data = df.groupby(['fwV_bin', 'heading_bin'])[neural_col].mean().unstack()

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap_data, origin='lower', aspect='auto',
               extent=[df[heading_col].min(), df[heading_col].max(),
                       df[fwV_col].min(), df[fwV_col].max()])
    plt.colorbar(label='Average Neural Activity')
    plt.xlabel('Heading')
    plt.ylabel('Forward Velocity (fwV)')

    # Plot the red dotted line for modal heading if provided
    if modal_heading is not None:
        plt.axvline(x=modal_heading, color='red', linestyle='--', linewidth=2, label=f'Modal Heading: {modal_heading}')
        plt.legend()

    # Display the heatmap
    plt.show()


def tuning_heatmap_2d(behavior_var1, behavior_var2, filtered_columns, neural_activity, 
                     neurons_to_plot, num_bins, example_path_results, trial_num, ax=None):
    """
    Plots 2D tuning heatmaps.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure
    
    bins1 = np.linspace(np.min(behavior_var1), np.max(behavior_var1), num_bins+1)
    bins2 = np.linspace(np.min(behavior_var2), np.max(behavior_var2), num_bins+1)
    
    binned_activity = np.zeros((num_bins, num_bins))
    
    for i in range(num_bins):
        for j in range(num_bins):
            indices = np.where(
                (behavior_var1 >= bins1[i]) & (behavior_var1 < bins1[i+1]) &
                (behavior_var2 >= bins2[j]) & (behavior_var2 < bins2[j+1])
            )[0]
            binned_activity[j, i] = np.mean(neural_activity[neurons_to_plot, indices])
    
    im = ax.imshow(binned_activity, origin='lower', aspect='auto', cmap='viridis')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f'Mean activity of {filtered_columns[neurons_to_plot]}')
    
    ax.set_xlabel(behavior_var1.name)
    ax.set_ylabel(behavior_var2.name)
    
    ax.set_xticks(np.linspace(0, num_bins-1, 5))
    ax.set_yticks(np.linspace(0, num_bins-1, 5))
    ax.set_xticklabels([f'{x:.2f}' for x in np.linspace(np.min(behavior_var1), np.max(behavior_var1), 5)])
    ax.set_yticklabels([f'{y:.2f}' for y in np.linspace(np.min(behavior_var2), np.max(behavior_var2), 5)])
    
    plt.savefig(f"{example_path_results}_{behavior_var1.name}_{behavior_var2.name}_heatmap_{trial_num}.png", 
                dpi=300, bbox_inches='tight')
    return fig, ax

def plot_heading_tuning_circular(behav_df, neural_df, unique_seg, filtered_columns, 
                               example_path_results, trial_num, unique_mode_headings=None, 
                               segment_column='block'):
    """
    Plots circular heading tuning.
    """
    n_neurons = len(filtered_columns)
    fig, axes = plt.subplots(1, n_neurons, figsize=(5*n_neurons, 5), subplot_kw={'projection': 'polar'})
    
    if n_neurons == 1:
        axes = [axes]
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_seg)))
    
    if unique_mode_headings is None:
        unique_mode_headings = []
        for seg in unique_seg:
            mask = behav_df[segment_column] == seg
            headings = behav_df.loc[mask, 'heading'].values
            mode_heading = circular_mode(headings)
            unique_mode_headings.append(mode_heading)
    
    for ax_idx, (ax, neuron) in enumerate(zip(axes, filtered_columns)):
        for i, seg in enumerate(unique_seg):
            mask = behav_df[segment_column] == seg
            ax.scatter(behav_df.loc[mask, 'heading'], neural_df.loc[mask, neuron], 
                      color=colors[i], alpha=0.1, label=f'Segment {seg}')
        
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_title(f"{neuron}")
        
        if ax_idx == 0:
            ax.set_ylabel("Neural Activity")
        
        for i, mode_hd in enumerate(unique_mode_headings):
            ax.plot([mode_hd, mode_hd], [0, ax.get_ylim()[1]], 
                   color=colors[i % len(colors)], linewidth=2, linestyle='--', 
                   label=f'Mode {i+1}')
        
        if ax_idx == n_neurons - 1:
            ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
    
    plt.tight_layout()
    plt.savefig(f"{example_path_results}scatter_segment_circular_{trial_num}.png", 
                dpi=300, bbox_inches='tight')

def plot_binned_heading_tuning(behav_df, neural_df, unique_seg, filtered_columns, 
                             example_path_results, trial_num, unique_mode_headings=None, 
                             n_bins=16, segment_column='block'):
    """
    Plots binned heading tuning.
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
            mask = behav_df[segment_column] == seg
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
    plt.savefig(f"{example_path_results}binned_heading_tuning_{trial_num}.png", 
                dpi=300, bbox_inches='tight')
    plt.show()

# Helper Functions
def binned_stats(headings, values, bins):
    """
    Calculate binned averages and standard errors.
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

def circular_mode(circular_data, method='kde', bins=360, num_bins=30):
    """
    Calculate the mode angle of circular data.
    """
    if method == 'kde':
        density = gaussian_kde(circular_data)
        x = np.linspace(0, 2*np.pi, bins)
        y = density(x)
        peaks, _ = find_peaks(y)
        if len(peaks) == 0:
            return None
        mode_idx = peaks[np.argmax(y[peaks])]
        return x[mode_idx]
    elif method == 'histogram':
        n, bin_edges = np.histogram(circular_data, bins=num_bins, range=(0, 2*np.pi))
        max_bin_index = np.argmax(n)
        mode_angle = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2
        return mode_angle
    else:
        raise ValueError("Invalid method. Choose 'kde' or 'histogram'.")

def plot_heading_histogram_circular(headings, n_bins=36, ax=None, density=True, offset=0, 
                                 direction='clockwise', zero_location='N'):
    """
    Plot a circular histogram of heading data.
    
    Args:
        headings (array-like): Array of heading values in radians
        n_bins (int): Number of bins for the histogram (default=36, i.e., 10° bins)
        ax (matplotlib.axes.Axes): Optional axes for plotting. If None, creates new figure
        density (bool): If True, plots probability density. If False, plots counts
        offset (float): Rotation offset in radians (default=0)
        direction (str): 'clockwise' or 'counterclockwise'
        zero_location (str): Location of 0 degrees ('N', 'S', 'E', or 'W')
    
    Returns:
        matplotlib.axes.Axes: The axes containing the plot
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='polar')
    
    # Convert headings to range [0, 2π] if needed
    headings = np.mod(headings, 2*np.pi)
    
    # Create histogram
    counts, bin_edges = np.histogram(headings, bins=n_bins, range=(0, 2*np.pi))
    if density:
        counts = counts / len(headings) / (2*np.pi/n_bins)
    
    # Get bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot histogram bars
    width = 2*np.pi/n_bins
    bars = ax.bar(bin_centers, counts, width=width, alpha=0.5)
    
    # Set direction and zero location
    ax.set_theta_direction(-1 if direction == 'clockwise' else 1)
    ax.set_theta_zero_location(zero_location)
    
    # Add grid and labels
    ax.grid(True)
    #if density:
        #ax.set_ylabel('Probability Density')
    #else:
        #ax.set_ylabel('Count')
    
    # Calculate and plot circular mode
    mode_angle = circular_mode(headings, method='histogram')
    if mode_angle is not None:
        r_max = np.max(counts)
        # Plot mode as a red dotted line from center to max radius
        ax.plot([mode_angle, mode_angle], [0, r_max], 'r--', linewidth=2,
                label=f'Mode: {np.degrees(mode_angle):.1f}°')
    
    ax.legend()
    
    return ax

def plot_heading_histogram_by_condition(df, heading_col='heading', condition_col=None, 
                                      n_bins=36, example_path_results=None, trial_num=None):
    """
    Plot circular histograms of heading data, optionally split by a condition.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        heading_col (str): Name of the column containing heading data
        condition_col (str): Name of the column to split data by (optional)
        n_bins (int): Number of bins for the histogram
        example_path_results (str): Path to save results (optional)
        trial_num (int): Trial number for saving (optional)
    """
    if condition_col is None:
        # Single plot for all data
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='polar')
        plot_heading_histogram_circular(df[heading_col].values, n_bins=n_bins, ax=ax)
        plt.title('Heading Distribution')
    else:
        # Separate plots for each condition
        conditions = df[condition_col].unique()
        n_conditions = len(conditions)
        fig = plt.figure(figsize=(6*n_conditions, 6))
        
        for i, condition in enumerate(conditions, 1):
            mask = df[condition_col] == condition
            ax = fig.add_subplot(1, n_conditions, i, projection='polar')
            plot_heading_histogram_circular(df.loc[mask, heading_col].values, n_bins=n_bins, ax=ax)
            plt.title(f'Heading Distribution - {condition}')
    
    if example_path_results and trial_num:
        condition_str = f"_{condition_col}" if condition_col else ""
        plt.savefig(f"{example_path_results}heading_histogram{condition_str}_trial_{trial_num}.png", 
                    dpi=300, bbox_inches='tight')
    plt.show() 


# plot on off set together
def plot_neural_activity_zero_2(df, pre_window_size, post_window_size, 
                              plot_columns=["MBON09L", "MBON09R", "MBON21L", "MBON21R"],
                              sigma=None, plot_kw='past_interval',
                              bounds=[10, 25, 100, 200], percentile_range=[0, 100],
                              baseline_duration=1, onset_shift=0.6):
    """
    Plots neural activity traces aligned to odor events by combining both onset‐ and offset–aligned events.
    
    For each event pair, the function computes an onset‐aligned trace and an offset–aligned trace,
    performs baseline subtraction using a fixed pre-event duration, and then groups all traces based
    on the value extracted from the column specified in `plot_kw`. The grouping bins are defined by
    `bounds`, and the color of each group is determined by normalizing the average grouping value.
    
    Parameters:
      df (DataFrame): DataFrame containing the data. Must include columns "time", "odor_state", and those in plot_columns.
      pre_window_size (int): Number of timepoints to include before the event.
      post_window_size (int): Number of timepoints to include after the event.
      plot_columns (list): List of neural activity column names to plot.
      sigma (float): If provided, applies Gaussian smoothing (except for non-numeric columns like 'heading').
      plot_kw (str): Column name used to extract a grouping value (e.g. past_interval) from the DataFrame at the event time.
      bounds (list): List of numeric bounds used to group the traces.
      percentile_range (list): Two-element list used to compute the lower and upper limits for normalization.
      baseline_duration (float): Duration (in seconds) before time zero used for baseline subtraction.
      onset_shift (float): A constant shift added to the time axis so that the event appears at the desired time.
    """

    # Ensure required columns exist
    for column in plot_columns:
        if column not in df.columns:
            raise ValueError(f"Column {column} not found in the dataframe.")
    if "time" not in df.columns:
        raise ValueError("Column 'time' not found in the dataframe.")
    
    # Optionally smooth the data
    if sigma:
        for column in plot_columns:
            if column != "heading":
                smoothed_column_name = f"{column}_smoothed"
                df[smoothed_column_name] = gaussian_filter1d(df[column], sigma)

    # Identify odor onsets and offsets (assumes binary odor_state)
    odor_onsets = df.index[(df["odor_state"].shift(1, fill_value=0) == 0) & (df["odor_state"] == 1)]
    odor_offsets = df.index[(df["odor_state"].shift(1, fill_value=0) == 1) & (df["odor_state"] == 0)]
    
    # Set up one subplot per neural activity column.
    n_plots = len(plot_columns)
    fig, axs = plt.subplots(n_plots // 2, 2, figsize=(12, 2 * n_plots))
    axs = axs.ravel()
    
    for i, orig_column in enumerate(plot_columns):
        # Use smoothed version if sigma is provided
        column = f"{orig_column}_smoothed" if sigma else orig_column
        
        combined_aligned = []  # List to store tuples of (time_trace, trace)
        combined_values = []   # List to store grouping values from plot_kw
        
        # Use the minimum number of events from onsets and offsets
        n_events = min(len(odor_onsets), len(odor_offsets))
        for j in range(n_events - 1):
            # --- Process onset event ---
            onset_idx = odor_onsets[j]
            start_onset = max(0, onset_idx - pre_window_size)
            end_onset = min(len(df), onset_idx + post_window_size)
            time_onset = df["time"].iloc[start_onset:end_onset] - df["time"].iloc[onset_idx]
            time_onset = time_onset.reset_index(drop=True) + onset_shift
            trace_onset = df[column].iloc[start_onset:end_onset].reset_index(drop=True)
            if time_onset.iloc[0] < 0 and time_onset.iloc[-1] > 0:
                baseline_onset = trace_onset[(time_onset < 0) & (time_onset >= -baseline_duration)].mean()
                trace_onset = trace_onset - baseline_onset
                combined_aligned.append((time_onset, trace_onset))
                val_onset = df.loc[onset_idx, plot_kw]
                combined_values.append(val_onset)
            
            # --- Process offset event ---
            offset_idx = odor_offsets[j]
            start_offset = max(0, offset_idx - pre_window_size)
            end_offset = min(len(df), offset_idx + post_window_size)
            time_offset = df["time"].iloc[start_offset:end_offset] - df["time"].iloc[offset_idx]
            time_offset = time_offset.reset_index(drop=True) + onset_shift
            trace_offset = df[column].iloc[start_offset:end_offset].reset_index(drop=True)
            if time_offset.iloc[0] < 0 and time_offset.iloc[-1] > 0:
                baseline_offset = trace_offset[(time_offset < 0) & (time_offset >= -baseline_duration)].mean()
                trace_offset = trace_offset - baseline_offset
                combined_aligned.append((time_offset, trace_offset))
                val_offset = df.loc[offset_idx, plot_kw]
                combined_values.append(val_offset)
        
        # Setup color normalization using the combined grouping values
        if combined_values:
            lower_bound, upper_bound = np.percentile(combined_values, percentile_range)
            norm = mpl.colors.Normalize(vmin=max(min(combined_values), lower_bound),
                                        vmax=min(max(combined_values), upper_bound))
            cmap = plt.cm.rainbow
        
        # Group all traces by defined bounds
        grouped_traces = {str(idx): [] for idx in range(len(bounds) + 1)}
        for idx, value in enumerate(combined_values):
            if value < bounds[0]:
                grouped_traces["0"].append(combined_aligned[idx][1])
            else:
                for b_idx in range(1, len(bounds)):
                    if bounds[b_idx - 1] <= value < bounds[b_idx]:
                        grouped_traces[str(b_idx)].append(combined_aligned[idx][1])
                        break
                else:
                    grouped_traces[str(len(bounds))].append(combined_aligned[idx][1])
        
        # Use a common time axis from the last computed trace (assumes similar length)
        if combined_aligned:
            common_time = combined_aligned[-1][0]
        else:
            common_time = pd.Series([-pre_window_size + i for i in range(pre_window_size + post_window_size)])
        
        # Plot each group as a single trace with error shading.
        for group, traces in grouped_traces.items():
            if traces:
                trace_df = pd.DataFrame(traces)
                mean_trace = trace_df.mean()
                stderr = trace_df.std() / np.sqrt(len(traces))
                # Gather group values for color mapping
                group_vals = [combined_values[i] for i in range(len(combined_values))
                              if ((combined_values[i] < bounds[0] and group == "0") or
                                  any(bounds[b_idx - 1] <= combined_values[i] < bounds[b_idx] and group == str(b_idx)
                                      for b_idx in range(1, len(bounds))) or
                                  (combined_values[i] >= bounds[-1] and group == str(len(bounds))))]
                if group_vals:
                    group_avg = np.mean(group_vals)
                    color = cmap(norm(group_avg))
                    axs[i].plot(common_time, mean_trace, color=color, linestyle='solid',
                                label=f"Group {group} (avg: {group_avg:.2f})")
                    axs[i].fill_between(common_time, mean_trace - stderr, mean_trace + stderr,
                                        color=color, alpha=0.3, linewidth=0)
        
        axs[i].axvline(x=0, color='red', linestyle='--', label='Odor Event')
        if combined_values:
            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=axs[i])
            cbar.set_label(f"{plot_kw.capitalize()} Value")
        axs[i].set_title(f"Neural Activity: {column}")
        axs[i].set_xlabel("Time (s, combined onset & offset alignment)")
        axs[i].set_ylabel("Activity")
    
    plt.tight_layout()
    plt.show()
