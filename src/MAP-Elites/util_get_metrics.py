import ast
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
import os

def read_csv_data(head, left_front, right_front, middle, rear, left_hind, right_hind):
    """Reads CSV data from a file and returns a DataFrame."""
    data = []
    for i in range(len(head)):
        frame_data = {
            'frame_id': i + 1,
            'head': ast.literal_eval(str(head[i])),
            'middle': ast.literal_eval(str(left_front[i])),
            'rear': ast.literal_eval(str(right_front[i])),
            'left_front': ast.literal_eval(str(middle[i])),
            'right_front': ast.literal_eval(str(rear[i])),
            'left_hind': ast.literal_eval(str(left_hind[i])),
            'right_hind': ast.literal_eval(str(right_hind[i]))
        }
        data.append(frame_data)
    return pd.DataFrame(data)

def compute_angle_360(head, middle, rear):
    """Computes the angle in 360-degree format based on head, middle, and rear positions."""
    try:
        A = np.array([head[0] - middle[0], head[1] - middle[1]])
        B = np.array([rear[0] - middle[0], rear[1] - middle[1]])

        if np.linalg.norm(A) == 0 or np.linalg.norm(B) == 0:
            return np.nan

        dot_product = np.dot(A, B)
        mag_A = np.linalg.norm(A)
        mag_B = np.linalg.norm(B)

        if mag_A == 0 or mag_B == 0:
            return np.nan

        cos_angle = dot_product / (mag_A * mag_B)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        angle_deg = np.degrees(angle)
        cross_product = A[0] * B[1] - A[1] * B[0]

        return angle_deg if cross_product < 0 else 360 - angle_deg
    except (TypeError, ValueError):
        return np.nan

def calculate_frequency_and_cycles(smoothed_data, sampling_rate):
    """Calculates frequency and cycles from the smoothed data."""
    min_prominence = np.ptp(smoothed_data) * 0.25  # Minimum prominence based on data range
    min_distance = 50  # Minimum distance between peaks

    peaks, _ = find_peaks(smoothed_data, prominence=min_prominence, distance=min_distance)
    troughs, _ = find_peaks(-smoothed_data, prominence=min_prominence, distance=min_distance)

    if len(peaks) >= 2:
        avg_period_samples = np.mean(np.diff(peaks))
        avg_period_seconds = avg_period_samples / sampling_rate
        frequency = 1 / avg_period_seconds if avg_period_seconds > 0 else 0
        cycles = frequency * len(smoothed_data) / sampling_rate
        return peaks, troughs, avg_period_samples, frequency, cycles
    elif len(troughs) >= 2:
        avg_period_samples = np.mean(np.diff(troughs))
        avg_period_seconds = avg_period_samples / sampling_rate
        frequency = 1 / avg_period_seconds if avg_period_seconds > 0 else 0
        cycles = frequency * len(smoothed_data) / sampling_rate
        return peaks, troughs, avg_period_samples, frequency, cycles
    else:
        return None, None, None, None, None

def get_nearest_valid_value(df, idx, limb):
    """Finds the nearest valid value for the given index."""
    # Search backwards for the nearest valid value
    for offset in range(1, len(df)):
        if idx - offset >= 0 and df[limb].iloc[idx - offset] != ('', ''):
            return df[limb].iloc[idx - offset]
        if idx + offset < len(df) and df[limb].iloc[idx + offset] != ('', ''):
            return df[limb].iloc[idx + offset]
    return (0, 0)  # Return a default if no valid value found

# Improved function for clarity, and now sums movement instead of averages
def calculate_limb_movements(df, peaks, troughs):
    """Calculates total movement for each limb between consecutive turning points (peaks and troughs)."""
    limb_movements = {
        'left_front': 0,
        'right_front': 0,
        'left_hind': 0,
        'right_hind': 0
    }

    # Combine and sort peaks and troughs
    turning_points = sorted(peaks + troughs)

    for i in range(len(turning_points) - 1):
        start_idx = turning_points[i]
        end_idx = turning_points[i + 1]

        for limb in limb_movements.keys():
            start_pos = df[limb].iloc[start_idx]
            end_pos = df[limb].iloc[end_idx]

            # Handle missing values
            if start_pos == ('', ''):
                start_pos = get_nearest_valid_value(df, start_idx, limb)
            if end_pos == ('', ''):
                end_pos = get_nearest_valid_value(df, end_idx, limb)

            # Calculate the Euclidean distance between start and end positions
            movement = np.linalg.norm(np.array(start_pos) - np.array(end_pos))
            limb_movements[limb] += movement  # Add to total movement for each limb

    return limb_movements



def analyze_file(head, left_front, right_front, middle, rear, left_hind, right_hind,
                 sampling_rate=20, plot=False):
    """Processes a single CSV file and returns results as a dictionary."""
    df = read_csv_data(head, left_front, right_front, middle, rear, left_hind, right_hind)

    df['angle_360'] = df.apply(lambda row: compute_angle_360(row['head'], row['middle'], row['rear']), axis=1)

    x_data = df['frame_id'].values
    y_data = df['angle_360'].values
    mask = ~np.isnan(y_data)
    x_data, y_data = x_data[mask], y_data[mask]

    # changed average bias, to hip_std
    hip_std = np.std(y_data)

    window_size = 40
    smoothed_y_data = np.convolve(y_data, np.ones(window_size) / window_size, mode='valid')
    x_data_smoothed = x_data[window_size - 1:]

    amplitude = np.percentile(smoothed_y_data, 75) - np.percentile(smoothed_y_data, 25)

    peaks, troughs, avg_period_samples, frequency, cycles = calculate_frequency_and_cycles(smoothed_y_data, sampling_rate)

    # Check if peaks and troughs are valid before proceeding
    if peaks is None or troughs is None:
        return None  # Return None instead of a dictionary

    limb_movements = calculate_limb_movements(df, peaks, troughs)

    results = {
        'hip': hip_std,
        'frequency': frequency,
        'cycles': cycles,
        'limb_movements': limb_movements
    }

    # Plotting if required
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(df['frame_id'], df['angle_360'].where(df['angle_360'] <= 180), color='red', label='Right Bend', linewidth=2)
        plt.plot(df['frame_id'], df['angle_360'].where(df['angle_360'] > 180), color='blue', label='Left Bend', linewidth=2)
        plt.plot(x_data_smoothed, smoothed_y_data, color='green', label='Moving Average', linewidth=2)

        if peaks is not None:
            plt.scatter(x_data_smoothed[peaks], smoothed_y_data[peaks], color='red', label='Peaks', zorder=5)
        if troughs is not None:
            plt.scatter(x_data_smoothed[troughs], smoothed_y_data[troughs], color='blue', label='Troughs', zorder=5)

        plt.axhline(y=180, color='gray', linestyle='--', label='180 Degrees (Middle)')
        plt.xlabel('Frame', fontsize=12)
        plt.ylabel('Angle (degrees)', fontsize=12)
        plt.title(f'360-Degree Angle at Middle Body Over Time - {os.path.basename("Plots")}', fontsize=14)
        plt.ylim(60, 300)
        plt.grid(True, linestyle='--', alpha=0.6)

        # Add amplitude and frequency information to the plot
        plt.text(0.05, 0.95, f'Amplitude: {amplitude:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.text(0.05, 0.90, f'Cycles: {cycles:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.text(0.05, 0.85, f'Bias: {bias:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname("Plots"), f'plot_{os.path.basename("1")}.png'))
        plt.close()

    return results

def get_metrics(head, left_front, right_front, middle, rear, left_hind, right_hind):
    results = analyze_file(head, left_front, right_front, middle, rear, left_hind, right_hind)

    bias = None; frequency = None; cycles = None; total_limb_movement = None
    if results is not None:
        bias = results.get('bias', None)
        frequency = results.get('frequency', None)
        cycles = results.get('cycles', None) # Not Used
        if 'limb_movements' in results.keys():
            total_limb_movement = (results['limb_movements']['left_front'] + results['limb_movements']['right_front'] +
                              results['limb_movements']['left_hind'] + results['limb_movements']['right_hind']) / 4

    return bias, frequency, total_limb_movement

"""
1000 Random individuals:

bias:                  Min = 6.61122569, Max = 313.065579, Avg. = 162.297145, Std. dev. = 46.7920882
frequency:             Min = 0.02587322, Max = 0.30303030, Avg. = 0.10415503, Std. dev. = 0.03886989
cycles:                Min = 1.11513583, Max = 13.0606060, Avg. = 4.48908189, Std. dev. = 1.67529240
average_limb_movement: Min = 0.0,        Max = 0.22503245, Avg. = 0.07154340, Std. dev. = 0.03713042

Observations:
average_limb_movement: Max = 0.2513393488246488



"""