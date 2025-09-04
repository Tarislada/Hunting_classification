import pandas as pd
from scipy.signal import savgol_filter
import os
import numpy as np

# Set the input and output directories
input_dir = '/media/tarislada/SAMSUNG/Doric임시/854_0304/representative/pose_extraction/csv'
output_dir = '/media/tarislada/SAMSUNG/Doric임시/854_0304/representative/postprocessing/savgol_pose'
# input_dir = 'SKH FP/pose_data'
# output_dir = 'SKH FP/savgol_pose_w59p7'
os.makedirs(output_dir, exist_ok=True)

# Parameters
confidence_threshold = 0.6  # Set a threshold for acceptable keypoint confidence
window_length = 59  # Window length for Savitzky-Golay filter
polyorder = 7  # Polynomial order for Savitzky-Golay filter

# Column indices for keypoint coordinates and confidence (0-based)
keypoint_indices = [7, 8, 27, 28, 21, 22]  # Indices for the keypoint x, y coordinates
confidence_indices = [29, 39, 35]  # Indices for the confidence columns (nose, body center, tail base)

# Ensure that the lengths of keypoint_indices and confidence_indices are consistent
if len(keypoint_indices) != len(confidence_indices) * 2:
    raise ValueError("Each keypoint (x, y) must correspond to a single confidence index. Check the lengths of keypoint_indices and confidence_indices.")

# Loop through each file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        # Construct full file path
        file_path = os.path.join(input_dir, filename)
        
        # Load the CSV file without headers
        pose_data = pd.read_csv(file_path, header=None)
        
        # Generate a complete frame range to fill in any missing frames
        frame_range = pd.DataFrame({0: range(int(pose_data[0].min()), int(pose_data[0].max()) + 1)})
        pose_data = pd.merge(frame_range, pose_data, on=0, how="left")
        
        # Extract keypoint and confidence columns based on indices
        keypoint_data = pose_data.iloc[:, keypoint_indices]
        confidence_data = pose_data.iloc[:, confidence_indices]
        
        # Apply Savitzky-Golay filter to the keypoint coordinates for smoothing and filling
        smoothed_keypoints = keypoint_data.apply(
            lambda col: savgol_filter(col.interpolate(limit_direction="both"), window_length=window_length, polyorder=polyorder)
        )
        
        # # Replace low-confidence keypoints with Savitzky-Golay smoothed values
        # for i in range(len(confidence_indices)):
        #     x_col_idx = keypoint_indices[2 * i]
        #     y_col_idx = keypoint_indices[2 * i + 1]
        #     conf_idx = confidence_indices[i]
            
        #     # Determine which rows have low confidence for the current keypoint
        #     low_confidence = confidence_data.iloc[:, i] < confidence_threshold
        #     print(f"Processing keypoint at columns {x_col_idx}, {y_col_idx} with confidence index {conf_idx}")
            
        #     # Replace low-confidence X and Y coordinates with smoothed values
        #     pose_data.loc[low_confidence, x_col_idx] = smoothed_keypoints.iloc[low_confidence, 2 * i].values
        #     pose_data.loc[low_confidence, y_col_idx] = smoothed_keypoints.iloc[low_confidence, 2 * i + 1].values

        # Replace missing values in keypoints with smoothed values (due to added frames)
        pose_data.iloc[:, keypoint_indices] = smoothed_keypoints

        # Save the filtered data to a new CSV in the output directory without headers
        output_path = os.path.join(output_dir, f'filtered_{filename}')
        pose_data.to_csv(output_path, index=False, header=False)
        print(f"Filtered data saved for {filename} as 'filtered_{filename}'")

print("All files processed and saved in the output directory.")