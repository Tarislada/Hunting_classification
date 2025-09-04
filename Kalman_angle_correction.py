import pandas as pd
from filterpy.kalman import KalmanFilter
import numpy as np
import os

# Define the directories
input_dir = '/media/tarislada/SAMSUNG/Doric임시/854_0304/representative/postprocessing/head_angle'
output_dir =  '/media/tarislada/SAMSUNG/Doric임시/854_0304/representative/postprocessing/kal_filt'
# input_dir = 'SKH FP/head_angle_w59p7'  # Replace with the directory where processed files are saved
# output_dir = 'SKH FP/kalman_filtered_w59p7'  # Replace with the directory where filtered files will be saved
os.makedirs(output_dir, exist_ok=True)
frame_rate = 30  # frames per second

# Kalman filter parameters
process_variance = 1e-5  # Adjust based on your data, controls process noise
measurement_variance = 1e-2  # Adjust based on your data, controls measurement noise

# Loop through each file in the processed directory
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_dir, filename)
        
        # Load the processed data, retaining body center and head angle columns
        df = pd.read_csv(file_path)
        df['smoothed_head_angle'] = np.nan  # Initialize column for smoothed angle
        df['smoothed_angle_velocity'] = np.nan  # Initialize column for smoothed angle velocity
        
        # Set up Kalman filter
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.x = np.array([[df['head_angle'].values[0]], [0]])  # initial state (angle and angular velocity)
        kf.F = np.array([[1, 1], [0, 1]])  # state transition matrix
        kf.H = np.array([[1, 0]])  # measurement function
        kf.P *= 10.0  # covariance matrix
        kf.R = measurement_variance  # measurement noise
        kf.Q = process_variance * np.eye(2)  # process noise
        
        # Apply the filter to each head_angle measurement
        smoothed_angles = []
        for angle in df['head_angle'].values:
            kf.predict()
            kf.update([angle])
            smoothed_angles.append(kf.x[0, 0])  # store the filtered angle

        # Add the smoothed head angle to the dataframe
        df['smoothed_head_angle'] = smoothed_angles

        # Calculate mean and standard deviation for smoothed_head_angle
        mean_angle = df['smoothed_head_angle'].mean()
        std_angle = df['smoothed_head_angle'].std()
        print(f"File: {filename}, Mean Angle: {mean_angle:.4f}, Standard Deviation: {std_angle:.4f}")

        # Define acceptable range for head angle outliers (within 3 standard deviations)
        lower_bound = mean_angle - 3 * std_angle
        upper_bound = mean_angle + 3 * std_angle

        # Identify and replace angle outliers with NaN
        angle_outliers = (df['smoothed_head_angle'] < lower_bound) | (df['smoothed_head_angle'] > upper_bound)
        df.loc[angle_outliers, 'smoothed_head_angle'] = np.nan

        # Interpolate to fill NaN values in smoothed_head_angle
        df['smoothed_head_angle'] = df['smoothed_head_angle'].interpolate()

        # Calculate angular velocity from the smoothed head angle
        df['smoothed_angle_velocity'] = df['smoothed_head_angle'].diff() / (1 / frame_rate)

        # Calculate mean and standard deviation for angular velocity
        mean_velocity = df['smoothed_angle_velocity'].mean()
        std_velocity = df['smoothed_angle_velocity'].std()
        velocity_lower_bound = mean_velocity - 2 * std_velocity
        velocity_upper_bound = mean_velocity + 2 * std_velocity

        # Identify frames where angular velocity is an outlier
        velocity_outliers = (df['smoothed_angle_velocity'] < velocity_lower_bound) | (df['smoothed_angle_velocity'] > velocity_upper_bound)
        
        # Replace head angle with NaN for frames where angular velocity is an outlier
        df.loc[velocity_outliers, 'smoothed_head_angle'] = np.nan

        # Interpolate to fill NaN values in smoothed_head_angle again after replacing outliers
        df['smoothed_head_angle'] = df['smoothed_head_angle'].interpolate()

        # Recalculate angular velocity after final interpolation
        df['smoothed_angle_velocity'] = df['smoothed_head_angle'].diff() / (1 / frame_rate)

        # Save only necessary columns to the output directory
        output_df = df[['frame', 'body center x', 'body center y', 'smoothed_head_angle', 
                        'nose x', 'nose y', 'tail base x', 'tail base y', 'smoothed_angle_velocity']]
        output_path = os.path.join(output_dir, f'kalman_filtered_{filename}')
        output_df.to_csv(output_path, index=False)
        print(f"Kalman-filtered and cleaned data saved for {filename} as 'kalman_filtered_{filename}'")

print("Kalman filtering, angular velocity cutoff, and rule-based cleaning applied to all files in the directory.")
