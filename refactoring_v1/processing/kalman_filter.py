"""
Kalman filtering for head angle smoothing and outlier removal.
Applies Kalman filter to head angles and removes outliers based on angular velocity.
"""

import pandas as pd
from filterpy.kalman import KalmanFilter
import numpy as np
import os
from pathlib import Path
from typing import Optional
from config.settings import settings

class KalmanAngleProcessor:
    """
    Applies Kalman filtering to head angles with outlier detection.
    Smooths head angles and removes outliers based on angular velocity.
    """
    
    def __init__(self, config=None):
        """
        Initialize the Kalman angle processor.
        
        Args:
            config: KalmanFilterParams object. If None, uses global settings.
        """
        self.config = config or settings.kalman_filter
    
    def setup_kalman_filter(self, initial_angle: float) -> KalmanFilter:
        """
        Set up Kalman filter for angle tracking.
        
        Args:
            initial_angle: Initial head angle value
            
        Returns:
            KalmanFilter: Configured Kalman filter
        """
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.x = np.array([[initial_angle], [0]])  # initial state (angle and angular velocity)
        kf.F = np.array([[1, 1], [0, 1]])  # state transition matrix
        kf.H = np.array([[1, 0]])  # measurement function
        kf.P *= 10.0  # covariance matrix
        kf.R = self.config.measurement_variance  # measurement noise
        kf.Q = self.config.process_variance * np.eye(2)  # process noise
        
        return kf
    
    def apply_kalman_filtering(self, angles: pd.Series) -> np.ndarray:
        """
        Apply Kalman filter to angle sequence.
        
        Args:
            angles: Series of head angles
            
        Returns:
            np.ndarray: Smoothed angles
        """
        if len(angles) == 0:
            return np.array([])
        
        # Set up Kalman filter
        kf = self.setup_kalman_filter(angles.iloc[0])
        
        # Apply the filter to each head_angle measurement
        smoothed_angles = []
        for angle in angles.values:
            kf.predict()
            kf.update([angle])
            smoothed_angles.append(kf.x[0, 0])  # store the filtered angle
        
        return np.array(smoothed_angles)
    
    def remove_outliers(self, df: pd.DataFrame, angle_col: str) -> pd.DataFrame:
        """
        Remove angle outliers based on statistical thresholds.
        
        Args:
            df: DataFrame with angles
            angle_col: Name of angle column
            
        Returns:
            pd.DataFrame: DataFrame with outliers removed and interpolated
        """
        df = df.copy()
        
        # Calculate mean and standard deviation for smoothed_head_angle
        mean_angle = df[angle_col].mean()
        std_angle = df[angle_col].std()
        
        print(f"  Mean Angle: {mean_angle:.4f}, Standard Deviation: {std_angle:.4f}")

        # Define acceptable range for head angle outliers (within 3 standard deviations)
        lower_bound = mean_angle - 3 * std_angle
        upper_bound = mean_angle + 3 * std_angle

        # Identify and replace angle outliers with NaN
        angle_outliers = (df[angle_col] < lower_bound) | (df[angle_col] > upper_bound)
        df.loc[angle_outliers, angle_col] = np.nan

        # Interpolate to fill NaN values in smoothed_head_angle
        df[angle_col] = df[angle_col].interpolate()
        
        return df
    
    def remove_velocity_outliers(self, df: pd.DataFrame, angle_col: str, velocity_col: str) -> pd.DataFrame:
        """
        Remove outliers based on angular velocity thresholds.
        
        Args:
            df: DataFrame with angles and velocities
            angle_col: Name of angle column
            velocity_col: Name of velocity column
            
        Returns:
            pd.DataFrame: DataFrame with velocity outliers removed
        """
        df = df.copy()
        
        # Calculate angular velocity from the smoothed head angle
        df[velocity_col] = df[angle_col].diff() / (1 / self.config.frame_rate)

        # Calculate mean and standard deviation for angular velocity
        mean_velocity = df[velocity_col].mean()
        std_velocity = df[velocity_col].std()
        velocity_lower_bound = mean_velocity - 2 * std_velocity
        velocity_upper_bound = mean_velocity + 2 * std_velocity

        # Identify frames where angular velocity is an outlier
        velocity_outliers = (df[velocity_col] < velocity_lower_bound) | (df[velocity_col] > velocity_upper_bound)
        
        # Replace head angle with NaN for frames where angular velocity is an outlier
        df.loc[velocity_outliers, angle_col] = np.nan

        # Interpolate to fill NaN values in smoothed_head_angle again after replacing outliers
        df[angle_col] = df[angle_col].interpolate()

        # Recalculate angular velocity after final interpolation
        df[velocity_col] = df[angle_col].diff() / (1 / self.config.frame_rate)
        
        return df
    
    def process_file(self, input_path: str, output_path: str) -> bool:
        """
        Process a single CSV file with Kalman filtering.
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to save filtered output
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load the processed data
            df = pd.read_csv(input_path)
            
            if 'head_angle' not in df.columns:
                raise ValueError("Input file must contain 'head_angle' column")
            
            # Initialize column for smoothed angle
            df['smoothed_head_angle'] = np.nan
            df['smoothed_angle_velocity'] = np.nan
            
            # Apply Kalman filtering
            smoothed_angles = self.apply_kalman_filtering(df['head_angle'])
            df['smoothed_head_angle'] = smoothed_angles
            
            print(f"  Applied Kalman filter with process_var={self.config.process_variance}, "
                  f"measurement_var={self.config.measurement_variance}")
            
            # Remove angle outliers
            df = self.remove_outliers(df, 'smoothed_head_angle')
            
            # Remove velocity outliers
            df = self.remove_velocity_outliers(df, 'smoothed_head_angle', 'smoothed_angle_velocity')
            
            # Save only necessary columns to the output directory
            output_columns = [
                'frame', 'body center x', 'body center y', 'smoothed_head_angle', 
                'nose x', 'nose y', 'tail base x', 'tail base y', 'smoothed_angle_velocity'
            ]
            
            output_df = df[output_columns]
            output_df.to_csv(output_path, index=False)
            
            return True
            
        except Exception as e:
            print(f"Error processing {Path(input_path).name}: {e}")
            return False
    
    def process_directory(self, input_dir: str, output_dir: str, 
                         file_pattern: str = "processed_*.csv") -> dict:
        """
        Process all CSV files in a directory.
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            file_pattern: File pattern to match (default: "processed_*.csv")
            
        Returns:
            dict: Processing results with success/failure counts
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all CSV files
        input_path = Path(input_dir)
        csv_files = list(input_path.glob(file_pattern))
        
        if not csv_files:
            print(f"No files matching '{file_pattern}' found in {input_dir}")
            return {"processed": 0, "failed": 0, "files": []}
        
        print(f"Found {len(csv_files)} files to process")
        
        successful = []
        failed = []
        
        for csv_file in csv_files:
            # Generate output filename: processed_xxx.csv -> kalman_filtered_xxx.csv
            base_name = csv_file.name.replace('processed_', '')
            output_file = Path(output_dir) / f"kalman_filtered_{base_name}"
            
            print(f"Processing {csv_file.name}...")
            
            if self.process_file(str(csv_file), str(output_file)):
                successful.append(csv_file.name)
                print(f"✓ Saved as {output_file.name}")
            else:
                failed.append(csv_file.name)
                print(f"✗ Failed to process {csv_file.name}")
        
        # Summary
        print(f"\nKalman filtering complete:")
        print(f"Successfully processed: {len(successful)} files")
        print(f"Failed to process: {len(failed)} files")
        
        return {
            "processed": len(successful),
            "failed": len(failed),
            "successful_files": successful,
            "failed_files": failed
        }
    
    def process(self, input_dir: Optional[str] = None, 
                output_dir: Optional[str] = None) -> dict:
        """
        Main processing method using configured directories.
        
        Args:
            input_dir: Override input directory (uses config if None)
            output_dir: Override output directory (uses config if None)
            
        Returns:
            dict: Processing results
        """
        input_dir = input_dir or str(settings.paths.head_angle_dir)
        output_dir = output_dir or str(settings.paths.kalman_filtered_dir)
        
        print("=== KALMAN FILTERING (Step 3) ===")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Parameters:")
        print(f"  - Frame rate: {self.config.frame_rate} fps")
        print(f"  - Process variance: {self.config.process_variance}")
        print(f"  - Measurement variance: {self.config.measurement_variance}")
        
        return self.process_directory(input_dir, output_dir)

def main():
    """Command line interface for Kalman filtering."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply Kalman filtering to head angles")
    parser.add_argument("--input-dir", help="Input directory (overrides config)")
    parser.add_argument("--output-dir", help="Output directory (overrides config)")
    parser.add_argument("--process-variance", type=float, help="Kalman process variance")
    parser.add_argument("--measurement-variance", type=float, help="Kalman measurement variance")
    parser.add_argument("--frame-rate", type=int, help="Video frame rate")
    
    args = parser.parse_args()
    
    # Override config if arguments provided
    if args.process_variance:
        settings.kalman_filter.process_variance = args.process_variance
    if args.measurement_variance:
        settings.kalman_filter.measurement_variance = args.measurement_variance
    if args.frame_rate:
        settings.kalman_filter.frame_rate = args.frame_rate
    
    # Create processor and run
    processor = KalmanAngleProcessor()
    results = processor.process(args.input_dir, args.output_dir)
    
    # Exit with error code if any files failed
    if results["failed"] > 0:
        exit(1)

if __name__ == "__main__":
    main()