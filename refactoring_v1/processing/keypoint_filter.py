"""
Keypoint filtering using Savitzky-Golay filter.
Smooths pose keypoint coordinates and handles low-confidence detections.
"""

import pandas as pd
from scipy.signal import savgol_filter
import os
import numpy as np
from pathlib import Path
from typing import List, Optional
from config.settings import settings

class KeypointFilter:
    """
    Applies Savitzky-Golay filtering to keypoint coordinates.
    Handles missing frames and low-confidence detections.
    """
    
    def __init__(self, config=None):
        """
        Initialize the keypoint filter.
        
        Args:
            config: KeypointFilterParams object. If None, uses global settings.
        """
        self.config = config or settings.keypoint_filter
        
    def process_file(self, input_path: str, output_path: str) -> bool:
        """
        Process a single CSV file with keypoint filtering.
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to save filtered output
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load the CSV file without headers
            pose_data = pd.read_csv(input_path, header=None)
            
            # Generate a complete frame range to fill in any missing frames
            frame_range = pd.DataFrame({0: range(int(pose_data[0].min()), int(pose_data[0].max()) + 1)})
            pose_data = pd.merge(frame_range, pose_data, on=0, how="left")
            
            # Extract keypoint and confidence columns based on indices
            keypoint_data = pose_data.iloc[:, self.config.keypoint_indices]
            confidence_data = pose_data.iloc[:, self.config.confidence_indices]
            
            # Apply Savitzky-Golay filter to the keypoint coordinates
            smoothed_keypoints = keypoint_data.apply(
                lambda col: savgol_filter(
                    col.interpolate(limit_direction="both"), 
                    window_length=self.config.window_length, 
                    polyorder=self.config.polyorder
                )
            )
            
            # Replace missing values in keypoints with smoothed values
            pose_data.iloc[:, self.config.keypoint_indices] = smoothed_keypoints
            
            # Save the filtered data without headers
            pose_data.to_csv(output_path, index=False, header=False)
            
            return True
            
        except Exception as e:
            print(f"Error processing {Path(input_path).name}: {e}")
            return False
    
    def process_directory(self, input_dir: str, output_dir: str, 
                         file_pattern: str = "*.csv") -> dict:
        """
        Process all CSV files in a directory.
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            file_pattern: File pattern to match (default: "*.csv")
            
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
            output_file = Path(output_dir) / f"filtered_{csv_file.name}"
            
            print(f"Processing {csv_file.name}...")
            
            if self.process_file(str(csv_file), str(output_file)):
                successful.append(csv_file.name)
                print(f"✓ Saved as {output_file.name}")
            else:
                failed.append(csv_file.name)
                print(f"✗ Failed to process {csv_file.name}")
        
        # Summary
        print(f"\nKeypoint filtering complete:")
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
        input_dir = input_dir or str(settings.paths.pose_data_dir)
        output_dir = output_dir or str(settings.paths.savgol_pose_dir)
        
        print("=== KEYPOINT FILTERING (Step 1) ===")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Parameters:")
        print(f"  - Window length: {self.config.window_length}")
        print(f"  - Polynomial order: {self.config.polyorder}")
        print(f"  - Confidence threshold: {self.config.confidence_threshold}")
        
        return self.process_directory(input_dir, output_dir)

def main():
    """Command line interface for keypoint filtering."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply keypoint filtering")
    parser.add_argument("--input-dir", help="Input directory (overrides config)")
    parser.add_argument("--output-dir", help="Output directory (overrides config)")
    parser.add_argument("--window-length", type=int, help="Savgol window length")
    parser.add_argument("--polyorder", type=int, help="Savgol polynomial order")
    
    args = parser.parse_args()
    
    # Override config if arguments provided
    if args.window_length:
        settings.keypoint_filter.window_length = args.window_length
    if args.polyorder:
        settings.keypoint_filter.polyorder = args.polyorder
    
    # Create processor and run
    processor = KeypointFilter()
    results = processor.process(args.input_dir, args.output_dir)
    
    # Exit with error code if any files failed
    if results["failed"] > 0:
        exit(1)

if __name__ == "__main__":
    main()