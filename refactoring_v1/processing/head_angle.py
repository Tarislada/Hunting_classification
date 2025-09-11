"""
Head angle calculation from pose keypoints.
Calculates head angles relative to body axis and angular velocity.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import List, Optional
from config.settings import settings

class HeadAngleProcessor:
    """
    Calculates head angles from pose keypoint data.
    Computes head direction relative to body axis and angular velocity.
    """
    
    def __init__(self, config=None):
        """
        Initialize the head angle processor.
        
        Args:
            config: HeadAngleParams object. If None, uses global settings.
        """
        self.config = config or settings.head_angle
        
        # Define column names for pose data (40 columns total)
        self.column_names = [
            'frame', 'id', 'box x', 'box y', 'box width', 'box height', 'box confidence',
            'nose x', 'nose y',
            'left ear x', 'left ear y', 'right ear x', 'right ear y',
            'left forelimb x', 'left forelimb y', 'right forelimb x', 'right forelimb y',
            'left hindlimb x', 'left hindlimb y', 'right hindlimb x', 'right hindlimb y',
            'tail base x', 'tail base y', 'tail mid x', 'tail mid y', 'tail end x', 'tail end y',
            'body center x', 'body center y',
            'nose conf', 'left ear conf', 'right ear conf', 'left forelimb conf', 'right forelimb conf',
            'left hindlimb conf', 'right hindlimb conf', 'tail root conf', 'tail mid conf', 'tail base conf', 'body center conf'
        ]
    
    def process_file(self, input_path: str, output_path: str) -> bool:
        """
        Process a single CSV file to calculate head angles.
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to save processed output
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load the CSV file
            df = pd.read_csv(input_path, header=None, names=self.column_names)
            
            # Fill missing frames
            complete_range = pd.DataFrame({'frame': range(int(df['frame'].min()), int(df['frame'].max()) + 1)})
            df_filled = pd.merge(complete_range, df, on='frame', how='left').ffill()
            df = df_filled

            # Sort and keep the row with the highest box confidence for each frame
            df.sort_values(by=['frame', 'box confidence'], ascending=[True, False], inplace=True)
            df.drop_duplicates(subset='frame', keep='first', inplace=True)

            # Select only the target columns
            df_filtered = df[self.config.target_columns]

            # Calculate head and body vectors
            df_filtered['head x'] = df_filtered['nose x'] - df_filtered['body center x']
            df_filtered['head y'] = df_filtered['nose y'] - df_filtered['body center y']
            df_filtered['body x'] = df_filtered['body center x'] - df_filtered['tail base x']
            df_filtered['body y'] = df_filtered['body center y'] - df_filtered['tail base y']

            # Create vectors
            head_vector = df_filtered[['head x', 'head y']].values
            body_vector = df_filtered[['body x', 'body y']].values

            # Calculate dot product, cross product, and magnitudes
            dot_product = np.sum(head_vector * body_vector, axis=1)
            cross_product = np.cross(head_vector, body_vector)
            magnitude_head = np.linalg.norm(head_vector, axis=1)
            magnitude_body = np.linalg.norm(body_vector, axis=1)

            # Calculate cosine angle and convert to degrees
            cosine_angle = dot_product / (magnitude_head * magnitude_body)
            angle_radians = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            angle_degrees = np.rad2deg(angle_radians)

            # Adjust angle sign based on cross product
            for i in range(len(angle_degrees)):
                if cross_product[i] > 0:
                    angle_degrees[i] = -angle_degrees[i]

            # Add head angle and time columns
            df_filtered['head_angle'] = angle_degrees
            df_filtered['Time'] = df_filtered['frame'] / self.config.frame_rate

            # Calculate angular velocity
            df_filtered['angular_velocity'] = df_filtered['head_angle'].diff() / (1 / self.config.frame_rate)

            # Keep final columns, including body center coordinates
            final_df = df_filtered[[
                'frame', 'Time', 'head_angle', 'angular_velocity', 
                'body center x', 'body center y', 'nose x', 'nose y', 
                'tail base x', 'tail base y'
            ]]

            # Save the final output
            final_df.to_csv(output_path, index=False)
            return True
            
        except Exception as e:
            print(f"Error processing {Path(input_path).name}: {e}")
            return False
    
    def process_directory(self, input_dir: str, output_dir: str, 
                         file_pattern: str = "filtered_*.csv") -> dict:
        """
        Process all CSV files in a directory.
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path  
            file_pattern: File pattern to match (default: "filtered_*.csv")
            
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
            # Generate output filename: filtered_xxx.csv -> processed_xxx.csv
            base_name = csv_file.name.replace('filtered_', '')
            output_file = Path(output_dir) / f"processed_{base_name}"
            
            print(f"Processing {csv_file.name}...")
            
            if self.process_file(str(csv_file), str(output_file)):
                successful.append(csv_file.name)
                print(f"✓ Saved as {output_file.name}")
            else:
                failed.append(csv_file.name)
                print(f"✗ Failed to process {csv_file.name}")
        
        # Summary
        print(f"\nHead angle calculation complete:")
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
        input_dir = input_dir or str(settings.paths.savgol_pose_dir)
        output_dir = output_dir or str(settings.paths.head_angle_dir)
        
        print("=== HEAD ANGLE CALCULATION (Step 2) ===")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Parameters:")
        print(f"  - Frame rate: {self.config.frame_rate} fps")
        print(f"  - Target columns: {len(self.config.target_columns)} columns")
        
        return self.process_directory(input_dir, output_dir)

def main():
    """Command line interface for head angle calculation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate head angles from pose data")
    parser.add_argument("--input-dir", help="Input directory (overrides config)")
    parser.add_argument("--output-dir", help="Output directory (overrides config)")
    parser.add_argument("--frame-rate", type=int, help="Video frame rate")
    
    args = parser.parse_args()
    
    # Override config if arguments provided
    if args.frame_rate:
        settings.head_angle.frame_rate = args.frame_rate
    
    # Create processor and run
    processor = HeadAngleProcessor()
    results = processor.process(args.input_dir, args.output_dir)
    
    # Exit with error code if any files failed
    if results["failed"] > 0:
        exit(1)

if __name__ == "__main__":
    main()