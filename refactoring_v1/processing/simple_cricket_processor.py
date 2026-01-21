"""
Simple cricket processing with basic interpolation and filtering only.
Bypasses complicated validation and adaptive smoothing.
"""

import numpy as np
import pandas as pd
from scipy. interpolate import interp1d
from scipy.signal import savgol_filter
import re
import os
from pathlib import Path
from typing import Tuple, Dict, Optional
from config. settings import settings


class SimpleCricketProcessor: 
    """
    Simplified cricket processor that only does basic interpolation and filtering.
    Bypasses complicated validation, reliability scoring, and adaptive smoothing. 
    """
    
    def __init__(self, config=None):
        """
        Initialize the simple cricket processor. 
        
        Args:
            config: Configuration object. If None, uses global settings.
        """
        self. config = config or settings.cricket_validation
        self.cricket_in_frame = None
        self. cricket_out_frame = None
    
    def get_cricket_frames(self, txt_content: str) -> Tuple[int, int]:
        """Extract cricket in/out frames from txt file."""
        cricket_in = re.search(r'cricket in frame:\s*(\d+)', txt_content, re.IGNORECASE)
        cricket_out = re.search(r'cricket out frame:\s*(\d+)', txt_content, re.IGNORECASE)
        
        cricket_in_frame = int(cricket_in.group(1)) if cricket_in else 0
        cricket_out_frame = int(cricket_out.group(1)) if cricket_out else float('inf')
        
        return cricket_in_frame, cricket_out_frame
    
    def simple_interpolate(self, data: pd. DataFrame) -> pd.DataFrame:
        """
        Simple linear interpolation for missing values.
        Only interpolates small gaps (< 30 frames).
        """
        max_gap = 30
        coords = ['x', 'y', 'w', 'h']
        
        for coord in coords:
            # Find valid indices
            valid_mask = data[coord] != -1
            valid_indices = data[valid_mask].index. to_numpy()
            valid_values = data. loc[valid_mask, coord].to_numpy()
            
            if len(valid_indices) < 2:
                # Not enough valid points to interpolate
                data[f'smoothed_{coord}'] = data[coord]
                continue
            
            # Create interpolation function
            interp_func = interp1d(valid_indices, valid_values, 
                                  kind='linear', 
                                  bounds_error=False,
                                  fill_value=(valid_values[0], valid_values[-1]))
            
            # Interpolate all indices
            all_indices = data. index.to_numpy()
            interpolated = interp_func(all_indices)
            
            # Only keep interpolation for small gaps
            result = data[coord].copy()
            for i in range(len(data)):
                if data.loc[i, coord] == -1:
                    # Find gap size
                    prev_valid = data.loc[: i][valid_mask[: i+1]].index
                    next_valid = data. loc[i: ][valid_mask[i:]].index
                    
                    gap_size = 0
                    if len(prev_valid) > 0 and len(next_valid) > 0:
                        gap_size = next_valid[0] - prev_valid[-1] - 1
                    
                    # Only interpolate if gap is small
                    if gap_size <= max_gap:
                        result.iloc[i] = interpolated[i]
            
            data[f'smoothed_{coord}'] = result
        
        return data
    
    def basic_filter(self, data: pd.DataFrame, window_length: int = 11, polyorder: int = 3) -> pd.DataFrame:
        """
        Apply basic Savitzky-Golay filtering to smoothed coordinates.
        Only filters where we have valid data.
        """
        coords = ['x', 'y', 'w', 'h']
        
        for coord in coords:
            smoothed_col = f'smoothed_{coord}'
            
            # Only filter valid segments
            valid_mask = data[smoothed_col] != -1
            
            if valid_mask.sum() >= window_length: 
                # Apply filter to valid segments
                filtered = data[smoothed_col]. copy()
                
                # Find contiguous valid segments
                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) > 0:
                    # Split into contiguous segments
                    segments = np.split(valid_indices, 
                                      np.where(np.diff(valid_indices) != 1)[0] + 1)
                    
                    for segment in segments:
                        if len(segment) >= window_length:
                            segment_data = data.loc[segment, smoothed_col]. values
                            filtered_segment = savgol_filter(segment_data, 
                                                            window_length, 
                                                            polyorder)
                            filtered.iloc[segment] = filtered_segment
                
                data[smoothed_col] = filtered
        
        return data
    
    def process_file(self, csv_path: str, txt_path: str, output_path:  str) -> bool:
        """Process a single pair of CSV and TXT files with simple interpolation and filtering."""
        try:
            # Load data
            data = pd.read_csv(csv_path, header=None)
            data = data.iloc[:, :7]  # Select only first 7 columns
            
            # Assign column names
            data.columns = ['frame', 'trackID', 'x', 'y', 'w', 'h', 'confidence']
            
            # Apply downsampling if configured
            if self.config.downsample_60fps:
                data = data[data['frame'] % 2 == 0].reset_index(drop=True)
                data['frame'] = data['frame'] // 2
            
            # Get cricket frames
            with open(txt_path, 'r', encoding='utf-8') as f:
                self.cricket_in_frame, self.cricket_out_frame = self.get_cricket_frames(f.read())
            
            print(f"  Cricket frames:  {self.cricket_in_frame} to {self.cricket_out_frame}")
            
            # Replace -1 with NaN for easier handling
            for col in ['x', 'y', 'w', 'h']:
                data[col] = data[col].replace(-1, np.nan)
            
            # Cut data from cricket_in_frame
            data = data[data['frame'] >= self.cricket_in_frame]. reset_index(drop=True)
            
            # Replace NaN back to -1 for processing
            for col in ['x', 'y', 'w', 'h']:
                data[col] = data[col].fillna(-1)
            
            # Simple interpolation
            data = self. simple_interpolate(data)
            
            # Basic filtering
            data = self.basic_filter(data)
            
            # Add status column (all marked as valid for simple processing)
            data['status'] = 'valid'
            data. loc[data['x'] == -1, 'status'] = 'missing'
            
            # Save output with minimal columns
            columns_to_save = [
                'frame', 'x', 'y', 'w', 'h', 'confidence',
                'smoothed_x', 'smoothed_y', 'smoothed_w', 'smoothed_h',
                'status'
            ]
            
            data[columns_to_save].to_csv(output_path, index=False)
            print(f"  ✓ Successfully processed (simple mode)")
            return True
            
        except Exception as e:
            print(f"  ✗ Error:  {str(e)}")
            return False
    
    def process_directory(self, cricket_dir: str, txt_dir: str, output_dir: str) -> dict:
        """Process all cricket tracking files in a directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        csv_files = list(Path(cricket_dir).glob('*.csv'))
        
        if not csv_files: 
            print(f"No CSV files found in {cricket_dir}")
            return {"processed": 0, "failed": 0, "files": []}
        
        print(f"Found {len(csv_files)} files to process")
        
        successful = []
        failed = []
        
        for csv_path in csv_files:
            base_name = csv_path.stem
            txt_path = Path(txt_dir) / f"{base_name}.txt"
            output_path = Path(output_dir) / f"crprocessed_{base_name}.csv"
            
            print(f"Processing {csv_path.name}...")
            
            if not txt_path.exists():
                print(f"  ✗ Missing TXT file: {txt_path.name}")
                failed.append(csv_path.name)
                continue
            
            if self.process_file(str(csv_path), str(txt_path), str(output_path)):
                successful.append(csv_path.name)
            else:
                failed.append(csv_path.name)
        
        print(f"\n{'='*60}")
        print(f"SIMPLE CRICKET PROCESSING COMPLETE")
        print(f"  ✓ Successful: {len(successful)} files")
        print(f"  ✗ Failed: {len(failed)} files")
        
        return {
            "processed":  len(successful),
            "failed": len(failed),
            "files": successful
        }
    
    def process(self, cricket_dir: Optional[str] = None, 
                txt_dir: Optional[str] = None,
                output_dir: Optional[str] = None) -> dict:
        """
        Main processing method using configured directories.
        
        Args:
            cricket_dir: Override cricket detection directory (uses config if None)
            txt_dir: Override txt directory (uses config if None) 
            output_dir: Override output directory (uses config if None)
            
        Returns:
            dict: Processing results
        """
        cricket_dir = cricket_dir or str(settings.paths.cricket_detection_dir)
        txt_dir = txt_dir or str(settings. paths.interval_txt_dir)
        output_dir = output_dir or str(settings.paths.cricket_processed_dir)
        
        print("=== SIMPLE CRICKET PROCESSING (Workaround Mode) ===")
        print(f"Cricket detection directory: {cricket_dir}")
        print(f"Interval txt directory: {txt_dir}")
        print(f"Output directory:  {output_dir}")
        print(f"Mode: Basic interpolation and filtering ONLY")
        print(f"  - No validation metrics")
        print(f"  - No reliability scoring")
        print(f"  - No adaptive smoothing")
        print(f"  - Downsample 60fps:  {self.config.downsample_60fps}")
        print()
        
        return self.process_directory(cricket_dir, txt_dir, output_dir)


def main():
    """Command line interface for simple cricket processing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Simple Cricket Processing (Workaround Mode)"
    )
    parser.add_argument("--cricket-dir", 
                       help="Cricket detection directory")
    parser.add_argument("--txt-dir",
                       help="Interval txt directory")
    parser.add_argument("--output-dir",
                       help="Output directory")
    
    args = parser.parse_args()
    
    processor = SimpleCricketProcessor()
    results = processor.process(args.cricket_dir, args.txt_dir, args.output_dir)
    
    # Exit with error code if any files failed
    if results["failed"] > 0:
        exit(1)


if __name__ == "__main__":
    main()