"""
Cricket detection validation and interpolation.
Processes cricket tracking data with validation, gap filling, and adaptive smoothing.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import re
import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from config.settings import settings

class CricketProcessor:
    """
    Processes cricket tracking data with validation and interpolation.
    Handles gap detection, reliability scoring, and adaptive smoothing.
    """
    
    def __init__(self, config=None):
        """
        Initialize the cricket processor.
        
        Args:
            config: CricketValidationParams object. If None, uses global settings.
        """
        self.config = config or settings.cricket_validation
        self.cricket_in_frame = None
        self.cricket_out_frame = None
                
    def get_cricket_frames(self, txt_content: str) -> Tuple[int, int]:
        """
        Extract cricket start frame and final frame from text file.
        
        Args:
            txt_content: Content of the text file containing frame information
            
        Returns:
            Tuple[int, int]: A tuple containing (cricket_in_frame, cricket_out_frame)
            
        Raises:
            ValueError: If required frames cannot be found in the text
        """
        # Extract cricket in frame
        in_match = re.search(r"cricket in 부터\((\d+)\)", txt_content)
        if not in_match:
            raise ValueError("Cricket start frame not found in text file.")
        cricket_in_frame = int(in_match.group(1))
        
        # First attempt: Look for the last 'consume' entry
        lines = txt_content.strip().split('\n')
        consume_lines = [line for line in lines if line.strip().endswith('consume')]
        
        if consume_lines:
            # If we found consume lines, use the last one
            last_consume = consume_lines[-1]
            try:
                cricket_out_frame = int(last_consume.split('\t')[1])
                return cricket_in_frame, cricket_out_frame
            except (IndexError, ValueError):
                pass  # If parsing fails, continue to the fallback method
        
        # Fallback method: Find the largest number in the text
        all_numbers = re.findall(r'\d+', txt_content)
        if not all_numbers:
            raise ValueError("No frame numbers found in text file")
        
        cricket_out_frame = max(int(num) for num in all_numbers)
        
        return cricket_in_frame, cricket_out_frame
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle -1 values and prepare data."""
        data = data.copy()
        
        # Replace -1 with NaN in coordinate and dimension columns
        for col in ['x', 'y', 'w', 'h']:
            data.loc[data[col] == -1, col] = np.nan
            
        # Replace -1 with 0 in confidence column
        data.loc[data['confidence'] == -1, 'confidence'] = 0
        
        return data
    
    def calculate_validation_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all validation metrics for the entire dataset."""
        # Calculate distance between consecutive points
        data['speed'] = np.sqrt(
            (data['x'].diff())**2 + (data['y'].diff())**2)
        
        # Calculate acceleration
        data['acceleration'] = data['speed'].diff()
        
        # Calculate jerk
        data['jerk'] = data['acceleration'].diff()
        
        # Calculate size (area)
        data['size'] = data['w'] * data['h']
        
        return data
    
    def calculate_reliability_score(self, data: pd.DataFrame, thresholds: Dict[str, float], 
                                    gap_sizes: pd.Series) -> pd.DataFrame:
        """Calculate combined reliability score using weighted criteria."""
        scores = pd.DataFrame(index=data.index)
        
        # Confidence score (exponential version)
        scores['conf_score'] = np.exp(-(1 - data['confidence']) / self.config.confidence_threshold)
        
        # Size score
        steepness = 0.7
        std_multiplier = 1
        scores['size_score'] = 1 / (1 + np.exp(-steepness * (data['size'] - thresholds['size_mean']+std_multiplier*thresholds['size_std'])/thresholds['size_std']))
        
        # For motion-based scores, be more lenient based on gap size
        gap_factor = 1 + np.log1p(gap_sizes / self.config.max_gap_threshold)
        
        # Motion scores (disabled in current version as per config)
        scores['speed_score'] = -1.0
        scores['accel_score'] = -1.0
        scores['jerk_score'] = -1.0
        
        # Get weights
        weights = {
            'conf_score': self.config.confidence_weight,
            'size_score': self.config.size_weight,
            'speed_score': self.config.speed_weight,
            'accel_score': self.config.accel_weight,
            'jerk_score': self.config.jerk_weight,
        }
        total_weight = sum(weights.values())
        
        # Small value to prevent log(0) or division by zero
        epsilon = 1e-10
        
        # Geometric mean with safety checks
        log_scores = np.sum([
            np.log(scores[score_name].clip(lower=epsilon)) * weight
            for score_name, weight in weights.items()
        ], axis=0) / total_weight
        geometric_score = np.exp(log_scores)
        
        # Harmonic mean with safety checks
        harmonic_denominator = np.sum([
            weight / scores[score_name].clip(lower=epsilon)
            for score_name, weight in weights.items()
        ], axis=0)
        harmonic_score = np.where(
            harmonic_denominator > 0,
            total_weight / harmonic_denominator,
            0.0
        )
            
        # Store all scores for debugging
        data['arithmetic_score'] = (
            scores['conf_score'] * self.config.confidence_weight +
            scores['size_score'] * self.config.size_weight +
            scores['speed_score'] * self.config.speed_weight +
            scores['accel_score'] * self.config.accel_weight +
            scores['jerk_score'] * self.config.jerk_weight
        ) / total_weight
        data['geometric_score'] = geometric_score
        data['harmonic_score'] = harmonic_score
            
        # Use geometric score as main reliability score
        data['reliability_score'] = geometric_score
        for col in scores.columns:
            data[col] = scores[col]
        
        return data
        
    def determine_thresholds(self, data: pd.DataFrame) -> Dict[str, float]:
        """Determine validation thresholds based on data distribution."""
        thresholds = {}
        
        metrics = ['size', 'speed', 'acceleration', 'jerk']
        filtered_data = data[(data['frame'] >= self.cricket_in_frame) & 
                    (data['frame'] <= self.cricket_in_frame + self.cricket_out_frame)]
        
        for metric in metrics:
            valid_data = filtered_data[metric].dropna()
            if len(valid_data) > 0:
                if metric in ['speed', 'acceleration', 'jerk']:
                    # For speed-like metrics (right-skewed/Poisson-like)
                    mean = valid_data.median()
                    std = np.median(np.abs(valid_data - mean))
                    thresholds[f'{metric}_mean'] = mean
                    thresholds[f'{metric}_std'] = std
                    thresholds[f'{metric}'] = mean + std * 27
                else:
                    # For more normally distributed metrics (size)
                    mean = valid_data.mean()
                    std = valid_data.std()
                    thresholds[f'{metric}_mean'] = mean
                    thresholds[f'{metric}_std'] = std
                    thresholds[f'{metric}'] = mean + std * 2
    
        return thresholds
    
    def validate_detections(self, data: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
        """Validate detections using both hard cutoff and reliability score."""
        # Initialize status column as 'missing'
        data['status'] = 'missing'
        data.loc[~data[['x', 'y', 'w', 'h']].isna().any(axis=1), 'status'] = 'to_validate'
        
        # First apply hard confidence cutoff
        data.loc[(data['status'] == 'to_validate') & 
                (data['confidence'] < self.config.confidence_threshold), 'status'] = 'invalidated'
        
        # Only proceed with reliability scoring for points that passed confidence check
        valid_mask = data['status'] == 'to_validate'
        
        if valid_mask.any():
            # Create temporary column for gap calculation
            data['temp_valid'] = valid_mask.astype(int)
            
            # Calculate gap sizes for remaining points
            data['gap_group'] = (valid_mask != valid_mask.shift()).cumsum()
            gap_sizes = data.groupby('gap_group')['temp_valid'].transform(lambda x: (1 - x).sum())
            
            # Calculate thresholds and reliability score
            thresholds = self.determine_thresholds(data[valid_mask])
            data = self.calculate_reliability_score(data, thresholds, gap_sizes)
            
            # Validate based on reliability score for remaining points
            data.loc[(data['status'] == 'to_validate') & 
                    (data['reliability_score'] >= self.config.reliability_threshold), 'status'] = 'validated'
            data.loc[(data['status'] == 'to_validate') & 
                    (data['reliability_score'] < self.config.reliability_threshold), 'status'] = 'invalidated'
            
            # Clean up temporary columns
            data = data.drop(['gap_group', 'temp_valid'], axis=1, errors='ignore')
        
        return data

    def detect_prolonged_immobility(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect and flag periods of prolonged immobility."""
        # Parameters
        speed_threshold = 0.1  # pixels per frame for considering as stationary
        time_threshold = 240  # if immobile for 240 frames
        
        # Calculate moving windows of speed
        is_stationary = data['speed'] < speed_threshold
        
        # Group consecutive stationary frames
        stationary_groups = (is_stationary != is_stationary.shift()).cumsum()
        
        # Calculate duration of each stationary period
        group_sizes = is_stationary.groupby(stationary_groups).transform('size')
        
        # Flag prolonged immobility
        data['prolonged_immobility'] = (group_sizes >= time_threshold) & is_stationary
        
        return data

    def process_gaps(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process gaps and determine where to use nose position."""
        # Find sequences of non-validated points
        data['valid'] = data['status'] == 'validated'
        data['gap_group'] = (data['valid'] != data['valid'].shift()).cumsum()
        
        # Calculate gap sizes
        gap_sizes = data.groupby('gap_group').size()
        
        # Mark gaps exceeding threshold for nose position
        data['use_nose_position'] = False
        for group in data[~data['valid']].groupby('gap_group'):
            if len(group[1]) >= self.config.max_gap_threshold:
                data.loc[group[1].index, 'use_nose_position'] = True
        
        # Clean up temporary columns
        data = data.drop(['valid', 'gap_group'], axis=1)
        
        return data
    
    def get_adaptive_window(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate adaptive window sizes based on gaps."""
        base_size = self.config.base_window_size
        
        # Initialize window sizes
        window_sizes = np.full(len(data), base_size)
        
        # Find gaps
        gaps = data['status'].isin(['missing', 'invalidated'])
        
        # Calculate distance to nearest gap for each point
        for i in range(len(data)):
            if not gaps[i]:
                # Find distance to nearest gap
                forward_gap = np.where(gaps[i:])[0]
                backward_gap = np.where(gaps[:i])[0]
                
                dist_forward = forward_gap[0] if len(forward_gap) > 0 else len(data)
                dist_backward = i - backward_gap[-1] if len(backward_gap) > 0 else i
                
                min_dist = min(dist_forward, dist_backward)
                
                # Adjust window size based on gap proximity
                factor = 1 + 0.2 * (1 - np.clip(min_dist / base_size, 0, 1))
                window_sizes[i] = int(base_size * factor)
        
        # Ensure window sizes are odd
        window_sizes = (window_sizes // 2) * 2 + 1
        
        return window_sizes
    
    def apply_smoothing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply adaptive Savitzky-Golay filtering."""
        # Get adaptive window sizes
        window_sizes = self.get_adaptive_window(data)
        
        # Apply smoothing to all coordinates
        for col in ['x', 'y', 'w', 'h']:
            # First interpolate missing values
            smoothed = data[col].copy()
            smoothed[data['status'] == 'invalidated'] = np.nan
            smoothed = smoothed.interpolate(method='linear')  # Long, missing values 
            
            # Then apply Savitzky-Golay filter with adaptive windows
            if len(smoothed) > 3:  # Need at least 4 points for smoothing
                # Create array for storing smoothed values
                final_smoothed = np.copy(smoothed)
                
                # Apply smoothing with different window sizes
                for window_size in np.unique(window_sizes):
                    if window_size > 3:  # Minimum window size check
                        # Get indices for this window size
                        window_mask = window_sizes == window_size
                        if window_mask.any():
                            # Apply smoothing to this segment
                            smooth_segment = savgol_filter(
                                smoothed,
                                window_size,
                                self.config.window_polyorder
                            )
                            # Store smoothed values for this window size
                            final_smoothed[window_mask] = smooth_segment[window_mask]
                
                smoothed = final_smoothed
            
            data[f'smoothed_{col}'] = smoothed
        
        return data
    
    def process_file(self, csv_path: str, txt_path: str, output_path: str) -> bool:
        """Process a single pair of CSV and TXT files."""
        try:
            # Load data
            data = pd.read_csv(csv_path, header=None)
            data = data.iloc[:, :7]  # Select only first 7 columns
            
            # Assign column names
            data.columns = ['frame', 'trackID', 'x', 'y', 'w', 'h', 'confidence']
            
            # Apply downsampling if configured
            if self.config.downsample_60fps:
                data = data[data['frame'] % 2 == 0].reset_index(drop=True)
                data['frame'] = data['frame'] // 2  # Divide by 2 and floor to integer

            # Get start frame
            with open(txt_path, 'r', encoding='utf-8') as f:
                self.cricket_in_frame, self.cricket_out_frame = self.get_cricket_frames(f.read())
            
            print(f"  Cricket frames: {self.cricket_in_frame} to {self.cricket_out_frame}")
            
            # Preprocess data
            data = self.preprocess_data(data)
            
            # Cut data from cricket_in_frame
            data = data[data['frame'] >= self.cricket_in_frame].reset_index(drop=True)
            
            # Calculate validation metrics
            data = self.calculate_validation_metrics(data)
            
            # Determine thresholds
            thresholds = self.determine_thresholds(data)
            
            # Validate detections
            data = self.validate_detections(data, thresholds)
            
            # Detect prolonged immobility
            data = self.detect_prolonged_immobility(data)
            
            # Process gaps
            data = self.process_gaps(data)
            
            # Apply smoothing
            data = self.apply_smoothing(data)
            
            # Save output
            columns_to_save = [
                # Original data
                'frame', 'x', 'y', 'w', 'h', 'confidence',
                # Filtered results
                'smoothed_x', 'smoothed_y', 'smoothed_w', 'smoothed_h',
                # Status flags
                'status', 'use_nose_position',
                # Motion metrics
                'speed', 'acceleration', 'jerk', 'size',
                # Individual scores for debugging
                'conf_score', 'size_score', 'speed_score', 'accel_score', 'jerk_score',
                # Final score
                'reliability_score', 'geometric_score', 'harmonic_score',
                # Prolonged immobility flag
                'prolonged_immobility'
            ]

            data[columns_to_save].to_csv(output_path, index=False)
            return True
            
        except Exception as e:
            print(f"Error processing {Path(csv_path).name}: {e}")
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
                print(f"  ✓ Saved as {output_path.name}")
            else:
                failed.append(csv_path.name)
        
        # Summary
        print(f"\nCricket processing complete:")
        print(f"Successfully processed: {len(successful)} files")
        print(f"Failed to process: {len(failed)} files")
        
        return {
            "processed": len(successful),
            "failed": len(failed),
            "successful_files": successful,
            "failed_files": failed
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
        txt_dir = txt_dir or str(settings.paths.interval_txt_dir)
        output_dir = output_dir or str(settings.paths.cricket_processed_dir)
        
        print("=== CRICKET PROCESSING (Step 4) ===")
        print(f"Cricket detection directory: {cricket_dir}")
        print(f"Interval txt directory: {txt_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Parameters:")
        print(f"  - Confidence threshold: {self.config.confidence_threshold}")
        print(f"  - Reliability threshold: {self.config.reliability_threshold}")
        print(f"  - Max gap threshold: {self.config.max_gap_threshold}")
        print(f"  - Downsample 60fps: {self.config.downsample_60fps}")
        
        return self.process_directory(cricket_dir, txt_dir, output_dir)

def main():
    """Command line interface for cricket processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process cricket detection data")
    parser.add_argument("--cricket-dir", help="Cricket detection directory (overrides config)")
    parser.add_argument("--txt-dir", help="Interval txt directory (overrides config)")
    parser.add_argument("--output-dir", help="Output directory (overrides config)")
    parser.add_argument("--confidence-threshold", type=float, help="Confidence threshold")
    parser.add_argument("--reliability-threshold", type=float, help="Reliability threshold")
    parser.add_argument("--max-gap-threshold", type=int, help="Maximum gap threshold")
    
    args = parser.parse_args()
    
    # Override config if arguments provided
    if args.confidence_threshold:
        settings.cricket_validation.confidence_threshold = args.confidence_threshold
    if args.reliability_threshold:
        settings.cricket_validation.reliability_threshold = args.reliability_threshold
    if args.max_gap_threshold:
        settings.cricket_validation.max_gap_threshold = args.max_gap_threshold
    
    # Create processor and run
    processor = CricketProcessor()
    results = processor.process(args.cricket_dir, args.txt_dir, args.output_dir)
    
    # Exit with error code if any files failed
    if results["failed"] > 0:
        exit(1)

if __name__ == "__main__":
    main()