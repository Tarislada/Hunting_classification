import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import re
import os
from pathlib import Path
from typing import Tuple, Dict, List
from dataclasses import dataclass

@dataclass
class ValidationParams:
    """Parameters for validation criteria."""
    max_gap_threshold: int = 30 # need increase
    base_window_size: int = 20 # test increase
    window_polyorder: int = 3
    # confidence_weight: float = 0.82
    confidence_weight: float = 1.0
    size_weight: float = 1.0
    distance_weight: float = 1.0
    speed_weight: float = 1.0
    accel_weight: float = 1.0
    jerk_weight: float = 1.0
    reliability_threshold: float = 0.65  # May also need to adjust due to speed_weight change
    confidence_threshold: float = 0.25
    downsample_60fps: bool = False  # Added parameter

class CricketDataProcessor:
    def __init__(self, params: ValidationParams = None):
        """Initialize processor with validation parameters."""
        self.params = params or ValidationParams()
        self.confidence_threshold = self.params.confidence_threshold
                
    def get_cricket_frames(self, txt_content: str) -> Tuple[int, int]:
        """
        Extract cricket start frame and final frame from text file.
        
        The method will:
        1. First try to find the last 'consume' entry
        2. If no 'consume' entry exists, find the largest number in the text
        
        Args:
            txt_content (str): Content of the text file containing frame information
            
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
    
    def  calculate_reliability_score(self, data: pd.DataFrame, thresholds: Dict[str, float], 
                                gap_sizes: pd.Series) -> pd.DataFrame:
        """Calculate combined reliability score using weighted criteria."""
        scores = pd.DataFrame(index=data.index)
        
        # Confidence score
        # exponential version
        scores['conf_score'] = np.exp(-(1 - data['confidence']) / self.confidence_threshold)
        cfsteepness = 5
        cfalpha = 0.725
        cfbeta = 0.425
        # scores['conf_score'] = 1 / (1 + np.exp(-cfsteepness * (data['confidence'] - cfalpha)/cfbeta))
        
        # Size score
        # steepness & std multipler at 1 works somewhat with exponential version of confidence score
        steepness = 0.7
        std_multiplier = 1
        # TODO: it seems like size score can be somewhat close to binary - 
        # shit seems like a size smaller than 0.001, but that becomes problematic when small crickets are in
        scores['size_score'] = 1 / (1 + np.exp(-steepness * (data['size'] - thresholds['size_mean']+std_multiplier*thresholds['size_std'])/thresholds['size_std']))
        # scores['size_score'] = 1 / (1 + np.exp(-steepness * (data['size'] - thresholds['size'])))

        # scores['size_score'] = np.where(data['size'] < thresholds['size'],
        #                             np.exp(-np.abs(data['size'] - thresholds['size_mean']) / thresholds['size_std']),1.0) # reject small size only
        # scores['size_score'] = scores['size_score']**1.15  # Increase the curve for size => calls for std increase to compensate
        
        # For motion-based scores, be more lenient based on gap size
        gap_factor = 1 + np.log1p(gap_sizes / self.params.max_gap_threshold)
        
        # Motion scores using the computed thresholds with gap adjustment
        # scores['speed_score'] = np.where(data['speed'] > thresholds['speed'],
        #                             np.exp(-(data['speed'] - thresholds['speed']) / (thresholds['speed'] * gap_factor)),
        #                             1.0)
        scores['speed_score'] = -1.0
        # scores['accel_score'] = np.where(np.abs(data['acceleration']) > thresholds['acceleration'],
                                        # np.exp(-(np.abs(data['acceleration']) - thresholds['acceleration']) / (thresholds['acceleration'] * gap_factor)),
        #                                 1.0)
        scores['accel_score'] = -1.0
        # scores['jerk_score'] = np.where(np.abs(data['jerk']) > thresholds['jerk'],
        #                             np.exp(-(np.abs(data['jerk']) - thresholds['jerk']) / (thresholds['jerk'] * gap_factor)),
        #                             1.0)
        scores['jerk_score'] = -1.0
        
        # Combined weighted score
        reliability_score = (
            scores['conf_score'] * self.params.confidence_weight +
            scores['size_score'] * self.params.size_weight +
            scores['speed_score'] * self.params.speed_weight +
            scores['accel_score'] * self.params.accel_weight +
            scores['jerk_score'] * self.params.jerk_weight
        )
        
        # Normalize to 0-1 range
        total_weight = (self.params.confidence_weight + self.params.size_weight +
                    self.params.speed_weight + self.params.accel_weight +
                    self.params.jerk_weight)
        reliability_score /= total_weight
        
        # Test Geometric mean and Harmonic mean
        # Get weights
        weights = {
            'conf_score': self.params.confidence_weight,
            'size_score': self.params.size_weight,
            'speed_score': self.params.speed_weight,
            'accel_score': self.params.accel_weight,
            'jerk_score': self.params.jerk_weight,
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
        # data['reliability_score'] = harmonic_score  # or geometric_score
        # Optional: store both for comparison
        data['arithmetic_score'] = reliability_score
        data['geometric_score'] = geometric_score
        data['harmonic_score'] = harmonic_score
            
        # Store all scores for debugging
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
                # For speed-like metrics (right-skewed/Poisson-like)
                if metric in ['speed', 'acceleration', 'jerk']:
                    # Could use different approach for skewed data
                    # Option 1: Use median + n*std
                    mean = valid_data.median() # median
                    # mean = valid_data.mean() # mean
                    # std = valid_data.std() # std
                    std = np.median(np.abs(valid_data - mean))
                    thresholds[f'{metric}_mean'] = mean
                    thresholds[f'{metric}_std'] = std
                    thresholds[f'{metric}'] = mean + std * 27
                    # speed needs higher std
                # elif metric == 'speed':
                #     mean = valid_data.median() # median
                #     # mean = valid_data.mean() # mean
                #     # std = valid_data.std() # std
                #     std = np.median(np.abs(valid_data - mean))
                #     thresholds[f'{metric}_mean'] = mean
                #     thresholds[f'{metric}_std'] = std
                #     thresholds[f'{metric}'] = mean + std * 27
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
                (data['confidence'] < self.confidence_threshold), 'status'] = 'invalidated'
        
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
                    (data['reliability_score'] >= self.params.reliability_threshold), 'status'] = 'validated'
            data.loc[(data['status'] == 'to_validate') & 
                    (data['reliability_score'] < self.params.reliability_threshold), 'status'] = 'invalidated'
            
            # Clean up temporary columns
            data = data.drop(['gap_group', 'temp_valid'], axis=1, errors='ignore')
        
        return data

    def detect_prolonged_immobility(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and flag periods of prolonged immobility.
        Returns DataFrame with added 'prolonged_immobility' column.
        """
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
            if len(group[1]) >= self.params.max_gap_threshold:
                data.loc[group[1].index, 'use_nose_position'] = True
        
        # Clean up temporary columns
        data = data.drop(['valid', 'gap_group'], axis=1)
        
        return data
    
    def get_adaptive_window(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate adaptive window sizes based on gaps."""
        base_size = self.params.base_window_size
        
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
            
            # This section is for testing purposes
            smoothed = data[col].copy()
            smoothed[data['status'] == 'invalidated'] = np.nan
            # smoothed = smoothed.interpolate(method='linear', limit=self.params.max_gap_threshold)
            smoothed = smoothed.interpolate(method='linear') # Long, missing values 

            # smoothed = data[col].interpolate(method='linear', limit=self.params.max_gap_threshold)
            
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
                                self.params.window_polyorder
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
            
            if self.params.downsample_60fps:  # Add this as a class parameter
                data = data[data['frame'] % 2 == 0].reset_index(drop=True)
                data['frame'] = data['frame'] // 2  # Divide by 2 and floor to integer

            # Get start frame
            with open(txt_path, 'r', encoding='utf-8') as f:
                self.cricket_in_frame, self.cricket_out_frame = self.get_cricket_frames(f.read())
            
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
            print(f"Successfully processed: {os.path.basename(csv_path)}")
            return True
            
        except Exception as e:
            print(f"Error processing {os.path.basename(csv_path)}: {str(e)}")
            return False

def process_directory(input_dir: str,
                     txt_dir: str,
                     output_dir: str,
                     params: ValidationParams = None) -> None:
    """Process all cricket tracking files in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    processor = CricketDataProcessor(params)
    
    csv_files = list(Path(input_dir).glob('*.csv'))
    successful = 0
    failed = 0
    
    for csv_path in csv_files:
        base_name = csv_path.stem
        txt_path = Path(txt_dir) / f"{base_name}.txt"
        output_path = Path(output_dir) / f"crprocessed_{base_name}.csv"
        
        if not txt_path.exists():
            print(f"Missing TXT file for {base_name}")
            failed += 1
            continue
            
        if processor.process_file(csv_path, txt_path, output_path):
            successful += 1
        else:
            failed += 1
    
    print(f"\nProcessing complete:\n"
          f"Successfully processed: {successful} files\n"
          f"Failed to process: {failed} files")

# Example usage
if __name__ == "__main__":
    
    # input_dir = '/media/tarislada/SAMSUNG/Doric임시/854_0304/representative/Cricket_extraction/csv'
    # txt_dir = '/media/tarislada/SAMSUNG/Doric임시/854_0304/representative/postprocessing/interval_txt'
    # output_dir = '/media/tarislada/SAMSUNG/Doric임시/854_0304/representative/postprocessing/cricket_process_test'
    
    input_dir = "SKH FP/cricket_dection5"    
    txt_dir = "SKH FP/interval_txt"
    output_dir = "SKH FP/FInalized_process/cricket_process_test5"
    params = ValidationParams(downsample_60fps=True,speed_weight=0.0,accel_weight=0.0,jerk_weight=0.0)
    
    process_directory(input_dir, txt_dir, output_dir, params)
    
    # Process a single file
    # processor = CricketDataProcessor()
    # processor.process_file('SKH FP/cricket_dection5/m18_t7.csv', 'SKH FP/interval_txt/m18_t7.txt', 'SKH FP/FInalized_process/cricket_process_test5/crprocessed_m18_t7.csv')