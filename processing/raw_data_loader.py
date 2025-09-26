import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import gc
import re

class RawDataLoader:
    """
    Loads and processes raw pose keypoints and cricket tracking data for sequence modeling.
    This class bypasses all feature engineering, working directly with the outputs from 
    keypoint_filter.py and cricket_interp.py.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.target_encoder = LabelEncoder()
    
    def load_raw_data(self, keypoint_files, cricket_files, label_files, batch_size=5):
        """
        Load and align data directly from keypoint_filter.py and cricket_interp.py outputs.
        
        Args:
            keypoint_files: List of paths to filtered keypoint CSV files
            cricket_files: List of paths to cricket tracking CSV files  
            label_files: List of paths to behavior label CSV files
            batch_size: Number of files to process at once (for memory efficiency)
            
        Returns:
            DataFrame containing aligned raw pose, cricket, and behavior data
        """
        print(f"Loading raw data from {len(keypoint_files)} files in batches of {batch_size}")
        
        # Check that we have the same number of files
        assert len(keypoint_files) == len(cricket_files) == len(label_files), \
               "Must provide equal numbers of keypoint, cricket, and label files"
        
        all_batches = []
        for i in range(0, len(keypoint_files), batch_size):
            batch_kp_files = keypoint_files[i:i+batch_size]
            batch_cricket_files = cricket_files[i:i+batch_size]
            batch_label_files = label_files[i:i+batch_size]
            
            print(f"Processing batch {i//batch_size + 1}/{(len(keypoint_files)-1)//batch_size + 1}")
            batch_data = self._load_and_align_batch(batch_kp_files, batch_cricket_files, batch_label_files)
            
            if batch_data is not None:
                all_batches.append(batch_data)
            
            # Force garbage collection
            gc.collect()
        
        if all_batches:
            all_data = pd.concat(all_batches, ignore_index=True)
            print(f"Loaded {len(all_data)} total frames")
            return all_data
        else:
            return None
            
    def _load_and_align_batch(self, keypoint_files, cricket_files, label_files):
        """Load and align a batch of raw data files"""
        batch_data = []

        for kp_file, cricket_file, label_file in zip(keypoint_files, cricket_files, label_files):
            try:
                # Extract animal ID and trial ID from filename
                kp_path = Path(kp_file)
                match = re.search(r'filtered_(m\d+)_(t\d+)\.csv', kp_path.name)
                if match:
                    animal_id, trial_id = match.groups()
                else:
                    print(f"Warning: Couldn't parse animal/trial ID from {kp_path.name}")
                    continue
                
                # Define column names for keypoint CSV
                keypoint_columns = [
                    'frame', 'animal_id', 
                    'box_x', 'box_y', 'box_width', 'box_height', 'box_confidence',
                    'nose_x', 'nose_y', 
                    'left_ear_x', 'left_ear_y', 
                    'right_ear_x', 'right_ear_y', 
                    'left_forepaw_x', 'left_forepaw_y', 
                    'right_forepaw_x', 'right_forepaw_y', 
                    'left_hindpaw_x', 'left_hindpaw_y', 
                    'right_hindpaw_x', 'right_hindpaw_y', 
                    'tail_root_x', 'tail_root_y', 
                    'tail_center_x', 'tail_center_y', 
                    'tail_tip_x', 'tail_tip_y', 
                    'body_center_x', 'body_center_y'
                ]
                
                # Add confidence columns for each keypoint
                keypoints = [
                    'nose', 'left_ear', 'right_ear', 
                    'left_forepaw', 'right_forepaw', 
                    'left_hindpaw', 'right_hindpaw',
                    'tail_root', 'tail_center', 'tail_tip', 
                    'body_center'
                ]
                
                confidence_columns = [f'{kp}_confidence' for kp in keypoints]
                keypoint_columns.extend(confidence_columns)
                
                # Load data with column names
                keypoints_df = pd.read_csv(
                    kp_file, 
                    header=None, 
                    names=keypoint_columns, 
                    index_col=False
                )
                print(f"Loaded keypoints: {len(keypoints_df)} frames")
                
                # Load cricket and label data
                cricket_df = pd.read_csv(cricket_file)
                labels_df = pd.read_csv(label_file)
                
                merged = pd.merge(keypoints_df, cricket_df, on='frame', how='left', suffixes=('', '_cricket'))
                print(f"After cricket merge: {len(merged)} frames")
                
                # Add a flag to indicate if cricket is detected
                merged['cricket_detected'] = (~merged['x'].isna()).astype(int)  # Assuming x is from cricket data

                # Fill cricket NaN values with zeros so the model can learn when cricket is not visible
                cricket_cols = [col for col in ['x', 'y', 'w', 'h', 'confidence',
                                'smoothed_x', 'smoothed_y', 'smoothed_w', 'smoothed_h', 'status',
                                'use_nose_position', 'speed', 'acceleration', 'jerk', 'size',
                                'conf_score', 'size_score', 'speed_score', 'accel_score', 'jerk_score',
                                'reliability_score', 'geometric_score', 'harmonic_score',
                                'prolonged_immobility']]
                for col in cricket_cols:
                    if col in merged.columns:
                        merged[col] = merged[col].fillna(0)
                
                # Merge with behavior labels - again keep all keypoint frames
                merged = pd.merge(merged, labels_df, on='frame', how='left', suffixes=('', '_label'))
                print(f"After label merge: {len(merged)} frames")
                
                # Override animal_id from filename
                merged['animal_id'] = animal_id  
                merged['trial_id'] = trial_id
                
                # Basic cleaning - only drop rows without behavior labels
                merged = merged.dropna(subset=['behavior'])  # Remove rows without labels
                print(f"After dropping missing labels: {len(merged)} frames")
                
                batch_data.append(merged)
                print(f"Added trial {animal_id}_{trial_id}: {len(merged)} frames to batch")
                
            except Exception as e:
                print(f"Error loading {Path(kp_file).stem}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if batch_data:
            combined = pd.concat(batch_data, ignore_index=True)
            print(f"Combined batch size: {len(combined)} frames")
            return combined
        return None

    def prepare_data(self, all_data):
        """
        Prepare the raw data for sequence modeling.
        This processes the combined raw keypoint and cricket data.
        
        Args:
            all_data: DataFrame containing the merged raw data
            
        Returns:
            X: Feature matrix (pandas DataFrame)
            y_encoded: Encoded target labels (numpy array)
            groups: Trial grouping information (pandas Series)
            feature_names: List of feature column names
        """
        print("Preparing raw pose and cricket data...")
        
        # Define columns to exclude from features
        exclude_cols = ['frame', 'behavior', 'animal_id', 'trial_id']
        
        # Select all numeric columns as features
        numeric_cols = all_data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        print(f"Using {len(feature_cols)} raw features")
        
        X = all_data[feature_cols].copy()
        y = all_data['behavior'].copy()
        groups = all_data['trial_id'].copy()
        
        # Encode target labels
        y_encoded = self.target_encoder.fit_transform(y)
        print(f"Encoded labels: {dict(zip(self.target_encoder.classes_, range(len(self.target_encoder.classes_))))}")
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Scale numeric features
        X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
        
        return X, y_encoded, groups, feature_cols
    
    def _handle_missing_values(self, X):
        """Fill missing values in raw data"""
        # For raw pose data, it's often better to fill NaNs with 0
        # as this often means "not detected" or "not visible"
        return X.fillna(0)

    def find_matching_files(self, keypoint_dir, cricket_dir, label_dir):
        """
        Find matching sets of keypoint, cricket, and label files based on animal ID and trial ID.
        
        File naming patterns:
        - Keypoint files: filtered_m14_t1.csv
        - Cricket files: crprocessed_m14_t3.csv
        - Label files: m20_t5_processed_labels.csv
        
        Args:
            keypoint_dir: Directory containing keypoint CSV files
            cricket_dir: Directory containing cricket CSV files
            label_dir: Directory containing behavior label CSV files
            
        Returns:
            Tuple of (keypoint_files, cricket_files, label_files)
        """
        import re
        keypoint_files = []
        cricket_files = []
        label_files = []
        
        # Convert to Path objects if they're strings
        keypoint_dir = Path(keypoint_dir)
        cricket_dir = Path(cricket_dir)
        label_dir = Path(label_dir)
        
        # First, index all available files by their animal_id and trial_id
        cricket_dict = {}
        label_dict = {}
        
        # Build dictionary of cricket files
        for cricket_file in cricket_dir.glob('crprocessed_*.csv'):
            # Extract animal_id and trial_id using regex
            match = re.search(r'crprocessed_(m\d+)_(t\d+)\.csv', cricket_file.name)
            if match:
                animal_id, trial_id = match.groups()
                key = f"{animal_id}_{trial_id}"
                cricket_dict[key] = cricket_file
        
        # Build dictionary of label files
        for label_file in label_dir.glob('*.csv'):
            # Extract animal_id and trial_id using regex
            match = re.search(r'(m\d+)_(t\d+)_processed_labels\.csv', label_file.name)
            if match:
                animal_id, trial_id = match.groups()
                key = f"{animal_id}_{trial_id}"
                label_dict[key] = label_file
        
        # Now find matches for each keypoint file
        for kp_file in keypoint_dir.glob('filtered_*.csv'):
            # Extract animal_id and trial_id from keypoint filename
            match = re.search(r'filtered_(m\d+)_(t\d+)\.csv', kp_file.name)
            if match:
                animal_id, trial_id = match.groups()
                key = f"{animal_id}_{trial_id}"
                
                # Check if matching cricket and label files exist
                if key in cricket_dict and key in label_dict:
                    keypoint_files.append(str(kp_file))
                    cricket_files.append(str(cricket_dict[key]))
                    label_files.append(str(label_dict[key]))
                    print(f"Found matching set for {key}")
        
        print(f"Found {len(keypoint_files)} complete matching file sets")
        return keypoint_files, cricket_files, label_files
