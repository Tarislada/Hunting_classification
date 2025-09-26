import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import gc
import psutil
import os
from pathlib import Path
import warnings

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False

warnings.filterwarnings('ignore')

class MemoryOptimizedFeatureEngineer:
    # ... (Copy the entire MemoryOptimizedFeatureEngineer class here) ...
    # ... (from line 39 to line 258 in your action_segmentation.py) ...
    def __init__(self, window_sizes=[15, 30], max_features_per_type=3, 
                 sequential_lags=[1, 2, 3], sequential_features=None):
        """
        Memory-optimized feature engineering with sequential features
        Args:
            window_sizes: Reduced window sizes for rolling features
            max_features_per_type: Limit number of rolling statistics per feature
            sequential_lags: Number of previous frames to include (e.g., [1, 2, 3])
            sequential_features: List of features to include from previous frames
        """
        self.window_sizes = window_sizes
        self.max_features_per_type = max_features_per_type
        self.sequential_lags = sequential_lags
        
        # Default sequential features (most important for behavioral transitions)
        self.sequential_features = sequential_features or [
            'head_angle', 'cricket_angle', 'relative_angle', 
            'distance', 'cricket_in_binocular', 'is_cricket_visible'
        ]
        
    def optimize_dtypes(self, df):
        """Optimize pandas dtypes to reduce memory usage"""
        df = df.copy()
        
        # Convert to more memory-efficient types
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to category if few unique values
                if len(df[col].unique()) < len(df) * 0.5:
                    df[col] = df[col].astype('category')
            elif df[col].dtype == 'float64':
                # Downcast to float32 if possible
                df[col] = pd.to_numeric(df[col], downcast='float')
            elif df[col].dtype == 'int64':
                # Downcast to smaller int if possible
                df[col] = pd.to_numeric(df[col], downcast='integer')
        
        return df
    
    def engineer_essential_features(self, df):
        """Engineer only the most essential features to save memory"""
        df = df.copy()
        
        # Parse coordinates (only if needed)
        coord_features = []
        for coord_col in ['tail_base', 'body_center', 'nose']:
            if coord_col in df.columns and df[coord_col].dtype == 'object':
                try:
                    df[f'{coord_col}_x'] = df[coord_col].str.extract(r'\(([-\d.]+),').astype('float32')
                    df[f'{coord_col}_y'] = df[coord_col].str.extract(r',\s*([-\d.]+)\)').astype('float32')
                    coord_features.extend([f'{coord_col}_x', f'{coord_col}_y'])
                    df = df.drop(coord_col, axis=1)  # Remove original to save memory
                except:
                    print(f"Warning: Could not parse coordinates from {coord_col}")
        
        # Essential motion features only
        essential_cols = ['head_angle', 'cricket_angle', 'distance']
        for col in essential_cols:
            if col in df.columns:
                df[f'{col}_velocity'] = df[col].diff().astype('float32')
        
        # Essential behavioral features
        if 'cricket_angle' in df.columns and 'head_angle' in df.columns:
            df['relative_angle'] = (df['cricket_angle'] - df['head_angle']).astype('float32')
            # Also calculate velocity for relative_angle
            df['relative_angle_velocity'] = df['relative_angle'].diff().astype('float32')
        
        # --- NEW: Add acceleration features (2nd derivative) ---
        # These features spike during rapid changes in movement.
        acceleration_cols = ['head_angle_velocity', 'distance_velocity', 'relative_angle_velocity']
        for col in acceleration_cols:
            if col in df.columns:
                df[f'{col.replace("_velocity", "_acceleration")}'] = df[col].diff().astype('float32')

        # --- NEW: Add change-point detection features ---
        # These measure the magnitude of change over different time windows.
        change_point_cols = ['distance', 'head_angle', 'relative_angle']
        for col in change_point_cols:
            if col in df.columns:
                for window in [3, 5, 10]:
                    df[f'{col}_change_{window}'] = (df[col] - df[col].shift(window)).astype('float32')
        
        # --- NEW: Add zero-crossing / direction change features ---
        # This counts how "jerky" or "indecisive" the movement is.
        if 'relative_angle' in df.columns:
            df['direction_changes_10'] = (np.sign(df['relative_angle'].diff()).diff() != 0).rolling(10).sum().astype('float32')

        # Binary behavioral indicators
        if 'cricket_use_nose_position' in df.columns:
            df['is_cricket_visible'] = (~df['cricket_use_nose_position']).astype('int8')
        if 'zone' in df.columns:
            df['cricket_in_binocular'] = (df['zone'] == 'binocular').astype('int8')
        if PYWT_AVAILABLE:
            print("Adding wavelet features...")
            wavelet_cols = ['distance', 'relative_angle']
            wavelet = 'db4' # A common choice for transient signal analysis
            
            for col in wavelet_cols:
                if col in df.columns:
                    # Define a function to apply to each trial group
                    def get_wavelet_features(group):
                        series = group[col].fillna(0) # Fill NaNs within the group for transform
                        
                        # The transform requires a minimum length. Skip if too short.
                        if len(series) < 20: 
                            return group
                        
                        # Decompose the signal
                        try:
                            coeffs = pywt.wavedec(series, wavelet, level=3)
                            # cA3, cD3, cD2, cD1
                            
                            # Reconstruct features and align them with the original index
                            # Energy of detail coefficients (high and mid-frequency)
                            group[f'{col}_dwt_d1_energy'] = pd.Series(np.square(coeffs[-1])).rolling(15, min_periods=1).mean().reindex(group.index, method='bfill').astype('float32')
                            group[f'{col}_dwt_d2_energy'] = pd.Series(np.square(coeffs[-2])).rolling(15, min_periods=1).mean().reindex(group.index, method='bfill').astype('float32')
                        except ValueError as e:
                            # This can happen if a trial is too short for the chosen level
                            # print(f"Wavelet transform failed for a group in '{col}': {e}")
                            pass # Leave columns as NaN, they will be imputed later
                        return group

                    # Apply the function to each trial group
                    df = df.groupby(['animal_id', 'trial_id'], group_keys=False).apply(get_wavelet_features)
        else:
            print("Warning: PyWavelets is not installed. Skipping wavelet feature generation.")

        # NOTE: Lag features are NOT needed for sequence models, so they are removed here.
        
        # LIMITED rolling features (only most important)
        priority_features = ['head_angle', 'cricket_angle', 'distance', 'relative_angle']
        priority_stats = ['mean', 'std']  # Reduced from 5 to 2 statistics
        
        for window in self.window_sizes:
            for col in priority_features:
                if col in df.columns:
                    for stat in priority_stats:
                        if stat == 'mean':
                            df[f'{col}_{stat}_{window}'] = df[col].rolling(window, center=True, min_periods=1).mean().astype('float32')
                        elif stat == 'std':
                            df[f'{col}_{stat}_{window}'] = df[col].rolling(window, center=True, min_periods=1).std().astype('float32')
        
        # Optimize dtypes
        df = self.optimize_dtypes(df)
        
        return df
    
    def get_feature_names(self, df):
        """Get list of engineered feature names"""
        exclude_cols = [
            'frame', 'behavior', 'animal_id', 'trial_id',
            'cricket_status', 'validation'
        ]
        return [col for col in df.columns if col not in exclude_cols and '_lag_' not in col]

class MemoryOptimizedDataPreparator:
    # ... (Copy the entire MemoryOptimizedDataPreparator class here) ...
    # ... (from line 260 to line 450 in your action_segmentation.py) ...
    def __init__(self):
        self.scaler = StandardScaler()
        self.target_encoder = LabelEncoder()
        
    def process_files_in_batches(self, feature_files, label_files, batch_size=5):
        """Process files in batches to manage memory"""
        print(f"Processing {len(feature_files)} files in batches of {batch_size}")
        
        all_batches = []
        for i in range(0, len(feature_files), batch_size):
            batch_feature_files = feature_files[i:i+batch_size]
            batch_label_files = label_files[i:i+batch_size]
            
            print(f"Processing batch {i//batch_size + 1}/{(len(feature_files)-1)//batch_size + 1}")
            batch_data = self.load_and_align_batch(batch_feature_files, batch_label_files)
            
            if batch_data is not None:
                all_batches.append(batch_data)
            
            # Force garbage collection
            gc.collect()
            
            # Memory check
            memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
            print(f"Current memory usage: {memory_usage:.1f} MB")
        
        if all_batches:
            return pd.concat(all_batches, ignore_index=True)
        else:
            return None
    
    def load_and_align_batch(self, feature_files, label_files):
        """Load and align a batch of files"""
        batch_data = []
        
        for feat_file, label_file in zip(feature_files, label_files):
            try:
                # Extract animal ID from filename
                # animal_id = Path(feat_file).stem.split('_')[0]
                parts = Path(feat_file).stem.replace('_validated', '').replace('_analysis', '').split('_')
                animal_id = parts[0]
                trial_id = parts[1] if len(parts) > 1 else 'unknown'
                
                # Load data with memory optimization
                features = pd.read_csv(feat_file, dtype={'frame': 'int32'})
                labels = pd.read_csv(label_file, dtype={'frame': 'int32'})
                
                # Align by frame
                merged = pd.merge(features, labels, on='frame', how='inner')
                merged['animal_id'] = animal_id
                merged['trial_id'] = trial_id
                                
                # Basic cleaning
                merged = merged.dropna(subset=['behavior'])  # Remove rows without labels
                
                batch_data.append(merged)
                
            except Exception as e:
                print(f"Error loading {Path(feat_file).stem}: {e}")
                continue
        
        if batch_data:
            return pd.concat(batch_data, ignore_index=True)
        return None
    
    def prepare_data(self, data, feature_engineer):
        """Prepare data for sequence models (no lag features needed)."""
        print("Engineering features for sequence model...")
        data_engineered = feature_engineer.engineer_essential_features(data)
        
        feature_cols = feature_engineer.get_feature_names(data_engineered)
        print(f"Using {len(feature_cols)} features.")
        
        X = data_engineered[feature_cols].copy()
        y = data_engineered['behavior'].copy()
        groups = data_engineered['trial_id'].copy() # Use trial_id for sequence grouping
        
        # Encode target labels
        y_encoded = self.target_encoder.fit_transform(y)
        print(f"Encoded labels: {dict(zip(self.target_encoder.classes_, range(len(self.target_encoder.classes_))))}")
        
        # Handle missing values
        X = self._handle_missing_values_efficient(X)
        
        # Scale numeric features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
        
        return X, y_encoded, groups
    
    def _handle_missing_values_efficient(self, X):
        """Efficient missing value handling"""
        for col in X.columns:
            if X[col].dtype == 'category':
                if 'unknown' not in X[col].cat.categories:
                    X[col] = X[col].cat.add_categories(['unknown'])
                X[col] = X[col].fillna('unknown')
                X[col] = X[col].cat.codes.astype('int8')
            elif X[col].dtype == 'object':
                X[col] = X[col].fillna('unknown').astype('category').cat.codes.astype('int8')
            elif X[col].dtype == 'bool':
                X[col] = X[col].fillna(False).astype('int8')
            else:
                X[col] = X[col].fillna(0) # Fill with 0 for sequence models
        
        return X
