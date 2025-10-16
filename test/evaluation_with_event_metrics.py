import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import re
from pathlib import Path
from abc import ABC, abstractmethod
import sys

# FIX THE IMPORT PATH
sys.path.append(str(Path(__file__).parent.parent))

from processing.raw_data_loader import RawDataLoader

class FeatureEngineer(ABC):
    @abstractmethod
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class EnhancedFeatureEngineer(FeatureEngineer):
    """
    Comprehensive feature engineering for attack vs consume discrimination.
    Focuses on motion dynamics, prey-relative features, and temporal patterns.
    """
    
    def __init__(self):
        self.feature_names = []
    
    def parse_coordinates(self, df):
        """Parse coordinate strings into x,y columns."""
        coord_features = ['tail_base', 'body_center', 'nose']
        
        for coord_col in coord_features:
            if coord_col in df.columns and df[coord_col].dtype == 'object':
                try:
                    df[f'{coord_col}_x'] = df[coord_col].str.extract(r'\(([-\d.]+),').astype('float32')
                    df[f'{coord_col}_y'] = df[coord_col].str.extract(r',\s*([-\d.]+)\)').astype('float32')
                    print(f"  Parsed {coord_col} coordinates")
                except:
                    print(f"  Warning: Could not parse {coord_col}")
        
        return df
    
    def add_basic_derivatives(self, df):
        """Add velocity and acceleration features."""
        print("  Adding velocity and acceleration features...")
        
        # 1st derivatives (velocity)
        velocity_cols = ['head_angle', 'cricket_angle', 'distance', 'cricket_speed']
        for col in velocity_cols:
            if col in df.columns:
                df[f'{col}_velocity'] = df[col].diff().astype('float32')
        
        # Relative angle (very important for behavior)
        if 'cricket_angle' in df.columns and 'head_angle' in df.columns:
            df['relative_angle'] = (df['cricket_angle'] - df['head_angle']).astype('float32')
            df['relative_angle_velocity'] = df['relative_angle'].diff().astype('float32')
        
        # 2nd derivatives (acceleration)
        acceleration_base = ['head_angle_velocity', 'distance_velocity', 'relative_angle_velocity']
        for col in acceleration_base:
            if col in df.columns:
                new_col = col.replace('_velocity', '_acceleration')
                df[new_col] = df[col].diff().astype('float32')
        
        return df
    
    def add_jerk_features(self, df):
        """
        Add 3rd derivative (jerk) features - CRITICAL for attack detection.
        Attack has extreme jerk during strike, consume has low jerk.
        """
        print("  Adding jerk features (3rd derivatives)...")
        
        jerk_base = ['head_angle_acceleration', 'distance_acceleration', 'relative_angle_acceleration']
        
        for col in jerk_base:
            if col in df.columns:
                # Basic jerk
                jerk_col = col.replace('_acceleration', '_jerk')
                df[jerk_col] = df[col].diff().astype('float32')
                
                # Jerk magnitude (absolute value)
                df[f'{jerk_col}_magnitude'] = np.abs(df[jerk_col]).astype('float32')
                
                # Rolling max jerk (captures peak strike moment)
                df[f'{jerk_col}_max_3'] = df[f'{jerk_col}_magnitude'].rolling(3, min_periods=1).max().astype('float32')
                df[f'{jerk_col}_max_5'] = df[f'{jerk_col}_magnitude'].rolling(5, min_periods=1).max().astype('float32')
        
        return df
    
    def add_spike_detection(self, df):
        """
        Add spike detection features - binary flags for extreme motion.
        Helps model identify attack moments vs sustained consume behavior.
        """
        print("  Adding spike detection features...")
        
        spike_cols = [
            ('distance_velocity', 'velocity'),
            ('distance_acceleration', 'acceleration'),
            ('head_angle_velocity', 'angular_velocity'),
            ('head_angle_acceleration', 'angular_acceleration')
        ]
        
        for col, label in spike_cols:
            if col in df.columns:
                # Define spike as value > mean + 2*std in rolling window
                rolling_mean = df[col].rolling(10, min_periods=1).mean()
                rolling_std = df[col].rolling(10, min_periods=1).std()
                
                spike_col = f'{label}_spike'
                df[spike_col] = (df[col] > (rolling_mean + 2 * rolling_std)).astype('int8')
        
        # Combined spike (both velocity and acceleration spike together)
        if 'velocity_spike' in df.columns and 'acceleration_spike' in df.columns:
            df['combined_motion_spike'] = (df['velocity_spike'] * df['acceleration_spike']).astype('int8')
        
        # Count spikes in window (attack = 1 spike, consume = multiple rhythmic spikes)
        if 'velocity_spike' in df.columns:
            df['spike_count_10'] = df['velocity_spike'].rolling(10, min_periods=1).sum().astype('float32')
        
        return df
    
    def add_peak_analysis(self, df):
        """
        Detect and count peaks in motion. 
        Consume has rhythmic peaks (chewing), attack has single dominant peak.
        """
        print("  Adding peak analysis features...")
        
        peak_cols = ['distance_velocity', 'distance_acceleration']
        
        for col in peak_cols:
            if col in df.columns:
                # Local maximum detection
                is_peak = (df[col] > df[col].shift(1)) & (df[col] > df[col].shift(-1))
                peak_col = f'{col}_is_peak'
                df[peak_col] = is_peak.astype('int8')
                
                # Count peaks in window
                df[f'{col}_peaks_10'] = df[peak_col].rolling(10, min_periods=1).sum().astype('float32')
        
        return df
    
    def add_prey_relative_features(self, df):
        """
        Add prey-relative features - MOST DISCRIMINATIVE for attack vs consume.
        Attack: rapid approach, high alignment, short duration
        Consume: stable at prey, less alignment critical, long duration
        """
        print("  Adding prey-relative features...")
        
        # We have distance and cricket_angle already, which gives us relative position
        
        if 'distance' in df.columns:
            # Rate of distance change (approach speed)
            if 'distance_velocity' not in df.columns:
                df['distance_velocity'] = df['distance'].diff().astype('float32')
            
            # Binary: actively approaching prey (distance decreasing rapidly)
            df['approaching_prey'] = (df['distance_velocity'] < -1.0).astype('int8')
            
            # Binary: at prey location (very close distance)
            distance_threshold = df['distance'].quantile(0.1)  # Bottom 10% of distances
            df['at_prey_location'] = (df['distance'] < distance_threshold).astype('int8')
            
            # Time spent at prey location (cumulative in window)
            df['time_at_prey_10'] = df['at_prey_location'].rolling(10, min_periods=1).sum().astype('float32')
            df['time_at_prey_20'] = df['at_prey_location'].rolling(20, min_periods=1).sum().astype('float32')
            
            # Rapid approach detection (extreme approach speed)
            distance_vel_std = df['distance_velocity'].rolling(20, min_periods=1).std()
            df['rapid_approach'] = (df['distance_velocity'] < -2 * distance_vel_std).astype('int8')
        
        # Head-prey alignment (how aligned is head angle with cricket angle)
        if 'head_angle' in df.columns and 'cricket_angle' in df.columns:
            if 'relative_angle' not in df.columns:
                df['relative_angle'] = (df['cricket_angle'] - df['head_angle']).astype('float32')
            
            # Alignment quality (lower absolute relative angle = better aligned)
            df['prey_alignment'] = np.abs(df['relative_angle']).astype('float32')
            df['well_aligned'] = (df['prey_alignment'] < 15).astype('int8')  # Within 15 degrees
        
        return df
    
    def add_stability_features(self, df):
        """
        Measure behavior stability/consistency.
        Consume is stable/stationary, attack is unstable/directed motion.
        """
        print("  Adding stability features...")
        
        # Direction stability (how consistent is heading)
        if 'head_angle' in df.columns:
            df['direction_stability'] = df['head_angle'].rolling(10, min_periods=1).std().astype('float32')
        
        # Position stability (how much movement)
        if 'nose_x' in df.columns and 'nose_y' in df.columns:
            nose_displacement = np.sqrt(df['nose_x'].diff()**2 + df['nose_y'].diff()**2)
            df['position_stability'] = nose_displacement.rolling(10, min_periods=1).std().astype('float32')
        
        # Direction changes (jerky behavior indicator)
        if 'relative_angle' in df.columns:
            sign_changes = (np.sign(df['relative_angle'].diff()).diff() != 0)
            df['direction_changes_10'] = sign_changes.rolling(10, min_periods=1).sum().astype('float32')
        
        return df
    
    def add_change_point_features(self, df):
        """
        Detect change points over different time scales.
        Helps identify behavior transitions.
        """
        print("  Adding change-point detection features...")
        
        change_cols = ['distance', 'head_angle', 'relative_angle']
        windows = [3, 5, 10]
        
        for col in change_cols:
            if col in df.columns:
                for window in windows:
                    change_col = f'{col}_change_{window}'
                    df[change_col] = (df[col] - df[col].shift(window)).astype('float32')
        
        return df
    
    def add_rolling_statistics(self, df):
        """
        Add essential rolling statistics for temporal context.
        Reduced set focusing on most important features.
        """
        print("  Adding rolling statistics...")
        
        priority_features = ['head_angle', 'cricket_angle', 'distance', 'relative_angle']
        windows = [10, 20]
        stats = ['mean', 'std']
        
        for col in priority_features:
            if col in df.columns:
                for window in windows:
                    for stat in stats:
                        stat_col = f'{col}_{stat}_{window}'
                        if stat == 'mean':
                            df[stat_col] = df[col].rolling(window, min_periods=1).mean().astype('float32')
                        elif stat == 'std':
                            df[stat_col] = df[col].rolling(window, min_periods=1).std().astype('float32')
        
        return df
    
    def add_behavioral_indicators(self, df):
        """Add binary behavioral state indicators."""
        print("  Adding behavioral indicators...")
        
        # Cricket visibility
        if 'cricket_use_nose_position' in df.columns:
            df['is_cricket_visible'] = (~df['cricket_use_nose_position']).astype('int8')
        
        # Binocular zone (important for attack targeting)
        if 'zone' in df.columns:
            df['cricket_in_binocular'] = (df['zone'] == 'binocular').astype('int8')
        
        return df

    def load_raw_data_for_xgboost(keypoint_dir: Path, cricket_dir: Path, label_dir: Path) -> Tuple[List[str], List[str]]:
        """
        Load raw pose + cricket data and return as fake 'feature files' and 'label files'
        that can be processed by the existing pipeline.
        
        Returns lists of file paths that will be used to load data.
        """
        print("\n" + "="*60)
        print("LOADING RAW POSE DATA FOR XGBOOST")
        print("="*60)
        
        raw_loader = RawDataLoader()
        
        # Find matching files
        keypoint_files, cricket_files, label_files = raw_loader.find_matching_files(
            keypoint_dir, cricket_dir, label_dir
        )
        
        print(f"Found {len(keypoint_files)} matching file sets")
        
        # We'll return the label files as-is, but we need to process the keypoint+cricket data
        # and save it to temporary CSV files that look like "feature files"
        
        from pathlib import Path
        import tempfile
        
        # Create a temporary directory to store processed raw features
        temp_dir = Path(tempfile.mkdtemp(prefix="xgb_raw_features_"))
        print(f"Temporary feature directory: {temp_dir}")
        
        processed_feature_files = []
        processed_label_files = []
        
        for kp_file, cr_file, lb_file in zip(keypoint_files, cricket_files, label_files):
            try:
                # Extract trial ID
                base_name = Path(kp_file).stem
                match = re.search(r'(m\d+_t\d+)', base_name)
                if not match:
                    print(f"Warning: Could not extract trial ID from {base_name}")
                    continue
                trial_id = match.group(1)
                
                # Load raw data for this trial
                keypoint_df = pd.read_csv(kp_file)
                cricket_df = pd.read_csv(cr_file)
                label_df = pd.read_csv(lb_file)
                
                # Merge keypoint and cricket data on frame
                merged = pd.merge(keypoint_df, cricket_df, on='frame', how='inner')
                
                # Add basic computed features that XGBoost needs
                # (distance, angles, etc. - the "raw" features before rolling windows/lags)
                if 'nose_x' in merged.columns and 'nose_y' in merged.columns:
                    if 'cricket_x' in merged.columns and 'cricket_y' in merged.columns:
                        # Calculate distance
                        merged['distance'] = np.sqrt(
                            (merged['nose_x'] - merged['cricket_x'])**2 + 
                            (merged['nose_y'] - merged['cricket_y'])**2
                        )
                        
                        # Calculate cricket angle relative to nose
                        merged['cricket_angle'] = np.arctan2(
                            merged['cricket_y'] - merged['nose_y'],
                            merged['cricket_x'] - merged['nose_x']
                        ) * 180 / np.pi
                
                # Calculate head angle if we have nose and body center
                if 'nose_x' in merged.columns and 'body_center_x' in merged.columns:
                    merged['head_angle'] = np.arctan2(
                        merged['nose_y'] - merged['body_center_y'],
                        merged['nose_x'] - merged['body_center_x']
                    ) * 180 / np.pi
                
                # Save to temporary CSV
                temp_feature_file = temp_dir / f"{trial_id}_raw_features.csv"
                merged.to_csv(temp_feature_file, index=False)
                
                processed_feature_files.append(str(temp_feature_file))
                processed_label_files.append(lb_file)
                
                print(f"  Processed {trial_id}: {len(merged)} frames")
                
            except Exception as e:
                print(f"Error processing {Path(kp_file).stem}: {e}")
                continue
        
        print(f"\nSuccessfully processed {len(processed_feature_files)} trials")
        return processed_feature_files, processed_label_files

    def engineer_features(self, df):
        """Main feature engineering pipeline."""
        print("\nEngineering features...")
        
        df = df.copy()
        
        # Step 1: Parse coordinates
        df = self.parse_coordinates(df)
        
        # Step 2: Basic derivatives
        df = self.add_basic_derivatives(df)
        
        # Step 3: HIGH PRIORITY - Jerk features
        df = self.add_jerk_features(df)
        
        # Step 4: HIGH PRIORITY - Spike detection
        df = self.add_spike_detection(df)
        
        # Step 5: Peak analysis
        df = self.add_peak_analysis(df)
        
        # Step 6: HIGH PRIORITY - Prey-relative features
        df = self.add_prey_relative_features(df)
        
        # Step 7: Stability features
        df = self.add_stability_features(df)
        
        # Step 8: Change-point detection
        df = self.add_change_point_features(df)
        
        # Step 9: Rolling statistics
        df = self.add_rolling_statistics(df)
        
        # Step 10: Behavioral indicators
        df = self.add_behavioral_indicators(df)
        
        print("Feature engineering complete!")
        
        return df
class MinimalFeatureEngineer(FeatureEngineer):
    """
    Minimal feature engineering for raw pose data.
    Only includes basic computations, NO temporal features (lags, rolling windows).
    """
    
    def __init__(self):
        self.feature_names = []
    
    def parse_coordinates(self, df):
        """Parse coordinate strings if needed."""
        coord_features = ['tail_base', 'body_center', 'nose']
        
        for coord_col in coord_features:
            if coord_col in df.columns and df[coord_col].dtype == 'object':
                try:
                    df[f'{coord_col}_x'] = df[coord_col].str.extract(r'\(([-\d.]+),').astype('float32')
                    df[f'{coord_col}_y'] = df[coord_col].str.extract(r',\s*([-\d.]+)\)').astype('float32')
                except:
                    pass
        
        return df
    
    def add_basic_kinematics(self, df):
        """Add instantaneous velocity only (no rolling windows or lags)."""
        
        # Distance velocity (approach speed)
        if 'distance' in df.columns:
            df['distance_velocity'] = df['distance'].diff().astype('float32')
        
        # Angle velocities
        if 'head_angle' in df.columns:
            df['head_angle_velocity'] = df['head_angle'].diff().astype('float32')
        
        if 'cricket_angle' in df.columns:
            df['cricket_angle_velocity'] = df['cricket_angle'].diff().astype('float32')
        
        # Relative angle
        if 'cricket_angle' in df.columns and 'head_angle' in df.columns:
            df['relative_angle'] = (df['cricket_angle'] - df['head_angle']).astype('float32')
            df['relative_angle_velocity'] = df['relative_angle'].diff().astype('float32')
        
        return df
    
    def engineer_features(self, df):
        """Main pipeline - ONLY basic features, no temporal engineering."""
        print("  Applying minimal feature engineering (raw features only)...")
        
        df = df.copy()
        df = self.parse_coordinates(df)
        df = self.add_basic_kinematics(df)
        
        return df        
def plot_prediction_bands(y_true, y_pred, encoder, title, save_dir):
    """
    Generates a band-style plot comparing true and predicted labels for a trial.
    """
    import matplotlib.patches as mpatches

    # Define a consistent color map with publication-quality labels
    color_map = {
        'attack': '#66c2a5',      # Teal
        'background': '#8da0cb',  # Blue
        'chasing': '#e78ac3',      # Pink/Purple
        'consume': '#ffd92f'      # Yellow
    }
    
    # Display labels for publication (present continuous tense)
    display_labels = {
        'attack': 'Attacking',
        'background': 'Other',
        'chasing': 'Chasing',
        'consume': 'Consuming'
    }
    
    # Get integer labels for the colors
    int_to_color = {encoder.transform([name])[0]: color for name, color in color_map.items() if name in encoder.classes_}
    
    # Create a 2xN array representing the two bands
    num_frames = len(y_true)
    bands = np.zeros((2, num_frames))
    bands[0, :] = y_true  # True
    bands[1, :] = y_pred  # Predicted
    
    # Create a colormap from our dictionary
    cmap_colors = [int_to_color.get(i, '#ffffff') for i in range(len(encoder.classes_))]
    cmap = plt.cm.colors.ListedColormap(cmap_colors)
    
    fig, ax = plt.subplots(figsize=(15, 2.5))
    
    # Display the bands
    ax.imshow(bands, cmap=cmap, aspect='auto', interpolation='nearest')
    
    # Configure plot aesthetics
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['True', 'Predicted'])
    ax.set_xlabel('Frame Number')
    ax.set_title(title)
    
    # Create a custom legend with display labels
    legend_patches = [mpatches.Patch(color=color, label=display_labels.get(name, name)) 
                     for name, color in color_map.items() if name in encoder.classes_]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.01, 1), loc='upper left')
    
    # Ensure the save directory exists
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{title.replace(' ', '_')}.png"
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved prediction band plot to {save_path}")
    
def plot_classification_metrics_bar(behavior_summaries: Dict, experiment_name: str, save_dir: Path) -> None:
    """
    Create a grouped bar chart showing precision, recall, and F1-score for each behavior class.
    """
    print("\nGenerating classification metrics bar chart...")
    
    # Filter to only the behaviors we care about
    behaviors_of_interest = ['chasing', 'attack', 'consume']
    
    # Display labels for publication
    display_labels = {
        'attack': 'Attacking',
        'chasing': 'Chasing',
        'consume': 'Consuming'
    }
    
    # Prepare data for plotting
    behaviors = []
    precision_means = []
    precision_stds = []
    recall_means = []
    recall_stds = []
    f1_means = []
    f1_stds = []
    
    for behavior in behaviors_of_interest:
        if behavior in behavior_summaries:
            behaviors.append(display_labels.get(behavior, behavior))  # Use display label
            precision_means.append(behavior_summaries[behavior]['precision_mean'])
            precision_stds.append(behavior_summaries[behavior]['precision_std'])
            recall_means.append(behavior_summaries[behavior]['recall_mean'])
            recall_stds.append(behavior_summaries[behavior]['recall_std'])
            f1_means.append(behavior_summaries[behavior]['f1_mean'])
            f1_stds.append(behavior_summaries[behavior]['f1_std'])
    
    if not behaviors:
        print("Warning: No behaviors found for plotting.")
        return
    
    # Set up the plot
    x = np.arange(len(behaviors))
    width = 0.25  # Width of each bar
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create the grouped bars
    bars1 = ax.bar(x - width, precision_means, width, yerr=precision_stds, 
                   label='Precision', capsize=5, color='#66c2a5', alpha=0.8)
    bars2 = ax.bar(x, recall_means, width, yerr=recall_stds,
                   label='Recall', capsize=5, color='#fc8d62', alpha=0.8)
    bars3 = ax.bar(x + width, f1_means, width, yerr=f1_stds,
                   label='F1-Score', capsize=5, color='#8da0cb', alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Behavior Class', fontsize=14, weight='bold')
    ax.set_ylabel('Score', fontsize=14, weight='bold')
    ax.set_title(f'Classification Performance by Behavior\n{experiment_name}', 
                 fontsize=16, weight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(behaviors, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    def add_value_labels(bars, values, stds):
        for bar, val, std in zip(bars, values, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                   f'{val:.2f}',
                   ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars1, precision_means, precision_stds)
    add_value_labels(bars2, recall_means, recall_stds)
    add_value_labels(bars3, f1_means, f1_stds)
    
    plt.tight_layout()
    
    # Save the plot
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"classification_metrics_{experiment_name.replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved classification metrics bar chart to: {save_path}")
def plot_feature_importance(model, feature_cols, top_n=20, save_dir=None, experiment_name=""):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_cols: List of feature column names
        top_n: Number of top features to display
        save_dir: Directory to save the plot
        experiment_name: Name for the plot file
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return None
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Get top N features
    top_features = importance_df.head(top_n)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
    plt.title(f'Top {top_n} Feature Importances - {experiment_name}')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    # Save if directory provided
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"feature_importance_{experiment_name.replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature importance plot to: {save_path}")
    
    plt.show()
    
    # Print top features
    print(f"\n{'='*60}")
    print(f"TOP {top_n} FEATURE IMPORTANCES - {experiment_name}")
    print('='*60)
    print(f"{'Rank':<6} {'Feature':<40} {'Importance':>12}")
    print("-"*60)
    for idx, row in enumerate(top_features.itertuples(), 1):
        print(f"{idx:<6} {row.feature:<40} {row.importance:>12.6f}")
    
    return importance_df

def run_experiment(feature_files: List[str], label_files: List[str], 
                  experiment_name: str = "Experiment",
                  use_smote: bool = True,
                  hyperparam_search: bool = True,
                  use_thresholding: bool = False,
                  feature_engineer= None # Pass the feature engineer
):
    """
    Run experiment with comprehensive feature engineering.
    SIMPLIFIED: Frame-level metrics only, no event metrics.
    """
    import re
    
    print(f"\n{'='*60}")
    print(f"{experiment_name}")
    print('='*60)
    
    # Initialize feature engineer
    # feature_engineer = EnhancedFeatureEngineer()
    if feature_engineer is None:
        feature_engineer = EnhancedFeatureEngineer()
    
    # Load and combine all data
    all_data = []
    for feat_file, label_file in zip(feature_files, label_files):
        try:
            base_name = Path(feat_file).stem.replace('_enhanced_features', '')
            match = re.search(r'(m\d+)', Path(feat_file).stem)
            if not match:
                print(f"Warning: Could not find animal ID in {Path(feat_file).stem}. Skipping file.")
                continue
            animal_id = match.group(1)
            trial_id = base_name # e.g., 'm14_t1'

            features = pd.read_csv(feat_file)
            labels = pd.read_csv(label_file)
            
            merged = pd.merge(features, labels, on='frame', how='inner')
            merged['animal_id'] = animal_id
            merged['trial_id'] = trial_id # <-- ADD TRIAL ID
            all_data.append(merged)
            
            print(f"Loaded {animal_id}: {len(merged)} frames")
            
        except Exception as e:
            print(f"Error loading files: {e}")
            continue
    
    if not all_data:
        print("No data loaded!")
        return None
    
    data = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal dataset: {len(data)} frames from {len(all_data)} files")
    
    # Print class distribution
    print("\nClass distribution:")
    behavior_counts = data['behavior'].value_counts()
    for behavior, count in behavior_counts.items():
        print(f"  {behavior}: {count} ({count/len(data)*100:.1f}%)")
    
    # Engineer features
    data = feature_engineer.engineer_features(data)
    
    # Prepare features and labels
    exclude_cols = ['frame', 'behavior', 'animal_id', 'trial_id', 'cricket_status', 'validation', 
                   'zone', 'tail_base', 'body_center', 'nose', 'cricket_use_nose_position','status','use_nose_position','prolonged_immobility']
    feature_cols = [col for col in data.columns if col not in exclude_cols and not col.startswith('Unnamed')]
    
    # Handle missing values
    data[feature_cols] = data[feature_cols].fillna(data[feature_cols].median())
    
    X = data[feature_cols].values
    
    # Encode labels
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(data['behavior'])
    groups = data['animal_id'].values
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Using {len(feature_cols)} features")
    print(f"Encoded behaviors: {dict(zip(target_encoder.classes_, range(len(target_encoder.classes_))))}")
    print(f"SMOTE: {'Enabled' if use_smote else 'Disabled'}")
    print(f"Hyperparameter search: {'Enabled' if hyperparam_search else 'Disabled'}")
    
    # Cross-validation setup
    try: 
        cv = GroupKFold(n_splits=4)
        cv_splits = list(cv.split(X, y, groups))
    except:
        cv = GroupKFold(n_splits=3)
        cv_splits = list(cv.split(X, y, groups))
    
    # Storage for CV results
    frame_scores = []
    detailed_fold_results = []
    all_y_test_agg = []
    all_y_pred_agg = []
    final_model = None

    print(f"\nRunning 4-fold cross-validation...")

    for fold, (train_idx, test_idx) in enumerate(cv_splits):
        print(f"\n{'='*40}")
        print(f"Fold {fold + 1}/4")
        print('='*40)
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Get test group for reporting
        test_groups = groups[test_idx]
        unique_test_animals = np.unique(test_groups)
        print(f"Testing on: {', '.join(unique_test_animals)}")
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Apply SMOTE if requested
        if use_smote:
            print(f"Applying SMOTE to training data...")
            original_counts = pd.Series(y_train).value_counts().to_dict()
            print(f"Original counts: {original_counts}")
            
            # Define SMOTE strategy
            sampling_strategy = {}
            behavior_to_id = {behavior: target_encoder.transform([behavior])[0] 
                            for behavior in target_encoder.classes_}
            
            for behavior, behavior_id in behavior_to_id.items():
                if behavior_id in original_counts:
                    original_count = original_counts[behavior_id]
                    
                    if behavior == 'attack':
                        # Aggressive boost for attack
                        sampling_strategy[behavior_id] = int(original_count * 1.1)
                    elif behavior == 'consume':
                        # Moderate boost for consume
                        sampling_strategy[behavior_id] = int(original_count * 1.05)
                    elif behavior == 'chasing':
                        # Slight boost for chasing
                        sampling_strategy[behavior_id] = int(original_count * 1.05)
            
            print(f"SMOTE target strategy: {sampling_strategy}")
            
            try:
                smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
                X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
                print(f"After SMOTE: {X_train_scaled.shape[0]} training samples")
                print(f"New counts: {pd.Series(y_train).value_counts().to_dict()}")
            except Exception as e:
                print(f"SMOTE failed: {e}. Continuing without SMOTE for this fold.")
        
        # Calculate sample weights
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        
        # Extra weight for attack
        # attack_id = target_encoder.transform(['attack'])[0]
        class_weights_dict = dict(zip(classes, class_weights))
        # if attack_id in class_weights_dict:
        #     class_weights_dict[attack_id] *= 1.5  # Extra 50% weight for attack
        
        class_weights = np.array([class_weights_dict[label] for label in y_train])
        
        # Model selection
        if hyperparam_search:
            print(f"Performing hyperparameter search...")
            param_distributions = {
                'n_estimators': [150, 200],
                'max_depth': [4, 5, 6],
                'learning_rate': [0.075, 0.1],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.5, 0.6, 0.7],
                'gamma': [0, 0.1, 0.2],
                'reg_alpha': [0.05, 0.1],
                'reg_lambda': [1.5, 2.0]
            }
            
            base_model = XGBClassifier(
                random_state=42,
                eval_metric='mlogloss'
            )
            
            search = RandomizedSearchCV(
                base_model,
                param_distributions,
                n_iter=15,
                cv=2,
                scoring='f1_macro',
                random_state=42,
                n_jobs=4
            )
            search.fit(X_train_scaled, y_train, sample_weight=class_weights)
            model = search.best_estimator_
            print(f"Best params: {search.best_params_}")
        else:
            model = XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.6,
                random_state=42,
                eval_metric='mlogloss'
            )
            model.fit(X_train_scaled, y_train, sample_weight=class_weights)
        # Store the final fold's model for feature importance
        if fold == len(cv_splits) - 1:
            final_model = model

        if use_thresholding:
            print("Predicting with custom threshold to balance precision/recall...")
            # 1. Get class probabilities instead of direct predictions
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # 2. Get the default prediction (the class with the highest probability)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # 3. Apply a custom threshold to increase precision for 'attack'
            if 'attack' in target_encoder.classes_:
                attack_label = target_encoder.transform(['attack'])[0]
                
                # THIS IS YOUR TUNABLE PARAMETER. Increase it to boost precision.
                # ATTACK_THRESHOLD = 0.58
                ATTACK_THRESHOLD = 0.595 
                
                # Find where the model initially predicted 'attack'
                attack_predictions_mask = (y_pred == attack_label)
                
                # Find where the model was "not confident enough" in its attack prediction
                low_confidence_mask = (y_pred_proba[:, attack_label] < ATTACK_THRESHOLD)
                
                # Identify frames to change: predicted 'attack' BUT with low confidence
                change_mask = attack_predictions_mask & low_confidence_mask
                
                if np.any(change_mask):
                    print(f"Reverting {np.sum(change_mask)} low-confidence 'attack' predictions...")
                    # Get the probabilities for the frames to change
                    probs_to_change = y_pred_proba[change_mask]
                    # Temporarily set the probability of the 'attack' class to 0
                    probs_to_change[:, attack_label] = 0
                    # Find the new best class (which is now the original second choice)
                    new_predictions = np.argmax(probs_to_change, axis=1)
                    # Apply the changes back to the main prediction array
                    y_pred[change_mask] = new_predictions
        else:
            # Standard prediction if thresholding is off
            y_pred = model.predict(X_test_scaled)

        test_trial_ids = data['trial_id'].iloc[test_idx].values
        unique_test_trials = np.unique(test_trial_ids)
        
        print(f"\nGenerating prediction plots for {len(unique_test_trials)} trial(s)...")
        for trial in unique_test_trials:
            # Get the mask for the current trial
            trial_mask = (test_trial_ids == trial)
            
            # Filter the true and predicted labels for this trial
            y_true_trial = y_test[trial_mask]
            y_pred_trial = y_pred[trial_mask]
            
            # Plot the bands for this specific trial
            plot_title = f"Prediction_vs_True_for_{trial}"
            save_directory = Path("/home/tarislada/Documents/Extra_python_projects/SKH FP/FInalized_process/prediction_plots/prediction_plots") / experiment_name.replace(" ", "_")
            plot_prediction_bands(y_true_trial, y_pred_trial, target_encoder, plot_title, save_directory)
            trial_df = data.iloc[test_idx][data['trial_id'].iloc[test_idx] == trial]
            frame_numbers = trial_df['frame'].values

            # Convert predicted integer labels back to string labels
            predicted_labels_str = target_encoder.inverse_transform(y_pred_trial)

            # Create the analysis DataFrame in the format TimingValidator expects
            analysis_df = pd.DataFrame({
                'frame': frame_numbers,
                'behavior': predicted_labels_str
            })

            # Define the output directory for these analysis files
            analysis_save_dir = Path("/home/tarislada/Documents/Extra_python_projects/SKH FP/FInalized_process/prediction_plots/timing_analysis_files") / experiment_name.replace(" ", "_")
            analysis_save_dir.mkdir(parents=True, exist_ok=True)

            # Save the file with the expected name format (e.g., "m14_t1_analysis.csv")
            analysis_file_path = analysis_save_dir / f"{trial}_analysis.csv"
            analysis_df.to_csv(analysis_file_path, index=False)
            print(f"  Saved timing analysis file to: {analysis_file_path}")

        all_y_test_agg.append(y_test)
        all_y_pred_agg.append(y_pred)
        
        # Calculate frame-level metrics
        frame_f1 = f1_score(y_test, y_pred, average='macro')
        frame_scores.append(frame_f1)
        
        # Per-behavior metrics
        frame_report = classification_report(y_test, y_pred, 
                                            target_names=target_encoder.classes_,
                                            output_dict=True,
                                            zero_division=0)
        
        # Print fold results
        print(f"\nFold {fold + 1} Results:")
        print(classification_report(y_test, y_pred, 
                                    target_names=target_encoder.classes_,
                                    zero_division=0))
        print(f"  Macro F1: {frame_f1:.3f}")
        
        detailed_fold_results.append({
            'fold': fold + 1,
            'frame_f1': frame_f1,
            'frame_report': frame_report
        })
    
    # Final CV results
    print(f"\n{'='*60}")
    print("FINAL CROSS-VALIDATION RESULTS")
    print('='*60)
    
    # Display labels for publication
    display_labels = {
        'attack': 'Attacking',
        'background': 'Other',
        'chasing': 'Chasing',
        'consume': 'Consuming'
    }
    
    # Aggregated confusion matrix
    y_true_all_folds = np.concatenate(all_y_test_agg)
    y_pred_all_folds = np.concatenate(all_y_pred_agg)
    
    cm = confusion_matrix(y_true_all_folds, y_pred_all_folds, normalize='true')
    
    # Convert class names to display labels for confusion matrix
    display_class_names = [display_labels.get(cls, cls) for cls in target_encoder.classes_]
    cm_df = pd.DataFrame(cm, index=display_class_names, columns=display_class_names)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Aggregated Cross-Validation Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    # Overall metrics
    cv_frame_f1_mean = np.mean(frame_scores)
    cv_frame_f1_std = np.std(frame_scores)
    
    print(f"\nOverall Performance:")
    print(f"  Frame-level Macro F1: {cv_frame_f1_mean:.3f} ± {cv_frame_f1_std:.3f}")
    
    # Per-behavior CV metrics
    print(f"\nPer-Behavior CV Performance:")
    print(f"{'Behavior':<20} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
    print("-" * 56)
    
    behavior_summaries = {}
    for behavior in target_encoder.classes_:
        # Get metrics for this behavior across folds
        behavior_precisions = [r['frame_report'][behavior]['precision'] for r in detailed_fold_results]
        behavior_recalls = [r['frame_report'][behavior]['recall'] for r in detailed_fold_results]
        behavior_f1s = [r['frame_report'][behavior]['f1-score'] for r in detailed_fold_results]
        
        p_mean, p_std = np.mean(behavior_precisions), np.std(behavior_precisions)
        r_mean, r_std = np.mean(behavior_recalls), np.std(behavior_recalls)
        f1_mean, f1_std = np.mean(behavior_f1s), np.std(behavior_f1s)
        
        print(f"{behavior:<20} {p_mean:.3f}±{p_std:.3f}  {r_mean:.3f}±{r_std:.3f}  {f1_mean:.3f}±{f1_std:.3f}")
        
        behavior_summaries[behavior] = {
            'precision_mean': p_mean,
            'precision_std': p_std,
            'recall_mean': r_mean,
            'recall_std': r_std,
            'f1_mean': f1_mean,
            'f1_std': f1_std
        }
    
    print("\n" + "="*60)
    save_dir = Path("/home/tarislada/Documents/Extra_python_projects/SKH FP/FInalized_process/prediction_plots/metrics_plots")
    plot_classification_metrics_bar(behavior_summaries, experiment_name, save_dir)
    
    if final_model is not None:
        feature_importance_df = plot_feature_importance(
            final_model, 
            feature_cols, 
            top_n=30,
            save_dir=save_dir,
            experiment_name=experiment_name
        )
        print(f"Feature importance analysis complete. Saved at {save_dir} directory.")
    else:
        feature_importance_df = None
        print("Warning: No model available for feature importance analysis")

    return {
        'cv_frame_f1_mean': cv_frame_f1_mean,
        'cv_frame_f1_std': cv_frame_f1_std,
        'behavior_summaries': behavior_summaries,
        'detailed_folds': detailed_fold_results,
        'feature_cols': feature_cols,
        'feature_importance': feature_importance_df  # NEW: Add to return dict
    }

if __name__ == "__main__":
    # --- CONFIGURATION ---
    FEATURE_TYPE = 'engineered'  # Options: 'raw' or 'engineered'
    USE_SMOTE = True
    HYPERPARAM_SEARCH = True
    USE_THRESHOLDING = True
    EXPERIMENT_NAME = f"XGBoost_{FEATURE_TYPE.capitalize()}_Features"
    
    print(f"\n{'='*80}")
    print(f"CONFIGURATION")
    print('='*80)
    print(f"Feature Type: {FEATURE_TYPE}")
    print(f"Use SMOTE: {USE_SMOTE}")
    print(f"Hyperparameter Search: {HYPERPARAM_SEARCH}")
    print(f"Use Thresholding: {USE_THRESHOLDING}")
    print(f"Experiment Name: {EXPERIMENT_NAME}")
    print('='*80)

    # Update paths
    feature_dir = Path("/home/tarislada/Documents/Extra_python_projects/SKH FP/FInalized_process/enhanced_3_3")
    label_dir = Path("/home/tarislada/Documents/Extra_python_projects/SKH FP/FInalized_process/enhanced_3_3")
        # --- Data Loading ---
    if FEATURE_TYPE == 'engineered':
        print("\nSearching for ENGINEERED feature files...")
        
        # Find matching files
        feature_files = []
        label_files = []
        
        for feature_path in sorted(feature_dir.glob('*_enhanced_features.csv')):
            base_name = feature_path.stem.replace('_enhanced_features', '')
            label_path = label_dir / f"{base_name}_expanded_labels.csv"
            
            if label_path.exists():
                feature_files.append(str(feature_path))
                label_files.append(str(label_path))
                print(f"  Found pair: {base_name}")
        
        print(f"\nFound {len(feature_files)} matching ENGINEERED feature file pairs")
        
        if len(feature_files) == 0:
            print("ERROR: No engineered feature files found!")
            exit(1)
        
        # Use the full EnhancedFeatureEngineer (already does engineering)
        feature_engineer_instance = EnhancedFeatureEngineer()
    
    elif FEATURE_TYPE == 'raw':
        print("\nLoading RAW pose data...")
        keypoint_dir = Path("SKH_FP/savgol_pose_w59p7")
        cricket_dir = Path("SKH_FP/FInalized_process/cricket_process_test5")
        label_dir = Path("/home/tarislada/Documents/Extra_python_projects/SKH FP/FInalized_process/Behavior_label")

        # Load raw data and create temporary feature files
        rawdataloader = RawDataLoader()
        print("Finding matching keypoint, cricket, and label files...")
        keypoint_files, cricket_files, label_files_raw = rawdataloader.find_matching_files(
            keypoint_dir, cricket_dir, label_dir
        )
        
        print(f"Found {len(keypoint_files)} matching file sets")
        
        if len(keypoint_files) == 0:
            print("ERROR: No matching raw data files found!")
            exit(1)
        
        # STEP 2: Load and merge the raw data
        print("Loading and processing raw data...")
        all_data_raw = rawdataloader.load_raw_data(
            keypoint_files, cricket_files, label_files_raw, batch_size=5
        )
        
        if all_data_raw is None or len(all_data_raw) == 0:
            print("ERROR: Failed to load raw data!")
            exit(1)
        
        # STEP 3: Prepare the data (scaling, encoding, etc.)
        print("Preparing raw features for XGBoost...")
        X_raw, y_raw, groups_raw, feature_names_raw = rawdataloader.prepare_data(all_data_raw)
        
        print(f"Raw features shape: {X_raw.shape}")
        print(f"Number of raw features: {X_raw.shape[1]}")
        print(f"First 10 features: {feature_names_raw[:10]}")
        
        # STEP 4: Save to temporary CSV files for the experiment pipeline
        # The experiment expects file paths, so we need to create temporary files
        import tempfile
        
        temp_dir = Path(tempfile.mkdtemp(prefix="xgb_raw_features_"))
        print(f"Creating temporary feature files in: {temp_dir}")
        
        feature_files = []
        label_files = []
        
        # Group data by trial_id and save each trial separately
        for trial_id in all_data_raw['trial_id'].unique():
            trial_data = all_data_raw[all_data_raw['trial_id'] == trial_id].copy()
            
            # IMPORTANT: Extract animal_id from the trial_data
            # The animal_id column should exist in all_data_raw from raw_data_loader
            animal_id = trial_data['animal_id'].iloc[0]  # Get the animal_id for this trial
            
            # Create filename with both animal_id and trial_id to match expected format
            # Format: m14_t1_raw_features.csv (matches the expected "m\d+" pattern)
            filename_base = f"{animal_id}_{trial_id}"  # e.g., "m14_t1"
            
            feature_columns = [col for col in trial_data.columns if col != 'behavior']
            
            # Save feature file (contains features + frame column)
            feature_file = temp_dir / f"{filename_base}_raw_features.csv"
            trial_data[feature_columns].to_csv(feature_file, index=False)
            feature_files.append(str(feature_file))
            
            # Save label file (contains frame + behavior columns)
            label_file = temp_dir / f"{filename_base}_raw_labels.csv"
            trial_data[['frame', 'behavior']].to_csv(label_file, index=False)
            label_files.append(str(label_file))
            
            print(f"  Saved {filename_base}: {len(trial_data)} frames")
        
        print(f"\nPrepared {len(feature_files)} RAW feature file pairs")
        
        # Use minimal feature engineer (no temporal features)
        feature_engineer_instance = MinimalFeatureEngineer()
    else:
        raise ValueError("Invalid FEATURE_TYPE")
    
    if len(feature_files) > 0:
        # Run with enhanced features
        # results_baseline = run_experiment(
        #     feature_files,
        #     label_files,
        #     experiment_name="Baseline (No SMOTE, Auto Weights)",
        #     hyperparam_search=True,
        #     use_thresholding=False # Explicitly disable
        # )
        results_baseline = None
        
        # Run with thresholding enabled
        results_thresholding = run_experiment(
            feature_files,
            label_files,
            experiment_name="Strategy 2: With Thresholding",
            hyperparam_search=True,
            use_thresholding=True, # Enable thresholding
            feature_engineer=feature_engineer_instance
        )
        
        # Print comparison
        if results_baseline and results_thresholding:
            print("\n" + "="*80)
            print("SMOTE COMPARISON")
            print("="*80)
            print(f"{'Method':<30} {'Overall F1':>15} {'Attack F1':>15}")
            print("-"*60)
            
            with_f1 = results_thresholding['cv_frame_f1_mean']
            with_std = results_thresholding['cv_frame_f1_std']
            with_attack = results_thresholding['behavior_summaries']['attack']['f1_mean']
            with_attack_std = results_thresholding['behavior_summaries']['attack']['f1_std']
            
            no_f1 = results_baseline['cv_frame_f1_mean']
            no_std = results_baseline['cv_frame_f1_std']
            no_attack = results_baseline['behavior_summaries']['attack']['f1_mean']
            no_attack_std = results_baseline['behavior_summaries']['attack']['f1_std']
            
            print(f"{'With Thresholding':<30} {with_f1:.3f}±{with_std:.3f}    {with_attack:.3f}±{with_attack_std:.3f}")
            print(f"{'Without Thresholding':<30} {no_f1:.3f}±{no_std:.3f}    {no_attack:.3f}±{no_attack_std:.3f}")
            print(f"{'Improvement':<30} {(with_f1-no_f1):+.3f}          {(with_attack-no_attack):+.3f}")
    else:
        print("No matching file pairs found!")