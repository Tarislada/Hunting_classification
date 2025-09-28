import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

def expand_attack_labels(labels_df: pd.DataFrame, 
                        before_frames: int = 5, 
                        after_frames: int = 10) -> pd.DataFrame:
    """
    Expand attack labels by specified frames before and after.
    
    Args:
        labels_df: DataFrame with 'frame' and 'behavior' columns
        before_frames: Number of frames to expand before each attack
        after_frames: Number of frames to expand after each attack
    
    Returns:
        DataFrame with expanded attack labels
    """
    expanded_df = labels_df.copy()
    
    # Find all attack instances
    attack_frames = labels_df[labels_df['behavior'] == 'attack']['frame'].values
    
    # Group consecutive attack frames into events
    if len(attack_frames) == 0:
        return expanded_df
    
    attack_events = []
    current_event_start = attack_frames[0]
    current_event_end = attack_frames[0]
    
    for i in range(1, len(attack_frames)):
        if attack_frames[i] == current_event_end + 1:
            # Continue current event
            current_event_end = attack_frames[i]
        else:
            # Save current event and start new one
            attack_events.append((current_event_start, current_event_end))
            current_event_start = attack_frames[i]
            current_event_end = attack_frames[i]
    
    # Don't forget the last event
    attack_events.append((current_event_start, current_event_end))
    
    print(f"  Found {len(attack_events)} attack events")
    
    # Expand each attack event
    for event_start, event_end in attack_events:
        expand_start = max(expanded_df['frame'].min(), event_start - before_frames)
        expand_end = min(expanded_df['frame'].max(), event_end + after_frames)
        
        # Only expand into 'background' or 'chasing' frames
        for frame in range(expand_start, expand_end + 1):
            if frame in expanded_df['frame'].values:
                current_behavior = expanded_df.loc[expanded_df['frame'] == frame, 'behavior'].values[0]
                if current_behavior in ['background', 'chasing']:
                    expanded_df.loc[expanded_df['frame'] == frame, 'behavior'] = 'attack'
    
    # Report statistics
    original_attacks = (labels_df['behavior'] == 'attack').sum()
    expanded_attacks = (expanded_df['behavior'] == 'attack').sum()
    print(f"  Attack frames: {original_attacks} -> {expanded_attacks} ({expanded_attacks/max(1,original_attacks):.1f}x)")
    
    return expanded_df

def add_limb_features(features_df: pd.DataFrame, 
                      pose_csv_path: str = None) -> pd.DataFrame:
    """
    Add limb-based features to existing feature dataframe.
    
    Args:
        features_df: Existing features dataframe from angle_val.py
        pose_csv_path: Path to kalman filtered pose CSV (if not already in features)
    
    Returns:
        DataFrame with added limb features
    """
    enhanced_df = features_df.copy()
    
    # Check if limb positions are already in the dataframe
    limb_columns = ['left_forelimb_x', 'left_forelimb_y', 'right_forelimb_x', 'right_forelimb_y',
                    'left_hindlimb_x', 'left_hindlimb_y', 'right_hindlimb_x', 'right_hindlimb_y']
    
    if pose_csv_path and not all(col in features_df.columns for col in limb_columns):
        # Load pose data to get limb positions
        pose_df = pd.read_csv(pose_csv_path)
        
        # Merge limb positions
        if 'frame' in pose_df.columns and 'frame' in features_df.columns:
            # Select only limb columns and frame
            limb_data = pose_df[['frame'] + [col for col in limb_columns if col in pose_df.columns]]
            enhanced_df = pd.merge(enhanced_df, limb_data, on='frame', how='left')
    
    # Calculate limb-based features
    if 'left_forelimb_x' in enhanced_df.columns:
        # Forelimb spread (reaching behavior)
        enhanced_df['forelimb_spread'] = np.sqrt(
            (enhanced_df['right_forelimb_x'] - enhanced_df['left_forelimb_x'])**2 +
            (enhanced_df['right_forelimb_y'] - enhanced_df['left_forelimb_y'])**2
        )
        
        # Hindlimb spread 
        enhanced_df['hindlimb_spread'] = np.sqrt(
            (enhanced_df['right_hindlimb_x'] - enhanced_df['left_hindlimb_x'])**2 +
            (enhanced_df['right_hindlimb_y'] - enhanced_df['left_hindlimb_y'])**2
        )
        
        # Body compression (spring-loading for pounce)
        # Average forelimb position
        avg_forelimb_x = (enhanced_df['left_forelimb_x'] + enhanced_df['right_forelimb_x']) / 2
        avg_forelimb_y = (enhanced_df['left_forelimb_y'] + enhanced_df['right_forelimb_y']) / 2
        
        # Average hindlimb position  
        avg_hindlimb_x = (enhanced_df['left_hindlimb_x'] + enhanced_df['right_hindlimb_x']) / 2
        avg_hindlimb_y = (enhanced_df['left_hindlimb_y'] + enhanced_df['right_hindlimb_y']) / 2
        
        # Body compression ratio
        enhanced_df['body_compression'] = np.sqrt(
            (avg_forelimb_x - avg_hindlimb_x)**2 +
            (avg_forelimb_y - avg_hindlimb_y)**2
        )
        
        # Limb velocities
        enhanced_df['forelimb_spread_velocity'] = enhanced_df['forelimb_spread'].diff()
        enhanced_df['body_compression_velocity'] = enhanced_df['body_compression'].diff()
        
        # Asymmetry features (for mid-pounce detection)
        enhanced_df['forelimb_asymmetry'] = np.abs(
            enhanced_df['left_forelimb_x'] - enhanced_df['right_forelimb_x']
        )
        
        print(f"  Added {6} limb-based features")
    else:
        print(f"  Warning: Limb positions not found, skipping limb features")
    
    return enhanced_df

def process_file_pair(feature_file: str, label_file: str, 
                     output_dir: Path,
                     before_frames: int = 5, after_frames: int = 10,
                     pose_csv_path: str = None) -> bool:
    """
    Process a single pair of feature and label files.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Extract base name
        feature_path = Path(feature_file)
        base_name = feature_path.stem.replace('_validated', '').replace('_analysis', '')
        
        print(f"\nProcessing {base_name}...")
        
        # Load files
        features_df = pd.read_csv(feature_file)
        labels_df = pd.read_csv(label_file)
        
        # Expand attack labels
        print("  Expanding attack labels...")
        expanded_labels = expand_attack_labels(labels_df, before_frames, after_frames)
        
        # Add limb features if pose data available
        if pose_csv_path:
            print("  Adding limb features...")
            enhanced_features = add_limb_features(features_df, pose_csv_path)
        else:
            enhanced_features = features_df
        
        # Save processed files
        output_feature_path = output_dir / f"{base_name}_enhanced_features.csv"
        output_label_path = output_dir / f"{base_name}_expanded_labels.csv"
        
        enhanced_features.to_csv(output_feature_path, index=False)
        expanded_labels.to_csv(output_label_path, index=False)
        
        print(f"  Saved: {output_feature_path.name} and {output_label_path.name}")
        
        # Print class distribution
        behavior_counts = expanded_labels['behavior'].value_counts()
        total_frames = len(expanded_labels)
        print(f"  Class distribution:")
        for behavior, count in behavior_counts.items():
            print(f"    {behavior}: {count} ({count/total_frames*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"  Error processing {base_name}: {e}")
        return False

def batch_process_experiments(feature_dir: Path, label_dir: Path, 
                             pose_dir: Path = None,
                             expansion_configs: list = None):
    """
    Run multiple experiments with different expansion configurations.
    
    Args:
        feature_dir: Directory with *_analysis.csv files
        label_dir: Directory with *_processed_labels.csv files  
        pose_dir: Optional directory with kalman filtered pose CSVs
        expansion_configs: List of (before, after, experiment_name) tuples
    """
    if expansion_configs is None:
        expansion_configs = [
            (3, 5, 'expand_3_5'),
            (5, 10, 'expand_5_10'),
            (5, 15, 'expand_5_15'),
            (7, 20, 'expand_7_20'),
        ]
    
    # Find matching file pairs
    feature_files = sorted(feature_dir.glob('*_analysis.csv'))
    
    for before, after, exp_name in expansion_configs:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {exp_name} (before={before}, after={after})")
        print('='*60)
        
        # Create output directory for this experiment
        output_dir = feature_dir.parent / f"enhanced_{exp_name}"
        output_dir.mkdir(exist_ok=True)
        
        successful = 0
        failed = 0
        
        for feature_path in feature_files:
            # Find corresponding label file
            base_name = feature_path.stem.replace('_validated', '').replace('_analysis', '')
            label_path = label_dir / f"{base_name}_processed_labels.csv"
            
            if not label_path.exists():
                print(f"Skipping {base_name}: label file not found")
                failed += 1
                continue
            
            # Find pose file if directory provided
            pose_path = None
            if pose_dir:
                pose_path = pose_dir / f"kalman_filtered_processed_filtered_{base_name}.csv"
                if not pose_path.exists():
                    pose_path = None
            
            # Process the file pair
            if process_file_pair(str(feature_path), str(label_path), 
                               output_dir, before, after, str(pose_path) if pose_path else None):
                successful += 1
            else:
                failed += 1
        
        print(f"\n{exp_name} complete: {successful} successful, {failed} failed")

if __name__ == "__main__":
    # Define paths
    feature_dir = Path("/home/tarislada/Documents/Extra_python_projects/SKH FP/FInalized_process/test_val_vid5")
    label_dir = Path("/home/tarislada/Documents/Hunting_classification/SKH_FP/FInalized_process/Behavior_label2")
    pose_dir = Path("/home/tarislada/Documents/Extra_python_projects/SKH FP/kalman_filtered_w59p7")  # Optional
    
    # Single experiment
    print("Running single experiment with 4 frames before, 2 frames after...")
    output_dir = feature_dir.parent / "enhanced_4_2"
    output_dir.mkdir(exist_ok=True)
    
    feature_files = sorted(feature_dir.glob('*_analysis.csv'))
    for feature_path in feature_files:  # Process first ALL files
        base_name = feature_path.stem.replace('_validated', '').replace('_analysis', '')
        label_path = label_dir / f"{base_name}_processed_labels.csv"
        
        if label_path.exists():
            pose_path = pose_dir / f"kalman_filtered_processed_filtered_{base_name}.csv" if pose_dir else None
            process_file_pair(str(feature_path), str(label_path), output_dir, 
                            before_frames=4, after_frames=2,
                            pose_csv_path=str(pose_path) if pose_path and pose_path.exists() else None)
    
    # Or run multiple experiments
    # batch_process_experiments(feature_dir, label_dir, pose_dir)