import pandas as pd
import numpy as np
from pathlib import Path

def process_new_label_file(input_file):
    """Process a single label file with the new 10-column format."""
    
    # Read file (handles both .xlsx and .csv)
    if input_file.suffix == '.xlsx':
        df = pd.read_excel(input_file)
    else:
        df = pd.read_csv(input_file)
    
    print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")
    
    # Find the trial duration (row with start and end values)
    trial_start = None
    trial_end = None
    
    for idx, row in df.iterrows():
        if pd.notna(row['start']) and pd.notna(row['end']):
            trial_start = int(row['start'])
            trial_end = int(row['end'])
            print(f"  Found trial duration: frames {trial_start}-{trial_end}")
            break
    
    if trial_start is None or trial_end is None:
        print("  Error: No trial start/end found")
        return pd.DataFrame(columns=['frame', 'behavior'])
    
    labeled_frames = []
    overlap_warnings = []
    
    # Process each row for behavior events
    for idx, row in df.iterrows():
        # Process approaching behavior (label as 'chasing')
        if pd.notna(row['approaching_start']) and pd.notna(row['approaching_end']):
            approach_start = int(row['approaching_start'])
            approach_end = int(row['approaching_end'])
            print(f"    Row {idx+1}: Approaching {approach_start}-{approach_end}")
            for frame in range(approach_start, approach_end + 1):
                labeled_frames.append({'frame': frame, 'behavior': 'chasing', 'row': idx+1})
        
        # Process turning behavior (label as 'chasing')
        if pd.notna(row['turning_start']) and pd.notna(row['turning_end']):
            turn_start = int(row['turning_start'])
            turn_end = int(row['turning_end'])
            print(f"    Row {idx+1}: Turning {turn_start}-{turn_end}")
            for frame in range(turn_start, turn_end + 1):
                labeled_frames.append({'frame': frame, 'behavior': 'chasing', 'row': idx+1})
        
        # Process attack behavior
        if pd.notna(row['attack_start']) and pd.notna(row['attack_end']):
            attack_start = int(row['attack_start'])
            attack_end = int(row['attack_end'])
            print(f"    Row {idx+1}: Attack {attack_start}-{attack_end}")
            for frame in range(attack_start, attack_end + 1):
                labeled_frames.append({'frame': frame, 'behavior': 'attack', 'row': idx+1})
        
        # Process consuming behavior
        if pd.notna(row['consuming_start']) and pd.notna(row['consuming_end']):
            consume_start = int(row['consuming_start'])
            consume_end = int(row['consuming_end'])
            print(f"    Row {idx+1}: Consuming {consume_start}-{consume_end}")
            for frame in range(consume_start, consume_end + 1):
                labeled_frames.append({'frame': frame, 'behavior': 'consume', 'row': idx+1})
    
    if not labeled_frames:
        print("  No behavior labels found")
        # Still create background for the trial duration
        all_frames = pd.DataFrame({'frame': range(trial_start, trial_end + 1)})
        all_frames['behavior'] = 'background'
        return all_frames[['frame', 'behavior']]
    
    # Convert to DataFrame
    labels_df = pd.DataFrame(labeled_frames)
    
    # Check for overlaps and handle with priority: attack > chasing > consume > background
    priority = {'attack': 4, 'chasing': 3, 'consume': 2, 'background': 1}
    labels_df['priority'] = labels_df['behavior'].map(priority)
    
    # Find overlapping frames and warn user
    frame_counts = labels_df['frame'].value_counts()
    overlapping_frames = frame_counts[frame_counts > 1].index
    
    if len(overlapping_frames) > 0:
        print(f"\n  ⚠️  WARNING: Found {len(overlapping_frames)} overlapping frames!")
        
        # Show details of overlaps
        for frame in sorted(overlapping_frames)[:10]:  # Show first 10 overlaps
            frame_behaviors = labels_df[labels_df['frame'] == frame]
            behaviors = frame_behaviors['behavior'].unique()
            rows = frame_behaviors['row'].unique()
            print(f"    Frame {frame}: {list(behaviors)} (from rows {list(rows)})")
        
        if len(overlapping_frames) > 10:
            print(f"    ... and {len(overlapping_frames) - 10} more overlapping frames")
        
        print(f"    Applying priority: attack > chasing > consume > background")
    
    # Resolve overlaps using priority
    labels_df = labels_df.sort_values(['frame', 'priority'], ascending=[True, False])
    labels_df = labels_df.drop_duplicates('frame', keep='first')
    labels_df = labels_df.drop(['priority', 'row'], axis=1)
    
    # Create complete frame sequence for the trial
    all_frames = pd.DataFrame({'frame': range(trial_start, trial_end + 1)})
    result = all_frames.merge(labels_df, on='frame', how='left')
    
    # Fill unlabeled frames with 'background'
    result['behavior'].fillna('background', inplace=True)
    
    print(f"  Generated: {len(result)} frames ({trial_start} to {trial_end})")
    print(f"  Behaviors: {result['behavior'].value_counts().to_dict()}")
    
    return result[['frame', 'behavior']]

def process_directory(input_dir, output_dir):
    """Process all label files in input directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Look for both .xlsx and .csv files
    label_files = list(input_path.glob('*_annotation.xlsx')) + list(input_path.glob('*_annotation.csv'))
    
    # If no annotation files found, try all xlsx/csv files
    if not label_files:
        label_files = list(input_path.glob('*.xlsx')) + list(input_path.glob('*.csv'))
    
    if not label_files:
        print(f"No .xlsx or .csv files found in {input_dir}")
        return
    
    print(f"Found {len(label_files)} files to process")
    
    for label_file in label_files:
        print(f"\nProcessing: {label_file.name}")
        
        try:
            result_df = process_new_label_file(label_file)
            
            if len(result_df) > 0:
                # Output filename: remove _annotation suffix if present, add _processed_labels
                output_name = label_file.stem.replace('_annotation', '') + '_processed_labels.csv'
                output_file = output_path / output_name
                
                result_df.to_csv(output_file, index=False)
                print(f"  ✓ Saved: {output_file.name}")
            else:
                print(f"  ✗ No data to save")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()

# Test function for single file
def test_single_file(file_path):
    """Test processing a single file and show sample output."""
    result = process_new_label_file(Path(file_path))
    
    print("\n" + "="*50)
    print("SAMPLE OUTPUT:")
    print("="*50)
    print(result.head(20))
    print("\nBehavior distribution:")
    print(result['behavior'].value_counts())
    
    # Show some transitions
    print("\nSample behavior transitions:")
    transitions = result[result['behavior'] != result['behavior'].shift()].head(10)
    for _, row in transitions.iterrows():
        print(f"Frame {row['frame']}: {row['behavior']}")
    
    return result

if __name__ == "__main__":
    # Update these paths for your setup
    input_dir = "SKH_FP/raw_behavior_label"  # Directory with the new .xlsx files
    output_dir = "SKH_FP/FInalized_process/Behavior_label2"     # Directory for output CSV files
    
    # Process entire directory
    process_directory(input_dir, output_dir)
    
    # For testing a single file:
    # result = test_single_file("m20_t7_annotation.xlsx")