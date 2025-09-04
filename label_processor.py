import pandas as pd
import numpy as np
from pathlib import Path

def process_label_file(input_file):
    """Process a single label file with the clean format."""
    
    # Read file (handles both .xlsx and .csv)
    if input_file.suffix == '.xlsx':
        df = pd.read_excel(input_file)
    else:
        df = pd.read_csv(input_file)
    
    print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    labeled_frames = []
    
    # Process approaching behavior
    for _, row in df.iterrows():
        if pd.notna(row['approaching_start']) and pd.notna(row['approaching_end']):
            start = int(row['approaching_start'])
            end = int(row['approaching_end'])
            for frame in range(start, end + 1):
                labeled_frames.append({'frame': frame, 'behavior': 'chasing'})
    
    # Process turning behavior  
    for _, row in df.iterrows():
        if pd.notna(row['turning_start']) and pd.notna(row['turning_end']):
            start = int(row['turning_start'])
            end = int(row['turning_end'])
            for frame in range(start, end + 1):
                labeled_frames.append({'frame': frame, 'behavior': 'non_visual_rotation'})
    
    # Process attack behavior
    for _, row in df.iterrows():
        if pd.notna(row['attack_start']) and pd.notna(row['attack_end']):
            start = int(row['attack_start'])
            end = int(row['attack_end'])
            for frame in range(start, end + 1):
                labeled_frames.append({'frame': frame, 'behavior': 'attack'})
    
    if not labeled_frames:
        print("  No labels found")
        return pd.DataFrame(columns=['frame', 'behavior'])
    
    # Convert to DataFrame
    labels_df = pd.DataFrame(labeled_frames)
    
    # Handle overlaps (priority: attack > chasing > non_visual_rotation)
    priority = {'attack': 3, 'chasing': 2, 'non_visual_rotation': 1}
    labels_df['priority'] = labels_df['behavior'].map(priority)
    labels_df = labels_df.sort_values('priority', ascending=False).drop_duplicates('frame', keep='first')
    
    # Create complete sequence from min to max frame
    min_frame, max_frame = labels_df['frame'].min(), labels_df['frame'].max()
    all_frames = pd.DataFrame({'frame': range(min_frame, max_frame + 1)})
    result = all_frames.merge(labels_df[['frame', 'behavior']], on='frame', how='left')
    result['behavior'].fillna('background', inplace=True)
    
    print(f"  Generated: {len(result)} frames ({min_frame} to {max_frame})")
    print(f"  Behaviors: {result['behavior'].value_counts().to_dict()}")
    
    return result[['frame', 'behavior']]

def process_directory(input_dir, output_dir):
    """Process all label files in input directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Look for both .xlsx and .csv files
    label_files = list(input_path.glob('*.xlsx')) + list(input_path.glob('*.csv'))
    
    if not label_files:
        print(f"No .xlsx or .csv files found in {input_dir}")
        return
    
    print(f"Found {len(label_files)} files to process")
    
    for label_file in label_files:
        print(f"\nProcessing: {label_file.name}")
        
        try:
            result_df = process_label_file(label_file)
            
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

if __name__ == "__main__":
    # Simple usage - just change these paths
    input_dir = "/home/tarislada/Documents/Extra_python_projects/SKH FP/FInalized_process/mouse_hunting_annotations"
    output_dir = "/home/tarislada/Documents/Extra_python_projects/SKH FP/FInalized_process/Behavior_label"
    
    process_directory(input_dir, output_dir)