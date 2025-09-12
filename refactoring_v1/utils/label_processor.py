"""
Label processing for behavior annotations.
Converts behavior annotation files to processed labels for classification.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, Any, Optional
from config.settings import settings

class LabelProcessor:
    """
    Processes behavior annotation files into standardized labels.
    Handles Excel and CSV annotation formats with behavior start/end times.
    """
    
    def __init__(self, config=None):
        """
        Initialize the label processor.
        
        Args:
            config: Configuration object. If None, uses global settings.
        """
        self.config = config or settings
        
        # Behavior priority for overlap resolution (higher number = higher priority)
        self.behavior_priority = {
            'attack': 3,
            'chasing': 2, 
            'non_visual_rotation': 1,
            'background': 0
        }
    
    def process_label_file(self, input_path: str) -> pd.DataFrame:
        """
        Process a single annotation file.
        
        Args:
            input_path: Path to annotation file (.xlsx or .csv)
            
        Returns:
            pd.DataFrame: Processed labels with 'frame' and 'behavior' columns
        """
        input_file = Path(input_path)
        
        # Read file (handles both .xlsx and .csv)
        try:
            if input_file.suffix == '.xlsx':
                df = pd.read_excel(input_file)
            else:
                df = pd.read_csv(input_file)
        except Exception as e:
            raise ValueError(f"Could not read annotation file {input_file.name}: {e}")
        
        print(f"    Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Validate expected columns
        expected_cols = ['approaching_start', 'approaching_end', 'turning_start', 'turning_end', 
                        'attack_start', 'attack_end']
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            print(f"    Warning: Missing columns: {missing_cols}")
        
        labeled_frames = []
        
        # Process approaching behavior (maps to 'chasing')
        for _, row in df.iterrows():
            if pd.notna(row.get('approaching_start')) and pd.notna(row.get('approaching_end')):
                start = int(row['approaching_start'])
                end = int(row['approaching_end'])
                for frame in range(start, end + 1):
                    labeled_frames.append({'frame': frame, 'behavior': 'chasing'})
        
        # Process turning behavior (maps to 'non_visual_rotation')
        for _, row in df.iterrows():
            if pd.notna(row.get('turning_start')) and pd.notna(row.get('turning_end')):
                start = int(row['turning_start'])
                end = int(row['turning_end'])
                for frame in range(start, end + 1):
                    labeled_frames.append({'frame': frame, 'behavior': 'non_visual_rotation'})
        
        # Process attack behavior
        for _, row in df.iterrows():
            if pd.notna(row.get('attack_start')) and pd.notna(row.get('attack_end')):
                start = int(row['attack_start'])
                end = int(row['attack_end'])
                for frame in range(start, end + 1):
                    labeled_frames.append({'frame': frame, 'behavior': 'attack'})
        
        if not labeled_frames:
            print("    Warning: No labels found in annotation file")
            return pd.DataFrame(columns=['frame', 'behavior'])
        
        # Convert to DataFrame
        labels_df = pd.DataFrame(labeled_frames)
        
        # Handle overlaps using priority system (attack > chasing > non_visual_rotation)
        labels_df['priority'] = labels_df['behavior'].map(self.behavior_priority)
        labels_df = labels_df.sort_values('priority', ascending=False).drop_duplicates('frame', keep='first')
        
        # Create complete sequence from min to max frame
        min_frame, max_frame = labels_df['frame'].min(), labels_df['frame'].max()
        all_frames = pd.DataFrame({'frame': range(min_frame, max_frame + 1)})
        result = all_frames.merge(labels_df[['frame', 'behavior']], on='frame', how='left')
        result['behavior'].fillna('background', inplace=True)
        
        print(f"    Generated: {len(result)} frames ({min_frame} to {max_frame})")
        
        # Show behavior distribution
        behavior_counts = result['behavior'].value_counts().to_dict()
        print(f"    Behaviors: {behavior_counts}")
        
        return result[['frame', 'behavior']]
    
    def process_file(self, input_path: str, output_path: str) -> bool:
        """
        Process a single annotation file and save results.
        
        Args:
            input_path: Path to input annotation file
            output_path: Path to save processed labels
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            result_df = self.process_label_file(input_path)
            
            if len(result_df) > 0:
                result_df.to_csv(output_path, index=False)
                return True
            else:
                print(f"    ✗ No data to save for {Path(input_path).name}")
                return False
                
        except Exception as e:
            print(f"    ✗ Error processing {Path(input_path).name}: {e}")
            return False
    
    def process_directory(self, input_dir: str, output_dir: str, 
                         file_pattern: str = "*") -> Dict[str, Any]:
        """
        Process all annotation files in a directory.
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            file_pattern: File pattern to match (default: all files)
            
        Returns:
            dict: Processing results with success/failure counts
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        input_path = Path(input_dir)
        
        # Look for both .xlsx and .csv files
        xlsx_files = list(input_path.glob(f'{file_pattern}.xlsx'))
        csv_files = list(input_path.glob(f'{file_pattern}.csv'))
        annotation_files = xlsx_files + csv_files
        
        if not annotation_files:
            print(f"No annotation files (.xlsx or .csv) found in {input_dir}")
            return {"processed": 0, "failed": 0, "files": []}
        
        print(f"Found {len(annotation_files)} annotation files to process")
        
        successful = []
        failed = []
        
        for annotation_file in annotation_files:
            print(f"  Processing: {annotation_file.name}")
            
            # Generate output filename: remove _annotation suffix if present, add _processed_labels
            base_name = annotation_file.stem.replace('_annotation', '')
            output_file = Path(output_dir) / f"{base_name}_processed_labels.csv"
            
            if self.process_file(str(annotation_file), str(output_file)):
                successful.append(annotation_file.name)
                print(f"    ✓ Saved: {output_file.name}")
            else:
                failed.append(annotation_file.name)
        
        # Summary
        print(f"\nLabel processing complete:")
        print(f"Successfully processed: {len(successful)} files")
        print(f"Failed to process: {len(failed)} files")
        
        return {
            "processed": len(successful),
            "failed": len(failed),
            "successful_files": successful,
            "failed_files": failed
        }
    
    def check_existing_labels(self, output_dir: str) -> bool:
        """
        Check if processed labels already exist in output directory.
        
        Args:
            output_dir: Directory to check for existing processed labels
            
        Returns:
            bool: True if processed labels exist, False otherwise
        """
        if not Path(output_dir).exists():
            return False
            
        processed_labels = list(Path(output_dir).glob('*_processed_labels.csv'))
        
        if processed_labels:
            print(f"Found {len(processed_labels)} existing processed label files:")
            for label_file in processed_labels[:5]:  # Show first 5
                print(f"  - {label_file.name}")
            if len(processed_labels) > 5:
                print(f"  ... and {len(processed_labels) - 5} more")
            return True
        
        return False
    
    def process(self, input_dir: Optional[str] = None, 
                output_dir: Optional[str] = None,
                force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Main processing method using configured directories.
        
        Args:
            input_dir: Override input directory (uses config if None)
            output_dir: Override output directory (uses config if None)
            force_reprocess: If True, reprocess even if labels exist
            
        Returns:
            dict: Processing results
        """
        input_dir = input_dir or str(self.config.paths.annotation_dir)
        output_dir = output_dir or str(self.config.paths.behavior_labels_dir)
        
        print("=== LABEL PROCESSING (Step 5.5) ===")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        
        # Check if processed labels already exist (unless forcing reprocess)
        if not force_reprocess and self.check_existing_labels(output_dir):
            return {
                'processed': 0,
                'failed': 0,
                'skipped': True,
                'reason': 'Processed labels already exist (use --force-reprocess to override)'
            }
        
        # Validate input directory
        if not Path(input_dir).exists():
            return {
                'processed': 0,
                'failed': 0,
                'error': f'Input directory does not exist: {input_dir}'
            }
        
        print(f"Processing behavior annotations...")
        print(f"Expected annotation format:")
        print(f"  - Columns: approaching_start/end, turning_start/end, attack_start/end")
        print(f"  - File formats: .xlsx or .csv")
        print(f"  - Behavior mapping: approaching→chasing, turning→non_visual_rotation")
        
        return self.process_directory(input_dir, output_dir)

def main():
    """Command line interface for label processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process behavior annotation files")
    parser.add_argument("--input-dir", help="Input directory (overrides config)")
    parser.add_argument("--output-dir", help="Output directory (overrides config)")
    parser.add_argument("--force-reprocess", action="store_true", 
                       help="Reprocess even if labels exist")
    
    args = parser.parse_args()
    
    # Create processor and run
    processor = LabelProcessor()
    results = processor.process(args.input_dir, args.output_dir, args.force_reprocess)
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        exit(1)
    elif results.get('skipped'):
        print(f"Skipped: {results['reason']}")
        exit(0)
    elif results["failed"] > 0:
        exit(1)

if __name__ == "__main__":
    main()