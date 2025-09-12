
"""
Timing ground truth preprocessor.
Converts expert timing annotations from Excel format to standardized CSV format.
"""

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

class TimingPreprocessor:
    """
    Preprocesses expert timing annotations into standardized format.
    Handles special formatting, missing data, and frame rate conversions.
    """
    
    def __init__(self, target_fps: int = 30):
        """
        Initialize the timing preprocessor.
        
        Args:
            target_fps: Target frame rate for standardized output (default: 30fps)
        """
        self.target_fps = target_fps
        
        # Column mapping from Excel headers to standardized names
        self.column_mapping = {
            'Cricket enterance': 'cricket_entrance',  # Note: keeping original spelling
            'latency to hunt(first pursuit)': 'latency_to_hunt', 
            'Latency to first attack onset': 'latency_to_attack',
            'hunting duration(latency to consumption)': 'hunting_duration'
        }
        
        # Standard output columns
        self.output_columns = [
            'animal_id', 'trial_id', 'cricket_entrance', 'latency_to_hunt', 
            'latency_to_attack', 'hunting_duration', 'notes', 'alternate_values'
        ]
    
    def parse_special_value(self, value: Any) -> Tuple[float, str, str]:
        """
        Parse special formatted values like "(4159)4233", "no hunting", etc.
        
        Args:
            value: Raw cell value from Excel
            
        Returns:
            Tuple of (main_value, notes, alternate_values)
        """
        if pd.isna(value) or value == "":
            return np.nan, "", ""
        
        # Convert to string for processing
        value_str = str(value).strip()
        
        # Handle "no hunting" case
        if "no hunting" in value_str.lower():
            return np.nan, "no_hunting", ""
        
        # Handle parentheses cases like "(4159)4233" or "1534(1864)"
        paren_pattern = r'\((\d+)\)'
        paren_match = re.search(paren_pattern, value_str)
        
        if paren_match:
            alternate_value = paren_match.group(1)
            # Extract main value (number outside parentheses)
            main_value_str = re.sub(paren_pattern, '', value_str).strip()
            
            try:
                main_value = float(main_value_str)
                notes = f"alternate_available"
                alternate_values = f"alternate:{alternate_value}"
                return main_value, notes, alternate_values
            except ValueError:
                # If we can't parse main value, use alternate
                try:
                    main_value = float(alternate_value)
                    notes = f"used_alternate"
                    alternate_values = f"original:{main_value_str}"
                    return main_value, notes, alternate_values
                except ValueError:
                    return np.nan, f"parse_error", f"original:{value_str}"
        
        # Try to parse as regular number
        try:
            return float(value_str), "", ""
        except ValueError:
            return np.nan, f"parse_error", f"original:{value_str}"
    
    def process_animal_sheet(self, sheet_data: List[List], animal_id: str) -> pd.DataFrame:
        """
        Process a single animal's sheet data.
        
        Args:
            sheet_data: Raw sheet data as list of lists
            animal_id: Animal identifier (e.g., 'm14')
            
        Returns:
            pd.DataFrame: Processed data for this animal
        """
        if len(sheet_data) < 2:
            print(f"  Warning: Sheet for {animal_id} has no data rows")
            return pd.DataFrame(columns=self.output_columns)
        
        # Get headers
        headers = sheet_data[0]
        print(f"  Headers: {headers}")
        
        # Process each data row
        processed_rows = []
        
        for row_idx, row in enumerate(sheet_data[1:], 1):
            # Skip empty rows
            if not row or all(pd.isna(cell) or cell == "" for cell in row):
                print(f"    Row {row_idx}: EMPTY - skipping")
                continue
            
            # Get trial ID from first column
            trial_id = str(row[0]).strip() if row and len(row) > 0 else f"row_{row_idx}"
            
            print(f"    Processing {trial_id}: {row[1:] if len(row) > 1 else 'no data'}")
            
            # Initialize processed row
            processed_row = {
                'animal_id': animal_id,
                'trial_id': trial_id,
                'cricket_entrance': np.nan,
                'latency_to_hunt': np.nan, 
                'latency_to_attack': np.nan,
                'hunting_duration': np.nan,
                'notes': "",
                'alternate_values': ""
            }
            
            # Process each timing column
            notes_list = []
            alternates_list = []
            
            for col_idx, col_name in enumerate(['cricket_entrance', 'latency_to_hunt', 'latency_to_attack', 'hunting_duration']):
                if col_idx + 1 < len(row):  # +1 because first column is trial_id
                    raw_value = row[col_idx + 1]
                    main_val, note, alternate = self.parse_special_value(raw_value)
                    
                    processed_row[col_name] = main_val
                    
                    if note:
                        notes_list.append(f"{col_name}:{note}")
                    if alternate:
                        alternates_list.append(f"{col_name}:{alternate}")
            
            # Combine notes and alternates
            processed_row['notes'] = ";".join(notes_list) if notes_list else ""
            processed_row['alternate_values'] = ";".join(alternates_list) if alternates_list else ""
            
            processed_rows.append(processed_row)
            
            # Show what we parsed
            timing_summary = {k: v for k, v in processed_row.items() if k not in ['animal_id', 'trial_id', 'notes', 'alternate_values']}
            if processed_row['notes']:
                print(f"      Parsed: {timing_summary} (notes: {processed_row['notes']})")
            else:
                print(f"      Parsed: {timing_summary}")
        
        return pd.DataFrame(processed_rows)
    
    def process_excel_file(self, excel_path: str, output_path: str) -> pd.DataFrame:
        """
        Process the complete Excel file with multiple animal sheets.
        
        Args:
            excel_path: Path to input Excel file
            output_path: Path to save standardized CSV
            
        Returns:
            pd.DataFrame: Complete processed dataset
        """
        print(f"Processing Excel file: {excel_path}")
        print(f"Target frame rate: {self.target_fps} fps")
        
        # Read Excel file
        try:
            excel_data = pd.read_excel(excel_path, sheet_name=None, header=None)
        except Exception as e:
            raise ValueError(f"Could not read Excel file: {e}")
        
        print(f"Found {len(excel_data)} sheets: {list(excel_data.keys())}")
        
        all_animals_data = []
        
        # Process each animal sheet
        for animal_id, sheet_df in excel_data.items():
            print(f"\n--- Processing animal: {animal_id} ---")
            
            # Convert DataFrame to list of lists for easier processing
            sheet_data = sheet_df.values.tolist()
            
            # Process this animal's data
            animal_data = self.process_animal_sheet(sheet_data, animal_id)
            
            if len(animal_data) > 0:
                all_animals_data.append(animal_data)
                print(f"  Processed {len(animal_data)} trials for {animal_id}")
            else:
                print(f"  No valid trials found for {animal_id}")
        
        # Combine all animals
        if all_animals_data:
            complete_data = pd.concat(all_animals_data, ignore_index=True)
        else:
            print("No data found in any sheets!")
            return pd.DataFrame(columns=self.output_columns)
        
        # Save to CSV
        complete_data.to_csv(output_path, index=False)
        
        # Summary report
        print(f"\n=== PROCESSING SUMMARY ===")
        print(f"Total trials processed: {len(complete_data)}")
        print(f"Animals: {complete_data['animal_id'].unique()}")
        print(f"Trials per animal:")
        for animal in complete_data['animal_id'].unique():
            animal_trials = complete_data[complete_data['animal_id'] == animal]['trial_id'].tolist()
            print(f"  {animal}: {animal_trials}")
        
        # Show special cases
        special_cases = complete_data[complete_data['notes'] != ""]
        if len(special_cases) > 0:
            print(f"\nSpecial cases found: {len(special_cases)}")
            for _, row in special_cases.iterrows():
                print(f"  {row['animal_id']} {row['trial_id']}: {row['notes']}")
        
        # Show data completeness
        print(f"\nData completeness:")
        for col in ['cricket_entrance', 'latency_to_hunt', 'latency_to_attack', 'hunting_duration']:
            complete_count = complete_data[col].notna().sum()
            total_count = len(complete_data)
            print(f"  {col}: {complete_count}/{total_count} ({complete_count/total_count*100:.1f}%)")
        
        print(f"\nSaved standardized data to: {output_path}")
        
        return complete_data
    
    def convert_frame_rate(self, df: pd.DataFrame, source_fps: int) -> pd.DataFrame:
        """
        Convert frame numbers from source fps to target fps.
        
        Args:
            df: DataFrame with timing data
            source_fps: Original frame rate of the data
            
        Returns:
            pd.DataFrame: DataFrame with converted frame numbers
        """
        if source_fps == self.target_fps:
            print(f"Frame rate already matches target ({self.target_fps} fps)")
            return df
        
        print(f"Converting from {source_fps} fps to {self.target_fps} fps")
        
        conversion_factor = self.target_fps / source_fps
        timing_columns = ['cricket_entrance', 'latency_to_hunt', 'latency_to_attack', 'hunting_duration']
        
        df_converted = df.copy()
        
        for col in timing_columns:
            df_converted[col] = df_converted[col] * conversion_factor
            # Round to nearest frame
            df_converted[col] = df_converted[col].round().astype('Int64')  # Int64 supports NaN
        
        return df_converted
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the processed timing data for logical consistency.
        
        Args:
            df: Processed timing DataFrame
            
        Returns:
            dict: Validation results
        """
        print(f"\n=== DATA VALIDATION ===")
        
        validation_results = {
            'total_trials': len(df),
            'valid_trials': 0,
            'issues': []
        }
        
        for _, row in df.iterrows():
            trial_id = f"{row['animal_id']} {row['trial_id']}"
            issues = []
            
            # Check timing sequence logic
            cricket_entrance = row['cricket_entrance']
            latency_hunt = row['latency_to_hunt']
            latency_attack = row['latency_to_attack'] 
            hunting_duration = row['hunting_duration']
            
            # Skip validation for "no hunting" cases
            if 'no_hunting' in str(row['notes']):
                print(f"  {trial_id}: No hunting - skipping validation")
                continue
            
            # Check if cricket entrance < latency to hunt
            if pd.notna(cricket_entrance) and pd.notna(latency_hunt):
                if cricket_entrance >= latency_hunt:
                    issues.append(f"Cricket entrance ({cricket_entrance}) >= hunt start ({latency_hunt})")
            
            # Check if hunt < attack
            if pd.notna(latency_hunt) and pd.notna(latency_attack):
                if latency_hunt >= latency_attack:
                    issues.append(f"Hunt start ({latency_hunt}) >= attack start ({latency_attack})")
            
            # Check if attack < consumption
            if pd.notna(latency_attack) and pd.notna(hunting_duration):
                if latency_attack >= hunting_duration:
                    issues.append(f"Attack start ({latency_attack}) >= consumption ({hunting_duration})")
            
            if issues:
                validation_results['issues'].append({
                    'trial': trial_id,
                    'issues': issues
                })
                print(f"  {trial_id}: ⚠️  {'; '.join(issues)}")
            else:
                validation_results['valid_trials'] += 1
                print(f"  {trial_id}: ✓ Valid sequence")
        
        print(f"\nValidation summary:")
        print(f"  Valid trials: {validation_results['valid_trials']}/{validation_results['total_trials']}")
        print(f"  Trials with issues: {len(validation_results['issues'])}")
        
        return validation_results

def main():
    """Command line interface for timing preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess timing ground truth data")
    parser.add_argument("--excel-file", required=True, help="Input Excel file path")
    parser.add_argument("--output-csv", required=True, help="Output CSV file path") 
    parser.add_argument("--target-fps", type=int, default=30, help="Target frame rate (default: 30)")
    parser.add_argument("--source-fps", type=int, default=30, help="Source frame rate (default: 30)")
    parser.add_argument("--validate", action="store_true", help="Run data validation")
    
    args = parser.parse_args()
    
    # Create processor
    processor = TimingPreprocessor(target_fps=args.target_fps)
    
    # Process Excel file
    try:
        df = processor.process_excel_file(args.excel_file, args.output_csv)
        
        # Convert frame rate if needed
        if args.source_fps != args.target_fps:
            df = processor.convert_frame_rate(df, args.source_fps)
            # Save the converted version
            converted_path = args.output_csv.replace('.csv', f'_converted_{args.target_fps}fps.csv')
            df.to_csv(converted_path, index=False)
            print(f"Saved frame-rate converted data to: {converted_path}")
        
        # Run validation if requested
        if args.validate:
            validation_results = processor.validate_data(df)
        
        print(f"\n✅ Processing complete!")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()