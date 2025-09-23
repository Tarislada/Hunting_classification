import pandas as pd
from pathlib import Path
import argparse

def split_predictions_into_trial_files(input_path: str, output_dir: str):
    """
    Splits a single predictions CSV file into multiple per-trial CSV files,
    formatted as *_analysis.csv for the TimingValidator.

    Args:
        input_path (str): Path to the consolidated predictions.csv file.
                          Expected columns: ['animal_id', 'trial_id', 'frame', 'behavior'].
        output_dir (str): Path to the directory where individual trial files will be saved.
    """
    print(f"Loading consolidated predictions from: {input_path}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return

    # Group by each unique trial
    grouped = df.groupby(['animal_id', 'trial_id'])
    
    if len(grouped) == 0:
        print("No trials found in the input file.")
        return

    print(f"Found {len(grouped)} unique trials. Splitting into individual files...")

    for (animal_id, trial_id), trial_df in grouped:
        # Construct the output filename that TimingValidator expects
        output_filename = f"{animal_id}_{trial_id}_analysis.csv"
        file_save_path = output_path / output_filename
        
        # The validator expects a 'behavior' column, which we already have.
        # We can save the minimal required columns.
        trial_df_to_save = trial_df[['frame', 'behavior']]
        
        trial_df_to_save.to_csv(file_save_path, index=False)
        print(f"  ✓ Saved {output_filename}")

    print(f"\nSuccessfully created {len(grouped)} trial files in: {output_dir}")


def main():
    """Command-line interface for the splitting script."""
    parser = argparse.ArgumentParser(
        description="Split a predictions CSV into per-trial files for TimingValidator."
    )
    parser.add_argument(
        "--input",
        default="predictions.csv",
        help="Path to the input predictions CSV file (default: predictions.csv)"
    )
    parser.add_argument(
        "--output-dir",
        default="validator_input",
        help="Directory to save the split trial files (default: validator_input)"
    )
    args = parser.parse_args()

    split_predictions_into_trial_files(args.input, args.output_dir)

if __name__ == "__main__":
    main()