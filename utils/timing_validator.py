"""
Timing validation for hunting behavior pipeline.
Compares pipeline predictions with expert ground truth timing annotations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TimingValidator:
    """
    Validates pipeline timing predictions against expert ground truth.
    Compares frame-level predictions for hunting behavior transitions.
    """
    
    def __init__(self, ground_truth_csv: str):
        """
        Initialize the timing validator.
        
        Args:
            ground_truth_csv: Path to standardized ground truth CSV file
        """
        self.ground_truth_csv = ground_truth_csv
        self.ground_truth = None
        self.pipeline_results = None
        self.comparison_results = None
        
        # Behavior mapping from pipeline to ground truth events
        self.behavior_mapping = {
            'latency_to_hunt': 'chasing',      # First pursuit = first chasing detection
            # 'latency_to_attack': 'attack',      # Attack onset = first attack detection
            'hunting_duration': 'consumption'   # Consumption = end of hunting sequence
        }
        
        print(f"Initialized TimingValidator with ground truth: {ground_truth_csv}")
    
    def load_ground_truth(self) -> pd.DataFrame:
        """Load and validate ground truth data."""
        try:
            self.ground_truth = pd.read_csv(self.ground_truth_csv)
            print(f"Loaded ground truth: {len(self.ground_truth)} trials")
            print(f"Animals: {self.ground_truth['animal_id'].unique()}")
            print(f"Trials per animal: {self.ground_truth.groupby('animal_id')['trial_id'].nunique().to_dict()}")
            
            # Show data completeness
            timing_cols = ['latency_to_hunt', 'hunting_duration'] #, 'latency_to_attack']
            for col in timing_cols:
                complete_count = self.ground_truth[col].notna().sum()
                total_count = len(self.ground_truth)
                print(f"  {col}: {complete_count}/{total_count} ({complete_count/total_count*100:.1f}% complete)")
            
            return self.ground_truth
            
        except Exception as e:
            raise ValueError(f"Could not load ground truth CSV: {e}")
    
    def extract_pipeline_timing(self, classification_results: pd.DataFrame, 
                               animal_id: str, trial_id: str) -> Dict[str, Optional[int]]:
        """
        Extract timing events from pipeline classification results for a single trial.
        
        Args:
            classification_results: Pipeline classification results DataFrame  
            animal_id: Animal identifier
            trial_id: Trial identifier
            
        Returns:
            dict: Extracted timing events {event_name: frame_number}
        """
        # Filter to this specific trial
        trial_data = classification_results[
            (classification_results['animal_id'] == animal_id) & 
            (classification_results['trial_id'] == trial_id)
        ].copy()
        
        if len(trial_data) == 0:
            print(f"    No pipeline data found for {animal_id} {trial_id}")
            return {
                'latency_to_hunt': None,
                # 'latency_to_attack': None, 
                'hunting_duration': None
            }
        
        # Sort by frame to ensure correct temporal order
        trial_data = trial_data.sort_values('frame')
        
        timing_events = {}
        
        # Extract latency to hunt (first chasing behavior)
        chasing_frames = trial_data[trial_data['behavior'] == 'chasing']
        if len(chasing_frames) > 0:
            timing_events['latency_to_hunt'] = chasing_frames['frame'].min()
        else:
            timing_events['latency_to_hunt'] = None
        
        # Extract latency to attack (first attack behavior)
        attack_frames = trial_data[trial_data['behavior'] == 'attack']
        # if len(attack_frames) > 0:
        #     timing_events['latency_to_attack'] = attack_frames['frame'].min()
        # else:
        timing_events['latency_to_attack'] = None
        
        # Extract hunting duration (last frame of hunting sequence)
        # This could be last attack frame, or we could define it differently
        # For now, let's use the last frame where behavior is not 'background'
        # hunting_frames = trial_data[trial_data['behavior'] != 'background']
        # # TODO: what if we do it as last attack frame?
        # if len(hunting_frames) > 0:
        #     timing_events['hunting_duration'] = hunting_frames['frame'].max()
        # else:
        #     timing_events['hunting_duration'] = None
        if len(attack_frames) > 0:
            timing_events['hunting_duration'] = attack_frames['frame'].max()+1
        else:
            # If there are no attack frames, the duration event is also considered missing.
            timing_events['hunting_duration'] = None
                
        return timing_events
    
    def load_pipeline_results(self, results_directory: str) -> pd.DataFrame:
        """
        Load and consolidate pipeline classification results.
        
        Args:
            results_directory: Directory containing pipeline analysis CSV files
            
        Returns:
            pd.DataFrame: Consolidated pipeline results
        """
        results_dir = Path(results_directory)
        
        # Find all analysis CSV files
        analysis_files = list(results_dir.glob('*_analysis.csv'))
        
        if not analysis_files:
            raise ValueError(f"No *_analysis.csv files found in {results_directory}")
        
        print(f"Found {len(analysis_files)} pipeline result files")
        
        all_results = []
        
        for file_path in analysis_files:
            try:
                # Extract animal and trial ID from filename
                # Assuming format like "m14_t1_validated_analysis.csv"
                filename = file_path.stem
                
                # Extract animal_id (e.g., m14, m17)
                animal_match = filename.split('_')[0]  # First part should be animal ID
                trial_match = filename.split('_')[1] if '_' in filename else 'unknown'   # Second part should be trial ID
                
                # Load the results
                df = pd.read_csv(file_path)
                
                # Add animal and trial identifiers
                df['animal_id'] = animal_match
                df['trial_id'] = trial_match
                df['source_file'] = file_path.name
                
                all_results.append(df)
                print(f"  Loaded {len(df)} frames from {animal_match} {trial_match}")
                
            except Exception as e:
                print(f"  Warning: Could not load {file_path.name}: {e}")
                continue
        
        if not all_results:
            raise ValueError("No pipeline results could be loaded")
        
        # Combine all results
        self.pipeline_results = pd.concat(all_results, ignore_index=True)
        
        print(f"Total pipeline results: {len(self.pipeline_results)} frames")
        print(f"Animals: {self.pipeline_results['animal_id'].unique()}")
        print(f"Behaviors: {self.pipeline_results['behavior'].unique()}")
        
        return self.pipeline_results
    
    def compare_timing(self) -> pd.DataFrame:
        """
        Compare pipeline timing predictions with ground truth.
        
        Returns:
            pd.DataFrame: Comparison results
        """
        if self.ground_truth is None:
            raise ValueError("Ground truth not loaded. Call load_ground_truth() first.")
        
        if self.pipeline_results is None:
            raise ValueError("Pipeline results not loaded. Call load_pipeline_results() first.")
        
        print("Comparing pipeline results with ground truth...")
        
        comparison_rows = []
        
        for _, gt_row in self.ground_truth.iterrows():
            animal_id = gt_row['animal_id']
            trial_id = gt_row['trial_id']
            
            print(f"\nProcessing {animal_id} {trial_id}:")
            
            # Extract pipeline timing for this trial
            pipeline_timing = self.extract_pipeline_timing(
                self.pipeline_results, animal_id, trial_id
            )
            
            # Compare each timing event
            comparison_row = {
                'animal_id': animal_id,
                'trial_id': trial_id,
                'notes': gt_row.get('notes', '')
            }
            
            timing_events = ['latency_to_hunt', 'hunting_duration'] #, 'latency_to_attack']
            
            for event in timing_events:
                gt_value = gt_row[event] if pd.notna(gt_row[event]) else None
                pipeline_value = pipeline_timing[event]
                
                comparison_row[f'gt_{event}'] = gt_value
                comparison_row[f'pipeline_{event}'] = pipeline_value
                
                # Calculate difference
                if gt_value is not None and pipeline_value is not None:
                    diff = pipeline_value - gt_value
                    comparison_row[f'diff_{event}'] = diff
                    abs_diff = abs(diff)
                    comparison_row[f'abs_diff_{event}'] = abs_diff
                    
                    print(f"  {event}: GT={gt_value}, Pipeline={pipeline_value}, Diff={diff} ({abs_diff} frames)")
                else:
                    comparison_row[f'diff_{event}'] = None
                    comparison_row[f'abs_diff_{event}'] = None
                    
                    if gt_value is None and pipeline_value is None:
                        print(f"  {event}: Both missing")
                    elif gt_value is None:
                        print(f"  {event}: GT missing, Pipeline={pipeline_value}")
                    else:
                        print(f"  {event}: GT={gt_value}, Pipeline missing")
            
            comparison_rows.append(comparison_row)
        
        self.comparison_results = pd.DataFrame(comparison_rows)
        
        print(f"\nComparison complete: {len(self.comparison_results)} trials")
        
        return self.comparison_results
    
    def generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics for the timing comparison."""
        if self.comparison_results is None:
            raise ValueError("No comparison results available. Call compare_timing() first.")
        
        print("\n=== TIMING VALIDATION SUMMARY ===")
        
        summary = {
            'total_trials': len(self.comparison_results),
            'events': {}
        }
        
        timing_events = ['latency_to_hunt', 'hunting_duration'] #, 'latency_to_attack']
        
        for event in timing_events:
            gt_col = f'gt_{event}'
            pipeline_col = f'pipeline_{event}'
            diff_col = f'diff_{event}'
            abs_diff_col = f'abs_diff_{event}'
            
            # Count available comparisons
            valid_comparisons = self.comparison_results[
                self.comparison_results[gt_col].notna() & 
                self.comparison_results[pipeline_col].notna()
            ]
            
            gt_available = self.comparison_results[gt_col].notna().sum()
            pipeline_available = self.comparison_results[pipeline_col].notna().sum()
            both_available = len(valid_comparisons)
            
            event_summary = {
                'gt_available': gt_available,
                'pipeline_available': pipeline_available, 
                'both_available': both_available,
                'comparison_rate': both_available / gt_available if gt_available > 0 else 0
            }
            
            if both_available > 0:
                differences = valid_comparisons[diff_col]
                abs_differences = valid_comparisons[abs_diff_col]
                
                event_summary.update({
                    'mean_diff': differences.mean(),
                    'std_diff': differences.std(),
                    'median_diff': differences.median(),
                    'mean_abs_diff': abs_differences.mean(),
                    'median_abs_diff': abs_differences.median(),
                    'max_abs_diff': abs_differences.max(),
                    'q25_abs_diff': abs_differences.quantile(0.25),
                    'q75_abs_diff': abs_differences.quantile(0.75)
                })
            
            summary['events'][event] = event_summary
            
            # Print summary
            print(f"\n{event.upper()}:")
            print(f"  Ground truth available: {gt_available}/{summary['total_trials']}")
            print(f"  Pipeline predictions: {pipeline_available}/{summary['total_trials']}")
            print(f"  Comparable pairs: {both_available}/{gt_available} ({event_summary['comparison_rate']*100:.1f}%)")
            
            if both_available > 0:
                print(f"  Mean difference: {event_summary['mean_diff']:.1f} ± {event_summary['std_diff']:.1f} frames")
                print(f"  Median difference: {event_summary['median_diff']:.1f} frames")
                print(f"  Mean absolute difference: {event_summary['mean_abs_diff']:.1f} frames")
                print(f"  Median absolute difference: {event_summary['median_abs_diff']:.1f} frames")
        
        return summary
    
    def create_validation_plots(self, output_dir: str) -> None:
        """Create validation plots and save to output directory."""
        if self.comparison_results is None:
            raise ValueError("No comparison results available. Call compare_timing() first.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timing_events = ['latency_to_hunt', 'hunting_duration'] #, 'latency_to_attack']
        
        # 1. Scatter plots: Ground Truth vs Pipeline Predictions
        # fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for i, event in enumerate(timing_events):
            gt_col = f'gt_{event}'
            pipeline_col = f'pipeline_{event}'
            
            # Get valid data points
            valid_data = self.comparison_results[
                self.comparison_results[gt_col].notna() & 
                self.comparison_results[pipeline_col].notna()
            ]
            
            if len(valid_data) > 0:
                axes[i].scatter(valid_data[gt_col], valid_data[pipeline_col], alpha=0.6)
                
                # Add perfect correlation line
                min_val = min(valid_data[gt_col].min(), valid_data[pipeline_col].min())
                max_val = max(valid_data[gt_col].max(), valid_data[pipeline_col].max())
                axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect match')
                
                axes[i].set_xlabel('Ground Truth (frames)')
                axes[i].set_ylabel('Pipeline Prediction (frames)')
                axes[i].set_title(f'{event.replace("_", " ").title()}\nn={len(valid_data)}')
                axes[i].legend()
                
                # Calculate and show correlation
                correlation = valid_data[gt_col].corr(valid_data[pipeline_col])
                axes[i].text(0.05, 0.95, f'r = {correlation:.3f}', 
                           transform=axes[i].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                axes[i].text(0.5, 0.5, 'No valid\ncomparisons', 
                           transform=axes[i].transAxes, ha='center', va='center')
                axes[i].set_title(f'{event.replace("_", " ").title()}\nn=0')
        
        plt.tight_layout()
        plt.savefig(output_path / 'timing_correlation_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Difference distribution plots
        # fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for i, event in enumerate(timing_events):
            diff_col = f'diff_{event}'
            
            # Get valid differences
            valid_diffs = self.comparison_results[self.comparison_results[diff_col].notna()][diff_col]
            
            if len(valid_diffs) > 0:
                axes[i].hist(valid_diffs, bins=20, alpha=0.7, edgecolor='black')
                axes[i].axvline(0, color='red', linestyle='--', alpha=0.8, label='Perfect match')
                axes[i].axvline(valid_diffs.mean(), color='green', linestyle='--', alpha=0.8, 
                              label=f'Mean: {valid_diffs.mean():.1f}')
                
                axes[i].set_xlabel('Difference (Pipeline - Ground Truth) frames')
                axes[i].set_ylabel('Frequency')
                axes[i].set_title(f'{event.replace("_", " ").title()} Differences\nn={len(valid_diffs)}')
                axes[i].legend()
            else:
                axes[i].text(0.5, 0.5, 'No valid\ncomparisons', 
                           transform=axes[i].transAxes, ha='center', va='center')
                axes[i].set_title(f'{event.replace("_", " ").title()}\nn=0')
        
        plt.tight_layout()
        plt.savefig(output_path / 'timing_difference_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # # 3. Per-animal comparison
        # animals = self.comparison_results['animal_id'].unique()
        
        # if len(animals) > 1:
        #     fig, ax = plt.subplots(figsize=(12, 8))
            
        #     # Create summary by animal
        #     animal_summaries = []
        #     for animal in animals:
        #         animal_data = self.comparison_results[self.comparison_results['animal_id'] == animal]
                
        #         for event in timing_events:
        #             abs_diff_col = f'abs_diff_{event}'
        #             valid_diffs = animal_data[animal_data[abs_diff_col].notna()][abs_diff_col]
                    
        #             if len(valid_diffs) > 0:
        #                 animal_summaries.append({
        #                     'animal': animal,
        #                     'event': event,
        #                     'mean_abs_diff': valid_diffs.mean(),
        #                     'n_trials': len(valid_diffs)
        #                 })
            
        #     if animal_summaries:
        #         summary_df = pd.DataFrame(animal_summaries)
                
        #         # Pivot for plotting
        #         pivot_df = summary_df.pivot(index='animal', columns='event', values='mean_abs_diff')
                
        #         # Create grouped bar plot
        #         pivot_df.plot(kind='bar', ax=ax, rot=45)
        #         ax.set_ylabel('Mean Absolute Difference (frames)')
        #         ax.set_title('Timing Accuracy by Animal')
        #         ax.legend(title='Event', bbox_to_anchor=(1.05, 1), loc='upper left')
                
        #         plt.tight_layout()
        #         plt.savefig(output_path / 'timing_accuracy_by_animal.png', dpi=300, bbox_inches='tight')
        #         plt.close()
                # --- MODIFIED: Changed from per-animal to per-trial comparison ---
        # 3. Per-trial comparison to spot outliers
        
        # Create a unique label for each trial (e.g., 'm14_t1')
        plot_data = self.comparison_results.copy()
        plot_data['trial_label'] = plot_data['animal_id'] + '_' + plot_data['trial_id']
        
        # Select the columns with absolute differences
        diff_cols = [f'abs_diff_{event}' for event in timing_events]
        
        # Filter out trials where there is no data to plot
        plot_data = plot_data.dropna(subset=diff_cols, how='all').set_index('trial_label')
        
        if not plot_data.empty:
            # Dynamically adjust figure height based on number of trials
            num_trials = len(plot_data)
            fig_height = max(8, num_trials * 0.4) # Ensure a minimum height
            
            fig, ax = plt.subplots(figsize=(12, fig_height))
            
            # Create the bar plot
            plot_data[diff_cols].plot(kind='barh', ax=ax, width=0.8) # Use horizontal bars for readability
            
            ax.set_xlabel('Absolute Difference (frames)')
            ax.set_ylabel('Trial')
            ax.set_title('Timing Accuracy by Trial')
            ax.grid(axis='x', linestyle='--', alpha=0.6)
            
            # Invert y-axis so trials are in a more natural order
            ax.invert_yaxis()
            
            # Clean up legend labels
            handles, labels = ax.get_legend_handles_labels()
            new_labels = [l.replace('abs_diff_', '').replace('_', ' ') for l in labels]
            ax.legend(handles, new_labels, title='Event', bbox_to_anchor=(1.02, 1), loc='upper left')
            
            plt.tight_layout()
            plt.savefig(output_path / 'timing_accuracy_by_trial.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            print("Skipping per-trial plot: No valid absolute differences to plot.")

        # --- NEW: 4. Summary bar plot with animals as samples ---
        print("Creating summary bar plot with per-animal means...")
        
        # Calculate per-animal means
        animal_means = []
        for animal in self.comparison_results['animal_id'].unique():
            animal_data = self.comparison_results[self.comparison_results['animal_id'] == animal]
            
            animal_summary = {'animal_id': animal}
            for event in timing_events:
                abs_diff_col = f'abs_diff_{event}'
                valid_diffs = animal_data[animal_data[abs_diff_col].notna()][abs_diff_col]
                
                if len(valid_diffs) > 0:
                    animal_summary[event] = valid_diffs.mean()
                else:
                    animal_summary[event] = np.nan
            
            animal_means.append(animal_summary)
        
        animal_means_df = pd.DataFrame(animal_means)
        
        # Calculate mean and SEM across animals for each event
        summary_stats = []
        for event in timing_events:
            valid_animal_means = animal_means_df[animal_means_df[event].notna()][event]
            
            if len(valid_animal_means) > 0:
                summary_stats.append({
                    'event': event.replace('_', ' ').title(),
                    'mean': valid_animal_means.mean(),
                    'sem': valid_animal_means.sem(),
                    'n_animals': len(valid_animal_means)
                })
        
        if summary_stats:
            summary_df = pd.DataFrame(summary_stats)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x_pos = np.arange(len(summary_df))
            bars = ax.bar(x_pos, summary_df['mean'], yerr=summary_df['sem'], 
                         capsize=10, alpha=0.8, color=['steelblue', 'coral'])
            
            ax.set_xlabel('Timing Event', fontsize=14, weight='bold')
            ax.set_ylabel('Mean Absolute Difference (frames)', fontsize=14, weight='bold')
            ax.set_title('Pipeline Timing Accuracy\n(Animals as Samples)', fontsize=16, weight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(summary_df['event'], fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add n= labels on bars
            for i, (bar, n) in enumerate(zip(bars, summary_df['n_animals'])):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + summary_df.iloc[i]['sem'],
                       f'n={int(n)} animals',
                       ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(output_path / 'timing_accuracy_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved summary bar plot (n={summary_df['n_animals'].iloc[0]} animals)")
        else:
            print("Skipping summary bar plot: No valid data")

        
        print(f"Validation plots saved to: {output_path}")
    
    def save_detailed_results(self, output_path: str) -> None:
        """Save detailed comparison results to CSV."""
        if self.comparison_results is None:
            raise ValueError("No comparison results available. Call compare_timing() first.")
        
        self.comparison_results.to_csv(output_path, index=False)
        print(f"Detailed results saved to: {output_path}")
    
    def run_complete_validation(self, pipeline_results_dir: str, output_dir: str) -> Dict[str, Any]:
        """Run complete validation pipeline."""
        print("🎯 STARTING COMPLETE TIMING VALIDATION")
        
        # Load data
        self.load_ground_truth()
        self.load_pipeline_results(pipeline_results_dir)
        
        # Compare timing
        self.compare_timing()
        
        # Generate summary
        summary = self.generate_summary_statistics()
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save results
        self.save_detailed_results(output_path / 'timing_comparison_detailed.csv')
        self.create_validation_plots(output_dir)
        
        # Save summary to JSON
        import json
        with open(output_path / 'timing_validation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n✅ Complete validation finished!")
        print(f"Results saved to: {output_path}")
        
        return summary

def main():
    """Command line interface for timing validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate pipeline timing predictions")
    parser.add_argument("--ground-truth", required=True, help="Ground truth CSV file")
    parser.add_argument("--pipeline-results", required=True, help="Pipeline results directory") 
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    
    args = parser.parse_args()
    
    try:
        validator = TimingValidator(args.ground_truth)
        summary = validator.run_complete_validation(args.pipeline_results, args.output_dir)
        
        print(f"\n🎉 Validation complete! Check {args.output_dir} for detailed results.")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()
#     python utils/timing_validator.py \
#   --ground-truth ground_truth_timing.csv \
#   --pipeline-results "/home/tarislada/Documents/Extra_python_projects/SKH FP/FInalized_process/prediction_plots/timing_analysis_files/Strategy_2:_With_Thresholding" \
#   --output-dir "/home/tarislada/Documents/Extra_python_projects/SKH FP/FInalized_process/prediction_plots/timing_analysis_files/timing_validation_results/"