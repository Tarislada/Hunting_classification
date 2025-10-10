import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import os
import glob
import re
from typing import List, Dict, Optional, Tuple
from scipy.stats import pearsonr, spearmanr

class TrackingDataVisualizer:
    """
    Visualizer for animal tracking pipeline data.
    Creates histograms and statistical summaries for distance, speed, cricket angle, and head angle.
    """
    
    def __init__(self, analysis_dir: str, cricket_dir: str, txt_dir:str, label_dir: str, output_dir: str, 
             instance_dir: Optional[str] = None,
             use_predictions: bool = False,  # <-- NEW FLAG
             prediction_dir: Optional[str] = None):  # <-- NEW PARAMETER
        """
        Initialize the visualizer with input and output directories.
        
        Args:
            analysis_dir: Directory containing *_analysis.csv files.
            cricket_dir: Directory containing crprocessed_*.csv files.
            txt_dir: Directory containing txt files for analysis window.
            label_dir: Directory containing *_processed_labels.csv files.
            output_dir: Directory to save visualization outputs.
            instance_dir: If provided, activates instance-based probability analysis.
        """
        self.analysis_dir = Path(analysis_dir)
        self.cricket_dir = Path(cricket_dir)
        self.txt_dir = Path(txt_dir)
        self.label_dir = Path(label_dir)
        self.output_dir = Path(output_dir)
        # --- MODIFIED: Store the path if it exists ---
        self.instance_dir = Path(instance_dir) if instance_dir else None
        self.use_instance_based = self.instance_dir is not None
        self.use_predictions = use_predictions
        self.prediction_dir = Path(prediction_dir) if prediction_dir else None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_analysis_data(self) -> Dict[str, pd.DataFrame]:
        """Load all analysis CSV files."""
        analysis_files = list(self.analysis_dir.glob('*_analysis.csv'))
        analysis_data = {}
        
        print(f"Found {len(analysis_files)} analysis files")
        
        for file_path in analysis_files:
            try:
                df = pd.read_csv(file_path)
                file_key = file_path.stem.replace('_analysis', '')
                analysis_data[file_key] = df
                print(f"Loaded {file_path.name}: {len(df)} frames")
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
                
        return analysis_data
    def load_prediction_data(self) -> Dict[str, pd.DataFrame]:
        """Load model prediction files (same format as labels, but from prediction directory)."""
        if not self.prediction_dir:
            return {}
            
        prediction_files = list(self.prediction_dir.glob('*_analysis.csv'))
        prediction_data = {}

        print(f"Found {len(prediction_files)} prediction files")

        for file_path in prediction_files:
            try:
                df = pd.read_csv(file_path)
                # Extract file key (e.g., 'm14_t1' from 'm14_t1_analysis.csv')
                file_key = file_path.stem.replace('_analysis', '')
                prediction_data[file_key] = df
                print(f"Loaded {file_path.name}: {len(df)} frames")
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
                
        return prediction_data
    
    def load_cricket_data(self) -> Dict[str, pd.DataFrame]:
        """Load all cricket processing CSV files."""
        cricket_files = list(self.cricket_dir.glob('crprocessed_*.csv'))
        cricket_data = {}
        
        print(f"Found {len(cricket_files)} cricket files")
        
        for file_path in cricket_files:
            try:
                df = pd.read_csv(file_path)
                file_key = file_path.stem.replace('crprocessed_', '')
                cricket_data[file_key] = df
                print(f"Loaded {file_path.name}: {len(df)} frames")
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
                
        return cricket_data
    
    def load_label_data(self) -> Dict[str, pd.DataFrame]:
        """Load all behavior label CSV files."""
        label_files = list(self.label_dir.glob('*_processed_labels.csv'))
        label_data = {}
        
        print(f"Found {len(label_files)} label files")
        
        for file_path in label_files:
            try:
                df = pd.read_csv(file_path)
                file_key = file_path.stem.replace('_processed_labels', '')
                label_data[file_key] = df
                print(f"Loaded {file_path.name}: {len(df)} frames")
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
                
        return label_data
    
    def load_instance_data(self) -> Dict[str, List[Dict]]:
        """Load instance data from raw Excel files."""
        if not self.instance_dir:
            return {}
            
        instance_files = list(self.instance_dir.glob('*.xlsx'))
        instance_data = {}
        print(f"\nLoading instance data from {len(instance_files)} Excel files...")

        for file_path in instance_files:
            try:
                df = pd.read_excel(file_path)
                match = re.search(r'(m\d+_t\d+)', file_path.stem)
                if not match:
                    print(f"  Warning: Could not extract a valid key from '{file_path.name}'. Skipping.")
                    continue
                file_key = match.group(1)
                
                # Standardize column names (e.g., 'chasing_start', 'chasing_end')
                df.columns = [col.lower().replace(' ', '_') for col in df.columns]
                
                # 1. Extract all approaching and turning events
                all_events = []
                event_types = {'approaching': ('approaching_start', 'approaching_end'),
                               'turning': ('turning_start', 'turning_end')}

                for behavior, (start_col, end_col) in event_types.items():
                    if start_col in df.columns and end_col in df.columns:
                        for _, row in df.iterrows():
                            if pd.notna(row[start_col]) and pd.notna(row[end_col]):
                                all_events.append({
                                    'start_frame': int(row[start_col]),
                                    'end_frame': int(row[end_col])
                                })
                
                if not all_events:
                    instance_data[file_key] = []
                    continue

                # 2. Sort all events by start frame
                all_events.sort(key=lambda x: x['start_frame'])

                # 3. Merge consecutive events
                merged_instances = []
                if all_events:
                    current_instance = all_events[0]
                    
                    for next_event in all_events[1:]:
                        # Check for immediate succession
                        if next_event['start_frame'] == current_instance['end_frame'] + 1:
                            # Merge by extending the end frame
                            current_instance['end_frame'] = next_event['end_frame']
                        else:
                            # Not consecutive, so finalize the current instance and start a new one
                            merged_instances.append(current_instance)
                            current_instance = next_event
                    
                    # Add the last processed instance
                    merged_instances.append(current_instance)

                # 4. Finalize all as 'chasing' behavior
                for inst in merged_instances:
                    inst['behavior'] = 'chasing'

                instance_data[file_key] = merged_instances
                print(f"  Loaded and merged into {len(merged_instances)} chasing instances from {file_path.name}")

            except Exception as e:
                print(f"Error loading instance file {file_path.name}: {e}")
        
        return instance_data

    def merge_data(self, analysis_data: Dict[str, pd.DataFrame], 
                   cricket_data: Dict[str, pd.DataFrame],
                   label_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge analysis, cricket, and label data based on matching file names."""
        merged_dfs = []
        
        # Use label_data as the primary key source, as it defines the ground truth
        for file_key, label_df in label_data.items():
            analysis_key = file_key + '_validated' # e.g., m14_t7 -> m14_t7_validated
            cricket_key = file_key # e.g., m14_t7

            if analysis_key in analysis_data and cricket_key in cricket_data:
                analysis_df = analysis_data[analysis_key]
                cricket_df = cricket_data[cricket_key]

                # Perform a three-way merge
                # 1. Merge analysis and cricket data
                temp_merge = pd.merge(analysis_df, cricket_df[['frame', 'speed', 'size', 'status']], on='frame', how='inner')
                # 2. Merge the result with label data
                full_merge = pd.merge(temp_merge, label_df, on='frame', how='inner')

                # Now, apply the time window filtering from the .txt file
                txt_path = self.txt_dir / f"{cricket_key}.txt"
                if txt_path.exists():
                    try:
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            cricket_in_frame, cricket_out_frame = self.get_cricket_frames(f.read())
                        
                        # Filter the fully merged dataframe
                        filtered_df = full_merge[
                            (full_merge['frame'] >= cricket_in_frame) & 
                            (full_merge['frame'] <= cricket_out_frame)
                        ].copy()
                        
                        filtered_df['file'] = cricket_key
                        merged_dfs.append(filtered_df)
                        print(f"Merged and filtered data for {file_key}: {len(filtered_df)} frames")

                    except Exception as e:
                        print(f"Error processing text file for {cricket_key}, skipping merge: {e}")
                else:
                    print(f"Warning: Text file not found for {cricket_key}, skipping merge.")
            else:
                print(f"Warning: Missing data for {file_key}. Analysis found: {analysis_key in analysis_data}, Cricket found: {cricket_key in cricket_data}")
                
        if merged_dfs:

            all_merged = pd.concat(merged_dfs, ignore_index=True)
            all_merged['frame'] = all_merged['frame'].astype(int)
            all_merged = self.calculate_mouse_speed(all_merged)
            return all_merged
        else:
            return pd.DataFrame()
    
    def calculate_statistics(self, data: pd.DataFrame) -> Dict:
        """Calculate summary statistics for each metric."""
        metrics = ['distance', 'cricket_angle', 'head_angle', 'speed', 'mouse_speed']
        stats = {}
        
        for metric in metrics:
            if metric in data.columns:
                valid_data = data[metric].dropna()
                if len(valid_data) > 0:
                    stats[metric] = {
                        'count': len(valid_data),
                        'mean': valid_data.mean(),
                        'median': valid_data.median(),
                        'std': valid_data.std(),
                        'min': valid_data.min(),
                        'max': valid_data.max(),
                        'q25': valid_data.quantile(0.25),
                        'q75': valid_data.quantile(0.75)
                    }
        
        return stats
    def calculate_mouse_speed(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate mouse speed based on body_center coordinates."""
        print("Calculating mouse speed...")
        if 'body_center' not in data.columns:
            print("Warning: 'body_center' column not found. Cannot calculate mouse speed.")
            data['mouse_speed'] = np.nan
            return data

        # Sort data by file and frame to ensure correct calculation
        coords = data['body_center'].str.strip('()').str.split(', ', expand=True)
        data['body_center_x'] = pd.to_numeric(coords[0], errors='coerce')
        data['body_center_y'] = pd.to_numeric(coords[1], errors='coerce')
        
        # Sort data by file and frame to ensure correct calculation
        data = data.sort_values(by=['file', 'frame']).reset_index(drop=True)
        
        # Calculate displacement per file
        data['dx'] = data.groupby('file')['body_center_x'].diff()
        data['dy'] = data.groupby('file')['body_center_y'].diff()
        
        # Speed is the hypotenuse of the displacement
        data['mouse_speed'] = np.sqrt(data['dx']**2 + data['dy']**2)
        
        # Remove intermediate columns
        data = data.drop(columns=['dx', 'dy'])
        print("Mouse speed calculation complete.")
        return data


    def get_cricket_frames(self, txt_content: str) -> Tuple[int, int]:
        """
        Extract cricket start frame and final frame from text file.
        Same logic as in cricket_interpv4.py
        """
        
        # Extract cricket in frame
        in_match = re.search(r"cricket in 부터\((\d+)\)", txt_content)
        if not in_match:
            raise ValueError("Cricket start frame not found in text file.")
        cricket_in_frame = int(in_match.group(1))
        
        # First attempt: Look for the last 'consume' entry
        lines = txt_content.strip().split('\n')
        try:
            # Find the line index where the 'attack' section starts
            attack_section_start_index = lines.index('attack')
            attack_numbers = []
            
            # Iterate through lines immediately following the 'attack' header
            for line in lines[attack_section_start_index + 1:]:
                # Stop if we hit a blank line or a non-numeric line (new section)
                if not line.strip() or not line.strip()[0].isdigit():
                    break
                
                # Extract the number from the start of the line
                num_match = re.match(r'\s*(\d+)', line)
                if num_match:
                    attack_numbers.append(int(num_match.group(1)))
            
            if attack_numbers:
                # If we found numbers, use the largest one and return
                cricket_out_frame = max(attack_numbers)
                print(f"Found cricket_out_frame from 'attack' section: {cricket_out_frame}")
                return cricket_in_frame, cricket_out_frame

        except ValueError:
            # This means 'attack' section was not found, so we proceed to fallbacks
            pass
        
        # Fallback 1: Look for the last 'consume' entry
        consume_lines = [line for line in lines if line.strip().endswith('consume')]
        if consume_lines:
            last_consume = consume_lines[-1]
            try:
                cricket_out_frame = int(last_consume.split('\t')[1])
                print(f"Found cricket_out_frame from 'consume' line: {cricket_out_frame}")
                return cricket_in_frame, cricket_out_frame
            except (IndexError, ValueError):
                pass
        
        # Fallback 2: Find the largest number in the text
        all_numbers = re.findall(r'\d+', txt_content)
        if not all_numbers:
            raise ValueError("No frame numbers found in text file")
        
        cricket_out_frame = max(int(num) for num in all_numbers)
        print(f"Found cricket_out_frame using fallback (max number in file): {cricket_out_frame}")
        
        return cricket_in_frame, cricket_out_frame


    def plot_histograms(self, data: pd.DataFrame, save_individual: bool = True) -> None:
        """Create histogram plots for each metric."""
        metrics_config = {
            'distance': {'title': 'Cricket-Mouse Distance', 'xlabel': 'Distance (pixels)', 'color': 'skyblue'},
            'speed': {'title': 'Cricket Movement Speed', 'xlabel': 'Speed (pixels/frame)', 'color': 'lightcoral'},
            'cricket_angle': {
                'title': 'Cricket Angle', 
                'xlabel': 'Angle (degrees)', 
                'color': 'lightgreen', 
                'xlim': (-45, 45),
                'bins': np.arange(-45, 45 + 1, 2.5) # Creates bins of width 5 from -45 to 45
            },
            'head_angle': {'title': 'Mouse Head Angle', 'xlabel': 'Angle (degrees)', 'color': 'plum'},
            'mouse_speed': {'title': 'Mouse Movement Speed', 'xlabel': 'Speed (pixels/frame)', 'color': 'gold'}
        }
        
        # Create individual histograms
        if save_individual:
            for metric, config in metrics_config.items():
                if metric in data.columns:
                    self._plot_single_histogram(data, metric, config)
        combined_metrics_config = {k: v for k, v in metrics_config.items() if k != 'speed'}

        # Create combined plot
        self._plot_combined_histograms(data, combined_metrics_config)
    
    def _plot_single_histogram(self, data: pd.DataFrame, metric: str, config: Dict) -> None:
        """Plot histogram for a single metric."""
        valid_data = data[metric].dropna()
        
        if len(valid_data) == 0:
            print(f"No valid data for {metric}")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(valid_data, bins=config.get('bins', 50), alpha=0.7, color=config['color'], edgecolor='black')
        ax1.set_title(f'{config["title"]} - Histogram')
        ax1.set_xlabel(config['xlabel'])
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        if 'xlim' in config:
            ax1.set_xlim(config['xlim'])
        
        # Add statistics text
        mean_val = valid_data.mean()
        median_val = valid_data.median()
        std_val = valid_data.std()
        
        stats_text = f'Count: {len(valid_data)}\n'
        stats_text += f'Mean: {mean_val:.2f}\n'
        stats_text += f'Median: {median_val:.2f}\n'
        stats_text += f'Std: {std_val:.2f}'
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Box plot
        ax2.boxplot(valid_data, vert=True, patch_artist=True,
                   boxprops=dict(facecolor=config['color'], alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        ax2.set_title(f'{config["title"]} - Box Plot')
        ax2.set_ylabel(config['xlabel'])
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{metric}_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved individual histogram for {metric}")
    
    def _plot_combined_histograms(self, data: pd.DataFrame, metrics_config: Dict) -> None:
        """Create a combined plot with all histograms."""
        num_metrics = len(metrics_config)
        nrows = (num_metrics + 1) // 2
        fig, axes = plt.subplots(nrows, 2, figsize=(16, 6 * nrows))
        axes = axes.flatten()
        
        for idx, (metric, config) in enumerate(metrics_config.items()):
            if metric in data.columns:
                valid_data = data[metric].dropna()
                
                if len(valid_data) > 0:
                    axes[idx].hist(valid_data, bins=config.get('bins', 40), alpha=0.7, 
                                 color=config['color'], edgecolor='black')
                    axes[idx].set_title(config['title'])
                    axes[idx].set_xlabel(config['xlabel'])
                    axes[idx].set_ylabel('Frequency')
                    axes[idx].grid(True, alpha=0.3)
                    if 'xlim' in config:
                        axes[idx].set_xlim(config['xlim'])
                    
                    # Add basic stats
                    mean_val = valid_data.mean()
                    median_val = valid_data.median()
                    axes[idx].axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.1f}')
                    axes[idx].axvline(median_val, color='blue', linestyle='--', alpha=0.7, label=f'Median: {median_val:.1f}')
                    axes[idx].legend()
                else:
                    axes[idx].text(0.5, 0.5, f'No data for {metric}', 
                                 ha='center', va='center', transform=axes[idx].transAxes)
            else:
                axes[idx].text(0.5, 0.5, f'{metric} not found', 
                             ha='center', va='center', transform=axes[idx].transAxes)
                
        # Hide the last empty subplot if we have an odd number of plots
        for i in range(num_metrics, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'combined_histograms.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved combined histogram plot")
    
    def plot_zone_distribution(self, data: pd.DataFrame) -> None:
        """Plot distribution of visual field zones."""
        if 'zone' in data.columns:
            zone_counts = data['zone'].value_counts()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Bar plot
            zone_counts.plot(kind='bar', ax=ax1, color=['green', 'orange', 'red', 'gray'])
            ax1.set_title('Visual Field Zone Distribution')
            ax1.set_xlabel('Zone')
            ax1.set_ylabel('Count')
            ax1.tick_params(axis='x', rotation=45)
            
            # Pie chart
            ax2.pie(zone_counts.values, labels=zone_counts.index, autopct='%1.1f%%', 
                   colors=['green', 'orange', 'red', 'gray'])
            ax2.set_title('Visual Field Zone Proportions')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'zone_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("Saved zone distribution plot")
            
    def plot_angle_speed_correlation(self, data: pd.DataFrame) -> None:
        """Plot correlation for both original and log-transformed speed, and return stats."""
        print("Plotting mouse angle-speed correlation (original and transformed)...")
        if 'head_angle' not in data.columns or 'mouse_speed' not in data.columns:
            print("Warning: 'head_angle' or 'mouse_speed' not available for correlation plot.")
            return None

        sample_data = data.sample(n=min(5000, len(data)), random_state=42).copy()
        sample_data['abs_head_angle'] = sample_data['head_angle'].abs()
        corr_data = sample_data[['abs_head_angle', 'mouse_speed']].dropna()
        
        if len(corr_data) < 2:
            print("Not enough data to calculate correlation.")
            return None

        all_corr_stats = {}

        # --- 1. Original Data Plot ---
        rho_orig, p_orig = spearmanr(corr_data['abs_head_angle'], corr_data['mouse_speed'])
        all_corr_stats['original'] = {'rho_value': rho_orig, 'p_value': p_orig}
        
        fig, ax1 = plt.subplots(figsize=(10, 8))
        sns.regplot(data=corr_data, x='abs_head_angle', y='mouse_speed', ax=ax1,
                    scatter_kws={'alpha': 0.3, 's': 15, 'color': 'lightcoral'},
                    line_kws={'color': 'red'})
        ax1.set_title('Mouse Head Angle vs. Speed Correlation (Original)')
        ax1.set_xlabel('Absolute Head Angle (degrees)')
        ax1.set_ylabel('Mouse Speed (pixels/frame)')
        ax1.grid(True, alpha=0.3)
        stats_text_orig = f"Spearman's ρ: {rho_orig:.3f}\np-value: {p_orig:.3g}"
        ax1.text(0.95, 0.95, stats_text_orig, transform=ax1.transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        fig.savefig(self.output_dir / 'angle_speed_correlation.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Saved original angle-speed correlation plot.")

        # --- 2. Log-Transformed Data Plot ---
        corr_data['log_speed'] = np.log1p(corr_data['mouse_speed']) # Use log1p to handle speed = 0
        rho_log, p_log = spearmanr(corr_data['abs_head_angle'], corr_data['log_speed'])
        all_corr_stats['transformed'] = {'rho_value': rho_log, 'p_value': p_log}

        fig, ax2 = plt.subplots(figsize=(10, 8))
        sns.regplot(data=corr_data, x='abs_head_angle', y='log_speed', ax=ax2,
                    scatter_kws={'alpha': 0.3, 's': 15, 'color': 'skyblue'},
                    line_kws={'color': 'blue'})
        ax2.set_title('Mouse Head Angle vs. Log-Transformed Speed Correlation')
        ax2.set_xlabel('Absolute Head Angle (degrees)')
        ax2.set_ylabel('Log(1 + Mouse Speed)')
        ax2.grid(True, alpha=0.3)
        stats_text_log = f"Spearman's ρ: {rho_log:.3f}\np-value: {p_log:.3g}"
        ax2.text(0.95, 0.95, stats_text_log, transform=ax2.transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        fig.savefig(self.output_dir / 'angle_speed_correlation_log_transformed.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Saved log-transformed angle-speed correlation plot.")
        
        return all_corr_stats

    def plot_binned_speed_vs_angle(self, data: pd.DataFrame) -> None:
        """Plot mouse speed as a bar plot binned by head angle."""
        print("Plotting binned speed vs. angle...")
        if 'head_angle' not in data.columns or 'mouse_speed' not in data.columns:
            print("Warning: 'head_angle' or 'mouse_speed' not available for binned plot.")
            return

        plot_data = data[['head_angle', 'mouse_speed']].dropna().copy()
        plot_data['abs_head_angle'] = plot_data['head_angle'].abs()

        # Define bins for the absolute head angle
        bins = [0, 15, 30, 45, 60, 90, 180]
        labels = ['0-15°', '15-30°', '30-45°', '45-60°', '60-90°', '>90°']
        plot_data['angle_bin'] = pd.cut(plot_data['abs_head_angle'], bins=bins, labels=labels, right=False)

        # Calculate mean, std deviation, and count for speed in each bin
        bin_stats = plot_data.groupby('angle_bin')['mouse_speed'].agg(['mean', 'std', 'count']).reset_index()

        if bin_stats.empty or bin_stats['count'].sum() == 0:
            print("No data to plot for binned speed vs. angle.")
            return

        # Create the bar plot
        plt.figure(figsize=(12, 7))
        ax = sns.barplot(data=bin_stats, x='angle_bin', y='mean', color='mediumseagreen')
        
        # Add error bars
        plt.errorbar(x=bin_stats.index, y=bin_stats['mean'], yerr=bin_stats['std'], fmt='none', c='black', capsize=5)
        
        ax.set_title('Mean Mouse Speed by Head Angle Range')
        ax.set_xlabel('Absolute Head Angle Range')
        ax.set_ylabel('Mean Mouse Speed (pixels/frame)')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Add count labels on top of bars
        for i, p in enumerate(ax.patches):
            count = bin_stats.loc[i, 'count']
            ax.annotate(f'n={int(count)}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 15), 
                        textcoords='offset points',
                        color='black')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'binned_speed_vs_angle.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved binned speed vs. angle plot.")

    def plot_relative_speed_by_behavior(self, data: pd.DataFrame) -> None:
        """Plot the mouse-cricket relative speed for different behaviors using a hybrid violin/strip plot."""
        print("Plotting relative speed by behavior...")
        required_cols = ['distance', 'behavior', 'file', 'frame']
        if not all(col in data.columns for col in required_cols):
            print(f"Warning: Missing one of required columns {required_cols} for relative speed plot.")
            return

        plot_data = data.copy()
        plot_data = plot_data.sort_values(by=['file', 'frame'])
        plot_data['relative_speed'] = plot_data.groupby('file')['distance'].diff()

        behaviors_of_interest = ['chasing', 'attack', 'consume']
        filtered_data = plot_data[plot_data['behavior'].isin(behaviors_of_interest)].dropna(subset=['relative_speed'])

        if filtered_data.empty:
            print("No data found for 'chasing', 'attack', or 'consuming' behaviors to plot relative speed.")
            return

        plt.figure(figsize=(10, 8))
        
        # --- MODIFIED: Hybrid Plot Logic ---
        # Behaviors with variance get a violin plot
        violin_behaviors = ['chasing', 'attack']
        violin_data = filtered_data[filtered_data['behavior'].isin(violin_behaviors)]

        # Behaviors with no variance (like 'consuming') get a strip plot
        strip_behaviors = ['consume']
        strip_data = filtered_data[filtered_data['behavior'].isin(strip_behaviors)]

        # 1. Draw the violin plots for chasing and attack
        if not violin_data.empty:
            ax = sns.violinplot(
                data=violin_data, 
                x='behavior', 
                y='relative_speed', 
                order=violin_behaviors,
                palette='muted',
                inner='quartile',
                # REMOVED cut=0 to get smooth violins
            )
        
        # 2. Draw the strip plot for consuming
        if not strip_data.empty:
            sns.stripplot(
                data=strip_data,
                x='behavior',
                y='relative_speed',
                order=strip_behaviors,
                color=sns.color_palette('muted')[2], # Match the 3rd color in the palette
                jitter=0.1,
                size=5,
                alpha=0.7
            )

        ax.axhline(0, color='red', linestyle='--', lw=2, label='No Distance Change')
        
        lower_bound = filtered_data['relative_speed'].quantile(0.02)
        upper_bound = filtered_data['relative_speed'].quantile(0.98)
        ax.set_ylim(lower_bound * 1.5, upper_bound * 1.5)

        ax.set_title('Mouse-Cricket Relative Speed by Behavior', fontsize=16, weight='bold')
        ax.set_xlabel('Behavior', fontsize=14)
        ax.set_ylabel('Speed (pixels/frame)', fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'relative_speed_by_behavior_hybrid.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved enhanced relative speed by behavior plot (hybrid).")
        
    def plot_zone_usage_by_behavior(self, data: pd.DataFrame) -> None:
        """Plot zone usage (binocular vs monocular) across key behaviors with statistical analysis."""
        print("Plotting zone usage by behavior with statistical analysis...")
        
        if 'behavior' not in data.columns or 'zone' not in data.columns:
            print("Warning: 'behavior' or 'zone' not available for zone usage analysis.")
            return

        # Prepare data
        plot_data = data.copy()
        plot_data['simple_zone'] = plot_data['zone'].replace({
            'right_monocular': 'Monocular',
            'left_monocular': 'Monocular',
            'binocular': 'Binocular',
            'out_of_sight': 'Out of Sight'  # <-- ADD THIS
        })

        behaviors_of_interest = ['chasing', 'attack', 'consume']
        filtered_data = plot_data[plot_data['behavior'].isin(behaviors_of_interest)].copy()
        filtered_data['behavior'] = filtered_data['behavior'].str.capitalize()

        if filtered_data.empty:
            print("No data found for key behaviors.")
            return

        # --- Statistical Analysis: Chi-square test ---
        from scipy.stats import chi2_contingency
        
        # Create contingency table for attack vs. non-attack
        attack_data = plot_data.copy()
        attack_data['is_attack'] = (attack_data['behavior'] == 'attack').astype(int)
        contingency_table = pd.crosstab(attack_data['is_attack'], attack_data['simple_zone'])
        
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Calculate Cramér's V (effect size)
        n = contingency_table.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
        
        # Calculate conditional probabilities
        attack_bino = len(filtered_data[(filtered_data['behavior'] == 'Attack') & 
                                        (filtered_data['simple_zone'] == 'Binocular')])
        attack_total = len(filtered_data[filtered_data['behavior'] == 'Attack'])
        p_bino_given_attack = attack_bino / attack_total if attack_total > 0 else 0
        
        print(f"\nStatistical Analysis:")
        print(f"  Chi-square: χ²={chi2:.2f}, p={p_value:.2e}, Cramér's V={cramers_v:.3f}")
        print(f"  P(Binocular | Attack) = {p_bino_given_attack:.3f}")

        # --- Create Grouped Bar Chart ---
        # Calculate proportions for each behavior
        behavior_zone_counts = filtered_data.groupby(['behavior', 'simple_zone']).size().unstack(fill_value=0)
        behavior_zone_props = behavior_zone_counts.div(behavior_zone_counts.sum(axis=1), axis=0)

        fig, ax = plt.subplots(figsize=(10, 7))
        
        # --- FIXED: Define colors for all possible zones ---
        colors = {
            'Binocular': '#4CAF50', 
            'Monocular': '#BDBDBD',
            'Out of Sight': '#FF6B6B'  # Add a color for out_of_sight (red-ish)
        }
        
        behavior_zone_props.plot(kind='bar', stacked=False, ax=ax, 
                                 color=colors,
                                 alpha=0.85, width=0.7)
        
        ax.set_title('Visual Field Usage During Key Behaviors', fontsize=16, weight='bold', pad=20)
        ax.set_xlabel('Behavior', fontsize=14, weight='bold')
        ax.set_ylabel('Proportion of Frames', fontsize=14, weight='bold')
        ax.set_ylim(0, 1.0)
        ax.set_xticklabels(behavior_zone_props.index, rotation=0, fontsize=12)
        ax.legend(title='Zone', fontsize=11, title_fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add statistical annotation
        stats_text = (f"Chi-square test: χ²={chi2:.2f}, p={p_value:.2e}\n"
                     f"Cramér's V={cramers_v:.3f}\n"
                     f"P(Binocular | Attack) = {p_bino_given_attack:.2%}")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'zone_usage_by_behavior_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved zone usage bar chart.")

        # --- Create Three Pie Charts ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # --- FIXED: Update colors dictionary to include all zones ---
        pie_colors = {
            'Binocular': '#4CAF50', 
            'Monocular': '#BDBDBD',
            'Out of Sight': '#FF6B6B'
        }
        
        for idx, behavior in enumerate(['Chasing', 'Attack', 'Consume']):
            behavior_data = filtered_data[filtered_data['behavior'] == behavior]
            zone_counts = behavior_data['simple_zone'].value_counts()
            
            if not zone_counts.empty:
                # Ensure all zones are present for consistent coloring
                for zone in ['Binocular', 'Monocular', 'Out of Sight']:
                    if zone not in zone_counts:
                        zone_counts[zone] = 0
                
                # Remove zones with zero count for cleaner pie chart
                zone_counts = zone_counts[zone_counts > 0]
                
                wedges, texts, autotexts = axes[idx].pie(
                    zone_counts, 
                    labels=zone_counts.index,
                    autopct='%1.1f%%',
                    colors=[pie_colors[zone] for zone in zone_counts.index],
                    startangle=90,
                    textprops={'fontsize': 11, 'weight': 'bold'}
                )
                
                # Make percentage text more visible
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontsize(12)
                
                axes[idx].set_title(f'{behavior}\n(n={len(behavior_data)} frames)', 
                                   fontsize=13, weight='bold', pad=10)
            else:
                axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center')
                axes[idx].set_title(behavior, fontsize=13, weight='bold')
        
        fig.suptitle('Visual Field Usage During Key Behaviors', 
                    fontsize=16, weight='bold', y=1.02)
        
        # Add statistical annotation to the figure
        fig.text(0.5, -0.05, stats_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'zone_usage_by_behavior_pies.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved zone usage pie charts.")

    def save_statistics_summary(self, data: pd.DataFrame, stats: Dict, chasing_stats: Dict, corr_stats: Optional[Dict]) -> None:
        """Save statistical summary to CSV and text files."""
        # Create summary DataFrame
        summary_df = pd.DataFrame(stats).T
        summary_df.to_csv(self.output_dir / 'statistics_summary.csv')
        
        # Create detailed text report
        with open(self.output_dir / 'data_summary_report.txt', 'w') as f:
            f.write("ANIMAL TRACKING DATA ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total frames analyzed: {len(data)}\n")
            f.write(f"Unique files: {data['file'].nunique() if 'file' in data.columns else 'N/A'}\n\n")
            
            for metric, metric_stats in stats.items():
                f.write(f"{metric.upper().replace('_', ' ')} STATISTICS:\n")
                f.write("-" * 30 + "\n")
                for stat_name, value in metric_stats.items():
                    f.write(f"{stat_name.capitalize()}: {value:.3f}\n")
                f.write("\n")

            if corr_stats:
                f.write("HEAD ANGLE VS SPEED CORRELATION:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Spearman's ρ: {corr_stats['original']['rho_value']:.3f}\n")
                f.write(f"P-value: {corr_stats['original']['p_value']:.3g}\n\n")

            if 'transformed' in corr_stats:
                f.write("HEAD ANGLE VS SPEED CORRELATION (LOG-TRANSFORMED):\n")
                f.write("-" * 30 + "\n")
                f.write(f"Spearman's ρ: {corr_stats['transformed']['rho_value']:.3f}\n")
                f.write(f"P-value: {corr_stats['transformed']['p_value']:.3g}\n\n")

            if 'zone' in data.columns:
                f.write("VISUAL FIELD ZONES:\n")
                f.write("-" * 30 + "\n")
                zone_counts = data['zone'].value_counts()
                total_zones = zone_counts.sum()
                for zone, count in zone_counts.items():
                    percentage = (count / total_zones) * 100
                    f.write(f"{zone}: {count} ({percentage:.1f}%)\n")
                f.write("\n")

            f.write("CHASING ZONE ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total chasing frames: {chasing_stats.get('total_chasing_frames', 0)}\n")
            f.write(f"Monocular chasing / Total chasing: {chasing_stats.get('mono_chasing_ratio', 0):.3f}\n")
            f.write(f"Binocular chasing / Total chasing: {chasing_stats.get('bino_chasing_ratio', 0):.3f}\n")
            if self.use_instance_based:
                f.write("INSTANCE-BASED TRANSITION PROBABILITIES:\n")
                f.write("-" * 30 + "\n")
                # f.write(f"P(Mono->Bino | Chasing Instance): {chasing_stats.get('prob_mono_to_bino_given_chasing', 0):.3f}\n")
                f.write(f"P(Bino Appearance | Chasing Instance): {chasing_stats.get('prob_mono_to_bino_given_chasing', 0):.3f}\n")
            else:
                f.write("FRAME-BASED TRANSITION PROBABILITIES:\n")
                f.write("-" * 30 + "\n")
                f.write(f"P(Mono->Bino | Chasing Frame): {chasing_stats.get('prob_mono_to_bino_given_chasing', 0):.3f}\n")
            
            f.write(f"P(Chasing | Mono->Bino Transition): {chasing_stats.get('prob_chasing_given_mono_to_bino', 0):.3f}\n")

        print("Saved statistical summaries")
    
    def analyze_chasing_zones(self, data: pd.DataFrame, instance_data: Dict[str, List[Dict]]) -> None:
        print("Analyzing visual zones and transitions during chasing...")
        results = {
            'mono_chasing_ratio': 0, 'bino_chasing_ratio': 0, 'total_chasing_frames': 0,
            'prob_mono_to_bino_given_chasing': 0, 'prob_chasing_given_mono_to_bino': 0
        }
        if 'behavior' not in data.columns or 'zone' not in data.columns:
            print("Warning: 'behavior' or 'zone' not available for chasing analysis.")
            return results

        # --- Common setup for both methods ---
        data['simple_zone'] = data['zone'].replace({'right_monocular': 'monocular', 'left_monocular': 'monocular'})
        data['prev_simple_zone'] = data.groupby('file')['simple_zone'].shift(1)
        mono_to_bino_mask = (data['prev_simple_zone'] == 'monocular') & (data['simple_zone'] == 'binocular')
        total_mono_to_bino_transitions = mono_to_bino_mask.sum()
        
        chasing_data = data[data['behavior'] == 'chasing']
        results['total_chasing_frames'] = len(chasing_data)

        if self.use_instance_based:
            print("Analyzing transitions based on TRUE INSTANCES from Excel files...")
            # --- TRUE INSTANCE-BASED LOGIC ---
            
            total_chasing_instances = 0
            num_chasing_instances_with_bino = 0
            num_chasing_instances_with_transition = 0

            # Iterate through each file's true instances
            for file_key, instances in instance_data.items():
                if not instances:
                    continue
                
                # Filter main data to just this file
                file_data = data[data['file'] == file_key]
                
                for instance in instances:
                    if instance['behavior'] == 'chasing':
                        total_chasing_instances += 1
                        
                        # Get the frames for this specific instance
                        instance_frames = file_data[
                            (file_data['frame'] >= instance['start_frame']) &
                            (file_data['frame'] <= instance['end_frame'])
                        ]
                        
                        # Check if a mono->bino transition exists within these frames
                        transition_occurred = (
                            (instance_frames['prev_simple_zone'] == 'monocular') &
                            (instance_frames['simple_zone'] == 'binocular')
                        ).any()
                        bino_appeared = (instance_frames['simple_zone'] == 'binocular').any()
                        if bino_appeared:
                            num_chasing_instances_with_bino += 1

                        if transition_occurred:
                            num_chasing_instances_with_transition += 1

            if total_chasing_instances > 0:
                results['prob_mono_to_bino_given_chasing'] = num_chasing_instances_with_bino / total_chasing_instances
            
            chasing_at_transition = (data.loc[mono_to_bino_mask, 'behavior'] == 'chasing').sum()
            if total_mono_to_bino_transitions > 0:
                results['prob_chasing_given_mono_to_bino'] = chasing_at_transition / total_mono_to_bino_transitions
            
            print(f"  Total true chasing instances found: {total_chasing_instances}")
            print(f"  P(Bino Appearance | Chasing Instance): {results['prob_mono_to_bino_given_chasing']:.3f}")

        else:
            print("Analyzing transitions based on FRAMES...")

            # --- FRAME-BASED LOGIC ---
            chasing_mono_to_bino_transitions = (mono_to_bino_mask & (data['behavior'] == 'chasing')).sum()
            mono_frames_in_chasing = (chasing_data['simple_zone'] == 'monocular').sum()

            if mono_frames_in_chasing > 0:
                results['prob_mono_to_bino_given_chasing'] = chasing_mono_to_bino_transitions / mono_frames_in_chasing
            
            if total_mono_to_bino_transitions > 0:
                results['prob_chasing_given_mono_to_bino'] = chasing_mono_to_bino_transitions / total_mono_to_bino_transitions
            
            print(f"  Total monocular chasing frames: {mono_frames_in_chasing}")
            print(f"  P(Mono->Bino | Chasing Frame): {results['prob_mono_to_bino_given_chasing']:.3f}")

        # --- Common calculations and cleanup ---
        if results['total_chasing_frames'] > 0:
            zone_counts = chasing_data['zone'].value_counts()
            results['mono_chasing_ratio'] = (zone_counts.get('right_monocular', 0) + zone_counts.get('left_monocular', 0)) / results['total_chasing_frames']
            results['bino_chasing_ratio'] = zone_counts.get('binocular', 0) / results['total_chasing_frames']
        
        print(f"  P(Chasing | Mono->Bino Transition): {results['prob_chasing_given_mono_to_bino']:.3f}")
        data.drop(columns=['simple_zone', 'prev_simple_zone'], inplace=True, errors='ignore')
        return results
    
    def run_complete_analysis(self) -> None:
        """Run the complete visualization analysis."""
        print("Starting data visualization analysis...")
        
        # Load data
        print("\n1. Loading data...")
        analysis_data = self.load_analysis_data()
        cricket_data = self.load_cricket_data()
        if self.use_predictions:
            print("📊 Using MODEL PREDICTIONS for behavior labels")
            if not self.prediction_dir:
                print("❌ Error: use_predictions=True but no prediction_dir provided!")
                return
            label_data = self.load_prediction_data()
        else:
            print("📊 Using GROUND TRUTH LABELS for behavior labels")
            label_data = self.load_label_data()

        instance_data = self.load_instance_data()
        
        if not analysis_data or not label_data: # <-- MODIFY THIS
            print("Error: No analysis or label data found!")
            return
        
        # Merge data
        print("\n2. Merging data...")
        merged_data = self.merge_data(analysis_data, cricket_data, label_data) # <-- MODIFY THIS
        
        if merged_data.empty:

            print("Error: No merged data available!")
            return
        
        print(f"Total merged data: {len(merged_data)} frames")
        
        # Calculate statistics
        print("\n3. Calculating statistics...")
        stats = self.calculate_statistics(merged_data)
        # --- ADD CHASING ANALYSIS ---
        chasing_stats = self.analyze_chasing_zones(merged_data, instance_data)
        
        # Create visualizations
        print("\n4. Creating histograms...")
        self.plot_histograms(merged_data)
        
        print("\n5. Creating zone distribution plots...")
        self.plot_zone_distribution(merged_data)
        
        print("\n6. Creating correlation plot...")
        corr_stats = self.plot_angle_speed_correlation(merged_data)
        self.plot_binned_speed_vs_angle(merged_data)
        self.plot_relative_speed_by_behavior(merged_data)
        self.plot_zone_usage_by_behavior(merged_data)
        
        # Save summaries
        print("\n7. Saving statistical summaries...")
        self.save_statistics_summary(merged_data, stats, chasing_stats, corr_stats)

        print(f"\nAnalysis complete! Results saved to: {self.output_dir}")

def main():
    """Main function to run the visualization analysis."""
    # Directory paths - adjust these to match your pipeline structure
    # ANALYSIS_DIR = "/media/tarislada/SAMSUNG/Doric임시/854_0304/representative/postprocessing/final_vid"
    # CRICKET_DIR = "/media/tarislada/SAMSUNG/Doric임시/854_0304/representative/postprocessing/cricket_process_test"
    # OUTPUT_DIR = "/media/tarislada/SAMSUNG/Doric임시/854_0304/representative/postprocessing/data_visualization"
    
    # Alternative paths for different setups
    ANALYSIS_DIR = "/home/tarislada/Documents/Extra_python_projects/SKH FP/FInalized_process/test_val_vid5"
    CRICKET_DIR = "/home/tarislada/Documents/Extra_python_projects/SKH FP/FInalized_process/cricket_process_test5"
    LABEL_DIR = "/home/tarislada/Documents/Hunting_classification/SKH_FP/FInalized_process/Behavior_label2"
    INSTANCE_DIR = "/home/tarislada/Documents/Hunting_classification/SKH_FP/raw_behavior_label" 
    TXT_DIR = "/home/tarislada/Documents/Extra_python_projects/SKH FP/interval_txt"
    PREDICTION_DIR = "/home/tarislada/Documents/Extra_python_projects/SKH FP/FInalized_process/prediction_plots/timing_analysis_files/Strategy_2:_With_Thresholding"
    
    OUTPUT_DIR = "/home/tarislada/Documents/Extra_python_projects/SKH FP/FInalized_process/data_visualization1"
    OUTPUT_DIR_PRED = "/home/tarislada/Documents/Extra_python_projects/SKH FP/FInalized_process/data_visualization1/predictions"
    
    # Create visualizer and run analysis
    visualizer = TrackingDataVisualizer(
        ANALYSIS_DIR, CRICKET_DIR, TXT_DIR, LABEL_DIR, 
        # OUTPUT_DIR, 
        OUTPUT_DIR_PRED,
        instance_dir=INSTANCE_DIR,
        use_predictions=True,
        prediction_dir=PREDICTION_DIR
    )
    visualizer.run_complete_analysis()

if __name__ == "__main__":
    main()