import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import os
import glob
import re
from typing import List, Dict, Optional, Tuple

class TrackingDataVisualizer:
    """
    Visualizer for animal tracking pipeline data.
    Creates histograms and statistical summaries for distance, speed, cricket angle, and head angle.
    """
    
    def __init__(self, analysis_dir: str, cricket_dir: str, txt_dir:str, output_dir: str):
        """
        Initialize the visualizer with input and output directories.
        
        Args:
            analysis_dir: Directory containing *_analysis.csv files from angle_valv3.py
            cricket_dir: Directory containing crprocessed_*.csv files from cricket_interpv4.py  
            txt_dir: Directory containing txt files that indicate the cricket_in frame and out frame
            output_dir: Directory to save visualization outputs
        """
        self.analysis_dir = Path(analysis_dir)
        self.cricket_dir = Path(cricket_dir)
        self.txt_dir = Path(txt_dir)
        self.output_dir = Path(output_dir)
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
    
    def merge_data(self, analysis_data: Dict[str, pd.DataFrame], 
                   cricket_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge analysis and cricket data based on matching file names."""
        merged_dfs = []
        
        for file_key in analysis_data.keys():
            cricket_key = file_key.replace('_validated', '')

            if cricket_key in cricket_data:
                analysis_df = analysis_data[file_key].copy()
                cricket_df = cricket_data[cricket_key].copy()

                txt_path = self.txt_dir / f"{cricket_key}.txt"
                
                if txt_path.exists():
                    try:
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            cricket_in_frame, cricket_out_frame = self.get_cricket_frames(f.read())
                        
                        print(f"Cricket frames for {cricket_key}: {cricket_in_frame} to {cricket_out_frame}")
                        
                        # Filter both dataframes to only include cricket interaction period
                        analysis_df = analysis_df[
                            (analysis_df['frame'] >= cricket_in_frame) & 
                            (analysis_df['frame'] <= cricket_out_frame)
                        ].copy()
                        
                        cricket_df = cricket_df[
                            (cricket_df['frame'] >= cricket_in_frame) & 
                            (cricket_df['frame'] <= cricket_out_frame)
                        ].copy()
                        
                        # Merge on frame number
                        merged_df = pd.merge(analysis_df, cricket_df[['frame', 'speed', 'size', 'status']], 
                                           on='frame', how='left')
                        merged_df['file'] = cricket_key  # Use the base name without '_validated'
                        merged_dfs.append(merged_df)
                        print(f"Merged filtered data for {file_key} -> {cricket_key}: {len(merged_df)} frames")
                        
                    except Exception as e:
                        print(f"Error processing text file for {cricket_key}: {e}")
                        # If text file processing fails, merge without filtering
                        merged_df = pd.merge(analysis_df, cricket_df[['frame', 'speed', 'size', 'status']], 
                                           on='frame', how='left')
                        merged_df['file'] = cricket_key
                        merged_dfs.append(merged_df)
                        print(f"Merged unfiltered data for {file_key} -> {cricket_key}: {len(merged_df)} frames (text file error)")
                else:
                    print(f"Warning: Text file not found for {cricket_key}, using unfiltered data")
                    # If no text file, merge without filtering
                    merged_df = pd.merge(analysis_df, cricket_df[['frame', 'speed', 'size', 'status']], 
                                       on='frame', how='left')
                    merged_df['file'] = cricket_key
                    merged_dfs.append(merged_df)
                    print(f"Merged unfiltered data for {file_key} -> {cricket_key}: {len(merged_df)} frames")
            else:
                print(f"Warning: No matching cricket data for {file_key} (looked for {cricket_key})")
                
        if merged_dfs:
            return pd.concat(merged_dfs, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def calculate_statistics(self, data: pd.DataFrame) -> Dict:
        """Calculate summary statistics for each metric."""
        metrics = ['distance', 'cricket_angle', 'head_angle', 'speed']
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
        consume_lines = [line for line in lines if line.strip().endswith('consume')]
        
        if consume_lines:
            # If we found consume lines, use the last one
            last_consume = consume_lines[-1]
            try:
                cricket_out_frame = int(last_consume.split('\t')[1])
                return cricket_in_frame, cricket_out_frame
            except (IndexError, ValueError):
                pass  # If parsing fails, continue to the fallback method
        
        # Fallback method: Find the largest number in the text
        all_numbers = re.findall(r'\d+', txt_content)
        if not all_numbers:
            raise ValueError("No frame numbers found in text file")
        
        cricket_out_frame = max(int(num) for num in all_numbers)
        
        return cricket_in_frame, cricket_out_frame


    def plot_histograms(self, data: pd.DataFrame, save_individual: bool = True) -> None:
        """Create histogram plots for each metric."""
        metrics_config = {
            'distance': {'title': 'Cricket-Mouse Distance', 'xlabel': 'Distance (pixels)', 'color': 'skyblue'},
            'speed': {'title': 'Cricket Movement Speed', 'xlabel': 'Speed (pixels/frame)', 'color': 'lightcoral'},
            'cricket_angle': {'title': 'Cricket Angle', 'xlabel': 'Angle (degrees)', 'color': 'lightgreen'},
            'head_angle': {'title': 'Mouse Head Angle', 'xlabel': 'Angle (degrees)', 'color': 'plum'}
        }
        
        # Create individual histograms
        if save_individual:
            for metric, config in metrics_config.items():
                if metric in data.columns:
                    self._plot_single_histogram(data, metric, config)
        
        # Create combined plot
        self._plot_combined_histograms(data, metrics_config)
    
    def _plot_single_histogram(self, data: pd.DataFrame, metric: str, config: Dict) -> None:
        """Plot histogram for a single metric."""
        valid_data = data[metric].dropna()
        
        if len(valid_data) == 0:
            print(f"No valid data for {metric}")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(valid_data, bins=50, alpha=0.7, color=config['color'], edgecolor='black')
        ax1.set_title(f'{config["title"]} - Histogram')
        ax1.set_xlabel(config['xlabel'])
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
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
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, (metric, config) in enumerate(metrics_config.items()):
            if metric in data.columns:
                valid_data = data[metric].dropna()
                
                if len(valid_data) > 0:
                    axes[idx].hist(valid_data, bins=40, alpha=0.7, 
                                 color=config['color'], edgecolor='black')
                    axes[idx].set_title(config['title'])
                    axes[idx].set_xlabel(config['xlabel'])
                    axes[idx].set_ylabel('Frequency')
                    axes[idx].grid(True, alpha=0.3)
                    
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
    
    def save_statistics_summary(self, data: pd.DataFrame, stats: Dict) -> None:
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
            
            if 'zone' in data.columns:
                f.write("VISUAL FIELD ZONES:\n")
                f.write("-" * 30 + "\n")
                zone_counts = data['zone'].value_counts()
                total_zones = zone_counts.sum()
                for zone, count in zone_counts.items():
                    percentage = (count / total_zones) * 100
                    f.write(f"{zone}: {count} ({percentage:.1f}%)\n")
        
        print("Saved statistical summaries")
    
    def run_complete_analysis(self) -> None:
        """Run the complete visualization analysis."""
        print("Starting data visualization analysis...")
        
        # Load data
        print("\n1. Loading data...")
        analysis_data = self.load_analysis_data()
        cricket_data = self.load_cricket_data()
        
        if not analysis_data:
            print("Error: No analysis data found!")
            return
        
        # Merge data
        print("\n2. Merging data...")
        merged_data = self.merge_data(analysis_data, cricket_data)
        
        if merged_data.empty:
            print("Error: No merged data available!")
            return
        
        print(f"Total merged data: {len(merged_data)} frames")
        
        # Calculate statistics
        print("\n3. Calculating statistics...")
        stats = self.calculate_statistics(merged_data)
        
        # Create visualizations
        print("\n4. Creating histograms...")
        self.plot_histograms(merged_data)
        
        print("\n5. Creating zone distribution plots...")
        self.plot_zone_distribution(merged_data)
        
        # Save summaries
        print("\n6. Saving statistical summaries...")
        self.save_statistics_summary(merged_data, stats)
        
        print(f"\nAnalysis complete! Results saved to: {self.output_dir}")

def main():
    """Main function to run the visualization analysis."""
    # Directory paths - adjust these to match your pipeline structure
    # ANALYSIS_DIR = "/media/tarislada/SAMSUNG/Doric임시/854_0304/representative/postprocessing/final_vid"
    # CRICKET_DIR = "/media/tarislada/SAMSUNG/Doric임시/854_0304/representative/postprocessing/cricket_process_test"
    # OUTPUT_DIR = "/media/tarislada/SAMSUNG/Doric임시/854_0304/representative/postprocessing/data_visualization"
    
    # Alternative paths for different setups
    ANALYSIS_DIR = "SKH FP/FInalized_process/test_val_vid4"
    CRICKET_DIR = "SKH FP/FInalized_process/cricket_process_test5"
    OUTPUT_DIR = "SKH FP/FInalized_process/data_visualization1"
    TXT_DIR = "/home/tarislada/Documents/Extra_python_projects/SKH FP/interval_txt"
    
    # Create visualizer and run analysis
    visualizer = TrackingDataVisualizer(ANALYSIS_DIR, CRICKET_DIR, TXT_DIR, OUTPUT_DIR)
    visualizer.run_complete_analysis()

if __name__ == "__main__":
    main()