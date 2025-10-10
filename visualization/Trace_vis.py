import cv2
import pandas as pd
import numpy as np
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
import matplotlib.patches as patches

@dataclass
class VisualizationConfig:
    """Configuration parameters for trace visualization."""
    # Frame selection
    trace_start: Optional[int] = None  # <-- ADD THIS: Start frame for the trace
    target_frame: Optional[int] = None  # None means use last frame
    
    # Skeleton configuration
    full_skeleton: bool = False  # True: all keypoints, False: nose/body/tail only
    
    # Cricket trace styling
    cricket_trace_color: Tuple[int, int, int] = (0, 255, 255)  # Yellow in BGR
    cricket_trace_thickness: int = 2
    cricket_trace_alpha: float = 0.8
    
    # Mouse nose trace styling
    mouse_trace_color: Tuple[int, int, int] = (255, 0, 0)  # Blue in BGR
    mouse_trace_thickness: int = 2
    mouse_trace_alpha: float = 0.8
    
    # Mouse skeleton styling
    skeleton_color: Tuple[int, int, int] = (0, 255, 0)  # Green in BGR
    skeleton_thickness: int = 3
    keypoint_radius: int = 5
    
    # Trace fading effect
    enable_fading: bool = True
    fade_length: int = 100  # Number of frames to fade over
    
    # Output configuration
    output_dpi: int = 300
    figure_size: Tuple[int, int] = (16, 9)  # Width, Height in inches

class TraceVisualizer:
    """Handles trace and skeleton visualization."""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        
    def load_data(self, pose_csv_path: str, cricket_csv_path: str, video_path: str):
        """Load required data files."""
        # Load pose data
        self.pose_data = pd.read_csv(pose_csv_path)
        
        # Load cricket data
        self.cricket_data = pd.read_csv(cricket_csv_path)
        
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Determine target frame
        if self.config.target_frame is None:
            self.target_frame = total_frames - 1
        else:
            self.target_frame = min(self.config.target_frame, total_frames - 1)
            
        # Get target frame image
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.target_frame)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {self.target_frame}")
            
        self.background_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()
        
        print(f"Loaded data up to frame {self.target_frame}")
        
    def filter_data_to_frame(self):
        """Filter data up to target frame."""
        
        start_frame = self.config.trace_start if self.config.trace_start is not None else 0
        
        self.pose_filtered = self.pose_data[
            (self.pose_data['frame'] >= start_frame) &
            (self.pose_data['frame'] <= self.target_frame)
        ].copy()

        if self.pose_filtered['nose x'].max() < 1.1 and self.pose_filtered['nose y'].max()<1.1:
            self.pose_normalized = True
        else:
            self.pose_normalized = False
        
        self.cricket_filtered = self.cricket_data[
            (self.cricket_data['frame'] >= start_frame) &
            (self.cricket_data['frame'] <= self.target_frame)
        ].copy()

        if self.cricket_filtered.empty:
            print("Warning: No cricket data in the specified frame range.")
            return

        # Remove invalid cricket positions
        valid_cricket = ~(
            self.cricket_filtered[['smoothed_x', 'smoothed_y']].isna().any(axis=1)
        )
        self.cricket_filtered = self.cricket_filtered[valid_cricket]
        
        # Check if normalized
        if self.cricket_filtered['smoothed_x'].max()< 1.1 and self.cricket_filtered['smoothed_y'].max()< 1.1:
            self.cricket_normalized = True
        else:
            self.cricket_normalized = False
        
    def get_trace_alphas(self, num_points: int) -> np.ndarray:
        """Calculate alpha values for fading effect."""
        if not self.config.enable_fading or num_points <= self.config.fade_length:
            return np.ones(num_points)
            
        alphas = np.ones(num_points)
        fade_start = max(0, num_points - self.config.fade_length)
        fade_range = np.linspace(0.1, 1.0, num_points - fade_start)
        alphas[fade_start:] = fade_range
        
        return alphas
        
    def draw_cricket_trace(self, ax):
        """Draw cricket movement trace."""
        if len(self.cricket_filtered) < 2:
            return
        
        if self.cricket_normalized == True:
            x_coords = self.cricket_filtered['smoothed_x'].values * self.background_frame.shape[1]
            y_coords = self.cricket_filtered['smoothed_y'].values * self.background_frame.shape[0]
        else:
            x_coords = self.cricket_filtered['smoothed_x'].values
            y_coords = self.cricket_filtered['smoothed_y'].values
        
        # Remove any remaining NaN values
        valid_mask = ~(np.isnan(x_coords) | np.isnan(y_coords))
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]
        
        if len(x_coords) < 2:
            return
            
        # Convert BGR to RGB for matplotlib
        color_rgb = (
            self.config.cricket_trace_color[2] / 255,
            self.config.cricket_trace_color[1] / 255,
            self.config.cricket_trace_color[0] / 255
        )
        
        if self.config.enable_fading:
            alphas = self.get_trace_alphas(len(x_coords))
            
            # Draw segments with varying alpha
            for i in range(len(x_coords) - 1):
                ax.plot(
                    [x_coords[i], x_coords[i + 1]],
                    [y_coords[i], y_coords[i + 1]],
                    color=color_rgb,
                    linewidth=self.config.cricket_trace_thickness,
                    alpha=alphas[i] * self.config.cricket_trace_alpha,
                    solid_capstyle='round'
                )
        else:
            ax.plot(
                x_coords, y_coords,
                color=color_rgb,
                linewidth=self.config.cricket_trace_thickness,
                alpha=self.config.cricket_trace_alpha,
                solid_capstyle='round'
            )
            
        # Mark start and end points
        ax.scatter(x_coords[0], y_coords[0], 
                  c=[color_rgb], s=50, marker='o', 
                  edgecolors='white', linewidth=2, 
                  alpha=self.config.cricket_trace_alpha, label='Cricket Start')
        ax.scatter(x_coords[-1], y_coords[-1], 
                  c=[color_rgb], s=80, marker='s', 
                  edgecolors='white', linewidth=2, 
                  alpha=self.config.cricket_trace_alpha, label='Cricket End')
                  
    def draw_mouse_nose_trace(self, ax):
        """Draw mouse nose movement trace."""
        if len(self.pose_filtered) < 2:
            return
            
        if self.pose_normalized == True:
            x_coords = self.pose_filtered['nose x'].values * self.background_frame.shape[1]
            y_coords = self.pose_filtered['nose y'].values * self.background_frame.shape[0]
        else:
            x_coords = self.pose_filtered['nose x'].values
            y_coords = self.pose_filtered['nose y'].values
        
        # Remove any NaN values
        valid_mask = ~(np.isnan(x_coords) | np.isnan(y_coords))
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]
        
        if len(x_coords) < 2:
            return
            
        # Convert BGR to RGB for matplotlib
        color_rgb = (
            self.config.mouse_trace_color[2] / 255,
            self.config.mouse_trace_color[1] / 255,
            self.config.mouse_trace_color[0] / 255
        )
        
        if self.config.enable_fading:
            alphas = self.get_trace_alphas(len(x_coords))
            
            # Draw segments with varying alpha
            for i in range(len(x_coords) - 1):
                ax.plot(
                    [x_coords[i], x_coords[i + 1]],
                    [y_coords[i], y_coords[i + 1]],
                    color=color_rgb,
                    linewidth=self.config.mouse_trace_thickness,
                    alpha=alphas[i] * self.config.mouse_trace_alpha,
                    solid_capstyle='round'
                )
        else:
            ax.plot(
                x_coords, y_coords,
                color=color_rgb,
                linewidth=self.config.mouse_trace_thickness,
                alpha=self.config.mouse_trace_alpha,
                solid_capstyle='round'
            )
            
        # Mark start and end points
        ax.scatter(x_coords[0], y_coords[0], 
                  c=[color_rgb], s=50, marker='o', 
                  edgecolors='white', linewidth=2, 
                  alpha=self.config.mouse_trace_alpha, label='Mouse Start')
        ax.scatter(x_coords[-1], y_coords[-1], 
                  c=[color_rgb], s=80, marker='s', 
                  edgecolors='white', linewidth=2, 
                  alpha=self.config.mouse_trace_alpha, label='Mouse End')
                  
    def draw_mouse_skeleton(self, ax):
        """Draw mouse skeleton at target frame."""
        # Get the last valid pose data
        target_pose = self.pose_filtered[
            self.pose_filtered['frame'] == self.target_frame
        ]
        
        if len(target_pose) == 0:
            # If no data for exact frame, use the last available frame
            target_pose = self.pose_filtered.iloc[-1:]
            
        if len(target_pose) == 0:
            return
            
        pose_row = target_pose.iloc[0]
        
        # Convert BGR to RGB for matplotlib
        color_rgb = (
            self.config.skeleton_color[2] / 255,
            self.config.skeleton_color[1] / 255,
            self.config.skeleton_color[0] / 255
        )
        
        if self.config.full_skeleton:
            self._draw_full_skeleton(ax, pose_row, color_rgb)
        else:
            self._draw_simple_skeleton(ax, pose_row, color_rgb)
            
    def _draw_simple_skeleton(self, ax, pose_row, color_rgb):
        """Draw simple skeleton: nose -> body center -> tail base."""
        try:
            # Get keypoint coordinates
            nose_x, nose_y = pose_row['nose x'], pose_row['nose y']
            body_x, body_y = pose_row['body center x'], pose_row['body center y']
            tail_x, tail_y = pose_row['tail base x'], pose_row['tail base y']
            
            # Check if coordinates are valid
            if any(np.isnan([nose_x, nose_y, body_x, body_y, tail_x, tail_y])):
                print("Warning: Invalid coordinates for skeleton drawing")
                return
                
            # Draw skeleton lines
            # Nose to body center
            ax.plot([nose_x, body_x], [nose_y, body_y],
                   color=color_rgb, linewidth=self.config.skeleton_thickness,
                   solid_capstyle='round')
            
            # Body center to tail base
            ax.plot([body_x, tail_x], [body_y, tail_y],
                   color=color_rgb, linewidth=self.config.skeleton_thickness,
                   solid_capstyle='round')
            
            # Draw keypoints
            keypoints = [
                (nose_x, nose_y, 'Nose'),
                (body_x, body_y, 'Body'),
                (tail_x, tail_y, 'Tail')
            ]
            
            for x, y, label in keypoints:
                ax.scatter(x, y, c=[color_rgb], s=self.config.keypoint_radius**2,
                          edgecolors='white', linewidth=2, zorder=10)
                          
        except KeyError as e:
            print(f"Missing required columns for skeleton: {e}")
            
    def _draw_full_skeleton(self, ax, pose_row, color_rgb):
        """Draw full skeleton with all available keypoints."""
        # Define skeleton connections based on the pose data structure
        # This assumes standard mouse pose keypoint order from the files
        skeleton_connections = [
            # Head connections
            ('nose x', 'nose y', 'body center x', 'body center y'),
            
            # Body to tail connections  
            ('body center x', 'body center y', 'tail base x', 'tail base y'),
        ]
        
        # Draw skeleton lines
        for connection in skeleton_connections:
            try:
                x1, y1, x2, y2 = [pose_row[col] for col in connection]
                if not any(np.isnan([x1, y1, x2, y2])):
                    ax.plot([x1, x2], [y1, y2],
                           color=color_rgb, linewidth=self.config.skeleton_thickness,
                           solid_capstyle='round')
            except KeyError:
                continue
                
        # Draw all keypoints
        keypoint_pairs = [
            ('nose x', 'nose y', 'Nose'),
            ('body center x', 'body center y', 'Body Center'),
            ('tail base x', 'tail base y', 'Tail Base'),
        ]
        
        for x_col, y_col, label in keypoint_pairs:
            try:
                x, y = pose_row[x_col], pose_row[y_col]
                if not any(np.isnan([x, y])):
                    ax.scatter(x, y, c=[color_rgb], s=self.config.keypoint_radius**2,
                              edgecolors='white', linewidth=2, zorder=10)
            except KeyError:
                continue
                
    def create_visualization(self, output_path: str):
        """Create and save the complete visualization."""
        # Filter data to target frame
        self.filter_data_to_frame()
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=self.config.figure_size, dpi=self.config.output_dpi)
        
        # Display background frame
        ax.imshow(self.background_frame, aspect='equal')
        
        # Draw traces and skeleton
        self.draw_cricket_trace(ax)
        self.draw_mouse_nose_trace(ax)
        self.draw_mouse_skeleton(ax)
        
        # Configure plot
        ax.set_xlim(0, self.background_frame.shape[1])
        ax.set_ylim(self.background_frame.shape[0], 0)  # Invert Y axis for image coordinates
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add title with frame information
        skeleton_type = "Full" if self.config.full_skeleton else "Simple"
        title = f"Mouse and Cricket Traces - Frame {self.target_frame} ({skeleton_type} Skeleton)"
        fig.suptitle(title, fontsize=14, y=0.95)
        
        # Add legend
        ax.legend(loc='upper right', framealpha=0.8)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.output_dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Visualization saved to: {output_path}")

def create_trace_visualization(video_path: str, 
                             pose_csv_path: str, 
                             cricket_csv_path: str, 
                             output_path: str,
                             config: Optional[VisualizationConfig] = None):
    """
    Create trace visualization with default or custom configuration.
    
    Args:
        video_path: Path to input video file
        pose_csv_path: Path to Kalman-filtered pose CSV
        cricket_csv_path: Path to processed cricket CSV  
        output_path: Path for output image
        config: Optional custom configuration
    """
    if config is None:
        config = VisualizationConfig()
        
    visualizer = TraceVisualizer(config)
    visualizer.load_data(pose_csv_path, cricket_csv_path, video_path)
    visualizer.create_visualization(output_path)

# Example usage
if __name__ == "__main__":
    # Example file paths - adjust these to your actual file paths
    video_path = "/home/tarislada/Documents/Extra_python_projects/SKH FP/video_file/m14_t2.mp4"
    pose_csv_path = "/home/tarislada/Documents/Extra_python_projects/SKH FP/kalman_filtered_w59p7/kalman_filtered_processed_filtered_m14_t2.csv"
    cricket_csv_path = "/home/tarislada/Documents/Extra_python_projects/SKH FP/FInalized_process/cricket_process_test5/crprocessed_m14_t2.csv"
    output_path = "/home/tarislada/Documents/Extra_python_projects/SKH FP/FInalized_process/traces.png"
    
    # Example 1: Default configuration (simple skeleton, last frame)
    # create_trace_visualization(video_path, pose_csv_path, cricket_csv_path, output_path)
    
    # Example 2: Custom configuration
    custom_config = VisualizationConfig(
        trace_start=2613,  # <-- EXAMPLE: Start the trace from frame 2500
        target_frame=2950,  # Specific frame instead of last
        full_skeleton=True,  # Show full skeleton
        # cricket_trace_color=(255, 165, 0),  # Orange cricket trace
        cricket_trace_color=(0,0,0),
        mouse_trace_color=(128, 0, 128),  # Purple mouse trace
        enable_fading=True,  # Enable trace fading effect
        fade_length=3400,  # Fade over 2000 frames
        output_dpi=300  # High resolution output
    )
    
    output_path_custom = "/home/tarislada/Documents/Extra_python_projects/SKH FP/FInalized_process/trace_vis_custom1.png"
    create_trace_visualization(video_path, pose_csv_path, cricket_csv_path, 
                             output_path_custom, custom_config)