"""
Central configuration for the hunting classification pipeline.
All paths and parameters are defined here to avoid hardcoding throughout the codebase.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

@dataclass
class DirectoryPaths:
    """All directory paths used in the pipeline."""
    # Input directories
    raw_video_dir: str = "SKH FP/video_file"
    pose_data_dir: str = "SKH FP/pose_data"
    cricket_detection_dir: str = "SKH FP/cricket_dection5"
    interval_txt_dir: str = "SKH FP/interval_txt"
    annotation_dir: str = "SKH FP/mouse_hunting_annotations"
    
    # Intermediate processing directories
    savgol_pose_dir: str = "SKH FP/savgol_pose_w59p7"
    head_angle_dir: str = "SKH FP/head_angle_w59p7"
    kalman_filtered_dir: str = "SKH FP/kalman_filtered_w59p7"
    cricket_processed_dir: str = "SKH FP/FInalized_process/cricket_process_test5"
    
    # Output directories
    final_videos_dir: str = "SKH FP/FInalized_process/test_val_vid5"
    behavior_labels_dir: str = "SKH FP/FInalized_process/Behavior_label"
    visualization_dir: str = "SKH FP/FInalized_process/data_visualization1"
    
    def __post_init__(self):
        """Convert string paths to Path objects and create directories if needed."""
        for field_name, value in self.__dict__.items():
            path_obj = Path(value)
            setattr(self, field_name, path_obj)
            # Optionally create directories
            # path_obj.mkdir(parents=True, exist_ok=True)

@dataclass
class KeypointFilterParams:
    """Parameters for keypoint filtering (Savitzky-Golay)."""
    confidence_threshold: float = 0.6
    window_length: int = 59
    polyorder: int = 7
    # Column indices for keypoint coordinates and confidence
    keypoint_indices: List[int] = None
    confidence_indices: List[int] = None
    
    def __post_init__(self):
        if self.keypoint_indices is None:
            self.keypoint_indices = [7, 8, 27, 28, 21, 22]  # nose, body center, tail base x,y
        if self.confidence_indices is None:
            self.confidence_indices = [29, 39, 35]  # corresponding confidence columns

@dataclass
class HeadAngleParams:
    """Parameters for head angle calculation."""
    frame_rate: int = 30
    # Column names for pose data
    target_columns: List[str] = None
    
    def __post_init__(self):
        if self.target_columns is None:
            self.target_columns = [
                'frame', 'box confidence', 'nose x', 'nose y', 
                'body center x', 'body center y', 'tail base x', 'tail base y'
            ]

@dataclass
class KalmanFilterParams:
    """Parameters for Kalman filtering of head angles."""
    frame_rate: int = 30
    process_variance: float = 1e-5
    measurement_variance: float = 1e-2

@dataclass
class CricketValidationParams:
    """Parameters for cricket detection validation."""
    max_gap_threshold: int = 30
    base_window_size: int = 20
    window_polyorder: int = 3
    confidence_weight: float = 1.0
    size_weight: float = 1.0
    speed_weight: float = 0.0  # Disabled in current version
    accel_weight: float = 0.0  # Disabled in current version
    jerk_weight: float = 0.0   # Disabled in current version
    reliability_threshold: float = 0.65
    confidence_threshold: float = 0.25
    downsample_60fps: bool = True

@dataclass
class VisualFieldParams:
    """Parameters for visual field analysis and angle validation."""
    # Line drawing parameters
    angle_line_length: int = 50
    nose_vector_length: int = 100
    tail_vector_length: int = 200
    
    # Frame processing
    frame_rate: int = 30
    base_distance_threshold: int = 75
    distance_scaling_factor: float = 0.025
    
    # Visual field classification
    binocular_threshold: float = 30.0    # Half-angle of binocular field (degrees)
    monocular_threshold: float = 140.0   # Maximum angle for monocular vision (degrees)
    
    # Kalman filter for cricket tracking
    process_variance: float = 1e-3
    measurement_variance: float = 1e-3
    
    # Visualization
    reference_circle_radius: int = 150
    show_angle_markers: bool = True
    marker_interval: int = 30
    
    # Video resolution
    x_resol: int = 1920
    y_resol: int = 1080

@dataclass
class ActionSegmentationParams:
    """Parameters for action segmentation/classification."""
    # Feature engineering
    window_sizes: List[int] = None
    batch_size: int = 3
    sequential_lags: List[int] = None
    max_features_per_type: int = 3
    
    # Sequential features for behavioral transitions
    sequential_features: List[str] = None
    
    # Cross-validation
    cv_folds: int = 3
    
    # Class weights for imbalanced data
    class_weights: Dict[str, float] = None
    
    # Model parameters
    random_state: int = 42
    
    def __post_init__(self):
        if self.window_sizes is None:
            self.window_sizes = [15, 30]
        if self.sequential_lags is None:
            self.sequential_lags = [1, 2, 3]
        if self.sequential_features is None:
            self.sequential_features = [
                'head_angle', 'cricket_angle', 'relative_angle', 
                'distance', 'cricket_in_binocular', 'is_cricket_visible', 'zone'
            ]
        if self.class_weights is None:
            self.class_weights = {
                'attack': 1.2,
                'non_visual_rotation': 1.0,
                'chasing': 0.94,
                'background': 1.0
            }

@dataclass
class VisualizationParams:
    """Parameters for visualization components."""
    # Trace visualization
    target_frame: Optional[int] = None  # None means use last frame
    full_skeleton: bool = False
    
    # Colors (BGR format for OpenCV)
    cricket_trace_color: Tuple[int, int, int] = (0, 255, 255)  # Yellow
    mouse_trace_color: Tuple[int, int, int] = (255, 0, 0)      # Blue
    skeleton_color: Tuple[int, int, int] = (0, 255, 0)         # Green
    
    # Trace styling
    cricket_trace_thickness: int = 2
    mouse_trace_thickness: int = 2
    skeleton_thickness: int = 3
    keypoint_radius: int = 5
    
    # Trace effects
    enable_fading: bool = True
    fade_length: int = 100
    
    # Output settings
    output_dpi: int = 300
    figure_size: Tuple[int, int] = (16, 9)

@dataclass
class PipelineSettings:
    """Complete pipeline configuration."""
    # Directory paths
    paths: DirectoryPaths = None
    
    # Processing parameters
    keypoint_filter: KeypointFilterParams = None
    head_angle: HeadAngleParams = None
    kalman_filter: KalmanFilterParams = None
    cricket_validation: CricketValidationParams = None
    visual_field: VisualFieldParams = None
    action_segmentation: ActionSegmentationParams = None
    visualization: VisualizationParams = None
    
    # General settings
    enable_progress_bar: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Initialize all parameter groups if not provided."""
        if self.paths is None:
            self.paths = DirectoryPaths()
        if self.keypoint_filter is None:
            self.keypoint_filter = KeypointFilterParams()
        if self.head_angle is None:
            self.head_angle = HeadAngleParams()
        if self.kalman_filter is None:
            self.kalman_filter = KalmanFilterParams()
        if self.cricket_validation is None:
            self.cricket_validation = CricketValidationParams()
        if self.visual_field is None:
            self.visual_field = VisualFieldParams()
        if self.action_segmentation is None:
            self.action_segmentation = ActionSegmentationParams()
        if self.visualization is None:
            self.visualization = VisualizationParams()

# Global settings instance
settings = PipelineSettings()

# Convenience function to update settings from a dictionary
def update_settings(config_dict: Dict[str, Any]) -> None:
    """Update settings from a dictionary (useful for YAML loading later)."""
    for section, params in config_dict.items():
        if hasattr(settings, section):
            section_obj = getattr(settings, section)
            for key, value in params.items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)
                else:
                    print(f"Warning: Unknown parameter {key} in section {section}")
        else:
            print(f"Warning: Unknown section {section}")

# Convenience function to get settings as dictionary
def get_settings_dict() -> Dict[str, Any]:
    """Convert settings to dictionary format."""
    return {
        'paths': settings.paths.__dict__,
        'keypoint_filter': settings.keypoint_filter.__dict__,
        'head_angle': settings.head_angle.__dict__,
        'kalman_filter': settings.kalman_filter.__dict__,
        'cricket_validation': settings.cricket_validation.__dict__,
        'visual_field': settings.visual_field.__dict__,
        'action_segmentation': settings.action_segmentation.__dict__,
        'visualization': settings.visualization.__dict__,
    }

if __name__ == "__main__":
    # Test the configuration
    print("Pipeline Settings:")
    print(f"Raw video directory: {settings.paths.raw_video_dir}")
    print(f"Kalman process variance: {settings.kalman_filter.process_variance}")
    print(f"Cricket confidence threshold: {settings.cricket_validation.confidence_threshold}")
    print(f"Action segmentation CV folds: {settings.action_segmentation.cv_folds}")