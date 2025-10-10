import cv2
import pandas as pd
import numpy as np
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional
from filterpy.kalman import KalmanFilter

@dataclass
class ValidationConfig:
    """Configuration parameters for angle validation visualization."""
    # Original parameters
    angle_line_length: int = 50
    nose_vector_length: int = 100
    tail_vector_length: int = 200
    frame_rate: int = 30
    base_distance_threshold: int = 75
    distance_scaling_factor: float = 0.025
    
    # Visual field parameters
    binocular_threshold: float = 30.0    # Half-angle of binocular field (degrees)
    monocular_threshold: float = 140.0   # Maximum angle for monocular vision (degrees)
    process_variance: float = 1e-3       # Kalman filter process noise
    measurement_variance: float = 1e-3    # Kalman filter measurement noise
    
    # Visualization parameters
    reference_circle_radius: int = 150    # Radius for angular markers
    show_angle_markers: bool = True       # Show degree markers
    marker_interval: int = 30             # Draw markers every 30 degrees
    
    # Video Resolution parameters
    x_resol: int = 1280 # X resolution of the video
    y_resol: int = 720 # Y resolution of the video
    
class AngleAnalyzer:
    """Handles Kalman filtering and visual field classification."""
    def __init__(self, config: ValidationConfig):
        self.config = config
        
        # Initialize Kalman filters for cricket measurements
        self.cricket_angle_kf = KalmanFilter(dim_x=2, dim_z=1)
        self.distance_kf = KalmanFilter(dim_x=2, dim_z=1)
        
        # Setup cricket angle Kalman filter
        self.cricket_angle_kf.F = np.array([[1., 1.], [0., 1.]])
        self.cricket_angle_kf.H = np.array([[1., 0.]])
        self.cricket_angle_kf.P *= 1.0
        self.cricket_angle_kf.R = config.measurement_variance
        self.cricket_angle_kf.Q = config.process_variance * np.array([[0.25, 0.5], [0.5, 1.]])
        
        # Setup distance Kalman filter
        self.distance_kf.F = np.array([[1., 1.], [0., 1.]])
        self.distance_kf.H = np.array([[1., 0.]])
        self.distance_kf.P *= 1.0
        self.distance_kf.R = config.measurement_variance
        self.distance_kf.Q = config.process_variance * np.array([[0.25, 0.5], [0.5, 1.]])
        
        self.initialized = False

    def calculate_head_angle(self, nose_x: float, nose_y: float, 
                           body_x: float, body_y: float) -> float:
        """Calculate head angle relative to vertical axis."""
        # Create head vector
        head_vector = np.array([nose_x - body_x, nose_y - body_y])
        
        # Create reference vector (0, 1) for vertical axis
        ref_vector = np.array([0, 1])
        # ref_vector = np.array([0, -1])
        
        # Calculate angle
        dot_product = np.dot(head_vector, ref_vector)
        head_mag = np.linalg.norm(head_vector)
        
        if head_mag == 0:
            return 0.0
            
        cos_angle = dot_product / (head_mag * 1.0)  # ref_vector magnitude is 1
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        angle_deg = np.rad2deg(angle_rad)
        
        # Determine sign using cross product
        cross_product = np.cross([head_vector[0], head_vector[1], 0], 
                               [ref_vector[0], ref_vector[1], 0])[2]
        # Note: Negative sign here to correct for image coordinates
        # cross_product = -np.cross([head_vector[0], head_vector[1], 0], 
        #                         [ref_vector[0], ref_vector[1], 0])[2]

        if cross_product > 0:
            angle_deg = -angle_deg
            
        return angle_deg

    def calculate_cricket_angle(self, cricket_x: float, cricket_y: float,
                              nose_x: float, nose_y: float, 
                              body_x: float, body_y: float) -> float:
        """Calculate cricket angle relative to vertical axis."""
        # Create cricket vector from nose to cricket
        cricket_vector = np.array([cricket_x - nose_x, cricket_y - nose_y])
        head_vector = np.array([nose_x - body_x, nose_y - body_y])
        
        # Create reference vector (0, 1) for vertical axis
        
        # ref_vector = np.array([0, 1])
        # ref_vector = np.array([0, -1])
        
        # Calculate angle
        # dot_product = np.dot(cricket_vector, ref_vector)
        dot_product = np.sum(cricket_vector * head_vector)
        cross_product = np.cross(cricket_vector, head_vector)
        # cricket_mag = np.linalg.norm(cricket_vector)
        cricket_mag = np.linalg.norm(cricket_vector)
        head_mag = np.linalg.norm(head_vector)
        
        # This is the cause of error when use_nose_position is True??
        if cricket_mag == 0:
            return 0.0
            
        # cos_angle = dot_product / (cricket_mag * 1.0)
        cos_angle = dot_product / (cricket_mag * head_mag)
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        angle_deg = np.rad2deg(angle_rad)
        
        # Determine sign using cross product
        # cross_product = np.cross([cricket_vector[0], cricket_vector[1], 0], 
        #                        [ref_vector[0], ref_vector[1], 0])[2]
        # cross_product = -np.cross([cricket_vector[0], cricket_vector[1], 0], 
        #                 [ref_vector[0], ref_vector[1], 0])[2

        if cross_product > 0:
            angle_deg = -angle_deg
            
        return angle_deg

    def calculate_distance(self, cricket_x: float, cricket_y: float,
                         nose_x: float, nose_y: float) -> float:
        """Calculate distance between cricket and nose."""
        return np.linalg.norm(np.array([nose_x, nose_y]) - np.array([cricket_x, cricket_y]))

    def classify_visual_field(self, head_angle: float, cricket_angle: float) -> str:
        """
        Classify cricket's position in visual field based on angle difference.
        
        Args:
            head_angle: Filtered head angle relative to vertical
            cricket_angle: Filtered cricket angle relative to vertical
        """
        # Calculate relative angle between head and cricket
        # relative_angle = cricket_angle - head_angle
        
        # Normalize to [-180, 180]
        # relative_angle = (relative_angle + 180) % 360 - 180
        
        relative_angle = cricket_angle
        
        # Classify based on relative angle
        if abs(relative_angle) <= self.config.binocular_threshold:
            return 'binocular'
        elif abs(relative_angle) <= self.config.monocular_threshold:
            return 'right_monocular' if relative_angle < 0 else 'left_monocular'
        else:
            return 'out_of_sight'

    def update_filters(self, cricket_angle: float, distance: float) -> Tuple[float, float]:
        """Update Kalman filters with new measurements."""
        if not self.initialized:
            self.cricket_angle_kf.x = np.array([[cricket_angle], [0.]])
            self.distance_kf.x = np.array([[distance], [0.]])
            self.initialized = True
            return cricket_angle, distance

        # Update cricket angle filter
        self.cricket_angle_kf.predict()
        self.cricket_angle_kf.update(np.array([cricket_angle]))

        # Update distance filter
        self.distance_kf.predict()
        self.distance_kf.update(np.array([distance]))

        return (self.cricket_angle_kf.x[0, 0], self.distance_kf.x[0, 0])
        
class VisualFieldVisualizer:
    """Handles visualization of angles and visual fields."""
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.colors = {
            'head_vector': (0, 255, 0),      # Green
            'cricket_vector': (0, 255, 255),  # Yellow
            'reference': (255, 255, 255),     # White
            'binocular': (0, 255, 0, 64),    # Green with alpha
            'left_monocular': (255, 0, 0, 64),  # Blue with alpha
            'right_monocular': (0, 0, 255, 64),  # Red with alpha
            'text': (255, 255, 255)          # White
        }

    def _to_int_tuple(self, point):
        """Convert a point's coordinates to integers."""
        if isinstance(point, tuple):
            return (int(point[0]), int(point[1]))
        else:
            raise TypeError("Expected a tuple of (x,y) coordinates")

    def draw_head_angle(self, frame: np.ndarray, 
                       body_center: Tuple[float, float],
                       head_angle: float,
                       line_length: int = 100) -> None:
        """Draw head direction vector."""
        angle_rad = np.deg2rad(head_angle)
        start_point = self._to_int_tuple(body_center)
        end_point = (
            # int(body_center[0] - line_length * np.sin(angle_rad)),
            # int(body_center[1] - line_length * np.cos(angle_rad))
            # int(body_center[0] + line_length * np.sin(angle_rad)),  # Changed sign
            # int(body_center[1] + line_length * np.cos(angle_rad))   # Changed sign
            int(body_center[0] + line_length * np.cos(angle_rad)),
            int(body_center[1] - line_length * np.sin(angle_rad))
        )
        
        cv2.line(frame, start_point, end_point, self.colors['head_vector'], 2, cv2.LINE_AA)

    def draw_cricket_angle(self, frame: np.ndarray,
                         nose_pos: Tuple[float, float],
                         cricket_angle: float,
                         line_length: int = 100) -> None:
        """Draw cricket direction vector."""
        angle_rad = np.deg2rad(cricket_angle)
        start_point = self._to_int_tuple(nose_pos)
        end_point = (
            # int(nose_pos[0] - line_length * np.sin(angle_rad)),
            # int(nose_pos[1] - line_length * np.cos(angle_rad))
            # int(nose_pos[0] + line_length * np.sin(angle_rad)),  # Changed sign
            # int(nose_pos[1] + line_length * np.cos(angle_rad))   # Changed sign
            int(nose_pos[0] + line_length * np.cos(angle_rad)),
            int(nose_pos[1] - line_length * np.sin(angle_rad))
        )
        cv2.line(frame, start_point, end_point, self.colors['cricket_vector'], 2, cv2.LINE_AA)

    def draw_visual_field_overlay(self, frame: np.ndarray,
                                nose_pos: Tuple[float, float],
                                head_angle: float,
                                zone: str) -> None:
        """Draw visual field overlay."""
        overlay = frame.copy()
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if zone in ['binocular', 'left_monocular', 'right_monocular']:
            nose_pos_int = self._to_int_tuple(nose_pos)
            
            if zone == 'binocular':
                start_angle = head_angle - self.config.binocular_threshold/2
                end_angle = head_angle + self.config.binocular_threshold/2
            elif zone == 'left_monocular':
                start_angle = head_angle - self.config.monocular_threshold/2
                end_angle = head_angle - self.config.binocular_threshold/2
            else:  # right_monocular
                start_angle = head_angle + self.config.binocular_threshold/2
                end_angle = head_angle + self.config.monocular_threshold/2

            # Convert angles to image coordinate system
            # start_angle = -start_angle
            # end_angle = -end_angle
            start_angle = start_angle + 90  # Adding 90 degrees to align with cv2.ellipse
            end_angle = end_angle + 90

            cv2.ellipse(mask, 
                       nose_pos_int,
                       (self.config.reference_circle_radius, self.config.reference_circle_radius),
                    #    90,  # Rotate 90 degrees to align with vertical reference
                        0,
                       start_angle, 
                       end_angle,
                       255, 
                       -1)

            color = self.colors[zone]
            colored_overlay = overlay.copy()
            colored_overlay[mask > 0] = color[:3]
            cv2.addWeighted(colored_overlay, color[3]/255.0, frame,
                          1 - color[3]/255.0, 0, frame)

    def add_metrics_display(self, frame: np.ndarray,
                          head_angle: float,
                          cricket_angle: float,
                          distance: float,
                          zone: str) -> None:
        """Add metrics display to frame."""
        metrics = [
            f"Head Angle: {head_angle:.1f}",
            f"Cricket Angle: {cricket_angle:.1f}",
            f"Distance: {distance:.1f}px",
            f"Zone: {zone}"
        ]

        for i, text in enumerate(metrics):
            cv2.putText(frame, text, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       self.colors['text'], 1, cv2.LINE_AA)

def process_frame(analyzer: AngleAnalyzer,
                 visualizer: VisualFieldVisualizer,
                 frame: np.ndarray,
                 pose_row: pd.Series,
                 cricket_pos: Optional[Tuple[float, float]],
                 cricket_row: Optional[pd.Series] = None,
                 previous_distance: float = None) -> Optional[dict]:
    """Process a single frame and return analysis results."""
    if cricket_pos is None:
        return None

    # Get positions
    # TODO: need to check for cases where keypoints and coordinates are normalized vs not
    # TODO: Also need to implement downsampling to 30hz, or....?
    tail_base = (pose_row['tail base x'], pose_row['tail base y'])
    body_center = (pose_row['body center x'], pose_row['body center y'])
    # body_center = (int(pose_row['body center x']), int(pose_row['body center y']))
    nose_pos = (pose_row['nose x'], pose_row['nose y'])
    # nose_pos = (int(pose_row['nose x']), int(pose_row['nose y']))
    cricket_pos = (cricket_pos[0], cricket_pos[1])

    # Calculate angles and distance
    dx = body_center[0] - tail_base[0]
    dy = body_center[1] - tail_base[1]
    baseline_angle = np.rad2deg(np.arctan2(-dy, dx))
    
    adjusted_head_angle = baseline_angle - pose_row['smoothed_head_angle']
    head_angle_end = (
        int(body_center[0] + 100 * np.cos(np.deg2rad(adjusted_head_angle))),
        int(body_center[1] - 100 * np.sin(np.deg2rad(adjusted_head_angle)))
    )
    cv2.line(frame, visualizer._to_int_tuple(body_center), head_angle_end, (0, 255, 0), 2)
    
    # head_angle = analyzer.calculate_head_angle(nose_pos[0], nose_pos[1],
    #                                          body_center[0], body_center[1])
    if nose_pos != cricket_pos:
        ##
        cricket_angle = analyzer.calculate_cricket_angle(cricket_pos[0], cricket_pos[1],
                                                    nose_pos[0], nose_pos[1],
                                                    body_center[0], body_center[1])
        distance = analyzer.calculate_distance(cricket_pos[0], cricket_pos[1],
                                            nose_pos[0], nose_pos[1])

        # Update filters
        # filtered_cricket_angle, filtered_distance = analyzer.update_filters(cricket_angle, distance)
        filtered_cricket_angle, filtered_distance = analyzer.update_filters(cricket_angle, distance)
        
        dxc = nose_pos[0] - body_center[0]
        dyc = nose_pos[1] - body_center[1]
        cricket_baseline = np.rad2deg(np.arctan2(-dyc, dxc))
        
        adjusted_cricket_angle = cricket_baseline - cricket_angle
    
        # Trying +-X, -+XX, --, ++
        cricket_angle_end = (
            int(nose_pos[0] + 100 * np.cos(np.deg2rad(adjusted_cricket_angle))),
            int(nose_pos[1] - 100 * np.sin(np.deg2rad(adjusted_cricket_angle)))
        )
        # cricket_angle_end = (
        #     int(nose_pos[0] + 100 * np.cos(np.deg2rad(-cricket_angle))),
        #     int(nose_pos[1] - 100 * np.sin(np.deg2rad(-cricket_angle)))
        # )
        cv2.line(frame, visualizer._to_int_tuple(nose_pos), cricket_angle_end, (255, 255, 0), 2)

        # draw line for debug cricket angle
        # cv2.line(frame, visualizer._to_int_tuple(nose_pos), visualizer._to_int_tuple(cricket_pos), (255, 255, 255), 2)
    
        # Classify visual field
        # zone = analyzer.classify_visual_field(head_angle, filtered_cricket_angle)
        zone = analyzer.classify_visual_field(adjusted_head_angle, cricket_angle)
    else:
        zone = 'binocular'
        cricket_angle = 0.0
        distance = 0.0
        filtered_cricket_angle, filtered_distance = analyzer.update_filters(cricket_angle, distance)

    # Extract cricket data from crprocessed
    cricket_status = cricket_row['status'] if cricket_row is not None else 'missing'
    cricket_reliability_score = cricket_row['reliability_score'] if cricket_row is not None else 0.0
    cricket_use_nose_position = cricket_row['use_nose_position'] if cricket_row is not None else False
    cricket_speed = cricket_row['speed'] if cricket_row is not None else 0.0
    cricket_size = cricket_row['size'] if cricket_row is not None else 0.0
    
    # Calculate simple temporal features (2 one-liners)
    distance_change = distance - previous_distance if previous_distance is not None else 0.0
    # frames_since_valid = 0 if cricket_status == 'validated' else 1  # Will be accumulated in process_video
    
    # Draw visualizations
    # visualizer.draw_body_angle(frame, tail_base, head_angle)
    # visualizer.draw_head_angle(frame, body_center, head_angle)
    # visualizer.draw_cricket_angle(frame, nose_pos, filtered_cricket_angle)
    # visualizer.draw_visual_field_overlay(frame, nose_pos, pose_row['smoothed_head_angle'], zone)
    # visualizer.draw_visual_field_overlay(frame, nose_pos, adjusted_head_angle, zone)
    # visualizer.add_metrics_display(frame, pose_row['smoothed_head_angle'], filtered_cricket_angle, 
    #                              filtered_distance, zone)
    visualizer.add_metrics_display(frame, pose_row['smoothed_head_angle'], cricket_angle, 
                                 distance, zone)

    return {
        'head_angle': pose_row['smoothed_head_angle'], # The angle on the video
        # 'head_angle': adjusted_head_angle, # the original angle
        'cricket_angle': cricket_angle,
        # 'cricket_angle': adjusted_cricket_angle,
        # 'filtered_cricket_angle': filtered_cricket_angle,
        'filtered_cricket_angle': filtered_cricket_angle,
        'distance': distance,
        'filtered_distance': filtered_distance,
        'zone': zone,
        'cricket_status': cricket_status,
        'cricket_reliability_score': cricket_reliability_score,
        'cricket_use_nose_position': cricket_use_nose_position,
        'cricket_speed': cricket_speed,
        'cricket_size': cricket_size,
        'distance_change': distance_change,
        'tail_base': tail_base,
        'body_center': body_center,
        'nose': nose_pos,
        # 'frames_since_valid_detection': frames_since_valid,
    }

def process_video(video_path: str, pose_data_path: str, bbox_data_path: str, 
                 output_path: str, config: ValidationConfig) -> bool:
    """Process a single video with visual field analysis."""
    try:
        # Initialize analyzers
        analyzer = AngleAnalyzer(config)
        visualizer = VisualFieldVisualizer(config)
        # TODO: shouldn't x_resol and y_resol be tied to the video resolution?
        resol_x = config.x_resol
        resol_y = config.y_resol
        downsampleflag = False
        
        # Load data
        pose_data = pd.read_csv(pose_data_path)
        # Check if the coordinates are normalized
        if pose_data['body center x'].values.max() <1.1:
            pose_data.loc[:, ['body center x', 'nose x', 'tail base x']] *= resol_x
            pose_data.loc[:, ['body center y', 'nose y', 'tail base y']] *= resol_y

        # TODO: same thing
        bbox_data = pd.read_csv(bbox_data_path)
        # cricket_data = pd.read_csv(bbox_data_path)
        
        # Initialize results DataFrame
        results = pd.DataFrame()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        # Initialize video writer
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = int(cap.get(cv2.CAP_PROP_FPS)) # FPS problem
        if original_fps > 30:
            pose_data = pose_data[pose_data['frame'] % 2 == 0].reset_index(drop=True)
            pose_data['frame'] = pose_data['frame'] // 2  # Divide by 2 and floor to integer
            downsampleflag = True
            fps = 30
        else:
            fps = original_fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Process each frame
        frame_idx = 0
        last_valid_cricket = None
        frames_since_valid = 0
        video_frame_idx = 0  # Original video frame index
        previous_distance = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: # or frame_idx >= len(pose_data):
                break
            video_frame_idx += 1
            if downsampleflag and video_frame_idx % 2 != 1:
                continue
                            
            # Get current frame data
            pose_row = pose_data.iloc[frame_idx]                
            # bbox_row = bbox_data[bbox_data['frame'] == frame_idx]
            # cricket_row = cricket_data[cricket_data['frame'] == frame_idx].iloc[0] if not cricket_data[cricket_data['frame'] == frame_idx].empty else None
            bbox_row = bbox_data[bbox_data['frame'] == frame_idx].iloc[0] if not bbox_data[bbox_data['frame'] == frame_idx].empty else None

            # Get cricket position
            cricket_pos = None
            # if not bbox_row.empty and not bbox_row[['smoothed_x', 'smoothed_y', 'smoothed_w', 'smoothed_h']].isnull().values.any():
            if bbox_row is not None and not bbox_row[['smoothed_x', 'smoothed_y', 'smoothed_w', 'smoothed_h']].isnull().any():
                # if bbox_row['use_nose_position'].values[0]:
                if bbox_row['use_nose_position']:
                    nose_x, nose_y = pose_row['nose x'], pose_row['nose y']
                    use_nose = True
                    
                    if last_valid_cricket is not None:
                        scaled_threshold = config.base_distance_threshold * (
                            1 + config.distance_scaling_factor * frames_since_valid
                        )
                        distance = np.sqrt(
                            (nose_x - last_valid_cricket[0])**2 + 
                            (nose_y - last_valid_cricket[1])**2
                        )
                        use_nose = distance <= scaled_threshold
                    
                    if use_nose:
                        cricket_pos = (nose_x, nose_y)
                        frames_since_valid += 1
                    else:
                        cricket_pos = last_valid_cricket
                        frames_since_valid += 1
                else:
                    # Convert normalized coordinates to pixel coordinates
                    # bbox_x = bbox_row['smoothed_x'].values[0] * resol_x
                    # bbox_y = bbox_row['smoothed_y'].values[0] * resol_y
                    # TODO: TEST resolution linking 
                    bbox_x = bbox_row['smoothed_x'] * frame_width
                    bbox_y = bbox_row['smoothed_y'] * frame_height
                    
                    cricket_pos = (bbox_x, bbox_y)
                    last_valid_cricket = cricket_pos
                    frames_since_valid = 0
                
                # # Draw cricket bounding box
                # bbox_w = bbox_row['smoothed_w'].values[0] * resol_x / 2
                # bbox_h = bbox_row['smoothed_h'].values[0] * resol_y / 2
                # TODO: test resolution linking
                bbox_w = bbox_row['smoothed_w'] * frame_width / 2
                bbox_h = bbox_row['smoothed_h'] * frame_height / 2
                top_left = (int(cricket_pos[0] - bbox_w), int(cricket_pos[1] - bbox_h))
                bottom_right = (int(cricket_pos[0] + bbox_w), int(cricket_pos[1] + bbox_h))
                color = (0, 255, 255) if bbox_row['use_nose_position'] else (255, 255, 0)
                cv2.rectangle(frame, top_left, bottom_right, color, 2)
            
            # Process frame and get results
            # frame_results = process_frame(analyzer, visualizer, frame, pose_row, cricket_pos)
            # frame_results = process_frame(analyzer, visualizer, frame, pose_row, cricket_pos, cricket_row, previous_distance)
            frame_results = process_frame(analyzer, visualizer, frame, pose_row, cricket_pos, bbox_row, previous_distance)
            frames_since_valid_counter = 0

            if frame_results is not None:
                # Add frame index and store results
                frame_results['frame'] = frame_idx
                # Add validation results row 
                frame_results['validation'] = bbox_row['status']
                frame_results['frames_since_valid_detection'] = frames_since_valid_counter
                if frame_results['cricket_status'] == 'validated':
                    frames_since_valid_counter = 0  # Reset for next frame
                else:
                    frames_since_valid_counter += 1  # Increment for next frame
                previous_distance = frame_results['distance']
                results = pd.concat([results, pd.DataFrame([frame_results])], 
                ignore_index=True)
            
            # Add frame number
            cv2.putText(frame, f"Frame: {frame_idx}", 
                    (frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
            
            # Write frame
            out.write(frame)
            frame_idx += 1
            # else:
            #     continue
            # video_frame_idx += 1    
        
        # Save results
        results_path = output_path.replace('.mp4', '_analysis.csv')
        results.to_csv(results_path, index=False)
        
        # Clean up
        cap.release()
        out.release()
        print(f"Successfully processed: {os.path.basename(video_path)}")
        return True
        
    except Exception as e:
        print(f"Error processing {os.path.basename(video_path)}: {str(e)}")
        return False

def process_directory(video_dir: str, pose_dir: str, bbox_dir: str, 
                     output_dir: str, config: Optional[ValidationConfig] = None) -> None:
    """Process all videos in directory."""
    if config is None:
        config = ValidationConfig()
    
    os.makedirs(output_dir, exist_ok=True)
    
    video_extensions = ['*.MP4', '*.mp4', '*.AVI', '*.avi', '*.MOV', '*.mov']
    video_files = []
    for ext in video_extensions:
        video_files.extend(Path(video_dir).glob(ext))
    successful = 0
    failed = 0
    
    for video_path in video_files:
        base_name = video_path.stem
        pose_path = Path(pose_dir) / f"kalman_filtered_processed_filtered_{base_name}.csv"
        bbox_path = Path(bbox_dir) / f"crprocessed_{base_name}.csv"
        output_path = Path(output_dir) / f"{base_name}_validated.mp4"
        
        print(f"\nProcessing {base_name}...")
        
        if not pose_path.exists():
            print(f"Missing pose data file for {base_name}")
            failed += 1
            continue
            
        if not bbox_path.exists():
            print(f"Missing bbox data file for {base_name}")
            failed += 1
            continue
        
        if process_video(str(video_path), str(pose_path), 
                        str(bbox_path), str(output_path), config):
            successful += 1
        else:
            failed += 1
    
    print(f"\nProcessing complete:")
    print(f"Successfully processed: {successful} videos")
    print(f"Failed to process: {failed} videos")

if __name__ == "__main__":
    # Directory paths
    # VIDEO_DIR = "/media/tarislada/SAMSUNG/Doric임시/854_0304/representative/raw_vid"
    # POSE_DIR = "/media/tarislada/SAMSUNG/Doric임시/854_0304/representative/postprocessing/kal_filt"
    # BBOX_DIR = "/media/tarislada/SAMSUNG/Doric임시/854_0304/representative/postprocessing/cricket_process_test"
    # OUTPUT_DIR = "/media/tarislada/SAMSUNG/Doric임시/854_0304/representative/postprocessing/final_vid"
    
    VIDEO_DIR = "SKH FP/video_file"
    POSE_DIR = "SKH FP/kalman_filtered_w59p7"
    BBOX_DIR = "SKH FP/FInalized_process/cricket_process_test5"
    OUTPUT_DIR = "SKH FP/FInalized_process/test_val_vid5"
    
    # Configure parameters
    config = ValidationConfig(
        angle_line_length=50,
        nose_vector_length=100,
        tail_vector_length=200,
        frame_rate=60,
        base_distance_threshold=75,
        distance_scaling_factor=0.025,
        binocular_threshold=30.0,
        monocular_threshold=140.0,  
        process_variance=1e-3,
        measurement_variance=1e-5,
        reference_circle_radius=150,
        show_angle_markers=True,
        marker_interval=30,
        x_resol=1920,
        y_resol=1080,
    )
    
    # Run batch processing
    # process_directory(VIDEO_DIR, POSE_DIR, BBOX_DIR, OUTPUT_DIR, config)

    # For single video testing:
    validator = VisualFieldVisualizer(config)
    video_path = '/home/tarislada/Documents/Extra_python_projects/SKH FP/video_file/m18_t7.mp4'
    pose_path = '/home/tarislada/Documents/Extra_python_projects/SKH FP/kalman_filtered_w59p7/kalman_filtered_processed_filtered_m18_t7.csv'
    bbox_path = '/home/tarislada/Documents/Extra_python_projects/SKH FP/FInalized_process/cricket_process_test4/crprocessed_m18_t7.csv'
    output_path = '/home/tarislada/Documents/Extra_python_projects/SKH FP/FInalized_process/test_val_vid5/m18_t7_validated.mp4'
    process_video(video_path, pose_path, bbox_path, output_path, config=config)