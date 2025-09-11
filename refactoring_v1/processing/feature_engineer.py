"""
Feature engineering for visual field analysis and angle validation.
Creates analysis features and validation videos from pose and cricket data.
"""

import cv2
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from filterpy.kalman import KalmanFilter
from config.settings import settings

class AngleAnalyzer:
    """Handles Kalman filtering and visual field classification."""
    
    def __init__(self, config=None):
        self.config = config or settings.visual_field
        
        # Initialize Kalman filters for cricket measurements
        self.cricket_angle_kf = KalmanFilter(dim_x=2, dim_z=1)
        self.distance_kf = KalmanFilter(dim_x=2, dim_z=1)
        
        # Setup cricket angle Kalman filter
        self.cricket_angle_kf.F = np.array([[1., 1.], [0., 1.]])
        self.cricket_angle_kf.H = np.array([[1., 0.]])
        self.cricket_angle_kf.P *= 1.0
        self.cricket_angle_kf.R = self.config.measurement_variance
        self.cricket_angle_kf.Q = self.config.process_variance * np.array([[0.25, 0.5], [0.5, 1.]])
        
        # Setup distance Kalman filter
        self.distance_kf.F = np.array([[1., 1.], [0., 1.]])
        self.distance_kf.H = np.array([[1., 0.]])
        self.distance_kf.P *= 1.0
        self.distance_kf.R = self.config.measurement_variance
        self.distance_kf.Q = self.config.process_variance * np.array([[0.25, 0.5], [0.5, 1.]])
        
        self.initialized = False

    def calculate_head_angle(self, nose_x: float, nose_y: float, 
                           body_x: float, body_y: float) -> float:
        """Calculate head angle relative to vertical axis."""
        # Create head vector
        head_vector = np.array([nose_x - body_x, nose_y - body_y])
        
        # Create reference vector (0, 1) for vertical axis
        ref_vector = np.array([0, 1])
        
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

        if cross_product > 0:
            angle_deg = -angle_deg
            
        return angle_deg

    def calculate_cricket_angle(self, cricket_x: float, cricket_y: float,
                              nose_x: float, nose_y: float, 
                              body_x: float, body_y: float) -> float:
        """Calculate cricket angle relative to head direction."""
        # Create cricket vector from nose to cricket
        cricket_vector = np.array([cricket_x - nose_x, cricket_y - nose_y])
        head_vector = np.array([nose_x - body_x, nose_y - body_y])
        
        # Calculate angle
        dot_product = np.sum(cricket_vector * head_vector)
        cross_product = np.cross(cricket_vector, head_vector)
        cricket_mag = np.linalg.norm(cricket_vector)
        head_mag = np.linalg.norm(head_vector)
        
        if cricket_mag == 0:
            return 0.0
            
        cos_angle = dot_product / (cricket_mag * head_mag)
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        angle_deg = np.rad2deg(angle_rad)
        
        # Determine sign using cross product
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
    
    def __init__(self, config=None):
        self.config = config or settings.visual_field
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
            start_angle = start_angle + 90  # Adding 90 degrees to align with cv2.ellipse
            end_angle = end_angle + 90

            cv2.ellipse(mask, 
                       nose_pos_int,
                       (self.config.reference_circle_radius, self.config.reference_circle_radius),
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
            f"Head Angle: {head_angle:.1f}°",
            f"Cricket Angle: {cricket_angle:.1f}°",
            f"Distance: {distance:.1f}px",
            f"Zone: {zone}"
        ]

        for i, text in enumerate(metrics):
            cv2.putText(frame, text, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       self.colors['text'], 1, cv2.LINE_AA)

class FeatureEngineer:
    """
    Main feature engineering processor.
    Combines pose and cricket data to create analysis features and validation videos.
    """
    
    def __init__(self, config=None):
        self.config = config or settings.visual_field
        self.analyzer = AngleAnalyzer(config)
        self.visualizer = VisualFieldVisualizer(config)
    
    def process_frame(self, frame: np.ndarray, pose_row: pd.Series, 
                     cricket_pos: Optional[Tuple[float, float]],
                     cricket_row: Optional[pd.Series] = None,
                     previous_distance: float = None) -> Optional[Dict[str, Any]]:
        """Process a single frame and return analysis results."""
        if cricket_pos is None:
            return None

        # Get positions
        tail_base = (pose_row['tail base x'], pose_row['tail base y'])
        body_center = (pose_row['body center x'], pose_row['body center y'])
        nose_pos = (pose_row['nose x'], pose_row['nose y'])
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
        cv2.line(frame, self.visualizer._to_int_tuple(body_center), head_angle_end, (0, 255, 0), 2)
        
        if nose_pos != cricket_pos:
            cricket_angle = self.analyzer.calculate_cricket_angle(
                cricket_pos[0], cricket_pos[1],
                nose_pos[0], nose_pos[1],
                body_center[0], body_center[1]
            )
            distance = self.analyzer.calculate_distance(
                cricket_pos[0], cricket_pos[1],
                nose_pos[0], nose_pos[1]
            )

            # Update filters
            filtered_cricket_angle, filtered_distance = self.analyzer.update_filters(cricket_angle, distance)
            
            dxc = nose_pos[0] - body_center[0]
            dyc = nose_pos[1] - body_center[1]
            cricket_baseline = np.rad2deg(np.arctan2(-dyc, dxc))
            
            adjusted_cricket_angle = cricket_baseline - cricket_angle
        
            cricket_angle_end = (
                int(nose_pos[0] + 100 * np.cos(np.deg2rad(adjusted_cricket_angle))),
                int(nose_pos[1] - 100 * np.sin(np.deg2rad(adjusted_cricket_angle)))
            )
            cv2.line(frame, self.visualizer._to_int_tuple(nose_pos), cricket_angle_end, (255, 255, 0), 2)
    
            # Classify visual field
            zone = self.analyzer.classify_visual_field(adjusted_head_angle, cricket_angle)
        else:
            zone = 'binocular'
            cricket_angle = 0.0
            distance = 0.0
            filtered_cricket_angle, filtered_distance = self.analyzer.update_filters(cricket_angle, distance)

        # Extract cricket data from processed cricket file
        cricket_status = cricket_row['status'] if cricket_row is not None else 'missing'
        cricket_reliability_score = cricket_row['reliability_score'] if cricket_row is not None else 0.0
        cricket_use_nose_position = cricket_row['use_nose_position'] if cricket_row is not None else False
        cricket_speed = cricket_row['speed'] if cricket_row is not None else 0.0
        cricket_size = cricket_row['size'] if cricket_row is not None else 0.0
        
        # Calculate simple temporal features
        distance_change = distance - previous_distance if previous_distance is not None else 0.0
        
        # Add metrics display to frame
        self.visualizer.add_metrics_display(frame, pose_row['smoothed_head_angle'], cricket_angle, 
                                           distance, zone)

        return {
            'head_angle': pose_row['smoothed_head_angle'],
            'cricket_angle': cricket_angle,
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
        }
    
    def process_video(self, video_path: str, pose_data_path: str, 
                     cricket_data_path: str, output_path: str) -> bool:
        """Process a single video with visual field analysis."""
        try:
            # Initialize analyzers
            self.analyzer = AngleAnalyzer(self.config)
            self.visualizer = VisualFieldVisualizer(self.config)
            
            resol_x = self.config.x_resol
            resol_y = self.config.y_resol
            downsampleflag = False
            
            # Load data
            pose_data = pd.read_csv(pose_data_path)
            # Check if the coordinates are normalized
            if pose_data['body center x'].values.max() < 1.1:
                pose_data.loc[:, ['body center x', 'nose x', 'tail base x']] *= resol_x
                pose_data.loc[:, ['body center y', 'nose y', 'tail base y']] *= resol_y

            cricket_data = pd.read_csv(cricket_data_path)
            
            # Initialize results DataFrame
            results = pd.DataFrame()
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
                
            # Initialize video writer
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            original_fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            if original_fps > 30:
                pose_data = pose_data[pose_data['frame'] % 2 == 0].reset_index(drop=True)
                pose_data['frame'] = pose_data['frame'] // 2
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
            video_frame_idx = 0
            previous_distance = None
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                video_frame_idx += 1
                if downsampleflag and video_frame_idx % 2 != 1:
                    continue
                                
                # Get current frame data
                if frame_idx >= len(pose_data):
                    break
                    
                pose_row = pose_data.iloc[frame_idx]                
                cricket_row = cricket_data[cricket_data['frame'] == frame_idx].iloc[0] if not cricket_data[cricket_data['frame'] == frame_idx].empty else None

                # Get cricket position
                cricket_pos = None
                if cricket_row is not None and not cricket_row[['smoothed_x', 'smoothed_y', 'smoothed_w', 'smoothed_h']].isnull().any():
                    if cricket_row['use_nose_position']:
                        nose_x, nose_y = pose_row['nose x'], pose_row['nose y']
                        use_nose = True
                        
                        if last_valid_cricket is not None:
                            scaled_threshold = self.config.base_distance_threshold * (
                                1 + self.config.distance_scaling_factor * frames_since_valid
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
                        cricket_x = cricket_row['smoothed_x'] * frame_width
                        cricket_y = cricket_row['smoothed_y'] * frame_height
                        
                        cricket_pos = (cricket_x, cricket_y)
                        last_valid_cricket = cricket_pos
                        frames_since_valid = 0
                    
                    # Draw cricket bounding box
                    cricket_w = cricket_row['smoothed_w'] * frame_width / 2
                    cricket_h = cricket_row['smoothed_h'] * frame_height / 2
                    top_left = (int(cricket_pos[0] - cricket_w), int(cricket_pos[1] - cricket_h))
                    bottom_right = (int(cricket_pos[0] + cricket_w), int(cricket_pos[1] + cricket_h))
                    color = (0, 255, 255) if cricket_row['use_nose_position'] else (255, 255, 0)
                    cv2.rectangle(frame, top_left, bottom_right, color, 2)
                
                # Process frame and get results
                frame_results = self.process_frame(frame, pose_row, cricket_pos, cricket_row, previous_distance)
                frames_since_valid_counter = 0

                if frame_results is not None:
                    # Add frame index and store results
                    frame_results['frame'] = frame_idx
                    frame_results['validation'] = cricket_row['status'] if cricket_row is not None else 'missing'
                    frame_results['frames_since_valid_detection'] = frames_since_valid_counter
                    
                    if frame_results['cricket_status'] == 'validated':
                        frames_since_valid_counter = 0
                    else:
                        frames_since_valid_counter += 1
                        
                    previous_distance = frame_results['distance']
                    results = pd.concat([results, pd.DataFrame([frame_results])], ignore_index=True)
                
                # Add frame number
                cv2.putText(frame, f"Frame: {frame_idx}", 
                        (frame.shape[1] - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
                
                # Write frame
                out.write(frame)
                frame_idx += 1
            
            # Save results
            results_path = output_path.replace('.mp4', '_analysis.csv')
            results.to_csv(results_path, index=False)
            
            # Clean up
            cap.release()
            out.release()
            
            return True
            
        except Exception as e:
            print(f"Error processing {Path(video_path).name}: {e}")
            return False
    
    def process_directory(self, video_dir: str, pose_dir: str, 
                         cricket_dir: str, output_dir: str) -> dict:
        """Process all videos in directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        video_extensions = ['*.MP4', '*.mp4', '*.AVI', '*.avi', '*.MOV', '*.mov']
        video_files = []
        for ext in video_extensions:
            video_files.extend(Path(video_dir).glob(ext))
            
        if not video_files:
            print(f"No video files found in {video_dir}")
            return {"processed": 0, "failed": 0, "files": []}
        
        print(f"Found {len(video_files)} videos to process")
        
        successful = []
        failed = []
        
        for video_path in video_files:
            base_name = video_path.stem
            pose_path = Path(pose_dir) / f"kalman_filtered_{base_name}.csv"
            cricket_path = Path(cricket_dir) / f"crprocessed_{base_name}.csv"
            output_path = Path(output_dir) / f"{base_name}_validated.mp4"
            
            print(f"Processing {base_name}...")
            
            if not pose_path.exists():
                print(f"  ✗ Missing pose data: {pose_path.name}")
                failed.append(base_name)
                continue
                
            if not cricket_path.exists():
                print(f"  ✗ Missing cricket data: {cricket_path.name}")
                failed.append(base_name)
                continue
            
            if self.process_video(str(video_path), str(pose_path), 
                                str(cricket_path), str(output_path)):
                successful.append(base_name)
                print(f"  ✓ Saved as {output_path.name}")
            else:
                failed.append(base_name)
        
        # Summary
        print(f"\nFeature engineering complete:")
        print(f"Successfully processed: {len(successful)} videos")
        print(f"Failed to process: {len(failed)} videos")
        
        return {
            "processed": len(successful),
            "failed": len(failed),
            "successful_files": successful,
            "failed_files": failed
        }
    
    def process(self, video_dir: Optional[str] = None,
                pose_dir: Optional[str] = None, 
                cricket_dir: Optional[str] = None,
                output_dir: Optional[str] = None) -> dict:
        """
        Main processing method using configured directories.
        
        Args:
            video_dir: Override video directory (uses config if None)
            pose_dir: Override pose directory (uses config if None)
            cricket_dir: Override cricket directory (uses config if None)
            output_dir: Override output directory (uses config if None)
            
        Returns:
            dict: Processing results
        """
        video_dir = video_dir or str(settings.paths.raw_video_dir)
        pose_dir = pose_dir or str(settings.paths.kalman_filtered_dir)
        cricket_dir = cricket_dir or str(settings.paths.cricket_processed_dir)
        output_dir = output_dir or str(settings.paths.final_videos_dir)
        
        print("=== FEATURE ENGINEERING (Step 5) ===")
        print(f"Video directory: {video_dir}")
        print(f"Pose directory: {pose_dir}")
        print(f"Cricket directory: {cricket_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Parameters:")
        print(f"  - Frame rate: {self.config.frame_rate} fps")
        print(f"  - Binocular threshold: {self.config.binocular_threshold}°")
        print(f"  - Monocular threshold: {self.config.monocular_threshold}°")
        print(f"  - Video resolution: {self.config.x_resol}x{self.config.y_resol}")
        
        return self.process_directory(video_dir, pose_dir, cricket_dir, output_dir)

def main():
    """Command line interface for feature engineering."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create analysis features and validation videos")
    parser.add_argument("--video-dir", help="Video directory (overrides config)")
    parser.add_argument("--pose-dir", help="Pose directory (overrides config)")
    parser.add_argument("--cricket-dir", help="Cricket directory (overrides config)")
    parser.add_argument("--output-dir", help="Output directory (overrides config)")
    parser.add_argument("--binocular-threshold", type=float, help="Binocular field threshold")
    parser.add_argument("--monocular-threshold", type=float, help="Monocular field threshold")
    
    args = parser.parse_args()
    
    # Override config if arguments provided
    if args.binocular_threshold:
        settings.visual_field.binocular_threshold = args.binocular_threshold
    if args.monocular_threshold:
        settings.visual_field.monocular_threshold = args.monocular_threshold
    
    # Create processor and run
    processor = FeatureEngineer()
    results = processor.process(args.video_dir, args.pose_dir, args.cricket_dir, args.output_dir)
    
    # Exit with error code if any files failed
    if results["failed"] > 0:
        exit(1)

if __name__ == "__main__":
    main()