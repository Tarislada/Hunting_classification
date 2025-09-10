import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple, Optional
import seaborn as sns

class PoseEvaluator:
    def __init__(self, keypoint_names: List[str]):
        """
        Initialize evaluator with keypoint names
        
        Args:
            keypoint_names: List of keypoint names (e.g., ['nose', 'body_center', 'tail_base'])
        """
        self.keypoint_names = keypoint_names
        self.num_keypoints = len(keypoint_names)
        
    def normalize_coordinates(self, df: pd.DataFrame, image_width: int = 1920, image_height: int = 1080) -> pd.DataFrame:
        """
        Convert pixel coordinates to normalized coordinates (0-1 range)
        
        Args:
            df: DataFrame with pixel coordinates
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            DataFrame with normalized coordinates
        """
        df_norm = df.copy()
        
        for keypoint in self.keypoint_names:
            x_col = f'{keypoint}_x'
            y_col = f'{keypoint}_y'
            
            if x_col in df_norm.columns and y_col in df_norm.columns:
                # Normalize x coordinates by image width
                df_norm[x_col] = df_norm[x_col] / image_width
                # Normalize y coordinates by image height  
                df_norm[y_col] = df_norm[y_col] / image_height
        
        return df_norm
    
    def denormalize_coordinates(self, df: pd.DataFrame, image_width: int = 1920, image_height: int = 1080) -> pd.DataFrame:
        """
        Convert normalized coordinates (0-1 range) to pixel coordinates
        
        Args:
            df: DataFrame with normalized coordinates
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            DataFrame with pixel coordinates
        """
        df_pixel = df.copy()
        
        for keypoint in self.keypoint_names:
            x_col = f'{keypoint}_x'
            y_col = f'{keypoint}_y'
            
            if x_col in df_pixel.columns and y_col in df_pixel.columns:
                # Convert x coordinates to pixels
                df_pixel[x_col] = df_pixel[x_col] * image_width
                # Convert y coordinates to pixels
                df_pixel[y_col] = df_pixel[y_col] * image_height
        
        return df_pixel
    def standardize_csv_format(self, df: pd.DataFrame, format_type: str) -> pd.DataFrame:
        """
        Convert different CSV formats to standardized format
        
        Args:
            df: Input dataframe
            format_type: 'yolo', 'deeplabcut', or 'ground_truth'
            
        Returns:
            Standardized dataframe
        """
        if format_type == 'yolo':
            return self._standardize_yolo_format(df)
        elif format_type == 'deeplabcut':
            return self._standardize_deeplabcut_format(df)
        elif format_type == 'ground_truth':
            return self._standardize_ground_truth_format(df)
        else:
            raise ValueError(f"Unknown format type: {format_type}")
    
    def _standardize_yolo_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert your model's CSV format to standard format"""
        # Your format: frame, animal_ID, bbox(4), bbox_conf, keypoints_xy(22), keypoints_conf(11)
        # Total 40 columns: 1+1+4+1+22+11 = 40
        
        standardized = pd.DataFrame()
        standardized['frame_id'] = df.iloc[:, 0]  # First column is frame
        
        # Extract keypoint coordinates (columns 6-27: indices 6 to 27)
        # Extract keypoint confidences (columns 28-38: indices 28 to 38)
        keypoint_start_idx = 6  # After frame, animal_ID, bbox(4), bbox_conf
        
        for i, keypoint in enumerate(self.keypoint_names):
            x_idx = keypoint_start_idx + i * 2      # x coordinates
            y_idx = keypoint_start_idx + i * 2 + 1  # y coordinates  
            conf_idx = keypoint_start_idx + 22 + i  # confidence columns start after all xy pairs
            
            standardized[f'{keypoint}_x'] = df.iloc[:, x_idx]
            standardized[f'{keypoint}_y'] = df.iloc[:, y_idx]
            standardized[f'{keypoint}_conf'] = df.iloc[:, conf_idx]
        
        return standardized
    
    def _standardize_deeplabcut_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert DeepLabCut format to standard format"""
        
        # Check if we have multi-level columns (typical DLC format)
        if hasattr(df.columns, 'levels') or isinstance(df.columns, pd.MultiIndex):
            # Already loaded with multi-level headers
            bodyparts = df.columns.get_level_values(1)
            coords = df.columns.get_level_values(2)
            
        else:
            # Need to create multi-level from header rows
            # The format is: scorer row, bodyparts row, coords row, then data
            if len(df) >= 3 and 'scorer' in str(df.iloc[0, 0]).lower():
                # Skip the scorer row, use bodyparts and coords rows
                bodyparts = df.iloc[1].values
                coords = df.iloc[2].values
                df = df.iloc[3:].reset_index(drop=True)  # Skip header rows
                
                # Convert to numeric
                df = df.apply(pd.to_numeric, errors='coerce')
            else:
                raise ValueError("Could not parse DLC format. Expected multi-level headers or header rows.")
        
        standardized = pd.DataFrame()
        standardized['frame_id'] = df.index
        
        # Map DLC keypoint names to our standardized names
        dlc_to_standard = {
            'nose': 'nose',
            'left_ear': 'left_ear', 
            'right_ear': 'right_ear',
            'left_forelimb': 'left_forelimb',
            'right_forelimb': 'right_forelimb', 
            'left_hindlimb': 'left_hindlimb',
            'right_hindlimb': 'right_hindlimb',
            'tail_base': 'tail_base',
            'tail_mid': 'tail_mid', 
            'tail_end': 'tail_end',
            'bodycenter': 'body_center'  # Note: DLC uses 'Bodycenter', we use 'body_center'
        }
        
        # Extract keypoints
        for dlc_name, std_name in dlc_to_standard.items():
            if std_name not in self.keypoint_names:
                continue
                
            # Find columns for this bodypart
            if hasattr(df.columns, 'levels'):
                # Multi-level columns
                bodypart_mask = (df.columns.get_level_values(1).str.lower() == dlc_name.lower())
                
                x_cols = df.columns[bodypart_mask & (df.columns.get_level_values(2) == 'x')]
                y_cols = df.columns[bodypart_mask & (df.columns.get_level_values(2) == 'y')]
                conf_cols = df.columns[bodypart_mask & (df.columns.get_level_values(2) == 'likelihood')]
                
            else:
                # Single level after processing header rows
                x_idx = None
                y_idx = None
                conf_idx = None
                
                for i, (bp, coord) in enumerate(zip(bodyparts, coords)):
                    if bp.lower() == dlc_name.lower():
                        if coord.lower() == 'x':
                            x_idx = i
                        elif coord.lower() == 'y':
                            y_idx = i
                        elif coord.lower() == 'likelihood':
                            conf_idx = i
                
                x_cols = [df.columns[x_idx]] if x_idx is not None else []
                y_cols = [df.columns[y_idx]] if y_idx is not None else []
                conf_cols = [df.columns[conf_idx]] if conf_idx is not None else []
            
            # Extract data
            if len(x_cols) > 0 and len(y_cols) > 0:
                standardized[f'{std_name}_x'] = df[x_cols[0]]
                standardized[f'{std_name}_y'] = df[y_cols[0]]
                
                if len(conf_cols)>0:
                    standardized[f'{std_name}_conf'] = df[conf_cols[0]]
                else:
                    standardized[f'{std_name}_conf'] = 1.0
            else:
                # Keypoint not found
                standardized[f'{std_name}_x'] = np.nan
                standardized[f'{std_name}_y'] = np.nan
                standardized[f'{std_name}_conf'] = 0.0
        
        return standardized
    
    def _standardize_ground_truth_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert ground truth format to standard format"""
        # This assumes df was created from txt files using load_ground_truth_from_txt
        return df
    
    def load_ground_truth_from_txt(self, txt_directory: str, frame_pattern: str = "KH_set5_frame_{:06d}.txt") -> pd.DataFrame:
        """
        Load ground truth from individual txt files
        
        Args:
            txt_directory: Directory containing txt files
            frame_pattern: Pattern for frame files (use {} for frame number)
            
        Returns:
            Standardized dataframe with ground truth
        """
        import os
        import glob
        
        # Find all txt files
        txt_files = glob.glob(os.path.join(txt_directory, "*.txt"))
        
        all_data = []
        
        for txt_file in txt_files:
            # Extract frame number from filename
            filename = os.path.basename(txt_file)
            try:
                # Try to extract frame number from filename
                frame_num = int(filename.split('_')[-1].split('.')[0])
            except:
                # If pattern doesn't match, skip this file
                continue
                
            # Read the txt file
            with open(txt_file, 'r') as f:
                line = f.readline().strip()
            
            if not line:
                continue
                
            # Parse YOLO format: class x_center y_center width height x1 y1 v1 x2 y2 v2 ...
            values = list(map(float, line.split()))
            
            if len(values) < 5:  # Need at least class + bbox
                continue
                
            # Skip class and bbox info (first 5 values)
            keypoint_data = values[5:]
            
            # Parse keypoints (groups of 3: x, y, visibility)
            frame_data = {'frame_id': frame_num}
            
            for i, keypoint in enumerate(self.keypoint_names):
                if i * 3 + 2 < len(keypoint_data):  # Check if we have x, y, visibility
                    x = keypoint_data[i * 3]
                    y = keypoint_data[i * 3 + 1]
                    visibility = keypoint_data[i * 3 + 2]
                    
                    frame_data[f'{keypoint}_x'] = x
                    frame_data[f'{keypoint}_y'] = y
                    # Convert visibility to confidence (2=visible->1.0, 1=occluded->0.5, 0=not labeled->0.0)
                    if visibility == 2:
                        frame_data[f'{keypoint}_conf'] = 1.0
                    elif visibility == 1:
                        frame_data[f'{keypoint}_conf'] = 0.5
                    else:
                        frame_data[f'{keypoint}_conf'] = 0.0
                else:
                    # Missing keypoint data
                    frame_data[f'{keypoint}_x'] = np.nan
                    frame_data[f'{keypoint}_y'] = np.nan
                    frame_data[f'{keypoint}_conf'] = 0.0
            
            all_data.append(frame_data)
        
        # Convert to DataFrame and sort by frame
        df = pd.DataFrame(all_data)
        df = df.sort_values('frame_id').reset_index(drop=True)
        
        return df
    
    def compute_l2_error(self, pred_df: pd.DataFrame, gt_df: pd.DataFrame, 
                        conf_threshold: float = 0.0) -> Dict[str, float]:
        """
        Compute L2 (Euclidean) error between predictions and ground truth
        
        Args:
            pred_df: Predictions in standardized format
            gt_df: Ground truth in standardized format
            conf_threshold: Minimum confidence threshold
            
        Returns:
            Dictionary with L2 errors per keypoint and overall
        """
        errors = {}
        all_errors = []
        
        for keypoint in self.keypoint_names:
            pred_x = pred_df[f'{keypoint}_x'].values
            pred_y = pred_df[f'{keypoint}_y'].values
            pred_conf = pred_df[f'{keypoint}_conf'].values
            
            gt_x = gt_df[f'{keypoint}_x'].values
            gt_y = gt_df[f'{keypoint}_y'].values
            
            # Filter by confidence
            valid_mask = (pred_conf >= conf_threshold) & \
                        ~np.isnan(pred_x) & ~np.isnan(pred_y) & \
                        ~np.isnan(gt_x) & ~np.isnan(gt_y)
            
            if np.sum(valid_mask) > 0:
                l2_errors = np.sqrt((pred_x[valid_mask] - gt_x[valid_mask])**2 + 
                                   (pred_y[valid_mask] - gt_y[valid_mask])**2)
                errors[keypoint] = np.mean(l2_errors)
                all_errors.extend(l2_errors)
            else:
                errors[keypoint] = np.nan
        
        errors['overall'] = np.mean(all_errors) if all_errors else np.nan
        return errors
    
    def compute_mpjpe(self, pred_df: pd.DataFrame, gt_df: pd.DataFrame,
                     conf_threshold: float = 0.0) -> float:
        """
        Compute Mean Per Joint Position Error (MPJPE)
        
        Args:
            pred_df: Predictions in standardized format
            gt_df: Ground truth in standardized format
            conf_threshold: Minimum confidence threshold
            
        Returns:
            MPJPE value
        """
        all_errors = []
        
        for keypoint in self.keypoint_names:
            pred_x = pred_df[f'{keypoint}_x'].values
            pred_y = pred_df[f'{keypoint}_y'].values
            pred_conf = pred_df[f'{keypoint}_conf'].values
            
            gt_x = gt_df[f'{keypoint}_x'].values
            gt_y = gt_df[f'{keypoint}_y'].values
            
            # Filter by confidence and valid coordinates
            valid_mask = (pred_conf >= conf_threshold) & \
                        ~np.isnan(pred_x) & ~np.isnan(pred_y) & \
                        ~np.isnan(gt_x) & ~np.isnan(gt_y)
            
            if np.sum(valid_mask) > 0:
                joint_errors = np.sqrt((pred_x[valid_mask] - gt_x[valid_mask])**2 + 
                                     (pred_y[valid_mask] - gt_y[valid_mask])**2)
                all_errors.extend(joint_errors)
        
        return np.mean(all_errors) if all_errors else np.nan
    
    def compute_pck(self, pred_df: pd.DataFrame, gt_df: pd.DataFrame,
                   threshold: float = 0.2, normalize_by: str = 'image_size',
                   image_size: Tuple[int, int] = (1920, 1080),
                   conf_threshold: float = 0.0) -> Dict[str, float]:
        """
        Compute Percentage of Correct Keypoints (PCK)
        
        Args:
            pred_df: Predictions in standardized format
            gt_df: Ground truth in standardized format
            threshold: Distance threshold as fraction of normalization factor
            normalize_by: 'image_size', 'bbox_size', or 'torso_size'
            image_size: Image dimensions for normalization
            conf_threshold: Minimum confidence threshold
            
        Returns:
            Dictionary with PCK values per keypoint and overall
        """
        pck_scores = {}
        all_correct = []
        
        # Determine normalization factor
        if normalize_by == 'image_size':
            # For normalized coordinates, use fraction of image diagonal
            norm_factor = np.sqrt(1.0**2 + 1.0**2)  # Diagonal of normalized space
        elif normalize_by == 'torso_size':
            # Use distance between nose and tail_base as normalization
            nose_x = gt_df['nose_x'].values
            nose_y = gt_df['nose_y'].values
            tail_x = gt_df['tail_base_x'].values
            tail_y = gt_df['tail_base_y'].values
            torso_lengths = np.sqrt((nose_x - tail_x)**2 + (nose_y - tail_y)**2)
            norm_factor = np.mean(torso_lengths[~np.isnan(torso_lengths)])
        else:
            norm_factor = 1.0
        
        threshold_norm = threshold * norm_factor
        
        for keypoint in self.keypoint_names:
            pred_x = pred_df[f'{keypoint}_x'].values
            pred_y = pred_df[f'{keypoint}_y'].values
            pred_conf = pred_df[f'{keypoint}_conf'].values
            
            gt_x = gt_df[f'{keypoint}_x'].values
            gt_y = gt_df[f'{keypoint}_y'].values
            
            # Filter by confidence
            valid_mask = (pred_conf >= conf_threshold) & \
                        ~np.isnan(pred_x) & ~np.isnan(pred_y) & \
                        ~np.isnan(gt_x) & ~np.isnan(gt_y)
            
            if np.sum(valid_mask) > 0:
                distances = np.sqrt((pred_x[valid_mask] - gt_x[valid_mask])**2 + 
                                  (pred_y[valid_mask] - gt_y[valid_mask])**2)
                correct = distances <= threshold_norm
                pck_scores[keypoint] = np.mean(correct)
                all_correct.extend(correct)
            else:
                pck_scores[keypoint] = np.nan
        
        pck_scores['overall'] = np.mean(all_correct) if all_correct else np.nan
        return pck_scores
    
    def compute_oks(self, pred_df: pd.DataFrame, gt_df: pd.DataFrame,
                   keypoint_sigmas: Optional[List[float]] = None,
                   conf_threshold: float = 0.0) -> Dict[str, float]:
        """
        Compute Object Keypoint Similarity (OKS)
        
        Args:
            pred_df: Predictions in standardized format
            gt_df: Ground truth in standardized format
            keypoint_sigmas: Standard deviations for each keypoint type
            conf_threshold: Minimum confidence threshold
            
        Returns:
            Dictionary with OKS values per frame and overall
        """
        if keypoint_sigmas is None:
            # Default sigmas for different keypoint types (adjusted for normalized coordinates)
            keypoint_sigmas = [0.025] * self.num_keypoints
        
        oks_scores = []
        
        for idx in range(len(pred_df)):
            frame_oks = []
            
            for i, keypoint in enumerate(self.keypoint_names):
                pred_x = pred_df[f'{keypoint}_x'].iloc[idx]
                pred_y = pred_df[f'{keypoint}_y'].iloc[idx]
                pred_conf = pred_df[f'{keypoint}_conf'].iloc[idx]
                
                gt_x = gt_df[f'{keypoint}_x'].iloc[idx]
                gt_y = gt_df[f'{keypoint}_y'].iloc[idx]
                
                # Skip if confidence too low or invalid coordinates
                if pred_conf < conf_threshold or np.isnan(pred_x) or np.isnan(pred_y) or \
                   np.isnan(gt_x) or np.isnan(gt_y):
                    continue
                
                # Compute distance
                d = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
                
                # Object scale for normalized coordinates
                # Use diagonal of normalized space as scale
                s = np.sqrt(1.0**2 + 1.0**2)
                
                # Compute OKS for this keypoint
                oks_kp = np.exp(-d**2 / (2 * s**2 * keypoint_sigmas[i]**2))
                frame_oks.append(oks_kp)
            
            if frame_oks:
                oks_scores.append(np.mean(frame_oks))
        
        return {'overall': np.mean(oks_scores) if oks_scores else np.nan}
    
    def run_evaluation(self, model_a_df: pd.DataFrame, model_b_df: pd.DataFrame,
                      gt_df: pd.DataFrame, model_a_name: str = "Model A",
                      model_b_name: str = "Model B") -> Dict:
        """
        Run comprehensive evaluation comparing two models
        
        Args:
            model_a_df: First model predictions (standardized format)
            model_b_df: Second model predictions (standardized format)
            gt_df: Ground truth (standardized format)
            model_a_name: Name for first model
            model_b_name: Name for second model
            
        Returns:
            Dictionary with all evaluation results
        """
        results = {}
        
        # L2 Error
        results['l2_error'] = {
            model_a_name: self.compute_l2_error(model_a_df, gt_df),
            model_b_name: self.compute_l2_error(model_b_df, gt_df)
        }
        
        # MPJPE
        results['mpjpe'] = {
            model_a_name: self.compute_mpjpe(model_a_df, gt_df),
            model_b_name: self.compute_mpjpe(model_b_df, gt_df)
        }
        
        # PCK at multiple thresholds
        thresholds = [0.1, 0.2, 0.5]
        results['pck'] = {}
        for threshold in thresholds:
            results['pck'][f'pck@{threshold}'] = {
                model_a_name: self.compute_pck(model_a_df, gt_df, threshold=threshold),
                model_b_name: self.compute_pck(model_b_df, gt_df, threshold=threshold)
            }
        
        # OKS
        results['oks'] = {
            model_a_name: self.compute_oks(model_a_df, gt_df),
            model_b_name: self.compute_oks(model_b_df, gt_df)
        }
        
        return results
    
    def visualize_results(self, results: Dict, save_path: Optional[str] = None):
        """
        Create visualizations of evaluation results
        
        Args:
            results: Results from run_evaluation
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. L2 Error comparison
        ax1 = axes[0, 0]
        models = list(results['l2_error'].keys())
        l2_overall = [results['l2_error'][model]['overall'] for model in models]
        ax1.bar(models, l2_overall)
        ax1.set_title('L2 Error Comparison')
        ax1.set_ylabel('L2 Error (pixels)')
        
        # 2. MPJPE comparison
        ax2 = axes[0, 1]
        mpjpe_values = [results['mpjpe'][model] for model in models]
        ax2.bar(models, mpjpe_values)
        ax2.set_title('MPJPE Comparison')
        ax2.set_ylabel('MPJPE (pixels)')
        
        # 3. PCK comparison
        ax3 = axes[1, 0]
        pck_thresholds = list(results['pck'].keys())
        width = 0.35
        x = np.arange(len(pck_thresholds))
        
        model_a_pck = [results['pck'][thresh][models[0]]['overall'] for thresh in pck_thresholds]
        model_b_pck = [results['pck'][thresh][models[1]]['overall'] for thresh in pck_thresholds]
        
        ax3.bar(x - width/2, model_a_pck, width, label=models[0])
        ax3.bar(x + width/2, model_b_pck, width, label=models[1])
        ax3.set_title('PCK Comparison')
        ax3.set_ylabel('PCK Score')
        ax3.set_xticks(x)
        ax3.set_xticklabels(pck_thresholds)
        ax3.legend()
        
        # 4. OKS comparison
        ax4 = axes[1, 1]
        oks_values = [results['oks'][model]['overall'] for model in models]
        ax4.bar(models, oks_values)
        ax4.set_title('OKS Comparison')
        ax4.set_ylabel('OKS Score')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Define your keypoints - adjust based on your actual keypoint names
    keypoints = [
        'nose', 'left_ear', 'right_ear', 'left_forelimb', 'right_forelimb',
        'left_hindlimb', 'right_hindlimb', 'tail_base', 'tail_mid', 'tail_end', 'body_center'
    ]
    
    # Initialize evaluator
    evaluator = PoseEvaluator(keypoints)
    
    # Load and standardize data
    print("Loading data...")
    
    # 1. Load your model's CSV (no header, 40 columns)
    your_model_df = pd.read_csv('/home/tarislada/YOLOprojects/YOLO_custom/KH/KH_bot_IR_v4_val_YOLO2.csv', header=None)
    your_model_std = evaluator.standardize_csv_format(your_model_df, 'yolo')
    
    # 2. Load DeepLabCut CSV
    try:
        dlc_df = pd.read_csv('/home/tarislada/YOLOprojects/YOLO_custom/KH//KH_bot_IR_v4_valDLC_Resnet50_test3Jul11shuffle1_snapshot_best-100.csv', header=[0, 1, 2])  # Multi-level headers
    except:
        dlc_df = pd.read_csv('/home/tarislada/YOLOprojects/YOLO_custom/KH//KH_bot_IR_v4_valDLC_Resnet50_test3Jul11shuffle1_snapshot_best-100.csv')  # Single level, will be parsed
    dlc_std = evaluator.standardize_csv_format(dlc_df, 'deeplabcut')
    
    # 3. Load ground truth from txt files
    gt_std = evaluator.load_ground_truth_from_txt('/home/tarislada/YOLOprojects/YOLO_custom/Dataset/KH/YOLO_format/Bot_IR_Hunting/KH_bot_IR_v4/labels/val')

    # Normalize model outputs to match ground truth (which is already normalized)
    print("Normalizing model coordinates...")
    your_model_std = evaluator.normalize_coordinates(your_model_std)
    dlc_std = evaluator.normalize_coordinates(dlc_std)

    # Filter to common frames (intersection of all datasets)
    common_frames = set(your_model_std['frame_id']) & set(dlc_std['frame_id']) & set(gt_std['frame_id'])
    print(f"Found {len(common_frames)} common frames")
    
    your_model_filtered = your_model_std[your_model_std['frame_id'].isin(common_frames)].reset_index(drop=True)
    dlc_filtered = dlc_std[dlc_std['frame_id'].isin(common_frames)].reset_index(drop=True)
    gt_filtered = gt_std[gt_std['frame_id'].isin(common_frames)].reset_index(drop=True)
    
    # Run evaluation
    print("Running evaluation...")
    results = evaluator.run_evaluation(your_model_filtered, dlc_filtered, gt_filtered, 
                                      "Your Model", "DeepLabCut")
    
    # Print results
    print("\n=== EVALUATION RESULTS ===")
    print(f"L2 Error - Your Model: {results['l2_error']['Your Model']['overall']:.2f} pixels")
    print(f"L2 Error - DeepLabCut: {results['l2_error']['DeepLabCut']['overall']:.2f} pixels")
    print(f"MPJPE - Your Model: {results['mpjpe']['Your Model']:.2f} pixels")
    print(f"MPJPE - DeepLabCut: {results['mpjpe']['DeepLabCut']:.2f} pixels")
    
    for threshold in ['pck@0.1', 'pck@0.2', 'pck@0.5']:
        print(f"{threshold} - Your Model: {results['pck'][threshold]['Your Model']['overall']:.3f}")
        print(f"{threshold} - DeepLabCut: {results['pck'][threshold]['DeepLabCut']['overall']:.3f}")
    
    print(f"OKS - Your Model: {results['oks']['Your Model']['overall']:.3f}")
    print(f"OKS - DeepLabCut: {results['oks']['DeepLabCut']['overall']:.3f}")
    
    # Create visualizations
    evaluator.visualize_results(results, 'pose_evaluation_results.svg')
    
    # Save detailed results
    import json
    with open('detailed_results.txt', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nEvaluation complete! Check 'pose_evaluation_results.png' and 'detailed_results.json'")