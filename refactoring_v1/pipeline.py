"""
Main pipeline for hunting behavior classification.
Orchestrates all processing steps from raw data to behavioral classification.
"""

import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import traceback

from config.settings import settings
from processing.keypoint_filter import KeypointFilter
from processing.head_angle import HeadAngleProcessor
from processing.kalman_filter import KalmanAngleProcessor
from processing.cricket_processor import CricketProcessor
from processing.feature_engineer import FeatureEngineer
from processing.action_classifier import ActionClassifier
from utils.label_processor import LabelProcessor

class HuntingClassificationPipeline:
    """
    Complete pipeline for hunting behavior classification.
    Runs all processing steps from raw data to behavioral predictions.
    """
    
    def __init__(self, config=None):
        """
        Initialize the pipeline.
        
        Args:
            config: PipelineSettings object. If None, uses global settings.
        """
        self.config = config or settings
        self.steps = {
            'keypoint_filter': {
                'processor': KeypointFilter,
                'description': 'Keypoint filtering with Savitzky-Golay smoothing',
                'required_inputs': ['pose_data_dir'],
                'outputs': ['savgol_pose_dir']
            },
            'head_angle': {
                'processor': HeadAngleProcessor,
                'description': 'Head angle calculation from pose keypoints',
                'required_inputs': ['savgol_pose_dir'],
                'outputs': ['head_angle_dir']
            },
            'kalman_filter': {
                'processor': KalmanAngleProcessor,
                'description': 'Kalman filtering and outlier removal',
                'required_inputs': ['head_angle_dir'],
                'outputs': ['kalman_filtered_dir']
            },
            'cricket_processor': {
                'processor': CricketProcessor,
                'description': 'Cricket detection validation and interpolation',
                'required_inputs': ['cricket_detection_dir', 'interval_txt_dir'],
                'outputs': ['cricket_processed_dir']
            },
            'feature_engineer': {
                'processor': FeatureEngineer,
                'description': 'Visual field analysis and feature engineering',
                'required_inputs': ['raw_video_dir', 'kalman_filtered_dir', 'cricket_processed_dir'],
                'outputs': ['final_videos_dir']
            },
            'label_processor': {
                'processor': LabelProcessor,
                'description': 'Convert behavior annotations to processed labels',
                'required_inputs': ['annotation_dir'],
                'outputs': ['behavior_labels_dir'],
                'optional': True
            },
            'action_classifier': {
                'processor': ActionClassifier,
                'description': 'XGBoost behavioral classification',
                'required_inputs': ['final_videos_dir', 'behavior_labels_dir'],
                'outputs': ['classification_results']
            }
        }
        
        self.results = {}
        self.start_time = None
        
    def validate_inputs(self, step_name: str) -> bool:
        """
        Validate that required input directories exist for a step.
        
        Args:
            step_name: Name of the processing step
            
        Returns:
            bool: True if all inputs exist, False otherwise
        """
        step_info = self.steps[step_name]
        
        # Special handling for optional label processor
        if step_name == 'label_processor':
            # Check if processed labels already exist
            behavior_labels_dir = getattr(self.config.paths, 'behavior_labels_dir')
            if Path(behavior_labels_dir).exists():
                processed_labels = list(Path(behavior_labels_dir).glob('*_processed_labels.csv'))
                if processed_labels:
                    print(f"  ✓ Found {len(processed_labels)} existing processed label files")
                    return 'skip'
            
            # Check if annotation directory exists
            annotation_dir = getattr(self.config.paths, 'annotation_dir')
            if not Path(annotation_dir).exists():
                print(f"  ⚠️  Annotation directory not found: {annotation_dir}")
                print(f"     Label processing will be skipped - classification may fail without labels")
                return 'skip'
        
        # Standard input validation for all other steps
        for input_dir in step_info['required_inputs']:
            dir_path = getattr(self.config.paths, input_dir)
            if not Path(dir_path).exists():
                print(f"  ✗ Missing input directory: {dir_path}")
                return 'invalid'
        
        return 'valid'
    
    def create_output_directories(self):
        """Create all output directories if they don't exist."""
        output_dirs = [
            self.config.paths.savgol_pose_dir,
            self.config.paths.head_angle_dir,
            self.config.paths.kalman_filtered_dir,
            self.config.paths.cricket_processed_dir,
            self.config.paths.final_videos_dir,
            self.config.paths.behavior_labels_dir,
            self.config.paths.visualization_dir
        ]
        
        for dir_path in output_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
        print("Created output directories")
    
    def run_step(self, step_name: str) -> Dict[str, Any]:
        """
        Run a single processing step.
        
        Args:
            step_name: Name of the step to run
            
        Returns:
            dict: Results from the processing step
        """
        if step_name not in self.steps:
            raise ValueError(f"Unknown step: {step_name}")
        
        step_info = self.steps[step_name]
        
        print(f"\n{'='*60}")
        print(f"STEP: {step_name.upper().replace('_', ' ')}")
        print(f"Description: {step_info['description']}")
        print(f"{'='*60}")
        
        # Validate inputs
        print("Validating inputs...")
        validation_result = self.validate_inputs(step_name)
        
        if validation_result == 'invalid':
            return {'success': False, 'error': 'Missing required inputs'}
        elif validation_result == 'skip':
            return {
                'success': True, 
                'skipped': True, 
                'processing_time': 0,
                'step_name': step_name,
                'reason': 'Step skipped (inputs not needed or outputs exist)'
            }
        
        print("  ✓ All inputs available")
        
        # Initialize processor
        processor_class = step_info['processor']
        
        try:
            step_start = time.time()
            
            # Run the processing step
            if step_name == 'keypoint_filter':
                processor = processor_class()
                result = processor.process()
                
            elif step_name == 'head_angle':
                processor = processor_class()
                result = processor.process()
                
            elif step_name == 'kalman_filter':
                processor = processor_class()
                result = processor.process()
                
            elif step_name == 'cricket_processor':
                processor = processor_class()
                result = processor.process()
                
            elif step_name == 'feature_engineer':
                processor = processor_class()
                result = processor.process()

            elif step_name == 'label_processor':
                processor = processor_class()
                result = processor.process()

            elif step_name == 'action_classifier':
                processor = processor_class()
                result = processor.process()
                
            step_time = time.time() - step_start
            
            # Add timing and success info
            result.update({
                'success': True,
                'processing_time': step_time,
                'step_name': step_name
            })
            
            print(f"\n✓ {step_name.replace('_', ' ').title()} completed successfully")
            print(f"  Processing time: {step_time:.1f} seconds")
            if result.get('skipped'):
                print(f"  Status: Skipped - {result.get('reason', 'No reason provided')}")
            else:
                if 'processed' in result:
                    print(f"  Files processed: {result['processed']}")
                if 'failed' in result:
                    print(f"  Files failed: {result['failed']}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error in {step_name}: {str(e)}"
            print(f"\n✗ {error_msg}")
            if self.config.enable_logging:
                traceback.print_exc()
            
            return {
                'success': False,
                'error': error_msg,
                'step_name': step_name,
                'processing_time': time.time() - step_start
            }
    
    def run_pipeline(self, steps: Optional[List[str]] = None, 
                    start_from: Optional[str] = None,
                    stop_at: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete pipeline or selected steps.
        
        Args:
            steps: List of specific steps to run. If None, runs all steps.
            start_from: Step name to start from (runs all subsequent steps)
            stop_at: Step name to stop at (inclusive)
            
        Returns:
            dict: Complete pipeline results
        """
        self.start_time = time.time()
        
        print("🎯 HUNTING BEHAVIOR CLASSIFICATION PIPELINE")
        print("=" * 60)
        print(f"Pipeline configuration:")
        print(f"  Raw video dir: {self.config.paths.raw_video_dir}")
        print(f"  Output base: {Path(self.config.paths.final_videos_dir).parent}")
        print(f"  Enable logging: {self.config.enable_logging}")
        
        # Create output directories
        self.create_output_directories()
        
        # Determine which steps to run
        all_steps = list(self.steps.keys())
        
        if steps is not None:
            # Run specific steps
            steps_to_run = [s for s in steps if s in all_steps]
        else:
            # Determine range of steps to run
            steps_to_run = all_steps
            
            if start_from:
                if start_from in all_steps:
                    start_idx = all_steps.index(start_from)
                    steps_to_run = all_steps[start_idx:]
                else:
                    raise ValueError(f"Unknown start step: {start_from}")
            
            if stop_at:
                if stop_at in all_steps:
                    stop_idx = all_steps.index(stop_at) + 1
                    if start_from:
                        start_idx = all_steps.index(start_from)
                        steps_to_run = all_steps[start_idx:stop_idx]
                    else:
                        steps_to_run = all_steps[:stop_idx]
                else:
                    raise ValueError(f"Unknown stop step: {stop_at}")
        
        print(f"\nRunning {len(steps_to_run)} steps: {' → '.join(steps_to_run)}")
        
        # Run each step
        pipeline_success = True
        total_files_processed = 0
        total_files_failed = 0
        
        for i, step_name in enumerate(steps_to_run, 1):
            print(f"\n[{i}/{len(steps_to_run)}] Starting {step_name}...")
            
            step_result = self.run_step(step_name)
            self.results[step_name] = step_result
            
            if not step_result['success']:
                print(f"\n❌ Pipeline failed at step: {step_name}")
                print(f"Error: {step_result['error']}")
                pipeline_success = False
                break
            
            # Accumulate statistics (skip skipped steps from counts)
            if not step_result.get('skipped', False):
                if 'processed' in step_result:
                    total_files_processed += step_result['processed']
                if 'failed' in step_result:
                    total_files_failed += step_result['failed']
        
        # Pipeline summary
        total_time = time.time() - self.start_time
        
        print(f"\n{'='*60}")
        if pipeline_success:
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        else:
            print("💥 PIPELINE FAILED!")
        print(f"{'='*60}")
        print(f"Total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Steps completed: {sum(1 for r in self.results.values() if r['success'])}/{len(steps_to_run)}")
        
        if total_files_processed > 0:
            print(f"Total files processed: {total_files_processed}")
        if total_files_failed > 0:
            print(f"Total files failed: {total_files_failed}")
        
        # Show step-by-step timing
        print(f"\nStep-by-step timing:")
        for step_name, result in self.results.items():
            status = "✓" if result['success'] else "✗"
            time_str = f"{result['processing_time']:.1f}s"
            print(f"  {status} {step_name.replace('_', ' ').title()}: {time_str}")
        
        # Final results for action classification
        if 'action_classifier' in self.results and self.results['action_classifier']['success']:
            classifier_results = self.results['action_classifier']
            if 'cv_score_mean' in classifier_results:
                print(f"\n🏆 FINAL CLASSIFICATION RESULTS:")
                print(f"  CV F1 Score: {classifier_results['cv_score_mean']:.4f} ± {classifier_results['cv_score_sem']:.4f}")
                print(f"  In-sample Accuracy: {classifier_results['classification_report']['accuracy']:.4f}")
                print(f"  Best XGBoost params: {classifier_results['best_params']}")
        
        return {
            'success': pipeline_success,
            'total_time': total_time,
            'steps_run': steps_to_run,
            'results': self.results,
            'files_processed': total_files_processed,
            'files_failed': total_files_failed
        }
    
    def run_individual_step(self, step_name: str) -> Dict[str, Any]:
        """
        Run a single step of the pipeline.
        
        Args:
            step_name: Name of the step to run
            
        Returns:
            dict: Step results
        """
        print(f"🎯 Running individual step: {step_name}")
        self.create_output_directories()
        return self.run_step(step_name)
    
    def list_steps(self):
        """Print information about all available pipeline steps."""
        print("📋 AVAILABLE PIPELINE STEPS:")
        print("=" * 60)
        
        for i, (step_name, step_info) in enumerate(self.steps.items(), 1):
            print(f"{i}. {step_name.replace('_', ' ').title()}")
            print(f"   Description: {step_info['description']}")
            print(f"   Inputs: {', '.join(step_info['required_inputs'])}")
            print(f"   Outputs: {', '.join(step_info['outputs'])}")
            print()

def main():
    """Command line interface for the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Hunting Behavior Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py                           # Run complete pipeline
  python pipeline.py --list-steps              # Show all available steps
  python pipeline.py --step keypoint_filter    # Run single step
  python pipeline.py --start-from cricket_processor  # Run from step onwards
  python pipeline.py --stop-at head_angle      # Run up to step (inclusive)
  python pipeline.py --steps keypoint_filter head_angle  # Run specific steps
        """
    )
    
    # Step selection arguments
    parser.add_argument("--step", help="Run a single step")
    parser.add_argument("--steps", nargs="+", help="Run specific steps")
    parser.add_argument("--start-from", help="Start from this step and run all subsequent")
    parser.add_argument("--stop-at", help="Stop at this step (inclusive)")
    parser.add_argument("--list-steps", action="store_true", help="List all available steps")
    
    # Configuration overrides
    parser.add_argument("--video-dir", help="Override raw video directory")
    parser.add_argument("--pose-dir", help="Override pose data directory")
    parser.add_argument("--cricket-dir", help="Override cricket detection directory")
    parser.add_argument("--output-dir", help="Override base output directory")
    parser.add_argument("--disable-logging", action="store_true", help="Disable error logging")
    
    args = parser.parse_args()
    
    # Override configuration if provided
    if args.video_dir:
        settings.paths.raw_video_dir = Path(args.video_dir)
    if args.pose_dir:
        settings.paths.pose_data_dir = Path(args.pose_dir)
    if args.cricket_dir:
        settings.paths.cricket_detection_dir = Path(args.cricket_dir)
    if args.output_dir:
        # Update all output directories to be relative to base output dir
        base_dir = Path(args.output_dir)
        settings.paths.savgol_pose_dir = base_dir / "savgol_pose"
        settings.paths.head_angle_dir = base_dir / "head_angle"
        settings.paths.kalman_filtered_dir = base_dir / "kalman_filtered"
        settings.paths.cricket_processed_dir = base_dir / "cricket_processed"
        settings.paths.final_videos_dir = base_dir / "final_videos"
        settings.paths.behavior_labels_dir = base_dir / "behavior_labels"
        settings.paths.visualization_dir = base_dir / "visualization"
    if args.disable_logging:
        settings.enable_logging = False
    
    # Create pipeline
    pipeline = HuntingClassificationPipeline()
    
    try:
        if args.list_steps:
            pipeline.list_steps()
        elif args.step:
            # Run single step
            result = pipeline.run_individual_step(args.step)
            if not result['success']:
                exit(1)
        else:
            # Run pipeline
            result = pipeline.run_pipeline(
                steps=args.steps,
                start_from=args.start_from,
                stop_at=args.stop_at
            )
            if not result['success']:
                exit(1)
                
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n💥 Pipeline failed with error: {e}")
        if settings.enable_logging:
            traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()