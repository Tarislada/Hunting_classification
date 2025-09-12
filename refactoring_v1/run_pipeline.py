#!/usr/bin/env python3
"""
Simple CLI runner for the hunting classification pipeline.
Provides easy commands for common pipeline operations.
"""

import sys
from pathlib import Path
from pipeline import HuntingClassificationPipeline
from config.settings import settings

def print_banner():
    """Print a nice banner."""
    banner = """
🎯 HUNTING BEHAVIOR CLASSIFICATION PIPELINE
============================================
A complete pipeline for analyzing mouse hunting behaviors
from raw video data to behavioral classifications.
    """
    print(banner)

def show_help():
    """Show help information."""
    help_text = """
USAGE:
    python run_pipeline.py [COMMAND] [OPTIONS]

COMMANDS:
    run               Run the complete pipeline (default)
    steps             List all available pipeline steps
    check             Check if all required input directories exist
    
    # Individual steps:
    filter            Run keypoint filtering only
    angles            Run head angle calculation only  
    kalman            Run Kalman filtering only
    cricket           Run cricket processing only
    features          Run feature engineering only
    classify          Run action classification only

OPTIONS:
    --video-dir PATH      Override raw video directory
    --pose-dir PATH       Override pose data directory  
    --cricket-dir PATH    Override cricket detection directory
    --output-dir PATH     Override base output directory
    --help, -h            Show this help message

EXAMPLES:
    python run_pipeline.py                    # Run complete pipeline
    python run_pipeline.py run                # Same as above
    python run_pipeline.py steps              # List all steps
    python run_pipeline.py check              # Check inputs
    python run_pipeline.py filter             # Run only keypoint filtering
    python run_pipeline.py labels             # Run only label processing
    python run_pipeline.py features           # Run only feature engineering
    python run_pipeline.py --output-dir ./results  # Custom output directory

PIPELINE STEPS:
    1. filter    - Keypoint filtering with Savitzky-Golay smoothing
    2. angles    - Head angle calculation from pose keypoints
    3. kalman    - Kalman filtering and outlier removal
    4. cricket   - Cricket detection validation and interpolation
    5. features  - Visual field analysis and feature engineering
    6. classify  - XGBoost behavioral classification

For advanced usage, use pipeline.py directly.
    """
    print(help_text)

def check_inputs():
    """Check if all required input directories exist."""
    print("🔍 CHECKING INPUT DIRECTORIES")
    print("=" * 40)
    
    required_dirs = {
        'Raw videos': settings.paths.raw_video_dir,
        'Pose data': settings.paths.pose_data_dir,
        'Cricket detection': settings.paths.cricket_detection_dir,
        'Interval txt files': settings.paths.interval_txt_dir,
        'Behavior annotations': settings.paths.annotation_dir
    }
    
    all_exist = True
    for name, dir_path in required_dirs.items():
        exists = Path(dir_path).exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {dir_path}")
        if not exists:
            all_exist = False
    
    print()
    if all_exist:
        print("🎉 All required input directories exist!")
    else:
        print("⚠️  Some input directories are missing.")
        print("   Make sure your data is in the correct locations")
        print("   or use --video-dir, --pose-dir, etc. to override paths")
    
    return all_exist

def run_command(command, **kwargs):
    """Run a pipeline command."""
    pipeline = HuntingClassificationPipeline()
    
    # Step name mapping
    step_mapping = {
        'filter': 'keypoint_filter',
        'angles': 'head_angle', 
        'kalman': 'kalman_filter',
        'cricket': 'cricket_processor',
        'features': 'feature_engineer',
        'classify': 'action_classifier'
    }
    
    try:
        if command == 'run':
            print("🚀 Running complete pipeline...")
            result = pipeline.run_pipeline()
            return result['success']
            
        elif command == 'steps':
            pipeline.list_steps()
            return True
            
        elif command == 'check':
            return check_inputs()
            
        elif command in step_mapping:
            step_name = step_mapping[command]
            print(f"🚀 Running step: {command}")
            result = pipeline.run_individual_step(step_name)
            return result['success']
            
        else:
            print(f"❌ Unknown command: {command}")
            print("Use 'python run_pipeline.py --help' for usage information")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return False
    except Exception as e:
        print(f"💥 Error: {e}")
        return False

def parse_args():
    """Simple argument parsing."""
    args = sys.argv[1:]
    
    # Extract options
    options = {}
    filtered_args = []
    
    i = 0
    while i < len(args):
        arg = args[i]
        
        if arg in ['--help', '-h']:
            show_help()
            sys.exit(0)
        elif arg == '--video-dir' and i + 1 < len(args):
            options['video_dir'] = args[i + 1]
            i += 2
        elif arg == '--pose-dir' and i + 1 < len(args):
            options['pose_dir'] = args[i + 1]
            i += 2
        elif arg == '--cricket-dir' and i + 1 < len(args):
            options['cricket_dir'] = args[i + 1]
            i += 2
        elif arg == '--output-dir' and i + 1 < len(args):
            options['output_dir'] = args[i + 1]
            i += 2
        else:
            filtered_args.append(arg)
            i += 1
    
    # Get command (default to 'run')
    command = filtered_args[0] if filtered_args else 'run'
    
    return command, options

def main():
    """Main entry point."""
    print_banner()
    
    command, options = parse_args()
    
    # Apply configuration overrides
    if 'video_dir' in options:
        settings.paths.raw_video_dir = Path(options['video_dir'])
        print(f"📁 Using video directory: {options['video_dir']}")
    
    if 'pose_dir' in options:
        settings.paths.pose_data_dir = Path(options['pose_dir'])
        print(f"📁 Using pose directory: {options['pose_dir']}")
    
    if 'cricket_dir' in options:
        settings.paths.cricket_detection_dir = Path(options['cricket_dir'])
        print(f"📁 Using cricket directory: {options['cricket_dir']}")
    
    if 'output_dir' in options:
        base_dir = Path(options['output_dir'])
        settings.paths.savgol_pose_dir = base_dir / "savgol_pose"
        settings.paths.head_angle_dir = base_dir / "head_angle"
        settings.paths.kalman_filtered_dir = base_dir / "kalman_filtered"
        settings.paths.cricket_processed_dir = base_dir / "cricket_processed"
        settings.paths.final_videos_dir = base_dir / "final_videos"
        settings.paths.behavior_labels_dir = base_dir / "behavior_labels"
        settings.paths.visualization_dir = base_dir / "visualization"
        print(f"📁 Using output directory: {options['output_dir']}")
    
    # Show current configuration
    print(f"\n📂 Current configuration:")
    print(f"   Raw videos: {settings.paths.raw_video_dir}")
    print(f"   Pose data: {settings.paths.pose_data_dir}")
    print(f"   Cricket detection: {settings.paths.cricket_detection_dir}")
    print(f"   Output base: {Path(settings.paths.final_videos_dir).parent}")
    
    # Run the command
    success = run_command(command, **options)
    
    if success:
        print("\n🎉 Operation completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Operation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()