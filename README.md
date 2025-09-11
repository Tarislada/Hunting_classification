# Hunting Behavior Classification Pipeline

A complete, automated pipeline for analyzing mouse hunting behaviors from raw video data to behavioral classifications using computer vision and machine learning.

## 🎯 Overview

This pipeline processes mouse hunting videos through multiple stages to automatically classify hunting behaviors such as chasing, attacking, and non-visual rotation. It combines pose estimation, cricket tracking, visual field analysis, and machine learning to provide robust behavioral classification.

## 🔧 Pipeline Architecture

The pipeline consists of 6 sequential processing steps:

```
Raw Videos + Pose Data → Behavioral Classifications
     ↓
1. Keypoint Filtering (Savitzky-Golay smoothing)
     ↓
2. Head Angle Calculation (relative to body axis)
     ↓  
3. Kalman Filtering (outlier removal and smoothing)
     ↓
4. Cricket Processing (validation and interpolation)
     ↓
5. Feature Engineering (visual field analysis)
     ↓
6. Action Classification (XGBoost ML model)
```

### Processing Steps

| Step | Description | Input | Output |
|------|-------------|-------|--------|
| **1. Keypoint Filter** | Applies Savitzky-Golay filtering to smooth pose keypoints and handle low-confidence detections | Raw pose CSV files | Filtered pose data |
| **2. Head Angle** | Calculates head angles relative to body axis and angular velocity | Filtered pose data | Head angle measurements |
| **3. Kalman Filter** | Applies Kalman filtering to smooth head angles and remove statistical outliers | Head angle data | Smoothed angles |
| **4. Cricket Processor** | Validates cricket detections, handles gaps, and applies adaptive smoothing | Cricket tracking CSV + timing txt | Processed cricket data |
| **5. Feature Engineer** | Combines pose and cricket data to create visual field features and validation videos | Videos + pose + cricket data | Analysis features + videos |
| **6. Action Classifier** | Uses XGBoost to classify hunting behaviors with hyperparameter optimization | Feature data + behavior labels | Behavioral predictions |

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd hunting_classification
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data structure:**
   ```
   your_data/
   ├── videos/              # Raw video files (.mp4, .avi, .mov)
   ├── pose_data/           # Pose estimation CSV files  
   ├── cricket_detection/   # Cricket tracking CSV files
   ├── interval_txt/        # Cricket timing text files
   └── annotations/         # Behavior annotation files
   ```

### Basic Usage

**Run the complete pipeline:**
```bash
python run_pipeline.py
```

**Check if your data is properly organized:**
```bash
python run_pipeline.py check
```

**Run with custom directories:**
```bash
python run_pipeline.py --video-dir /path/to/videos --output-dir /path/to/results
```

## 📋 Detailed Usage

### Simple Commands

```bash
# Run complete pipeline
python run_pipeline.py

# List all available steps  
python run_pipeline.py steps

# Run individual steps
python run_pipeline.py filter    # Keypoint filtering only
python run_pipeline.py angles    # Head angle calculation only
python run_pipeline.py kalman    # Kalman filtering only
python run_pipeline.py cricket   # Cricket processing only
python run_pipeline.py features  # Feature engineering only
python run_pipeline.py classify  # Action classification only

# Check input directories
python run_pipeline.py check
```

### Advanced Usage

```bash
# Run from a specific step onwards
python pipeline.py --start-from cricket_processor

# Run up to a specific step
python pipeline.py --stop-at head_angle

# Run specific steps only
python pipeline.py --steps keypoint_filter kalman_filter

# Run single step with custom parameters
python pipeline.py --step keypoint_filter --window-length 71 --polyorder 5

# Custom directories for complete pipeline
python pipeline.py --video-dir /data/videos --pose-dir /data/pose --output-dir /results
```

## ⚙️ Configuration

All parameters are centralized in `config/settings.py`. You can easily modify:

### Directory Paths
```python
# Input directories
raw_video_dir = "data/videos"
pose_data_dir = "data/pose"  
cricket_detection_dir = "data/cricket"
interval_txt_dir = "data/intervals"

# Output directories  
final_videos_dir = "results/videos"
behavior_labels_dir = "results/labels"
```

### Processing Parameters
```python
# Keypoint filtering
window_length = 59
polyorder = 7
confidence_threshold = 0.6

# Kalman filtering
process_variance = 1e-5
measurement_variance = 1e-2

# Cricket validation
reliability_threshold = 0.65
max_gap_threshold = 30

# Visual field analysis
binocular_threshold = 30.0  # degrees
monocular_threshold = 140.0 # degrees

# XGBoost classification
cv_folds = 3
class_weights = {
    'attack': 1.2,
    'chasing': 0.94, 
    'non_visual_rotation': 1.0,
    'background': 1.0
}
```

## 📊 Expected Input Data

### 1. Video Files
- **Format:** .mp4, .avi, .mov
- **Content:** Mouse hunting behavior recordings
- **Naming:** Consistent naming scheme (e.g., `m17_t1.mp4`)

### 2. Pose Data
- **Format:** CSV files with pose keypoint coordinates
- **Columns:** 40 columns including frame, animal ID, bounding box, keypoints (x,y), confidence scores
- **Keypoints:** nose, ears, limbs, tail segments, body center

### 3. Cricket Detection
- **Format:** CSV files with cricket bounding boxes
- **Columns:** frame, trackID, x, y, w, h, confidence
- **Content:** Cricket position tracking throughout videos

### 4. Timing Files
- **Format:** Text files with cricket interaction timing
- **Content:** Cricket entry/exit frames and consumption events
- **Example:**
  ```
  cricket in 부터(1234)
  5678    consume
  ```

### 5. Behavior Annotations
- **Format:** CSV/Excel files with behavior labels
- **Columns:** approaching_start, approaching_end, turning_start, turning_end, attack_start, attack_end
- **Content:** Manual annotations of hunting behaviors

## 📈 Output

### Generated Files
- **Validation Videos:** Videos with visual overlays showing angles, zones, and tracking
- **Feature CSV Files:** Analysis features for each frame (`*_analysis.csv`)  
- **Classification Results:** Behavioral predictions with confidence scores
- **Visualization Plots:** Confusion matrices, sequence comparisons, feature importance

### Analysis Features
- Head angles and angular velocity
- Cricket angles relative to head direction  
- Visual field classification (binocular/monocular/out-of-sight)
- Distance and speed measurements
- Temporal features and behavioral transitions

## 🔧 Troubleshooting

### Common Issues

**1. "Missing input directory" error:**
- Check that your data directories exist
- Use `python run_pipeline.py check` to verify paths
- Override paths with `--video-dir`, `--pose-dir`, etc.

**2. "No files found" error:**
- Verify file naming conventions match expectations
- Check that CSV files have the expected column structure
- Ensure video and CSV files have matching base names

**3. Memory issues:**
- Reduce `batch_size` in action segmentation settings
- Process fewer files at once
- Monitor memory usage during processing

**4. Low classification performance:**
- Check behavior annotation quality and consistency
- Adjust class weights in configuration
- Verify that cricket timing files are accurate

### Performance Tips

- **Memory:** The pipeline is optimized for memory efficiency but large datasets may require batch processing
- **Speed:** Feature engineering is the most time-intensive step due to video processing
- **Accuracy:** Classification performance depends heavily on annotation quality and cricket tracking accuracy

## 🧪 Development

### Adding New Processing Steps

1. Create a new processor class in `processing/`
2. Follow the standard interface pattern:
   ```python
   class NewProcessor:
       def __init__(self, config=None):
           self.config = config or settings.new_processor
       
       def process_file(self, input_path, output_path):
           # Process single file
           pass
           
       def process_directory(self, input_dir, output_dir):
           # Process all files in directory
           pass
           
       def process(self, input_dir=None, output_dir=None):
           # Main method using config defaults
           pass
   ```
3. Add configuration parameters to `config/settings.py`
4. Register the step in `pipeline.py`

### Testing Individual Components

```bash
# Test specific processors
python -m processing.keypoint_filter --input-dir data/pose --output-dir test_output
python -m processing.head_angle --input-dir test_output --output-dir test_output2

# Test with custom parameters
python -m processing.kalman_filter --process-variance 1e-4 --measurement-variance 1e-3
```

## 📝 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{hunting_classification_pipeline,
  title={Hunting Behavior Classification Pipeline},
  author={[Your Name]},
  year={2024},
  url={[repository-url]}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the existing code style and patterns
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

[Your chosen license]

## 🆘 Support

For questions and support:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review the configuration documentation

---

**Note:** This pipeline is designed for research use. Ensure you have appropriate permissions for any data you process and follow your institution's guidelines for animal research data.