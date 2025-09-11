# Setup Guide

This guide will help you set up the Hunting Behavior Classification Pipeline on your system.

## 🔧 System Requirements

- **Python:** 3.8 or higher
- **RAM:** Minimum 8GB, recommended 16GB+ for large datasets
- **Storage:** Varies based on data size (videos can be large)
- **OS:** Windows, macOS, or Linux

## 📦 Installation Steps

### 1. Clone or Download

```bash
git clone <repository-url>
cd hunting_classification
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using conda
conda create -n hunting python=3.9
conda activate hunting

# OR using venv
python -m venv hunting_env
# On Windows:
hunting_env\Scripts\activate
# On macOS/Linux:
source hunting_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python run_pipeline.py steps
```

If this command shows the pipeline steps without errors, your installation is successful!

## 📁 Data Organization

Create your data directory structure:

```
your_project/
├── hunting_classification/     # This repository
│   ├── config/
│   ├── processing/
│   ├── pipeline.py
│   └── run_pipeline.py
└── data/                      # Your data (create this)
    ├── videos/                # Raw video files
    ├── pose_data/             # Pose estimation CSV files
    ├── cricket_detection/     # Cricket tracking CSV files
    ├── interval_txt/          # Cricket timing text files
    └── annotations/           # Behavior annotation files
```

## ⚙️ Configuration

### 1. Update Paths

Edit `config/settings.py` to point to your data directories:

```python
@dataclass
class DirectoryPaths:
    # Update these paths to match your data location
    raw_video_dir: str = "/path/to/your/data/videos"
    pose_data_dir: str = "/path/to/your/data/pose_data"
    cricket_detection_dir: str = "/path/to/your/data/cricket_detection"
    interval_txt_dir: str = "/path/to/your/data/interval_txt"
    annotation_dir: str = "/path/to/your/data/annotations"
    
    # Output directories (will be created automatically)
    final_videos_dir: str = "/path/to/your/results/final_videos"
    # ... other output paths
```

### 2. Test Configuration

```bash
python run_pipeline.py check
```

This will verify that all your input directories exist and contain the expected files.

## 🧪 Test Run

### 1. Small Test Dataset

Start with a small subset of your data:
- 1-2 video files
- Corresponding pose and cricket data
- Behavior annotations

### 2. Run Individual Steps

Test each step individually:

```bash
python run_pipeline.py filter
python run_pipeline.py angles
python run_pipeline.py kalman
python run_pipeline.py cricket
python run_pipeline.py features
python run_pipeline.py classify
```

### 3. Full Pipeline Test

```bash
python run_pipeline.py
```

## 🔍 Troubleshooting

### Common Installation Issues

**1. OpenCV installation problems:**
```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

**2. XGBoost compilation issues:**
```bash
# Try installing from conda-forge
conda install -c conda-forge xgboost
```

**3. FilterPy not found:**
```bash
pip install filterpy
```

**4. Memory issues during installation:**
```bash
pip install --no-cache-dir -r requirements.txt
```

### File Format Issues

**1. Video files not recognized:**
- Ensure videos are in supported formats (.mp4, .avi, .mov)
- Check file permissions

**2. CSV parsing errors:**
- Verify CSV files have the expected number of columns
- Check for encoding issues (use UTF-8)

**3. Path issues:**
- Use absolute paths in configuration
- Ensure forward slashes (/) in paths on all platforms

### Performance Issues

**1. Slow processing:**
- Reduce batch size in action segmentation
- Process smaller video subsets
- Use SSD storage for better I/O performance

**2. Memory errors:**
- Reduce `window_sizes` in feature engineering
- Process files in smaller batches
- Monitor memory usage with task manager

## 📊 Data Format Validation

### Video Files
- **Format:** .mp4, .avi, .mov
- **Codec:** H.264 recommended
- **Resolution:** Any (will be detected automatically)

### Pose Data CSV
Expected format:
```
frame,id,box_x,box_y,box_w,box_h,box_conf,nose_x,nose_y,...
0,0,100,150,50,80,0.9,125,165,...
```

### Cricket Detection CSV
Expected format:
```
frame,trackID,x,y,w,h,confidence
0,0,200,300,20,15,0.8
```

### Timing Text Files
Expected format:
```
cricket in 부터(1234)
5678    consume
9012    consume
```

## 🚀 Ready to Go!

Once setup is complete:

1. ✅ Dependencies installed
2. ✅ Data organized
3. ✅ Paths configured  
4. ✅ Test run successful

You're ready to process your full dataset!

```bash
python run_pipeline.py
```

## 📞 Getting Help

If you encounter issues:

1. Check this setup guide
2. Review the main README.md
3. Run diagnostic commands:
   ```bash
   python run_pipeline.py check
   python -c "import cv2, pandas, numpy, sklearn, xgboost; print('All imports successful!')"
   ```
4. Open an issue with error messages and system information