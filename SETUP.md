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
## 🎬 Video Preprocessing (Recommended)

### Why Downsample to 30fps?

If your videos are recorded at 60fps, **downsampling to 30fps is highly recommended** because:

✅ **Eliminates frame synchronization issues** between video, pose, and cricket data  
✅ **Reduces file sizes by 50%** (faster processing, less storage)  
✅ **30fps is sufficient** for behavioral analysis (no information loss)  
✅ **Prevents flickering** in visualization videos  
✅ **Simplifies the entire pipeline** (no frame number conversion logic needed)

### Downsample Videos Before Processing

```bash
cd refactoring_v1

# Downsample all videos in a directory to 30fps
python -m processing.video_downsampler --input-dir ../data/videos --output-dir ../data/videos_30fps --fps 30

# The downsampler will:
# - Skip videos already at 30fps (just copy them)
# - Downsample 60fps videos to 30fps
# - Warn about videos below 30fps and copy them as-is
```

**Recommended workflow:**
1. Downsample all raw videos to 30fps first
2. Use the downsampled videos for all subsequent processing steps
3. This avoids frame alignment issues throughout the pipeline

## 🐛 Cricket Processing: Two Processors Available

There are **two separate cricket processors** you can choose from. They are **different scripts in the same directory**, not a configuration toggle.

### Standard Processor (`cricket_processor.py`)

Located in `refactoring_v1/processing/cricket_processor.py`

**Features:**
- Full validation with reliability scoring
- Adaptive Kalman smoothing
- Multi-factor validation (confidence, size, movement)
- Nose position fallback after validation

**Use for:**
- Final analysis
- Publication-quality results
- When you need detailed validation metrics

**Run:**
```bash
cd refactoring_v1
python -m processing.cricket_processor --input-dir ../data/cricket_detection --interval-dir ../data/interval_txt --output-dir ../results/cricket_processed
```

### Simple Processor (`simple_cricket_processor.py`)

Located in `refactoring_v1/processing/simple_cricket_processor.py`

**Features:**
- Basic linear interpolation for gaps
- Savitzky-Golay smoothing
- Simple confidence filtering
- Immediate nose position fallback

**Use for:**
- Quick testing
- Initial data exploration
- When processing speed matters more than validation detail

**Run:**
```bash
cd refactoring_v1
python -m processing.simple_cricket_processor --input-dir ../data/cricket_detection --interval-dir ../data/interval_txt --output-dir ../results/cricket_simple
```

### Comparison

| Feature | Standard Processor | Simple Processor |
|---------|-------------------|------------------|
| **File** | `cricket_processor.py` | `simple_cricket_processor.py` |
| **Location** | `refactoring_v1/processing/` | `refactoring_v1/processing/` |
| **Speed** | Slower | Faster |
| **Validation** | Comprehensive | Basic |
| **Smoothing** | Adaptive Kalman | Savitzky-Golay |
| **Gap Filling** | Validated interpolation | Linear interpolation |
| **Output Columns** | More detailed | Standard only |

**Choose based on your needs** - run the appropriate script manually. The main pipeline (`refactoring_v1/pipeline.py`) integrates one or the other.

## 🧪 Test Run

### 1. Small Test Dataset

Start with a small subset of your data:
- 1-2 video files
- Corresponding pose and cricket data
- Behavior annotations

### 2-1. Downsample Videos (if needed)

```bash
cd refactoring_v1
python -m processing.video_downsampler --input-dir ../test_data/videos --output-dir ../test_data/videos_30fps --fps 30
```

### 2-2. Run Individual Steps

Test each step individually:

```bash
python run_pipeline.py filter
python run_pipeline.py angles
python run_pipeline.py kalman
python run_pipeline.py cricket
python run_pipeline.py cricket-simple
python run_pipeline.py features
python run_pipeline.py classify
```

### 3. Full Pipeline Test

```bash
python run_pipeline.py
```
Note: Check `refactoring_v1/pipeline.py` to see which cricket processor is currently integrated into the main pipeline.

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
- Verify framerate with: `python -c "import cv2; cap = cv2.VideoCapture('your_video.mp4'); print(f'FPS: {cap.get(cv2.CAP_PROP_FPS)}')"`

**2. CSV parsing errors:**
- Verify CSV files have the expected number of columns
- Check for encoding issues (use UTF-8)
- **For cricket CSVs**: Both 7-column and 8-column formats are supported (extra column auto-detected and skipped)

**3. Path issues:**
- Use absolute paths in configuration
- Ensure forward slashes (/) in paths on all platforms

### Frame Synchronization Issues

**Symptoms:**
- Cricket bounding boxes appear in wrong locations
- Flickering in output videos
- "Frame not found" warnings
- Cricket coordinates don't match actual position

**Solutions:**

**1. Downsample all videos to 30fps FIRST:**
```bash
cd refactoring_v1
python -m processing.video_downsampler --input-dir ../data/videos --output-dir ../data/videos_30fps --fps 30
```
Then use the `videos_30fps` directory for all processing.

**2. Verify frame numbers match across files:**
```bash
# Check cricket CSV frame numbers
head -20 data/cricket_detection/your_file_Cricket.csv

# Check pose CSV frame numbers  
head -20 data/pose_data/your_file_pose.csv

# They should have matching frame numbers
```

**3. Check if cricket processor kept original frame numbers:**
- After running cricket processor, verify the output CSV has the same frame numbers as input
- The processors should NOT divide frame numbers by 2 (this bug was fixed)

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

**7-column format:**
```
frame,trackID,x,y,w,h,confidence
0,0,200,300,20,15,0.8
```
**8-column format (also supported):**
```
frame,trackID,unknown,x,y,w,h,confidence
1312,0,0,979.67,123.31,58.31,39.02,0.91
1314,0,0,983.88,120.96,58.17,39.29,0.91
```

*Both processors automatically detect the format and handle column alignment.*

### Timing Text Files
Expected format:
```
cricket in 부터(1234)
5678    consume
9012    consume
```

## 🚀 Recommended Workflow

### For New Datasets

1. **Downsample all videos to 30fps first**
   ```bash
   cd refactoring_v1
   python -m processing.video_downsampler --input-dir ../data/raw_videos --output-dir ../data/videos_30fps --fps 30
   ```

2. **Quick test with simple cricket processor**
   ```bash
   cd refactoring_v1
   # Use simple processor for fast initial check
   python -m processing.simple_cricket_processor \
       --input-dir ../data/cricket_detection \
       --interval-dir ../data/interval_txt \
       --output-dir ../results/cricket_test
   ```

3. **Run a test with feature engineering**
   ```bash
   cd refactoring_v1
   python run_pipeline.py features
   # Check output video for correct cricket visualization
   ```

4. **If needed, reprocess with standard cricket processor**
   ```bash
   cd refactoring_v1
   python -m processing.cricket_processor \
       --input-dir ../data/cricket_detection \
       --interval-dir ../data/interval_txt \
       --output-dir ../results/cricket_final
   ```

5. **Full pipeline run**
   ```bash
   cd refactoring_v1
   python run_pipeline.py
   ```

### Checklist Before Production Run

- ✅ All videos downsampled to 30fps
- ✅ Cricket processor chosen (standard or simple)
- ✅ Test run completed successfully
- ✅ Output videos reviewed for correct visualization
- ✅ All paths in `refactoring_v1/config/settings.py` configured correctly
- ✅ Sufficient disk space for outputs

## 📞 Getting Help

If you encounter issues:

1. Check this setup guide
2. Review the main README.md
3. Run diagnostic commands:
   ```bash
   python -c "import cv2, pandas, numpy, sklearn, xgboost; print('All imports successful!')"
   ```
4. Verify video framerate:
   ```bash
   python -c "import cv2; cap = cv2.VideoCapture('your_video.mp4'); print(f'FPS: {cap.get(cv2.CAP_PROP_FPS)}')"
   ```
5. Check cricket CSV format:
   ```bash
   head -5 your_cricket_file.csv
   # Should have 7 or 8 columns
   ```
6. Open an issue with:
   - Error messages (full traceback)
   - System information (OS, Python version)
   - Video framerate (`cv2.CAP_PROP_FPS`)
   - Cricket CSV column count
   - Which cricket processor you used

## 💡 Tips for Best Results

- **Always downsample 60fps videos to 30fps first** - this prevents 90% of synchronization issues
- **Use simple processor for initial testing** to quickly identify data issues
- **Check output videos** before running full analysis - verify cricket boxes appear in correct positions
- **Keep original frame numbers** - the processors no longer divide frame numbers
- **Monitor disk space** - downsampled videos + outputs can be large
- **Process in batches** if you have many files to avoid memory issues
- **Keep backups** of raw data before processing
- **Work in refactoring_v1 directory** - this is the current active pipeline