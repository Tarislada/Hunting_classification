# Hunting_classification
KH hunting project

## Order of scripts
Scripts are in a chain; run in the order specified for now.

0. (raw video, txt file with cricket enter/consume timing) -> pose csv files (using Pose detection model), cricket csv file
1. keypoint_filter.py -> filters keypoints initially using Savitzky-Golay filter
2. bulk_head_angle.py -> calculates head angles
3. kalman_angle_correction -> filters head angles using kalman filter
4. cricket_interp.py -> looks for valid cricket and filter.
5. angle_val.py -> creates feature files, videos with filtered angle and keypoint
6. action_segmentation -> action segmentation using XGboost.
