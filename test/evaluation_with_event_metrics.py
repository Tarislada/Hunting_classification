import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

class EventMetrics:
    """Calculate event-level metrics for action segmentation."""
    
    @staticmethod
    def extract_events(predictions: np.ndarray, behavior_id: int) -> List[Tuple[int, int]]:
        """Extract continuous events of a specific behavior."""
        events = []
        in_event = False
        start = None
        
        for i, pred in enumerate(predictions):
            if pred == behavior_id and not in_event:
                in_event = True
                start = i
            elif pred != behavior_id and in_event:
                in_event = False
                events.append((start, i - 1))
        
        if in_event:
            events.append((start, len(predictions) - 1))
            
        return events
    
    @staticmethod
    def calculate_iou(event1: Tuple[int, int], event2: Tuple[int, int]) -> float:
        """Calculate IoU between two temporal events."""
        intersection_start = max(event1[0], event2[0])
        intersection_end = min(event1[1], event2[1])
        
        if intersection_start <= intersection_end:
            intersection = intersection_end - intersection_start + 1
        else:
            intersection = 0
            
        union = (event1[1] - event1[0] + 1) + (event2[1] - event2[0] + 1) - intersection
        
        return intersection / union if union > 0 else 0
    
    @staticmethod
    def calculate_event_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                               behavior_id: int, 
                               iou_thresholds: List[float] = [0.1, 0.25, 0.5, 0.75]) -> Dict:
        """
        Calculate precision, recall, F1, and mAP for event detection.
        """
        true_events = EventMetrics.extract_events(y_true, behavior_id)
        pred_events = EventMetrics.extract_events(y_pred, behavior_id)
        
        if len(true_events) == 0:
            return {
                'mAP': 0.0,
                'avg_f1': 0.0,
                'n_true_events': 0,
                'n_pred_events': len(pred_events),
                'n_true_frames': 0,
                'details': {}
            }
        
        # Count total frames for this behavior
        n_true_frames = np.sum(y_true == behavior_id)
        
        metrics_by_threshold = {}
        f1_scores = []
        
        for iou_thresh in iou_thresholds:
            true_positives = 0
            matched_true = set()
            
            for pred_event in pred_events:
                best_iou = 0
                best_match = None
                
                for i, true_event in enumerate(true_events):
                    if i not in matched_true:
                        iou = EventMetrics.calculate_iou(pred_event, true_event)
                        if iou > best_iou:
                            best_iou = iou
                            best_match = i
                
                if best_iou >= iou_thresh and best_match is not None:
                    true_positives += 1
                    matched_true.add(best_match)
            
            precision = true_positives / len(pred_events) if len(pred_events) > 0 else 0
            recall = true_positives / len(true_events)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics_by_threshold[f'IoU_{iou_thresh:.2f}'] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': true_positives,
                'fp': len(pred_events) - true_positives,
                'fn': len(true_events) - true_positives
            }
            f1_scores.append(f1)
        
        return {
            'mAP': np.mean(f1_scores),
            'avg_f1': np.mean(f1_scores),
            'n_true_events': len(true_events),
            'n_pred_events': len(pred_events),
            'n_true_frames': int(n_true_frames),
            'details': metrics_by_threshold
        }

def evaluate_with_event_metrics(y_true, y_pred, target_encoder, groups=None):
    """
    Comprehensive evaluation with both frame-level and event-level metrics.
    """
    class_names = target_encoder.classes_
    
    # Frame-level metrics
    print("\n" + "="*60)
    print("FRAME-LEVEL METRICS")
    print("="*60)
    
    frame_report = classification_report(y_true, y_pred, 
                                        target_names=class_names,
                                        output_dict=True,
                                        zero_division=0)
    
    print(classification_report(y_true, y_pred, 
                               target_names=class_names,
                               zero_division=0))
    
    # Event-level metrics
    print("\n" + "="*60)
    print("EVENT-LEVEL METRICS")
    print("="*60)
    
    event_results = {}
    for behavior in class_names:
        if behavior != 'background':  # Skip background for event metrics
            behavior_id = target_encoder.transform([behavior])[0]
            event_metrics = EventMetrics.calculate_event_metrics(
                y_true, y_pred, behavior_id
            )
            event_results[behavior] = event_metrics
            
            print(f"\n{behavior.upper()}:")
            print(f"  True events: {event_metrics['n_true_events']}")
            print(f"  Predicted events: {event_metrics['n_pred_events']}")
            print(f"  Total frames: {event_metrics['n_true_frames']}")
            print(f"  Event mAP: {event_metrics['mAP']:.3f}")
            
            # Show performance at different IoU thresholds
            print(f"  Performance by IoU threshold:")
            for thresh_name, metrics in event_metrics['details'].items():
                iou_val = thresh_name.split('_')[1]
                print(f"    IoU≥{iou_val}: P={metrics['precision']:.2f}, "
                      f"R={metrics['recall']:.2f}, F1={metrics['f1']:.2f}")
    
    # Combined summary
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    
    print(f"\n{'Behavior':<20} {'Frame F1':>10} {'Event mAP':>10} {'Improvement':>12}")
    print("-" * 52)
    
    for behavior in class_names:
        if behavior != 'background':
            frame_f1 = frame_report[behavior]['f1-score']
            event_map = event_results[behavior]['mAP']
            improvement = event_map - frame_f1
            
            print(f"{behavior:<20} {frame_f1:>10.3f} {event_map:>10.3f} "
                  f"{'+' if improvement > 0 else ''}{improvement:>11.3f}")
    
    # Overall scores
    frame_macro_f1 = frame_report['macro avg']['f1-score']
    event_macro_map = np.mean([r['mAP'] for r in event_results.values()])
    
    print("-" * 52)
    print(f"{'Macro Average':<20} {frame_macro_f1:>10.3f} {event_macro_map:>10.3f} "
          f"{'+' if event_macro_map > frame_macro_f1 else ''}"
          f"{event_macro_map - frame_macro_f1:>11.3f}")
    
    return {
        'frame_metrics': frame_report,
        'event_metrics': event_results,
        'summary': {
            'frame_macro_f1': frame_macro_f1,
            'event_macro_map': event_macro_map,
            'improvement': event_macro_map - frame_macro_f1
        }
    }

def run_experiment(feature_files: List[str], label_files: List[str], 
                  experiment_name: str = "Experiment",
                  use_smote: bool = True,
                  hyperparam_search: bool = True):
    """
    Run experiment with proper CV evaluation, SMOTE, and hyperparameter search.
    Reports ONLY cross-validation metrics, not in-sample.
    """
    from sklearn.model_selection import RandomizedSearchCV
    from imblearn.over_sampling import SMOTE
    from collections import defaultdict
    import re
    
    print(f"\n{'='*60}")
    print(f"{experiment_name}")
    print('='*60)
    
    # Load and combine all data
    all_data = []
    for feat_file, label_file in zip(feature_files, label_files):
        try:
            match = re.search(r'(m\d+)', Path(feat_file).stem)
            if not match:
                print(f"Warning: Could not find animal ID in {Path(feat_file).stem}. Skipping file.")
                continue
            animal_id = match.group(1)

            features = pd.read_csv(feat_file)
            labels = pd.read_csv(label_file)
            
            merged = pd.merge(features, labels, on='frame', how='inner')
            merged['animal_id'] = animal_id
            all_data.append(merged)
            
            print(f"Loaded {animal_id}: {len(merged)} frames")
            
        except Exception as e:
            print(f"Error loading files: {e}")
            continue
    
    if not all_data:
        print("No data loaded!")
        return None
    
    data = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal dataset: {len(data)} frames from {len(all_data)} files")
    
    # Print class distribution
    print("\nClass distribution:")
    for behavior, count in data['behavior'].value_counts().items():
        print(f"  {behavior}: {count} ({count/len(data)*100:.1f}%)")
    
    # Prepare features and labels
    exclude_cols = ['frame', 'behavior', 'animal_id', 'cricket_status', 'validation', 
                   'zone', 'tail_base', 'body_center', 'nose']
    feature_cols = [col for col in data.columns if col not in exclude_cols and not col.startswith('Unnamed')]
    
    # Handle missing values
    data[feature_cols] = data[feature_cols].fillna(data[feature_cols].median())
    
    X = data[feature_cols].values
    
    # Encode labels
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(data['behavior'])
    groups = data['animal_id'].values
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Using {len(feature_cols)} features")
    print(f"SMOTE: {'Enabled' if use_smote else 'Disabled'}")
    print(f"Hyperparameter search: {'Enabled' if hyperparam_search else 'Disabled'}")
    
    # Cross-validation setup
    n_animals = len(np.unique(groups))
    print(f"\nFound {n_animals} unique animals. Using LeaveOneGroupOut for cross-validation.")
    cv = GroupKFold(n_splits=4)
    # cv = LeaveOneGroupOut()
    cv_splits = list(cv.split(X, y, groups))
    
    # Storage for CV results
    frame_scores = []
    event_scores_by_behavior = defaultdict(list)
    all_event_maps = []
    detailed_fold_results = []
    all_y_test_agg = []
    all_y_pred_agg = []
    
    print(f"\nRunning 4-fold cross-validation...")
    # print(f"\nRunning {len(cv_splits)}-fold Leave-One-Group-Out cross-validation...")
    
    for fold, (train_idx, test_idx) in enumerate(cv_splits):
        print(f"\n{'='*40}")
        print(f"Fold {fold + 1}/4")
        # print(f"Fold {fold + 1}/{n_animals}")
        print('='*40)
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Get test group for reporting
        test_groups = groups[test_idx]
        unique_test_animals = np.unique(test_groups)
        print(f"Testing on: {', '.join(unique_test_animals)}")
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Apply SMOTE if requested (only on training data!)
        if use_smote:
            print(f"Applying SMOTE to training data...")
            # 1. Get current sample counts
            original_counts = pd.Series(y_train).value_counts().to_dict()
            print(f"Original counts: {original_counts}")
            
            # --- MODIFIED: Update SMOTE strategy for new labels ---
            # 2. Define your target counts for the new classes
            # Get the integer labels for your classes
            sampling_strategy = {}
            
            # Safely get labels, only if they exist in the data
            if 'attack' in target_encoder.classes_:
                attack_label = target_encoder.transform(['attack'])[0]
                sampling_strategy[attack_label] = int(original_counts.get(attack_label, 0) * 1.0) # Increase attack
            
            if 'chasing' in target_encoder.classes_:
                chasing_label = target_encoder.transform(['chasing'])[0]
                sampling_strategy[chasing_label] = int(original_counts.get(chasing_label, 0) * 1.05) # Increase chasing
            
            if 'consume' in target_encoder.classes_:
                consume_label = target_encoder.transform(['consume'])[0]
                sampling_strategy[consume_label] = int(original_counts.get(consume_label, 0) * 1.15) # Increase consume
            
            # SMOTE will keep other classes (like background) as they are.
            
            print(f"SMOTE target strategy: {sampling_strategy}")
            
            # 3. Apply SMOTE with the defined strategy
            if sampling_strategy:
                smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
                X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
                print(f"After SMOTE: {X_train_scaled.shape[0]} training samples")
                print(f"New counts: {pd.Series(y_train).value_counts().to_dict()}")
            else:
                print("No target classes for SMOTE found in this fold. Skipping.")
        
        # Calculate sample weights for training
        classes = np.unique(y_train)
        # Automatic weights (inverse ratio)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        sample_weights = np.array([class_weights[label] for label in y_train])
        # manual_weights = {
        #     'background': 1.86,
        #     'chasing': 5.69,
        #     'attack': 7.5,
        #     'consume': 4.565
        # }

        # class_weights_dict = {}
        # for class_name, weight in manual_weights.items():
        #     if class_name in target_encoder.classes_:
        #         # Get the integer label for the class name
        #         class_label = target_encoder.transform([class_name])[0]
        #         class_weights_dict[class_label] = weight
        
        # print(f"Using manual class weights: {class_weights_dict}")

        # # 3. Create the final sample_weights array to pass to the model.
        # #    This maps each sample in the training set to its corresponding weight.
        # sample_weights = np.array([class_weights_dict.get(label, 1.0) for label in y_train])

        # Model selection
        if hyperparam_search:
            print(f"Performing hyperparameter search...")
            param_distributions = {
                    'n_estimators': [150, 200],
                    'max_depth': [3, 4, 5],
                    'learning_rate': [0.075, 0.1],
                    'subsample': [0.8, 0.9, 0.95],       # Add subsampling
                    'colsample_bytree': [0.4, 0.5, 0.6],# Add feature sampling
                    'gamma': [0, 0.01, 0.1],             # Add gamma for pruning
                    'reg_alpha': [0.01, 0.05, 0.1],        # Add L1 regularization
                    'reg_lambda': [1.25, 1.5, 1.75]              # Add L2 regularization
            }
            
            base_model = XGBClassifier(
                random_state=42,
                eval_metric='mlogloss'
            )
            
            # Note: We're doing nested CV here (search within the training fold)
            search = RandomizedSearchCV(
                base_model,
                param_distributions,
                n_iter=10,  # Reduced for speed
                cv=2,  # Simple 2-fold within training data
                scoring='f1_macro',
                random_state=42,
                n_jobs=-1
            )
            search.fit(X_train_scaled, y_train, sample_weight=sample_weights)
            model = search.best_estimator_
            print(f"Best params: {search.best_params_}")
        else:
            # Use fixed hyperparameters
            model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='mlogloss'
            )
            model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        
        # Predict on test set
        y_pred = model.predict(X_test_scaled)
        # --- NEW: Store predictions for aggregated confusion matrix ---
        all_y_test_agg.append(y_test)
        all_y_pred_agg.append(y_pred)
        
        # Calculate frame-level metrics
        frame_f1 = f1_score(y_test, y_pred, average='macro')
        frame_scores.append(frame_f1)
        
        # Calculate event metrics for each behavior
        fold_event_results = {}
        for behavior in target_encoder.classes_:
            if behavior != 'background':
                behavior_id = target_encoder.transform([behavior])[0]
                event_metrics = EventMetrics.calculate_event_metrics(
                    y_test, y_pred, behavior_id
                )
                event_scores_by_behavior[behavior].append(event_metrics['mAP'])
                fold_event_results[behavior] = event_metrics
        
        event_map = np.mean([r['mAP'] for r in fold_event_results.values()])
        all_event_maps.append(event_map)
        
        # Per-behavior frame-level F1
        frame_report = classification_report(y_test, y_pred, 
                                            target_names=target_encoder.classes_,
                                            output_dict=True,
                                            zero_division=0)
        
        # Print fold results
        print(f"\nFold {fold + 1} Results:")
        # --- NEW: Print full classification report for the fold ---
        print("\n--- Frame-Level Report ---")
        print(classification_report(y_test, y_pred, 
                                    target_names=target_encoder.classes_,
                                    zero_division=0))

        print(f"  Overall Frame F1: {frame_f1:.3f}")
        print(f"  Overall Event mAP: {event_map:.3f}")
        print(f"\n  Per-behavior performance:")
        for behavior in target_encoder.classes_:
            if behavior != 'background':
                behavior_frame_f1 = frame_report[behavior]['f1-score']
                behavior_event_map = fold_event_results[behavior]['mAP']
                print(f"    {behavior}:")
                print(f"      Frame F1: {behavior_frame_f1:.3f}")
                print(f"      Event mAP: {behavior_event_map:.3f}")
                print(f"      Events: {fold_event_results[behavior]['n_true_events']} true, "
                      f"{fold_event_results[behavior]['n_pred_events']} predicted")
        
        detailed_fold_results.append({
            'fold': fold + 1,
            'frame_f1': frame_f1,
            'event_map': event_map,
            'frame_report': frame_report,
            'event_results': fold_event_results
        })
    
    # Calculate final CV metrics
    print(f"\n{'='*60}")
    print("FINAL CROSS-VALIDATION RESULTS (This is what you report!)")
    print('='*60)
    print("\n" + "="*60)
    print("AGGREGATED CONFUSION MATRIX (from all CV folds)")
    print("="*60)
    
    # Concatenate all test and prediction arrays from the folds
    y_true_all_folds = np.concatenate(all_y_test_agg)
    y_pred_all_folds = np.concatenate(all_y_pred_agg)
    
    # Generate the confusion matrix
    cm = confusion_matrix(y_true_all_folds, y_pred_all_folds, normalize='true')
    cm_df = pd.DataFrame(cm, index=target_encoder.classes_, columns=target_encoder.classes_)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Aggregated Cross-Validation Confusion Matrix (Normalized by True Label)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    print("\n" + "="*60)
    print("These are your reportable metrics - all from cross-validation!")

    
    # Overall metrics
    cv_frame_f1_mean = np.mean(frame_scores)
    cv_frame_f1_std = np.std(frame_scores)
    cv_event_map_mean = np.mean(all_event_maps)
    cv_event_map_std = np.std(all_event_maps)
    
    print(f"\nOverall Performance:")
    print(f"  Frame-level Macro F1: {cv_frame_f1_mean:.3f} ± {cv_frame_f1_std:.3f}")
    print(f"  Event-level Macro mAP: {cv_event_map_mean:.3f} ± {cv_event_map_std:.3f}")
    print(f"  Improvement: {((cv_event_map_mean - cv_frame_f1_mean) / cv_frame_f1_mean * 100):.1f}%")
    
    # Per-behavior CV metrics
    print(f"\nPer-Behavior CV Performance:")
    print(f"{'Behavior':<20} {'Frame F1':>12} {'Event mAP':>12} {'Improvement':>12}")
    print("-" * 56)
    
    behavior_summaries = {}
    for behavior in target_encoder.classes_:
        if behavior != 'background':
            # Get frame F1 for this behavior across folds
            behavior_frame_f1s = [r['frame_report'][behavior]['f1-score'] 
                                 for r in detailed_fold_results]
            frame_mean = np.mean(behavior_frame_f1s)
            frame_std = np.std(behavior_frame_f1s)
            
            # Get event mAP for this behavior across folds
            event_mean = np.mean(event_scores_by_behavior[behavior])
            event_std = np.std(event_scores_by_behavior[behavior])
            
            improvement = ((event_mean - frame_mean) / frame_mean * 100) if frame_mean > 0 else 0
            
            print(f"{behavior:<20} {frame_mean:.3f}±{frame_std:.3f}  "
                  f"{event_mean:.3f}±{event_std:.3f}  {improvement:+.1f}%")
            
            behavior_summaries[behavior] = {
                'frame_f1_mean': frame_mean,
                'frame_f1_std': frame_std,
                'event_map_mean': event_mean,
                'event_map_std': event_std
            }
    
    print("\n" + "="*60)
    print("These are your reportable metrics - all from cross-validation!")
    print("DO NOT use in-sample metrics for reporting!")
    print("="*60)
    
    return {
        'cv_frame_f1_mean': cv_frame_f1_mean,
        'cv_frame_f1_std': cv_frame_f1_std,
        'cv_event_map_mean': cv_event_map_mean,
        'cv_event_map_std': cv_event_map_std,
        'behavior_summaries': behavior_summaries,
        'detailed_folds': detailed_fold_results
    }

def compare_experiments(base_dir: Path, experiment_names: List[str]):
    """
    Compare results across different label expansion experiments.
    """
    results_summary = {}
    
    for exp_name in experiment_names:
        exp_dir = base_dir / f"enhanced_{exp_name}"
        
        if not exp_dir.exists():
            print(f"Skipping {exp_name}: directory not found")
            continue
        
        # Find all feature and label files
        feature_files = sorted(exp_dir.glob('*_enhanced_features.csv'))
        label_files = sorted(exp_dir.glob('*_expanded_labels.csv'))
        
        if len(feature_files) != len(label_files):
            print(f"Warning: Mismatched files in {exp_name}")
            continue
        
        # Run experiment
        results = run_experiment(
            [str(f) for f in feature_files],
            [str(f) for f in label_files],
            experiment_name=exp_name
        )
        
        if results:
            results_summary[exp_name] = results
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPARISON")
    print('='*80)
    print(f"{'Experiment':<20} {'CV Frame F1':>15} {'CV Event mAP':>15} {'Improvement':>15}")
    print("-"*80)
    
    for exp_name, results in results_summary.items():
        frame_f1 = results['cv_frame_f1']
        event_map = results['cv_event_map']
        improvement = ((event_map - frame_f1) / frame_f1) * 100 if frame_f1 > 0 else 0
        
        print(f"{exp_name:<20} {frame_f1:.3f} ± {results['cv_frame_std']:.3f}    "
              f"{event_map:.3f} ± {results['cv_event_std']:.3f}    "
              f"{improvement:+.1f}%")
    
    return results_summary

if __name__ == "__main__":
    # For single experiment
    base_dir = Path("/home/tarislada/Documents/Extra_python_projects/SKH FP/FInalized_process")
    exp_dir = base_dir / "enhanced_6_4"
    
    if exp_dir.exists():
        feature_files = sorted(exp_dir.glob('*_enhanced_features.csv'))
        label_files = sorted(exp_dir.glob('*_expanded_labels.csv'))
        
        print(f"Found {len(feature_files)} enhanced files")
        
        # Run with SMOTE and hyperparameter search
        results = run_experiment(
            [str(f) for f in feature_files],
            [str(f) for f in label_files],
            experiment_name="Label Expansion (6 before, 4 after) with SMOTE",
            use_smote=True,
            hyperparam_search=True
        )
        
        # Also try without SMOTE for comparison
        print("\n" + "="*80)
        print("RUNNING COMPARISON WITHOUT SMOTE")
        print("="*80)
        
        results_no_smote = run_experiment(
            [str(f) for f in feature_files],
            [str(f) for f in label_files],
            experiment_name="Label Expansion (6 before, 4 after) without SMOTE",
            use_smote=False,
            hyperparam_search=True
        )
        
        # Compare results
        if results and results_no_smote:
            print("\n" + "="*80)
            print("SMOTE COMPARISON")
            print("="*80)
            print(f"{'Method':<20} {'Frame F1':>15} {'Event mAP':>15}")
            print("-"*50)
            print(f"{'With SMOTE':<20} {results['cv_frame_f1_mean']:.3f} ± {results['cv_frame_f1_std']:.3f}  "
                  f"{results['cv_event_map_mean']:.3f} ± {results['cv_event_map_std']:.3f}")
            print(f"{'Without SMOTE':<20} {results_no_smote['cv_frame_f1_mean']:.3f} ± {results_no_smote['cv_frame_f1_std']:.3f}  "
                  f"{results_no_smote['cv_event_map_mean']:.3f} ± {results_no_smote['cv_event_map_std']:.3f}")
    
    # Or compare multiple experiments
    # compare_experiments(base_dir, ['expand_3_5', 'expand_5_10', 'expand_5_15', 'expand_7_20'])