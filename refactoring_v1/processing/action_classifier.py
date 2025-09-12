"""
Action segmentation using XGBoost classifier.
Processes behavioral features to classify hunting behaviors with hyperparameter optimization.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import gc
import psutil
import os
from typing import List, Dict, Optional, Tuple, Any
from config.settings import settings

warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Memory-optimized feature engineering with sequential features."""
    
    def __init__(self, config=None):
        self.config = config or settings.action_segmentation
        
    def optimize_dtypes(self, df):
        """Optimize pandas dtypes to reduce memory usage."""
        df = df.copy()
        
        for col in df.columns:
            if df[col].dtype == 'object':
                if len(df[col].unique()) < len(df) * 0.5:
                    df[col] = df[col].astype('category')
            elif df[col].dtype == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')
            elif df[col].dtype == 'int64':
                df[col] = pd.to_numeric(df[col], downcast='integer')
        
        return df
    
    def engineer_essential_features(self, df):
        """Engineer only the most essential features to save memory."""
        df = df.copy()
        
        # Parse coordinates (only if needed)
        coord_features = []
        for coord_col in ['tail_base', 'body_center', 'nose']:
            if coord_col in df.columns and df[coord_col].dtype == 'object':
                try:
                    df[f'{coord_col}_x'] = df[coord_col].str.extract(r'\(([-\d.]+),').astype('float32')
                    df[f'{coord_col}_y'] = df[coord_col].str.extract(r',\s*([-\d.]+)\)').astype('float32')
                    coord_features.extend([f'{coord_col}_x', f'{coord_col}_y'])
                    df = df.drop(coord_col, axis=1)
                except:
                    print(f"Warning: Could not parse coordinates from {coord_col}")
        
        # Essential motion features only
        essential_cols = ['head_angle', 'cricket_angle', 'distance']
        for col in essential_cols:
            if col in df.columns:
                df[f'{col}_velocity'] = df[col].diff().astype('float32')
        
        # Essential behavioral features
        if 'cricket_angle' in df.columns and 'head_angle' in df.columns:
            df['relative_angle'] = (df['cricket_angle'] - df['head_angle']).astype('float32')
        
        # Binary behavioral indicators
        if 'cricket_use_nose_position' in df.columns:
            df['is_cricket_visible'] = (~df['cricket_use_nose_position']).astype('int8')
        if 'zone' in df.columns:
            df['cricket_in_binocular'] = (df['zone'] == 'binocular').astype('int8')
        
        # LIMITED rolling features (only most important)
        priority_features = ['head_angle', 'cricket_angle', 'distance', 'relative_angle']
        priority_stats = ['mean', 'std']
        
        for window in self.config.window_sizes:
            for col in priority_features:
                if col in df.columns:
                    for stat in priority_stats:
                        if stat == 'mean':
                            df[f'{col}_{stat}_{window}'] = df[col].rolling(window, center=True, min_periods=1).mean().astype('float32')
                        elif stat == 'std':
                            df[f'{col}_{stat}_{window}'] = df[col].rolling(window, center=True, min_periods=1).std().astype('float32')
        
        # Optimize dtypes
        df = self.optimize_dtypes(df)
        
        return df
    
    def get_feature_names(self, df):
        """Get list of engineered feature names."""
        exclude_cols = [
            'frame', 'behavior', 'animal_id',
            'cricket_status', 'validation'
        ]
        return [col for col in df.columns if col not in exclude_cols]

class DataPreparator:
    """Data preparation with memory optimization."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.target_encoder = LabelEncoder()
        
    def process_files_in_batches(self, feature_files, label_files, batch_size=5):
        """Process files in batches to manage memory."""
        print(f"Processing {len(feature_files)} files in batches of {batch_size}")
        
        all_batches = []
        for i in range(0, len(feature_files), batch_size):
            batch_feature_files = feature_files[i:i+batch_size]
            batch_label_files = label_files[i:i+batch_size]
            
            print(f"Processing batch {i//batch_size + 1}/{(len(feature_files)-1)//batch_size + 1}")
            batch_data = self.load_and_align_batch(batch_feature_files, batch_label_files)
            
            if batch_data is not None:
                all_batches.append(batch_data)
            
            gc.collect()
            
            memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            print(f"Current memory usage: {memory_usage:.1f} MB")
        
        if all_batches:
            return pd.concat(all_batches, ignore_index=True)
        else:
            return None
    
    def load_and_align_batch(self, feature_files, label_files):
        """Load and align a batch of files."""
        batch_data = []
        
        for feat_file, label_file in zip(feature_files, label_files):
            try:
                animal_id = Path(feat_file).stem.split('_')[0]
                
                features = pd.read_csv(feat_file, dtype={'frame': 'int32'})
                labels = pd.read_csv(label_file, dtype={'frame': 'int32'})
                
                merged = pd.merge(features, labels, on='frame', how='inner')
                merged['animal_id'] = animal_id
                
                merged = merged.dropna(subset=['behavior'])
                
                batch_data.append(merged)
                
            except Exception as e:
                print(f"Error loading {Path(feat_file).stem}: {e}")
                continue
        
        if batch_data:
            return pd.concat(batch_data, ignore_index=True)
        return None
    
    def prepare_data(self, data, feature_engineer):
        """Prepare data with memory optimization and sequential feature handling."""
        print("Engineering features...")
        data_engineered = feature_engineer.engineer_essential_features(data)
        
        feature_cols = feature_engineer.get_feature_names(data_engineered)
        print(f"Using {len(feature_cols)} features (including sequential)")
        
        X = data_engineered[feature_cols].copy()
        y = data_engineered['behavior'].copy()
        groups = data_engineered['animal_id'].copy()
        
        # Remove rows with NaN in target or too many NaN features
        valid_label_mask = ~y.isna()
        max_lag = max(feature_engineer.config.sequential_lags) if feature_engineer.config.sequential_lags else 0
        
        if 'animal_id' in data_engineered.columns and max_lag > 0:
            print(f"Removing first {max_lag} frames per animal due to sequential lag features...")
            def remove_initial_frames(group):
                return group.iloc[max_lag:]
            
            valid_indices = data_engineered.groupby('animal_id').apply(remove_initial_frames).index.get_level_values(1)
            sequential_mask = X.index.isin(valid_indices)
        else:
            sequential_mask = X.index >= max_lag
        
        final_mask = valid_label_mask & sequential_mask
        X, y, groups = X[final_mask], y[final_mask], groups[final_mask]
        
        print(f"Removed {(~final_mask).sum()} rows due to missing labels or sequential lag NaNs")
        print(f"Final dataset size: {len(X)} samples")
        
        # Encode target labels
        y_encoded = self.target_encoder.fit_transform(y)
        print(f"Encoded labels: {dict(zip(self.target_encoder.classes_, range(len(self.target_encoder.classes_))))}")
        
        # Handle missing values efficiently
        X = self._handle_missing_values_efficient(X)
        
        # Feature selection to reduce dimensionality
        X_selected = self._select_important_features(X, y_encoded, max_features=60)
        
        # Scale only numeric features
        numeric_cols = X_selected.select_dtypes(include=[np.number]).columns
        X_scaled = X_selected.copy()
        X_scaled[numeric_cols] = self.scaler.fit_transform(X_selected[numeric_cols])
        
        return X_scaled, y_encoded, groups, list(X_scaled.columns)
    
    def _handle_missing_values_efficient(self, X):
        """Efficient missing value handling."""
        for col in X.columns:
            if X[col].dtype == 'category':
                if X[col].isna().any():
                    if 'unknown' not in X[col].cat.categories:
                        X[col] = X[col].cat.add_categories(['unknown'])
                    X[col] = X[col].fillna('unknown')
                X[col] = X[col].cat.codes.astype('int8')
            elif X[col].dtype == 'object':
                X[col] = X[col].fillna('unknown')
                X[col] = X[col].astype('category').cat.codes.astype('int8')
            elif X[col].dtype == 'bool':
                X[col] = X[col].fillna(False).astype('int8')
            else:
                X[col] = X[col].fillna(X[col].median())
        
        return X
    
    def _select_important_features(self, X, y, max_features=50):
        """Select most important features to reduce memory."""
        try:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            categorical_cols = X.select_dtypes(exclude=[np.number]).columns
            
            if len(numeric_cols) > max_features:
                selector = SelectKBest(score_func=f_classif, k=max_features-len(categorical_cols))
                X_numeric_selected = selector.fit_transform(X[numeric_cols], y)
                selected_numeric_cols = numeric_cols[selector.get_support()]
                
                X_selected = pd.concat([
                    pd.DataFrame(X_numeric_selected, columns=selected_numeric_cols, index=X.index),
                    X[categorical_cols]
                ], axis=1)
                
                print(f"Selected {len(selected_numeric_cols)} numeric + {len(categorical_cols)} categorical features")
                return X_selected
            else:
                return X
        except Exception as e:
            print(f"Feature selection failed: {e}, using all features")
            return X

class XGBoostClassifier:
    """XGBoost classifier with hyperparameter optimization."""
    
    def __init__(self, config=None):
        self.config = config or settings.action_segmentation
        self.model = None
        self.feature_importance = None
        
    def train_model(self, X, y, groups, cv_splits, class_weights=None):
        """Train XGBoost model with hyperparameter tuning."""
        classes = np.unique(y)

        # Calculate class weights
        if class_weights is None:
            print("No class weights provided, calculating balanced weights as default.")
            class_weights_array = compute_class_weight('balanced', classes=classes, y=y)
            class_weights = dict(zip(classes, class_weights_array))
        
        print(f"Training with {len(classes)} classes: {dict(zip(range(len(classes)), classes))}")
        print(f"Class distribution: {np.bincount(y)}")
        print(f"Applying class weights: {class_weights}")
        
        sample_weights = np.array([class_weights[label] for label in y])
        
        # Define XGBoost hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6],
            'learning_rate': [0.1, 0.2],
            # 'subsample': [0.8, 0.9, 1.0],
            # 'colsample_bytree': [0.8, 0.9, 1.0],
            # 'reg_alpha': [0, 0.1, 1],
            # 'reg_lambda': [1, 1.5, 2]
        }
        
        print(f"\nTraining XGBoost with hyperparameter search...")
        
        try:
            # Create base model
            base_model = XGBClassifier(
                random_state=self.config.random_state,
                eval_metric='mlogloss',
                use_label_encoder=False
            )
            
            # Hyperparameter tuning
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                cv=cv_splits,
                scoring='f1_macro',
                n_iter=3,  # Increased iterations for better hyperparameter search
                random_state=self.config.random_state,
                n_jobs=-1,
                error_score='raise'
            )
            
            # Fit with sample weights
            print("Fitting model with sample weights...")
            search.fit(X, y, groups=groups, sample_weight=sample_weights)
            
            self.model = search.best_estimator_
            
            # Calculate CV scores
            best_index = search.best_index_
            fold_scores = [search.cv_results_[f'split{i}_test_score'][best_index] for i in range(search.n_splits_)]
            
            mean_score = np.mean(fold_scores)
            sem_score = np.std(fold_scores) / np.sqrt(len(fold_scores))
            
            print(f"Best XGBoost score: {mean_score:.4f} ± {sem_score:.4f} (SEM)")
            print(f"Best XGBoost params: {search.best_params_}")
            
            # Feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
            
            return {
                'mean': mean_score,
                'sem': sem_score,
                'all_folds': fold_scores,
                'best_params': search.best_params_
            }
            
        except Exception as e:
            print(f"XGBoost training failed: {e}")
            return None
    
    def predict(self, X):
        """Make predictions using trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        return self.model.predict_proba(X)

class Evaluator:
    """Simple evaluation with sequence visualization."""
    
    def evaluate_model(self, y_true, y_pred, target_encoder, groups=None, show_plot=False, num_plot_trials=3):
        """Evaluate model with optional sequence plots."""
        class_names = target_encoder.classes_
        
        # Classification report
        report = classification_report(y_true, y_pred, 
                                     target_names=class_names, 
                                     output_dict=True,
                                     zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
        
        if show_plot:
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
            
            if groups is not None:
                unique_trials = groups.unique()
                
                if len(unique_trials) > num_plot_trials:
                    plot_trials = np.random.choice(unique_trials, num_plot_trials, replace=False)
                    print(f"\nShowing sequence plots for {num_plot_trials} random trials (out of {len(unique_trials)} total).")
                else:
                    plot_trials = unique_trials
                    print(f"\nShowing sequence plots for all {len(unique_trials)} trials.")

                for trial_id in plot_trials:
                    trial_mask = (groups == trial_id)
                    y_true_trial = y_true[trial_mask]
                    y_pred_trial = y_pred[trial_mask]
                    
                    if isinstance(y_true_trial, pd.Series):
                        y_true_trial = y_true_trial.values
                    if isinstance(y_pred_trial, pd.Series):
                        y_pred_trial = y_pred_trial.values

                    fig, ax = plt.subplots(1, 1, figsize=(15, 4))
                    self.plot_sequence_comparison(y_true_trial, y_pred_trial, class_names, ax)
                    ax.set_title(f'Behavioral Sequence Comparison - Trial: {trial_id}')
                    plt.show()

                # Overall confusion matrix
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', 
                           xticklabels=class_names, 
                           yticklabels=class_names)
                plt.title('Overall Confusion Matrix (All Trials)')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.show()

            else:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
                sns.heatmap(cm, annot=True, fmt='d', 
                           xticklabels=class_names, 
                           yticklabels=class_names, ax=ax1)
                ax1.set_title('Confusion Matrix')
                ax1.set_ylabel('True Label')
                ax1.set_xlabel('Predicted Label')
                self.plot_sequence_comparison(y_true, y_pred, class_names, ax2)
                plt.tight_layout()
                plt.show()
        
        return report, cm
    
    def plot_sequence_comparison(self, y_true, y_pred, class_names, ax=None):
        """Plot sequence comparison efficiently using broken_barh."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(15, 4))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
        color_dict = {i: colors[i] for i in range(len(class_names))}
        
        def find_segments(y):
            if len(y) == 0:
                return {}
            change_points = np.where(np.diff(y) != 0)[0] + 1
            all_points = np.concatenate(([0], change_points, [len(y)]))
            
            segments_by_behavior = {i: [] for i in range(len(class_names))}
            for i in range(len(all_points) - 1):
                start = all_points[i]
                duration = all_points[i+1] - start
                behavior = y[start]
                segments_by_behavior[behavior].append((start, duration))
            return segments_by_behavior

        # Plot true sequence (bottom)
        true_segments = find_segments(y_true)
        for behavior_code, segments in true_segments.items():
            if segments:
                ax.broken_barh(segments, (0, 0.8), facecolors=color_dict[behavior_code])

        # Plot predicted sequence (top)
        pred_segments = find_segments(y_pred)
        for behavior_code, segments in pred_segments.items():
            if segments:
                ax.broken_barh(segments, (1, 0.8), facecolors=color_dict[behavior_code])
        
        # Formatting
        ax.set_ylim(-0.5, 2.5)
        ax.set_xlim(0, len(y_true))
        ax.set_yticks([0.4, 1.4])
        ax.set_yticklabels(['True', 'Predicted'])
        ax.set_xlabel('Frame Number')
        ax.set_title('Behavioral Sequence Comparison')
        
        # Create legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color_dict[i], 
                                       label=class_names[i]) 
                          for i in range(len(class_names))]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Add accuracy text
        accuracy = np.mean(y_true == y_pred)
        ax.text(0.02, 0.95, f'Frame-wise Accuracy: {accuracy:.3f}', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return ax

class ActionClassifier:
    """
    Main action classification processor.
    Processes behavioral features to classify hunting behaviors using XGBoost.
    """
    
    def __init__(self, config=None):
        self.config = config or settings.action_segmentation
        self.feature_engineer = FeatureEngineer(self.config)
        self.data_preparator = DataPreparator()
        self.classifier = XGBoostClassifier(self.config)
        self.evaluator = Evaluator()
        
    def process_files(self, feature_files: List[str], label_files: List[str]) -> Dict[str, Any]:
        """Process feature and label files for action classification."""
        print("=== ACTION CLASSIFICATION (Step 7) ===")
        
        # Check available memory
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"Available RAM: {memory_gb:.1f} GB")
        
        # Process in batches
        print("Loading data in batches...")
        data = self.data_preparator.process_files_in_batches(
            feature_files, label_files, self.config.batch_size
        )
        
        if data is None:
            print("Failed to load any data!")
            return None
        
        print(f"Total data shape: {data.shape}")
        
        # Prepare data
        print("Preparing features...")
        X, y, groups, feature_cols = self.data_preparator.prepare_data(data, self.feature_engineer)
        
        print(f"Final feature matrix shape: {X.shape}")
        print(f"Memory usage: {X.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

        # Process class weights
        model_class_weights = None
        if self.config.class_weights:
            print(f"Using custom class weights: {self.config.class_weights}")
            encoder_classes = list(self.data_preparator.target_encoder.classes_)
            model_class_weights = {
                encoder_classes.index(cls): weight
                for cls, weight in self.config.class_weights.items()
                if cls in encoder_classes
            }
            print(f"Mapped to encoded labels: {model_class_weights}")
        
        # Create CV splits
        cv = GroupKFold(n_splits=self.config.cv_folds)
        cv_splits = list(cv.split(X, y, groups=groups))
        
        # Train model
        print("Training XGBoost model...")
        training_results = self.classifier.train_model(X, y, groups, cv_splits, class_weights=model_class_weights)
        
        if training_results is None:
            print("Model training failed!")
            return None
        
        # Evaluate model
        print("Evaluating model...")
        y_pred = self.classifier.predict(X)
        report, cm = self.evaluator.evaluate_model(
            y, y_pred, self.data_preparator.target_encoder, 
            groups=groups, show_plot=True, num_plot_trials=3
        )
        
        # Results summary
        results = {
            'data_shape': data.shape,
            'feature_shape': X.shape,
            'cv_score_mean': training_results['mean'],
            'cv_score_sem': training_results['sem'],
            'best_params': training_results['best_params'],
            'classification_report': report,
            'confusion_matrix': cm,
            'feature_columns': feature_cols,
            'feature_importance': self.classifier.feature_importance
        }
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Data processed: {results['data_shape']}")
        print(f"Features used: {results['feature_shape'][1]} (including sequential)")
        print(f"CV F1 Score: {results['cv_score_mean']:.4f} ± {results['cv_score_sem']:.4f}")
        print(f"In-sample Accuracy: {report['accuracy']:.4f}")
        print(f"In-sample Macro F1: {report['macro avg']['f1-score']:.4f}")
        print(f"Best Parameters: {results['best_params']}")
        
        # Show feature importance
        if results.get('feature_importance') is not None:
            print(f"\nTop 10 most important features:")
            print(results['feature_importance'].head(10))
            
            # Count sequential features in top 10
            top_features = results['feature_importance'].head(10)
            sequential_in_top10 = sum(1 for feat in top_features['feature'] 
                                    if '_lag_' in feat or 'changed' in feat or 'consistent' in feat)
            print(f"Sequential features in top 10: {sequential_in_top10}")
        
        return results
    
    def process(self, feature_dir: Optional[str] = None,
                label_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Main processing method using configured directories.
        
        Args:
            feature_dir: Override feature directory (uses config if None)
            label_dir: Override label directory (uses config if None)
            
        Returns:
            dict: Processing results
        """
        feature_dir = feature_dir or str(settings.paths.final_videos_dir)
        label_dir = label_dir or str(settings.paths.behavior_labels_dir)
        
        print(f"Feature directory: {feature_dir}")
        print(f"Label directory: {label_dir}")
        print(f"Parameters:")
        print(f"  - Window sizes: {self.config.window_sizes}")
        print(f"  - Sequential lags: {self.config.sequential_lags}")
        print(f"  - CV folds: {self.config.cv_folds}")
        print(f"  - Class weights: {self.config.class_weights}")
        
        # Find and pair files
        feature_files = []
        label_files = []

        print(f"Searching for feature files in: {feature_dir}")
        for feature_path in sorted(Path(feature_dir).glob('*_analysis.csv')):
            base_name = feature_path.stem.replace('_validated', '').replace('_analysis', '')
            label_path = Path(label_dir) / f"{base_name}_processed_labels.csv"
            
            if label_path.exists():
                feature_files.append(str(feature_path))
                label_files.append(str(label_path))
            else:
                print(f"Warning: Skipping {feature_path.name} because corresponding label file was not found at {label_path}")
        
        if not feature_files:
            print("No matching feature and label files found!")
            return None
        
        print(f"Found {len(feature_files)} matching file pairs")
        
        return self.process_files(feature_files, label_files)

def main():
    """Command line interface for action classification."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify hunting behaviors using XGBoost")
    parser.add_argument("--feature-dir", help="Feature directory (overrides config)")
    parser.add_argument("--label-dir", help="Label directory (overrides config)")
    parser.add_argument("--cv-folds", type=int, help="Number of CV folds")
    parser.add_argument("--batch-size", type=int, help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Override config if arguments provided
    if args.cv_folds:
        settings.action_segmentation.cv_folds = args.cv_folds
    if args.batch_size:
        settings.action_segmentation.batch_size = args.batch_size
    
    # Create processor and run
    processor = ActionClassifier()
    results = processor.process(args.feature_dir, args.label_dir)
    
    if results is None:
        exit(1)

if __name__ == "__main__":
    main()