import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import gc
import psutil
import os
warnings.filterwarnings('ignore')

class MemoryOptimizedFeatureEngineer:
    def __init__(self, window_sizes=[15, 30], max_features_per_type=3, 
                 sequential_lags=[1, 2, 3], sequential_features=None):
        """
        Memory-optimized feature engineering with sequential features
        Args:
            window_sizes: Reduced window sizes for rolling features
            max_features_per_type: Limit number of rolling statistics per feature
            sequential_lags: Number of previous frames to include (e.g., [1, 2, 3])
            sequential_features: List of features to include from previous frames
        """
        self.window_sizes = window_sizes
        self.max_features_per_type = max_features_per_type
        self.sequential_lags = sequential_lags
        
        # Default sequential features (most important for behavioral transitions)
        self.sequential_features = sequential_features or [
            'head_angle', 'cricket_angle', 'relative_angle', 
            'distance', 'cricket_in_binocular', 'is_cricket_visible'
        ]
        
    def optimize_dtypes(self, df):
        """Optimize pandas dtypes to reduce memory usage"""
        df = df.copy()
        
        # Convert to more memory-efficient types
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to category if few unique values
                if len(df[col].unique()) < len(df) * 0.5:
                    df[col] = df[col].astype('category')
            elif df[col].dtype == 'float64':
                # Downcast to float32 if possible
                df[col] = pd.to_numeric(df[col], downcast='float')
            elif df[col].dtype == 'int64':
                # Downcast to smaller int if possible
                df[col] = pd.to_numeric(df[col], downcast='integer')
        
        return df
    
    def engineer_essential_features(self, df):
        """Engineer only the most essential features to save memory"""
        df = df.copy()
        
        # Parse coordinates (only if needed)
        coord_features = []
        for coord_col in ['tail_base', 'body_center', 'nose']:
            if coord_col in df.columns and df[coord_col].dtype == 'object':
                try:
                    df[f'{coord_col}_x'] = df[coord_col].str.extract(r'\(([-\d.]+),').astype('float32')
                    df[f'{coord_col}_y'] = df[coord_col].str.extract(r',\s*([-\d.]+)\)').astype('float32')
                    coord_features.extend([f'{coord_col}_x', f'{coord_col}_y'])
                    df = df.drop(coord_col, axis=1)  # Remove original to save memory
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
        priority_stats = ['mean', 'std']  # Reduced from 5 to 2 statistics
        
        for window in self.window_sizes:
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
        """Get list of engineered feature names"""
        exclude_cols = [
            'frame', 'behavior', 'animal_id',
            'cricket_status', 'validation'
        ]
        return [col for col in df.columns if col not in exclude_cols]

class MemoryOptimizedDataPreparator:
    def __init__(self):
        self.scaler = StandardScaler()
        self.target_encoder = LabelEncoder()
        
    def process_files_in_batches(self, feature_files, label_files, batch_size=5):
        """Process files in batches to manage memory"""
        print(f"Processing {len(feature_files)} files in batches of {batch_size}")
        
        all_batches = []
        for i in range(0, len(feature_files), batch_size):
            batch_feature_files = feature_files[i:i+batch_size]
            batch_label_files = label_files[i:i+batch_size]
            
            print(f"Processing batch {i//batch_size + 1}/{(len(feature_files)-1)//batch_size + 1}")
            batch_data = self.load_and_align_batch(batch_feature_files, batch_label_files)
            
            if batch_data is not None:
                all_batches.append(batch_data)
            
            # Force garbage collection
            gc.collect()
            
            # Memory check
            memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
            print(f"Current memory usage: {memory_usage:.1f} MB")
        
        if all_batches:
            return pd.concat(all_batches, ignore_index=True)
        else:
            return None
    
    def load_and_align_batch(self, feature_files, label_files):
        """Load and align a batch of files"""
        batch_data = []
        
        for feat_file, label_file in zip(feature_files, label_files):
            try:
                # Extract animal ID from filename
                animal_id = Path(feat_file).stem.split('_')[0]
                
                # Load data with memory optimization
                features = pd.read_csv(feat_file, dtype={'frame': 'int32'})
                labels = pd.read_csv(label_file, dtype={'frame': 'int32'})
                
                # Align by frame
                merged = pd.merge(features, labels, on='frame', how='inner')
                merged['animal_id'] = animal_id
                
                # Basic cleaning
                merged = merged.dropna(subset=['behavior'])  # Remove rows without labels
                
                batch_data.append(merged)
                
            except Exception as e:
                print(f"Error loading {Path(feat_file).stem}: {e}")
                continue
        
        if batch_data:
            return pd.concat(batch_data, ignore_index=True)
        return None
    
    def prepare_data(self, data, feature_engineer):
        """Prepare data with memory optimization and sequential feature handling"""
        print("Engineering features...")
        data_engineered = feature_engineer.engineer_essential_features(data)
        
        # Get feature columns
        feature_cols = feature_engineer.get_feature_names(data_engineered)
        print(f"Using {len(feature_cols)} features (including sequential)")
        
        # Prepare X and y
        X = data_engineered[feature_cols].copy()
        y = data_engineered['behavior'].copy()
        groups = data_engineered['animal_id'].copy()
        
        # Remove rows with NaN in target or too many NaN features (from sequential lags)
        # Keep rows that have valid labels and at most 25% missing features
        valid_label_mask = ~y.isna()
        max_lag = max(feature_engineer.sequential_lags) if feature_engineer.sequential_lags else 0
        
        # For sequential features, we need to remove the first max_lag frames per animal
        if 'animal_id' in data_engineered.columns and max_lag > 0:
            print(f"Removing first {max_lag} frames per animal due to sequential lag features...")
            # Group by animal and remove first max_lag frames
            def remove_initial_frames(group):
                return group.iloc[max_lag:]
            
            valid_indices = data_engineered.groupby('animal_id').apply(remove_initial_frames).index.get_level_values(1)
            sequential_mask = X.index.isin(valid_indices)
        else:
            # If no animal grouping, just remove first max_lag frames globally
            sequential_mask = X.index >= max_lag
        
        # Combine masks
        final_mask = valid_label_mask & sequential_mask
        X, y, groups = X[final_mask], y[final_mask], groups[final_mask]
        
        print(f"Removed {(~final_mask).sum()} rows due to missing labels or sequential lag NaNs")
        print(f"Final dataset size: {len(X)} samples")
        
        # Encode target labels - THIS WAS THE MISSING PIECE!
        y_encoded = self.target_encoder.fit_transform(y)
        print(f"Encoded labels: {dict(zip(self.target_encoder.classes_, range(len(self.target_encoder.classes_))))}")
        
        # Handle missing values efficiently
        X = self._handle_missing_values_efficient(X)
        
        # Feature selection to reduce dimensionality
        X_selected = self._select_important_features(X, y_encoded, max_features=60)  # Increased slightly for sequential features
        
        # Scale only numeric features
        numeric_cols = X_selected.select_dtypes(include=[np.number]).columns
        X_scaled = X_selected.copy()
        X_scaled[numeric_cols] = self.scaler.fit_transform(X_selected[numeric_cols])
        
        return X_scaled, y_encoded, groups, list(X_scaled.columns)
    
    def _handle_missing_values_efficient(self, X):
        """Efficient missing value handling"""
        for col in X.columns:
            if X[col].dtype == 'category':
                # Handle categorical columns specially
                if X[col].isna().any():
                    # Add 'unknown' to categories if not already there
                    if 'unknown' not in X[col].cat.categories:
                        X[col] = X[col].cat.add_categories(['unknown'])
                    X[col] = X[col].fillna('unknown')
                # Convert to codes
                X[col] = X[col].cat.codes.astype('int8')
            elif X[col].dtype == 'object':
                X[col] = X[col].fillna('unknown')
                # Label encode string categorical
                X[col] = X[col].astype('category').cat.codes.astype('int8')
            elif X[col].dtype == 'bool':
                X[col] = X[col].fillna(False).astype('int8')
            else:
                X[col] = X[col].fillna(X[col].median())
        
        return X
    
    def _select_important_features(self, X, y, max_features=50):
        """Select most important features to reduce memory"""
        from sklearn.feature_selection import SelectKBest, f_classif
        
        try:
            # Use only numeric features for feature selection
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

class LightweightClassifier:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_models = {}
        self.feature_importance = {}
        
    def train_multiple_models(self, X, y, groups, cv_splits, class_weights=None):
        """Train multiple models efficiently"""
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
        
        # Define models with reduced hyperparameter grids
        model_configs = {
            # 'rf': {
            #     'model': RandomForestClassifier(random_state=self.random_state, class_weight=class_weights),
            #     'params': {
            #         'n_estimators': [100, 200],
            #         'max_depth': [10, 20],
            #         'min_samples_split': [5, 10]
            #     }
            # },
            'xgb': {
                'model': XGBClassifier(random_state=self.random_state, eval_metric='mlogloss'),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6],
                    'learning_rate': [0.1, 0.2]
                },
                'use_sample_weight': True
            },
        #     'lr': {
        #         'model': LogisticRegression(random_state=self.random_state, class_weight=class_weights, max_iter=1000),
        #         'params': {
        #             'C': [0.1, 1, 10],
        #             'penalty': ['l2']  # Simplified to avoid solver issues
        #         }
        #     }
        }
        
        scores = {}
        
        for model_name, config in model_configs.items():
            print(f"\nTraining {model_name}...")
            
            try:
                # Hyperparameter tuning with reduced iterations
                search = RandomizedSearchCV(
                    config['model'],
                    config['params'],
                    cv=cv_splits,
                    scoring='f1_macro',
                    n_iter=6,  # Reduced iterations for speed
                    random_state=self.random_state,
                    n_jobs=-1,
                    error_score='raise'
                )
                # --- MODIFIED: Pass sample_weight to fit if required by the model ---
                fit_params = {}
                if config.get('use_sample_weight', False):
                    print(f"Passing sample_weight to {model_name}.fit()")
                    fit_params['sample_weight'] = sample_weights

                search.fit(X, y, groups=groups, **fit_params)
                self.best_models[model_name] = search.best_estimator_
                
                best_index = search.best_index_
                fold_scores = [search.cv_results_[f'split{i}_test_score'][best_index] for i in range(search.n_splits_)]
                
                mean_score = np.mean(fold_scores)
                sem_score = np.std(fold_scores) / np.sqrt(len(fold_scores))

                scores[model_name] = {
                    'mean': mean_score,
                    'sem': sem_score,
                    'all_folds': fold_scores
                }
                
                print(f"Best {model_name} score: {mean_score:.4f} ± {sem_score:.4f} (SEM)")
                print(f"Best {model_name} params: {search.best_params_}")
                
                # Feature importance for tree-based models
                if hasattr(search.best_estimator_, 'feature_importances_'):
                    importance = pd.DataFrame({
                        'feature': X.columns,
                        'importance': search.best_estimator_.feature_importances_
                    }).sort_values('importance', ascending=False)
                    self.feature_importance[model_name] = importance
                
            except Exception as e:
                print(f"{model_name} training failed: {e}")
                continue
        
        # Create ensemble if we have multiple models
        if len(self.best_models) > 1:
            print("\nCreating ensemble...")
            self.ensemble = VotingClassifier([
                (name, model) for name, model in self.best_models.items()
            ], voting='soft')
            
            self.ensemble.fit(X, y)
            print("Ensemble created successfully")
        
        return scores
    
    def predict(self, X, model_name='ensemble'):
        """Make predictions using specified model"""
        if model_name == 'ensemble' and hasattr(self, 'ensemble'):
            return self.ensemble.predict(X)
        elif model_name in self.best_models:
            return self.best_models[model_name].predict(X)
        else:
            # Default to first available model
            first_model = list(self.best_models.keys())[0]
            print(f"Model {model_name} not found, using {first_model}")
            return self.best_models[first_model].predict(X)
    
    def predict_proba(self, X, model_name='ensemble'):
        """Get prediction probabilities"""
        if model_name == 'ensemble' and hasattr(self, 'ensemble'):
            return self.ensemble.predict_proba(X)
        elif model_name in self.best_models:
            return self.best_models[model_name].predict_proba(X)
        else:
            # Default to first available model
            first_model = list(self.best_models.keys())[0]
            return self.best_models[first_model].predict_proba(X)

class SimpleEvaluator:
    def evaluate_model(self, y_true, y_pred, target_encoder, show_plot=False):
        """Simple evaluation"""
        # Get class names
        class_names = target_encoder.classes_
        
        # Classification report
        report = classification_report(y_true, y_pred, 
                                     target_names=class_names, 
                                     output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Only show detailed output and plot if requested
        if show_plot:
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, target_names=class_names))
            
            # Create subplots for confusion matrix and sequence plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', 
                       xticklabels=class_names, 
                       yticklabels=class_names, ax=ax1)
            ax1.set_title('Confusion Matrix')
            ax1.set_ylabel('True Label')
            ax1.set_xlabel('Predicted Label')
            
            # Plot sequence comparison
            self.plot_sequence_comparison(y_true, y_pred, class_names, ax2)
            
            plt.tight_layout()
            plt.show()
        
        return report, cm
    
    def plot_sequence_comparison(self, y_true, y_pred, class_names, ax=None):
        """Plot sequence comparison efficiently using broken_barh."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(15, 4))
        
        # Create color map for behaviors
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
        color_dict = {i: colors[i] for i in range(len(class_names))}
        
        # Helper function to find contiguous segments
        def find_segments(y):
            if len(y) == 0:
                return {}
            # Find where behavior changes
            change_points = np.where(np.diff(y) != 0)[0] + 1
            # Add start and end points
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
        ax.set_yticks([0.4, 1.4]) # Centered ticks
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

# Main Lightweight Pipeline
class LightweightBehavioralPipeline:
    def __init__(self, window_sizes=[15, 30], batch_size=5, 
                 sequential_lags=[1, 2, 3], sequential_features=None,
                 class_weight=None):
        self.feature_engineer = MemoryOptimizedFeatureEngineer(
            window_sizes=window_sizes,
            sequential_lags=sequential_lags,
            sequential_features=sequential_features
        )
        self.data_preparator = MemoryOptimizedDataPreparator()
        self.classifier = LightweightClassifier()
        self.evaluator = SimpleEvaluator()
        self.batch_size = batch_size
        self.class_weight = class_weight
        
    def run_pipeline(self, feature_files, label_files, cv_folds=3):
        """Run the lightweight pipeline"""
        print("=== MEMORY-OPTIMIZED BEHAVIORAL CLASSIFICATION PIPELINE ===")
        
        # Check available memory
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"Available RAM: {memory_gb:.1f} GB")
        
        # Process in batches
        print("Loading data in batches...")
        data = self.data_preparator.process_files_in_batches(
            feature_files, label_files, self.batch_size
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

        # --- NEW: Process class weights ---
        model_class_weights = None
        if self.class_weight:
            if self.class_weight == 'balanced':
                print("Using 'balanced' class weights.")
                classes = np.unique(y)
                weights = compute_class_weight('balanced', classes=classes, y=y)
                model_class_weights = dict(zip(classes, weights))
            elif isinstance(self.class_weight, dict):
                print(f"Using custom class weights: {self.class_weight}")
                # Map string class names to integer labels used by the model
                encoder_classes = list(self.data_preparator.target_encoder.classes_)
                model_class_weights = {
                    encoder_classes.index(cls): weight
                    for cls, weight in self.class_weight.items()
                    if cls in encoder_classes
                }
                print(f"Mapped to encoded labels: {model_class_weights}")
        
        # Create CV splits
        cv = GroupKFold(n_splits=cv_folds)
        cv_splits = list(cv.split(X, y, groups=groups))
        
        # Train model
        print("Training models...")
        scores = self.classifier.train_multiple_models(X, y, groups, cv_splits, class_weights=model_class_weights)
        
        # Evaluate all models
        print("Evaluating models...")
        results = {}
        
        # Evaluate all models
        print("Evaluating models...")
        results = {}
        
        # Evaluate individual models
        for model_name in self.classifier.best_models.keys():
            y_pred = self.classifier.predict(X, model_name)
            report, cm = self.evaluator.evaluate_model(y, y_pred, self.data_preparator.target_encoder, show_plot=False)
            results[model_name] = {
                'cv_score_mean': scores.get(model_name, {}).get('mean', 0),
                'cv_score_sem': scores.get(model_name, {}).get('sem', 0),
                'classification_report': report,
                'confusion_matrix': cm
            }
            print(f"\n{model_name.upper()} Results:")
            print(f"CV Score: {results[model_name]['cv_score_mean']:.4f} ± {results[model_name]['cv_score_sem']:.4f} (SEM)")
            print(f"In-sample Accuracy: {report['accuracy']:.4f}")
            print(f"In-sample Macro F1: {report['macro avg']['f1-score']:.4f}")
        
        # Evaluate ensemble if available
        if hasattr(self.classifier, 'ensemble'):
            y_pred_ensemble = self.classifier.predict(X, 'ensemble')
            report_ensemble, cm_ensemble = self.evaluator.evaluate_model(y, y_pred_ensemble, self.data_preparator.target_encoder, show_plot=True)
            avg_cv_mean = np.mean([s['mean'] for s in scores.values()])
            avg_cv_sem = np.mean([s['sem'] for s in scores.values()]) # Note: This is a simplification

            results['ensemble'] = {
                'cv_score_mean': avg_cv_mean,
                'cv_score_sem': avg_cv_sem,
                'classification_report': report_ensemble,
                'confusion_matrix': cm_ensemble
            }
            print(f"\nENSEMBLE Results:")
            print(f"In-sample Accuracy: {report_ensemble['accuracy']:.4f}")
            print(f"In-sample Macro F1: {report_ensemble['macro avg']['f1-score']:.4f}")

        elif self.classifier.best_models:
            # If no ensemble, show plot for best individual model
            best_individual = max(results.keys(), key=lambda x: results[x]['classification_report']['macro avg']['f1-score'])
            y_pred_best = self.classifier.predict(X, best_individual)
            self.evaluator.evaluate_model(y, y_pred_best, self.data_preparator.target_encoder, show_plot=True)
        
        return {
            'data_shape': data.shape,
            'feature_shape': X.shape,
            'individual_scores': scores,
            'results': results,
            'feature_columns': feature_cols,
            'feature_importance': self.classifier.feature_importance
        }

# Example usage
if __name__ == "__main__":
    # Define file paths - UPDATE THESE PATHS
    feature_dir = Path("/home/tarislada/Documents/Extra_python_projects/SKH FP/FInalized_process/test_val_vid5")
    label_dir = Path("/home/tarislada/Documents/Extra_python_projects/SKH FP/FInalized_process/Behavior_label")
    
    # --- This block automatically finds and pairs your files ---
    feature_files = []
    label_files = []

    # Assuming a consistent naming convention:
    # Feature file: {base_name}_debugging_analysis.csv
    # Label file:   {base_name}_processed_labels.csv
    
    print(f"Searching for feature files in: {feature_dir}")
    for feature_path in sorted(feature_dir.glob('*_analysis.csv')):
        # Extract the base name (e.g., 'm14_t1')
        base_name = feature_path.stem.replace('_validated', '').replace('_analysis', '')
        
        # Construct the expected corresponding label file path
        label_path = label_dir / f"{base_name}_processed_labels.csv"
        
        # Add the pair to the lists if the label file exists
        if label_path.exists():
            feature_files.append(str(feature_path))
            label_files.append(str(label_path))
        else:
            print(f"Warning: Skipping {feature_path.name} because corresponding label file was not found at {label_path}")
    # For testing with just a few files first
    if len(feature_files) > 5:
        print("WARNING: Processing more than 5 files. Consider testing with fewer files first.")
    
    # Initialize lightweight pipeline with sequential features
    pipeline = LightweightBehavioralPipeline(
        window_sizes=[15, 30],          # Reduced temporal windows
        batch_size=3,                   # Process 3 files at a time
        sequential_lags=[1, 2, 3],      # Include previous 1, 2, 3 frames
        sequential_features=[           # Key features for behavioral transitions
            'head_angle', 'cricket_angle', 'relative_angle', 
            'distance', 'cricket_in_binocular', 'is_cricket_visible',
            'zone'  # Important for behavioral context
        ],
        class_weight={
            'attack': 1.2,
            'non_visual_rotation': 1.0,
            'chasing': 0.94,
            'background': 1.0
        }
    )
    
    # Run the pipeline
    try:
        print("=== ENHANCED BEHAVIORAL CLASSIFICATION WITH SEQUENTIAL FEATURES ===")
        print("This version includes:")
        print("- Features from previous 1, 2, 3 frames for temporal context")
        print("- Behavioral transition detection (zone changes, direction consistency)")
        print("- Memory-optimized processing")
        print("- Sequence visualization comparing true vs predicted behaviors")
        
        results = pipeline.run_pipeline(feature_files, label_files, cv_folds=3)
        
        if results:
            print(f"\n=== FINAL RESULTS ===")
            print(f"Data processed: {results['data_shape']}")
            print(f"Features used: {results['feature_shape'][1]} (including sequential)")
            
            # Show best model
            best_model = None
            best_f1 = 0
            
            for model_name, model_results in results['results'].items():
                f1_score = model_results['classification_report']['macro avg']['f1-score']
                cv_mean = model_results['cv_score_mean']
                cv_sem = model_results['cv_score_sem']
                print(f"{model_name}: CV F1={cv_mean:.4f} ± {cv_sem:.4f}, In-sample F1={f1_score:.4f}")
                
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_model = model_name
            
            print(f"\nBest model: {best_model} (F1: {best_f1:.4f})")
            
            # Show feature importance if available
            if results.get('feature_importance'):
                print(f"\nTop 10 features ({best_model}):")
                if best_model in results['feature_importance']:
                    top_features = results['feature_importance'][best_model].head(10)
                    print(top_features)
                    
                    # Count sequential features in top 10
                    sequential_in_top10 = sum(1 for feat in top_features['feature'] 
                                            if '_lag_' in feat or 'changed' in feat or 'consistent' in feat)
                    print(f"\nSequential features in top 10: {sequential_in_top10}")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()