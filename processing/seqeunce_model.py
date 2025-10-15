import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt

# Import the refactored data processing classes
from data_processing import MemoryOptimizedFeatureEngineer, MemoryOptimizedDataPreparator
from raw_data_loader import RawDataLoader

# --- 1. Custom PyTorch Dataset ---
class HuntingSequenceDataset(Dataset):
    def __init__(self, features, labels, groups, sequence_length):
        self.sequence_length = sequence_length
        self.features = features
        self.labels = labels
        self.groups = groups
        
        self.sequences = []
        self.sequence_labels = []
        
        # Create sequences respecting trial boundaries
        for trial_id in np.unique(self.groups):
            trial_mask = (self.groups == trial_id)
            X_trial = self.features[trial_mask]
            y_trial = self.labels[trial_mask]
            
            if len(X_trial) < self.sequence_length:
                continue

            for i in range(len(X_trial) - self.sequence_length + 1):
                self.sequences.append(X_trial[i : i + self.sequence_length])
                # The label for a sequence is the label of the last frame
                self.sequence_labels.append(y_trial[i + self.sequence_length - 1])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.sequence_labels[idx]
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# --- 2. TCN Model Architecture ---
class TCN(nn.Module):
    def __init__(self, num_inputs, num_classes, num_channels=[64, 128], kernel_size=3, dropout=0.5):
        super(TCN, self).__init__()
        layers = []
        for i, n_out in enumerate(num_channels):
            dilation_size = 2 ** i
            n_in = num_inputs if i == 0 else num_channels[i-1]
            layers += [
                nn.Conv1d(n_in, n_out, kernel_size, padding=(kernel_size-1) * dilation_size, dilation=dilation_size),
                nn.BatchNorm1d(n_out),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        # PyTorch Conv1d expects (N, C, L), so we permute (Batch, Seq_len, Features) -> (Batch, Features, Seq_len)
        x = x.permute(0, 2, 1)
        out = self.network(x)
        out = self.pool(out).squeeze(-1)
        return self.linear(out)

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_output):
        # lstm_output shape: (Batch, Seq_len, Hidden_size)
        # Calculate attention scores
        attn_weights = self.attention(lstm_output).squeeze(2)
        # Apply softmax to get a probability distribution
        soft_attn_weights = F.softmax(attn_weights, 1)
        
        # Multiply the output by the attention weights to get a weighted sum
        # (Batch, 1, Seq_len) x (Batch, Seq_len, Hidden_size) -> (Batch, 1, Hidden_size)
        context = torch.bmm(soft_attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        
        return context

class GRU(nn.Module):
    def __init__(self, num_inputs, num_classes, hidden_size=128, num_layers=2, dropout=0.5):
        super(GRU, self).__init__()
        self.gru = nn.GRU(
            input_size=num_inputs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, # Crucial for (N, L, C) input shape
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.linear = nn.Linear(hidden_size*2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (Batch, Seq_len, Features)
        # gru_out shape: (Batch, Seq_len, Hidden_size)
        # h_n shape: (Num_layers, Batch, Hidden_size)
        gru_out, _ = self.gru(x)
        
        # We only need the output from the last time step for classification
        last_time_step_out = gru_out[:, -1, :]
        
        out = self.dropout(last_time_step_out)
        return self.linear(out)

class LSTM(nn.Module):
    def __init__(self, num_inputs, num_classes, hidden_size=128, num_layers=2, dropout=0.5):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=num_inputs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, # Crucial for (N, L, C) input shape
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.linear = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (Batch, Seq_len, Features)
        # lstm_out shape: (Batch, Seq_len, Hidden_size)
        # (h_n, c_n) are the final hidden and cell states
        lstm_out, _ = self.lstm(x)
        
        # We only need the output from the last time step
        last_time_step_out = lstm_out[:, -1, :]
        
        out = self.dropout(last_time_step_out)
        return self.linear(out)

class LSTMwAtt(nn.Module):
    def __init__(self, num_inputs, num_classes, hidden_size=128, num_layers=2, dropout=0.5):
        super(LSTMwAtt, self).__init__()
        self.lstm = nn.LSTM(
            input_size=num_inputs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True # Attention works well with bidirectional
        )
        self.attention = Attention(hidden_size * 2) # Use hidden_size * 2 for bidirectional
        self.linear = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # Pass the entire sequence output to the attention layer
        context = self.attention(lstm_out)
        
        out = self.dropout(context)
        return self.linear(out)
    
class WeightedF1Loss(nn.Module):
    def __init__(self, weights):
        super(WeightedF1Loss, self).__init__()
        self.weights = weights

    def forward(self, y_hat, y):
        # y_hat shape: (Batch, Num_Classes)
        # y shape: (Batch)
        
        # Apply softmax to get probabilities
        y_hat = F.softmax(y_hat, dim=1)
        
        # Convert target y to one-hot encoding
        y_one_hot = F.one_hot(y, num_classes=y_hat.shape[1]).float()
        
        tp = torch.sum(y_hat * y_one_hot, dim=0)
        fp = torch.sum(y_hat * (1 - y_one_hot), dim=0)
        fn = torch.sum((1 - y_hat) * y_one_hot, dim=0)

        soft_f1 = 2*tp / (2*tp + fp + fn + 1e-16)
        cost = 1 - soft_f1
        
        # Apply weights
        weighted_cost = cost * self.weights.to(y_hat.device)
        
        return torch.mean(weighted_cost)


# --- 3. Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    SEQUENCE_LENGTH = 100
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.00025
    MODEL_TYPE = 'lstm' # Options: 'tcn', 'gru', 'lstm', 'lstm_att'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOSS_FUNCTION = 'weighted_ce' # Options: 'weighted_ce', 'weighted_f1'
    USE_OVERSAMPLING = True # Set to True to enable oversampling
    
    # --- NEW: Feature type configuration ---
    FEATURE_TYPE = 'raw'  # Options: 'raw' or 'engineered'

    print(f"Using device: {DEVICE}")
    print(f"Using model type: {MODEL_TYPE}")
    print(f"Using loss function: {LOSS_FUNCTION}")
    print(f"Oversampling enabled: {USE_OVERSAMPLING}")

    # --- Data Loading and Preparation ---
    feature_dir = Path("/home/tarislada/Documents/Extra_python_projects/SKH FP/FInalized_process/test_val_vid5")
    label_dir = Path("/home/tarislada/Documents/Extra_python_projects/SKH FP/FInalized_process/Behavior_label")
    keypoint_dir = Path("SKH_FP/savgol_pose_w59p7")
    cricket_dir = Path("SKH_FP/FInalized_process/cricket_process_test5")

    if FEATURE_TYPE == 'raw':
        print("\n" + "="*60)
        print("LOADING RAW POSE FEATURES")
        print("="*60)
        
        # Load raw data
        raw_data_loader = RawDataLoader()

        # Find matching files
        keypoint_files, cricket_files, label_files = raw_data_loader.find_matching_files(
            keypoint_dir, cricket_dir, label_dir
        )
        
        all_data = raw_data_loader.load_raw_data(keypoint_files, cricket_files, label_files, batch_size=5)
        X, y, groups, feature_names = raw_data_loader.prepare_data(all_data)
        
        # Store the encoder for later use
        target_encoder = raw_data_loader.target_encoder
        
        # Print some key features to verify we have the raw pose data
        print(f"First 10 feature names: {feature_names[:10]}...")
        print(f"Total raw features: {len(feature_names)}")
    
    elif FEATURE_TYPE == 'engineered':
        print("\n" + "="*60)
        print("LOADING ENGINEERED FEATURES")
        print("="*60)
        
        # Find and pair files
        feature_files, label_files = [], []
        for feature_path in sorted(feature_dir.glob('*_analysis.csv')):
            base_name = feature_path.stem.replace('_validated', '').replace('_analysis', '')
            label_path = label_dir / f"{base_name}_processed_labels.csv"
            if label_path.exists():
                feature_files.append(str(feature_path))
                label_files.append(str(label_path))

        # Instantiate data processing tools
        # Note: We don't need lag features for the sequence model
        feature_engineer = MemoryOptimizedFeatureEngineer(sequential_lags=None)
        data_preparator = MemoryOptimizedDataPreparator()

        # Load and process data
        all_data = data_preparator.process_files_in_batches(feature_files, label_files, batch_size=5)
        X, y, groups = data_preparator.prepare_data(all_data, feature_engineer)
        
        # Store the encoder for later use
        target_encoder = data_preparator.target_encoder
        
        print(f"Total engineered features: {X.shape[1]}")
    
    else:
        raise ValueError("FEATURE_TYPE must be 'raw' or 'engineered'")
    
    print(f"Total samples: {len(X)}")
    print(f"Class distribution: {np.unique(y, return_counts=True)}")    
    # --- Create Datasets and DataLoaders ---
    # Split data by trials to prevent data leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups))

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    groups_train, groups_val = groups.iloc[train_idx], groups.iloc[val_idx]

    train_dataset = HuntingSequenceDataset(X_train.values, y_train, groups_train.values, SEQUENCE_LENGTH)
    val_dataset = HuntingSequenceDataset(X_val.values, y_val, groups_val.values, SEQUENCE_LENGTH)

    train_sampler = None
    shuffle_train = True
    if USE_OVERSAMPLING:
        print("Implementing oversampling for training data...")
        # Get labels for each sequence in the training set
        train_labels = np.array(train_dataset.sequence_labels)
        
        # Calculate weights for each class
        class_sample_count = np.array([len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in train_labels])
        samples_weight = torch.from_numpy(samples_weight).double()
        
        # Create the sampler
        train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        shuffle_train = False # Sampler and shuffle are mutually exclusive

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Number of training sequences: {len(train_dataset)}")
    print(f"Number of validation sequences: {len(val_dataset)}")

    # --- Model Training ---
    num_features = X.shape[1]
    num_classes = len(np.unique(y))
    if MODEL_TYPE == 'tcn':
        model = TCN(num_inputs=num_features, num_classes=num_classes).to(DEVICE)
    elif MODEL_TYPE == 'gru':
        model = GRU(num_inputs=num_features, num_classes=num_classes).to(DEVICE)
    elif MODEL_TYPE == 'lstm':
        model = LSTM(num_inputs=num_features, num_classes=num_classes).to(DEVICE)
    elif MODEL_TYPE == 'lstm_att':
        model = LSTMwAtt(num_inputs=num_features, num_classes=num_classes).to(DEVICE)
    else:
        raise ValueError("Invalid MODEL_TYPE. Choose 'tcn', 'gru', 'lstm', or 'lstm_att'.")
    print(model)
    
    # train_sequence_labels = np.array(train_dataset.sequence_labels)
    # class_weights = compute_class_weight('balanced', classes=np.unique(train_sequence_labels), y=train_sequence_labels)
    # class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    # class_to_idx = {cls: i for i, cls in enumerate(raw_data_loader.target_encoder.classes_)}
    class_to_idx = {cls: i for i, cls in enumerate(target_encoder.classes_)} 
    
    # Define custom weights. Higher values force the model to focus more on that class.
    # We are heavily penalizing mistakes on 'attack' and 'non_visual_rotation'.
    manual_weights = {
        'background': 0.2,
        'chasing': 1.0,
        'attack': 1.1,
        'non_visual_rotation': 1.1
    }
    
    # Create the weight tensor in the correct order based on the encoder's mapping
    class_weights = torch.zeros(num_classes)
    for class_name, weight in manual_weights.items():
        if class_name in class_to_idx:
            class_weights[class_to_idx[class_name]] = weight
    
    class_weights_tensor = class_weights.to(DEVICE)
    print(f"Using Manual Class Weights: {class_weights_tensor.cpu().numpy()}")
    print(f"Corresponding to classes: {target_encoder.classes_}")  # CHANGED
        
    if LOSS_FUNCTION == 'weighted_ce':
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    elif LOSS_FUNCTION == 'weighted_f1':
        criterion = WeightedF1Loss(weights=class_weights_tensor)
    elif LOSS_FUNCTION == 'focal_loss':
        try:
            from torchvision.ops import sigmoid_focal_loss
            criterion = sigmoid_focal_loss(num_classes=num_classes, alpha=class_weights_tensor, gamma=2.0)
        except ImportError:
            raise ImportError("torchmetrics is not installed. Please install it to use FocalLoss.")
    else:
        raise ValueError("Invalid LOSS_FUNCTION. Choose 'weighted_ce' or 'weighted_f1'.")
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = EPOCHS * len(train_loader)
    
    # OneCycleLR typically uses a max_lr that's 10-20x the base learning rate
    max_lr = LEARNING_RATE * 10
    
    # Create the OneCycleLR scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=0.3,  # Spend 30% of training time in the increasing phase
        anneal_strategy='cos',  # Use cosine annealing
        div_factor=10,  # Initial learning rate will be max_lr/25
        final_div_factor=25  # Final learning rate will be max_lr/10000
    )

    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)
    
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
                outputs = model(sequences)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        # class_names = raw_data_loader.target_encoder.classes_
        class_names = target_encoder.classes_  # CHANGED

        try:
            # Get the index for the 'attack' class to calculate its specific F1 score
            attack_idx = list(class_names).index('attack')
            attack_f1 = f1_score(all_labels, all_preds, labels=[attack_idx], average='micro', zero_division=0)
        except ValueError:
            attack_f1 = 0.0 # Handle case where 'attack' might not be in the validation batch

        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        print(f"Epoch {epoch+1}/{EPOCHS} -> Val Loss: {val_loss:.4f}| Val Acc: {val_acc:.4f} | Macro F1: {macro_f1:.4f} | Attack F1: {attack_f1:.4f}")
        # scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = f'best_{MODEL_TYPE}_{FEATURE_TYPE}_model.pth'  # CHANGED
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 25: # Increased patience
                print("Early stopping.")
                break
    
    # --- Final Evaluation ---
    print("\n" + "="*60)
    print("FINAL MODEL EVALUATION ON VALIDATION SET")
    print("="*60)
    
    # Load the best model
    model_save_path = f'best_{MODEL_TYPE}_{FEATURE_TYPE}_model.pth'
    print(f"Loading best model from: {model_save_path}")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    # Re-run validation with the best model
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Print detailed evaluation
    class_names = target_encoder.classes_
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    
    # Calculate and print macro metrics
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print(f"\nMacro F1 Score: {macro_f1:.4f}")
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title(f'Confusion Matrix - {MODEL_TYPE.upper()} ({FEATURE_TYPE.capitalize()} Features)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    confusion_matrix_path = f'{MODEL_TYPE}_{FEATURE_TYPE}_confusion_matrix.png'
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nConfusion matrix saved to: {confusion_matrix_path}")
    
    print("\n" + "="*60)