"""
Evidential Deep Learning Model Testing and Uncertainty Analysis

This module provides comprehensive testing and visualization capabilities for
evidential deep learning models, including uncertainty quantification and
spatial analysis across multiple datasets.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

# Add project paths
project_root = Path('/Users/evhoxha/projects/impact_echo_uncertainty_estimation')
sys.path.insert(0, str(project_root))
for subdir in ['dataloaders', 'models', 'data', 'weights']:
    sys.path.insert(0, str(project_root / subdir))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloaders.dataloader import ImpactEchoDatasetClassifier
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from utils.utils import *

import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

from models.evidential_model import create_model
from losses.evidential_loss import dirichlet_kl_divergence, evidential_loss
from training.trainer import evaluate_evidential_classifier

# Configuration constants
DEFAULT_EXPERIMENT_NAME = "evidential_transformer"
DEFAULT_MODEL_NAME = "evidential_transformer"

# Dataset spatial configurations
DATASET_SHAPES = {
    252: (9, 28),      # DS1 and DS3 slabs
    1178: (31, 38),    # CCNY May 2022
    646: (19, 34),     # CCNY June 2022
    1496: (44, 34),    # CCNY Nov 2023
}


def load_model_checkpoint(model_path, device):
    """
    Load model from checkpoint or state_dict.
    
    Args:
        model_path: Path to model file
        device: Target device for model
        
    Returns:
        Loaded model in eval mode
    """
    model = create_model().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"Loaded model from checkpoint (epoch {epoch})")
    else:
        model.load_state_dict(checkpoint)
        print(f"Loaded model from state dict")
    
    model.eval()
    return model


def load_ds3_multi_slab_data_into_torch_tensor(device, X_path='data/X_overlayed_860.npy', 
                                               y_path='data/y_overlayed.npy'):
    """
    Load DS3 data and split into 4 separate slabs.
    
    DS3 has 1008 samples = 4 slabs × 252 samples each.
    Each slab has the same spatial arrangement as DS1: 252 samples = 9×28 grid.
    
    Args:
        device: Target device for tensors
        X_path: Path to features
        y_path: Path to labels
        
    Returns:
        Dictionary of slab data with keys 'DS3-1' through 'DS3-4'
    """
    print("Loading DS3 multi-slab data...")
    
    X_data = np.load(X_path)
    y_data = np.flip(np.load(y_path)) if y_path else None

    if y_data is not None:
        y_data = np.where(y_data > 0, 1, 0)

    print(f"Full DS3 data shape: X={X_data.shape}, y={y_data.shape if y_data is not None else 'None'}")
    
    total_samples = len(X_data)
    expected_total = 1008
    samples_per_slab = 252
    
    if total_samples != expected_total:
        print(f"Warning: Expected {expected_total} samples but found {total_samples}")
        samples_per_slab = total_samples // 4
        print(f"Using {samples_per_slab} samples per slab instead")
    
    X_tensor = torch.tensor(X_data, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(np.copy(np.flip(np.load(y_path))), dtype=torch.long).to(device) if y_data is not None else None
    
    slabs = {}
    for i in range(4):
        start_idx = i * samples_per_slab
        end_idx = min((i + 1) * samples_per_slab, total_samples)
        
        slab_name = f"DS3-{i+1}"
        slabs[slab_name] = {
            'X': X_tensor[start_idx:end_idx],
            'y': y_tensor[start_idx:end_idx] if y_tensor is not None else None,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'samples': end_idx - start_idx
        }
        
        print(f"{slab_name}: samples {start_idx}-{end_idx-1} ({end_idx - start_idx} total)")
    
    return slabs


def ensure_binary_predictions(predictions):
    """Convert predictions to binary (0, 1) format."""
    pred_array = predictions.numpy() if hasattr(predictions, 'numpy') else predictions
    return np.where(pred_array > 0, 1, 0)


def process_ds3_slab_with_dataloader(X_data, y_data, model, device, slab_name):
    """
    Process DS3 slab using DataLoader for consistent tensor handling.
    
    Args:
        X_data: Input features
        y_data: Labels
        model: Model to use for inference
        device: Target device
        slab_name: Name of the slab for logging
        
    Returns:
        Tuple of (accuracy, predictions, total_unc, epistemic_unc, 
                 aleatoric_unc, confidences, targets, alphas)
    """
    print(f"Processing {slab_name} using DataLoader approach...")
    
    class TempDS3Dataset(torch.utils.data.Dataset):
        def __init__(self, X_data, y_data):
            self.X_data = X_data.cpu().numpy()
            self.y_data = y_data.cpu().numpy() if y_data is not None else None
            if self.y_data is not None:
                self.y_data = np.where(self.y_data > 0, 1, 0)
            
        def __len__(self):
            return len(self.X_data)
            
        def __getitem__(self, idx):
            X = torch.tensor(self.X_data[idx], dtype=torch.float32)
            y = torch.tensor(self.y_data[idx], dtype=torch.long) if self.y_data is not None else torch.tensor(0, dtype=torch.long)
            return X, y
    
    temp_dataset = TempDS3Dataset(X_data, y_data)
    temp_loader = DataLoader(dataset=temp_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    return evaluate_evidential_classifier(model, temp_loader, device)


def process_ds3_slab_in_batches(X_data, y_data, model, device, slab_name, batch_size=32):
    """
    Process DS3 slab in batches to manage memory efficiently.
    
    Args:
        X_data: Input features
        y_data: Labels
        model: Model for inference
        device: Target device
        slab_name: Name for logging
        batch_size: Batch size for processing
        
    Returns:
        Tuple of inference results
    """
    print(f"Processing {slab_name} in batches of {batch_size}...")
    
    n_samples = len(X_data)
    all_predictions = []
    all_epistemic = []
    all_aleatoric = []
    all_total_unc = []
    all_confidences = []
    all_alphas = []
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_X = X_data[i:end_idx]
            batch_y = y_data[i:end_idx] if y_data is not None else None
            
            batch_X = batch_X.view(batch_X.size(0), 1, batch_X.size(1))
            
            prob, epistemic, aleatoric, total_unc, confidence, alpha_sum = model.predict_with_uncertainty(batch_X)
            
            if batch_y is not None:
                predicted = torch.argmax(prob.squeeze(0), dim=1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
            
            all_predictions.append(prob)
            all_epistemic.append(epistemic)
            all_aleatoric.append(aleatoric)
            all_total_unc.append(total_unc)
            all_confidences.append(confidence)
            all_alphas.append(alpha_sum)
    
    predictions = torch.cat(all_predictions, dim=1)
    epistemic_unc = torch.cat(all_epistemic, dim=1)
    aleatoric_unc = torch.cat(all_aleatoric, dim=1)
    total_uncertainties = torch.cat(all_total_unc, dim=1)
    alphas = torch.cat(all_alphas, dim=1)
    confidences = torch.cat(all_confidences, dim=0)
    
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    targets = y_data if y_data is not None else torch.zeros(n_samples, dtype=torch.long)
    
    return accuracy, predictions, total_uncertainties, epistemic_unc, aleatoric_unc, confidences, targets, alphas


def test_full_evidential_model_on_datasets_single_batch(model_path):
    """
    Test model on all datasets, processing DS3 slabs as single batches.
    
    Args:
        model_path: Path to model checkpoint
        
    Returns:
        Dictionary of results for each dataset
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model_checkpoint(model_path, device)
    
    datasets = [
        {'name': 'DS1 Test', 'X_path': 'data/X_test_860.npy', 'y_path': 'data/y_test.npy'},
        {'name': 'DS3-1', 'X_path': 'data/X_overlayed_860.npy', 'y_path': 'data/y_overlayed.npy', 'slab_idx': 0},
        {'name': 'DS3-2', 'X_path': 'data/X_overlayed_860.npy', 'y_path': 'data/y_overlayed.npy', 'slab_idx': 1},
        {'name': 'DS3-3', 'X_path': 'data/X_overlayed_860.npy', 'y_path': 'data/y_overlayed.npy', 'slab_idx': 2},
        {'name': 'DS3-4', 'X_path': 'data/X_overlayed_860.npy', 'y_path': 'data/y_overlayed.npy', 'slab_idx': 3},
        {'name': 'CCNY May 2022', 'X_path': 'data/X_our_slab_size860.npy', 'y_path': None},
        {'name': 'CCNY June 2022', 'X_path': 'data/X_our_slab_size860.npy', 'y_path': None},
        {'name': 'CCNY Nov 2023', 'X_path': 'data/nov2023_non_resampled.npy', 'y_path': None},
    ]
    
    results = {}
    ds3_slabs = None
    
    for dataset_info in datasets:
        dataset_name = dataset_info['name']
        X_path = dataset_info['X_path']
        y_path = dataset_info['y_path']
        
        print(f"\n=== Testing on {dataset_name} ===")
        
        try:
            if 'DS3-' in dataset_name:
                slab_idx = dataset_info['slab_idx']
                
                if ds3_slabs is None:
                    ds3_slabs = load_ds3_multi_slab_data_into_torch_tensor(device, X_path, y_path)
                
                slab_name = f"DS3-{slab_idx+1}"
                slab_data = ds3_slabs[slab_name]
                X_data = slab_data['X']
                y_data = slab_data['y']
                
                print(f"Testing DS3 slab {slab_idx+1}: {slab_data['samples']} samples (single batch)")
                
                X_data = X_data.view(X_data.size(0), 1, X_data.size(1))
                
                with torch.no_grad():
                    prob, epistemic, aleatoric, total_unc, confidence, alpha_sum = model.predict_with_uncertainty(X_data)
                
                if y_data is not None:
                    predicted = torch.argmax(prob.squeeze(0), dim=1)
                    accuracy = (predicted == y_data).float().mean().item() * 100
                    targets = y_data
                else:
                    accuracy = 0.0
                    targets = torch.zeros(len(X_data), dtype=torch.long)
                
                predictions = prob
                epistemic_unc = epistemic
                aleatoric_unc = aleatoric
                alphas = alpha_sum
                confidences = confidence.unsqueeze(0) if confidence.dim() == 1 else confidence
                
            elif y_path is not None:
                test_dataset = ImpactEchoDatasetClassifier([X_path], y_path=[y_path], array_size=860)
                test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=2)
                print(f"Testing supervised dataset on {len(test_dataset)} samples...")
                
                (accuracy, predictions, total_unc, epistemic_unc, 
                 aleatoric_unc, confidences, targets, alphas) = evaluate_evidential_classifier(model, test_loader, device)
                
            else:
                print(f"Testing unsupervised dataset: {dataset_name}")
                
                if 'May' in dataset_name or 'June' in dataset_name:
                    X_may, X_june = load_ccny_sep2022_data_into_torch_tensor(device, X_path)
                    X_data = X_may if 'May' in dataset_name else X_june
                    print(f"Testing CCNY {dataset_name.split()[-2]} data: {len(X_data)} samples")
                elif 'Nov' in dataset_name:
                    X_data = load_ccny_nov2023_data_into_torch_tensor2(device=device, X_path=X_path)
                    print(f"Testing CCNY Nov 2023 data: {len(X_data)} samples")
                else:
                    print(f"Skipping unknown unsupervised dataset: {dataset_name}")
                    continue
                
                with torch.no_grad():
                    prob, epistemic, aleatoric, total_unc, confidence, alpha_sum = model.predict_with_uncertainty(X_data)
                
                targets = torch.zeros(len(X_data), dtype=torch.long)
                accuracy = 0.0
                predictions = prob
                epistemic_unc = epistemic
                aleatoric_unc = aleatoric
                alphas = alpha_sum
                confidences = confidence.unsqueeze(0) if confidence.dim() == 1 else confidence
            
            if y_path is not None or 'DS3-' in dataset_name:
                print(f"Accuracy: {accuracy:.2f}%")
            else:
                print("Unsupervised dataset - no accuracy calculated")
            
            print(f"Mean Total Uncertainty: {total_unc.mean():.6f}")
            print(f"Mean Epistemic Uncertainty: {epistemic_unc.mean():.6f}")
            print(f"Mean Aleatoric Uncertainty: {aleatoric_unc.mean():.6f}")
            print(f"Mean Confidence: {confidences.mean():.6f}")
            print(f"Mean Evidence Strength: {alphas.mean():.6f}")
            
            results[dataset_name] = {
                'accuracy': accuracy,
                'predictions': predictions,
                'total_uncertainties': total_unc,
                'epistemic_uncertainties': epistemic_unc,
                'aleatoric_uncertainties': aleatoric_unc,
                'confidences': confidences,
                'targets': targets,
                'alphas': alphas
            }
            
        except FileNotFoundError as e:
            print(f"Dataset not found for {dataset_name}: {e}")
            continue
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


def calculate_detailed_accuracy_metrics(pred_classes, targets, class_names=None):
    """
    Calculate comprehensive accuracy metrics.
    
    Args:
        pred_classes: Predicted class labels
        targets: True class labels
        class_names: Optional class names for display
        
    Returns:
        Dictionary of metrics including TP, FP, TN, FN, precision, recall, F1, etc.
    """
    if class_names is None:
        class_names = ['No Defect', 'Defect']
    
    if hasattr(pred_classes, 'cpu'):
        pred_classes = pred_classes.cpu().numpy()
    if hasattr(targets, 'cpu'):
        targets = targets.cpu().numpy()
    
    pred_classes = pred_classes.flatten()
    targets = targets.flatten()
    
    targets = np.where(targets > 0, 1, 0)
    valid_mask = targets >= 0
    pred_classes = pred_classes[valid_mask]
    targets = targets[valid_mask]

    if len(targets) == 0:
        print("Warning: No valid targets found for accuracy calculation")
        return None
    
    print(f"\n=== Detailed Defect Detection Accuracy Metrics ===")
    print(f"Total samples evaluated: {len(targets)}")
    print(f"Class distribution: {class_names[0]}: {np.sum(targets == 0)}, {class_names[1]}: {np.sum(targets == 1)}")
    
    cm = confusion_matrix(targets, pred_classes)
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              {class_names[0]:<12} {class_names[1]:<12}")
    print(f"Actual {class_names[0]:<8} {cm[0,0]:<12} {cm[0,1]:<12}")
    print(f"       {class_names[1]:<8} {cm[1,0]:<12} {cm[1,1]:<12}")
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        if len(np.unique(targets)) == 1:
            if targets[0] == 0:
                tn = np.sum((targets == 0) & (pred_classes == 0))
                fp = np.sum((targets == 0) & (pred_classes == 1))
                fn, tp = 0, 0
            else:
                tp = np.sum((targets == 1) & (pred_classes == 1))
                fn = np.sum((targets == 1) & (pred_classes == 0))
                tn, fp = 0, 0
        else:
            print("Warning: Unexpected confusion matrix shape")
            return None
    
    print(f"\n=== Binary Classification Metrics ===")
    print(f"True Positives (TP):   {tp}")
    print(f"True Negatives (TN):   {tn}")
    print(f"False Positives (FP):  {fp}")
    print(f"False Negatives (FN):  {fn}")
    
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print(f"\n=== Performance Metrics ===")
    print(f"Overall Accuracy:      {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision (PPV):       {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall (Sensitivity):  {recall:.4f} ({recall*100:.2f}%)")
    print(f"Specificity:           {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"Negative Pred. Value:  {npv:.4f} ({npv*100:.2f}%)")
    print(f"F1 Score:              {f1_score:.4f}")
    
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    print(f"\n=== Error Rates ===")
    print(f"False Positive Rate:   {false_positive_rate:.4f} ({false_positive_rate*100:.2f}%)")
    print(f"False Negative Rate:   {false_negative_rate:.4f} ({false_negative_rate*100:.2f}%)")
    
    if np.sum(targets == 0) > 0:
        no_defect_accuracy = np.sum((targets == 0) & (pred_classes == 0)) / np.sum(targets == 0)
        print(f"\n=== Class-Specific Accuracy ===")
        print(f"{class_names[0]} Detection Accuracy: {no_defect_accuracy:.4f} ({no_defect_accuracy*100:.2f}%)")
    
    if np.sum(targets == 1) > 0:
        defect_accuracy = np.sum((targets == 1) & (pred_classes == 1)) / np.sum(targets == 1)
        if 'no_defect_accuracy' not in locals():
            print(f"\n=== Class-Specific Accuracy ===")
        print(f"{class_names[1]} Detection Accuracy: {defect_accuracy:.4f} ({defect_accuracy*100:.2f}%)")
    
    balanced_accuracy = (recall + specificity) / 2
    print(f"\nBalanced Accuracy:     {balanced_accuracy:.4f} ({balanced_accuracy*100:.2f}%)")
    
    try:
        precision_sk, recall_sk, f1_sk, support = precision_recall_fscore_support(
            targets, pred_classes, average='binary', zero_division=0
        )
        print(f"\n=== Sklearn Verification ===")
        print(f"Sklearn Precision:     {precision_sk:.4f}")
        print(f"Sklearn Recall:        {recall_sk:.4f}")
        print(f"Sklearn F1:            {f1_sk:.4f}")
        
        print(f"\n=== Detailed Classification Report ===")
        report = classification_report(targets, pred_classes, target_names=class_names, zero_division=0)
        print(report)
        
    except Exception as e:
        print(f"Warning: Could not generate sklearn verification: {e}")
    
    return {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'npv': npv,
        'f1_score': f1_score,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate,
        'balanced_accuracy': balanced_accuracy,
        'confusion_matrix': cm
    }


def get_spatial_shape(n_samples, dataset_name):
    """
    Determine the spatial shape for a dataset.
    
    Args:
        n_samples: Number of samples
        dataset_name: Name of the dataset
        
    Returns:
        Tuple of (rows, cols) for spatial arrangement
    """
    if n_samples in DATASET_SHAPES:
        return DATASET_SHAPES[n_samples]
    
    # Pattern matching
    name_upper = dataset_name.upper()
    if 'DS1' in name_upper or 'TEST' in name_upper:
        return (9, 28)
    elif 'MAY' in name_upper:
        return (31, 38)
    elif 'JUNE' in name_upper:
        return (19, 34)
    elif 'NOV' in name_upper:
        return (44, 34)
    elif 'OVERLAY' in name_upper or 'DS3' in name_upper:
        return (9, 28)
    
    # Fallback to rectangular arrangement
    factors = [(i, n_samples // i) for i in range(1, int(np.sqrt(n_samples)) + 1) if n_samples % i == 0]
    if factors:
        return min(factors, key=lambda x: abs(x[0] - x[1]))
    
    grid_size = int(np.sqrt(n_samples))
    return (grid_size, grid_size)


def pad_or_truncate_to_shape(data_arrays, shape):
    """
    Pad or truncate data arrays to match expected spatial shape.
    
    Args:
        data_arrays: List of numpy arrays to adjust
        shape: Target (rows, cols) shape
        
    Returns:
        List of adjusted arrays
    """
    expected_samples = shape[0] * shape[1]
    n_samples = len(data_arrays[0])
    
    if n_samples == expected_samples:
        return data_arrays
    
    adjusted = []
    for arr in data_arrays:
        if n_samples < expected_samples:
            pad_size = expected_samples - n_samples
            arr = np.pad(arr, (0, pad_size), mode='constant', constant_values=np.nan)
        else:
            arr = arr[:expected_samples]
        adjusted.append(arr)
    
    return adjusted


