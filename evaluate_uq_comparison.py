"""
Unified evaluation script comparing three UQ methods:
  1. Evidential Deep Learning (single forward pass)
  2. MC Dropout (T=50 stochastic passes)
  3. Deep Ensemble (M=5 models)

Outputs a comparison table for the paper.
"""
import warnings
warnings.filterwarnings("ignore")

import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from dataloaders.dataloader import ImpactEchoDatasetClassifier
from models.evidential_model import create_model
from models.standard_model import create_standard_model
from models.mc_dropout import mc_dropout_predict
from models.deep_ensemble import load_ensemble, ensemble_predict


def load_evidential_model(path, device):
    model = create_model().to(device)
    checkpoint = torch.load(path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def load_mc_dropout_model(path, device):
    model = create_standard_model().to(device)
    checkpoint = torch.load(path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model


def evaluate_evidential(model, X, device):
    """Evaluate evidential model and return predictions + uncertainty."""
    model.eval()
    with torch.no_grad():
        prob, epistemic, aleatoric, total_unc, confidence, alpha_sum = model.predict_with_uncertainty(X)
    prob = prob.squeeze(0) if prob.dim() == 3 else prob
    total_unc = total_unc.squeeze(0) if total_unc.dim() == 2 else total_unc
    epistemic = epistemic.squeeze(0) if epistemic.dim() == 2 else epistemic
    aleatoric = aleatoric.squeeze(0) if aleatoric.dim() == 2 else aleatoric
    pred_classes = prob.argmax(dim=-1)
    return pred_classes.cpu(), total_unc.cpu(), epistemic.cpu(), aleatoric.cpu(), confidence.cpu()


def evaluate_mc_dropout(model, X, T=50):
    """Evaluate MC Dropout model."""
    mean_prob, total_unc, epistemic, aleatoric, confidence = mc_dropout_predict(model, X, T=T)
    pred_classes = mean_prob.argmax(dim=-1)
    return pred_classes.cpu(), total_unc.cpu(), epistemic.cpu(), aleatoric.cpu(), confidence.cpu()


def evaluate_ensemble(models, X):
    """Evaluate Deep Ensemble."""
    mean_prob, total_unc, epistemic, aleatoric, confidence = ensemble_predict(models, X)
    pred_classes = mean_prob.argmax(dim=-1)
    return pred_classes.cpu(), total_unc.cpu(), epistemic.cpu(), aleatoric.cpu(), confidence.cpu()


def compute_metrics(pred_classes, targets, total_unc):
    """Compute classification and UQ quality metrics."""
    y_true = targets.numpy()
    y_pred = pred_classes.numpy()
    unc = total_unc.numpy()

    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, zero_division=0) * 100
    rec = recall_score(y_true, y_pred, zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100

    # Per-class accuracy
    defect_mask = y_true == 1
    nondef_mask = y_true == 0
    defect_acc = accuracy_score(y_true[defect_mask], y_pred[defect_mask]) * 100 if defect_mask.sum() > 0 else 0
    nondef_acc = accuracy_score(y_true[nondef_mask], y_pred[nondef_mask]) * 100 if nondef_mask.sum() > 0 else 0

    # Misclassification detection AUROC: can uncertainty identify wrong predictions?
    correct = (y_pred == y_true).astype(float)
    misclass_auroc = 0.0
    if len(np.unique(correct)) > 1:
        # Higher uncertainty should correlate with incorrect predictions
        misclass_auroc = roc_auc_score(1 - correct, unc) * 100

    return {
        'accuracy': acc,
        'defect_acc': defect_acc,
        'nondef_acc': nondef_acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'misclass_auroc': misclass_auroc,
        'mean_total_unc': float(unc.mean()),
        'std_total_unc': float(unc.std()),
    }


def measure_inference_time(fn, *args, n_runs=10):
    """Measure average inference time."""
    # Warmup
    fn(*args)
    times = []
    for _ in range(n_runs):
        start = time.time()
        fn(*args)
        times.append(time.time() - start)
    return np.mean(times), np.std(times)


def print_comparison_table(results, dataset_name):
    """Print a formatted comparison table."""
    print(f"\n{'='*90}")
    print(f"UQ Method Comparison - {dataset_name}")
    print(f"{'='*90}")
    header = f"{'Method':<20} {'Acc%':>6} {'Def%':>6} {'NDef%':>6} {'Prec%':>6} {'Rec%':>6} {'F1%':>6} {'AUROC%':>7} {'Time(ms)':>10}"
    print(header)
    print("-" * 90)
    for name, r in results.items():
        time_str = f"{r['time_mean']*1000:.1f}+-{r['time_std']*1000:.1f}"
        print(f"{name:<20} {r['accuracy']:>6.1f} {r['defect_acc']:>6.1f} {r['nondef_acc']:>6.1f} "
              f"{r['precision']:>6.1f} {r['recall']:>6.1f} {r['f1']:>6.1f} {r['misclass_auroc']:>7.1f} {time_str:>10}")
    print("-" * 90)

    # Uncertainty decomposition
    print(f"\n{'Method':<20} {'Total Unc':>15} {'Epistemic':>15} {'Aleatoric':>15}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<20} {r['mean_total_unc']:.4f}+-{r['std_total_unc']:.4f}"
              f"   {r['mean_epi']:.4f}+-{r['std_epi']:.4f}"
              f"   {r['mean_ale']:.4f}+-{r['std_ale']:.4f}")
    print()


def print_latex_table(results, dataset_name):
    """Print LaTeX-formatted table for the paper."""
    print(f"\n% LaTeX table for {dataset_name}")
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(f"\\caption{{UQ Method Comparison on {dataset_name}.}}")
    print(f"\\label{{tab:uq_comparison_{dataset_name.lower().replace(' ', '_')}}}")
    print(r"\begin{tabular}{l|ccc|cc|c|c}")
    print(r"\toprule")
    print(r"\textbf{Method} & \textbf{Overall} & \textbf{Defect} & \textbf{Non-Defect} & \textbf{Precision} & \textbf{Recall} & \textbf{Misclass.} & \textbf{Inference} \\")
    print(r"& \textbf{Acc. (\%)} & \textbf{Acc. (\%)} & \textbf{Acc. (\%)} & \textbf{(\%)} & \textbf{(\%)} & \textbf{AUROC (\%)} & \textbf{Time (ms)} \\")
    print(r"\midrule")
    for name, r in results.items():
        time_ms = r['time_mean'] * 1000
        print(f"{name} & {r['accuracy']:.1f} & {r['defect_acc']:.1f} & {r['nondef_acc']:.1f} "
              f"& {r['precision']:.1f} & {r['recall']:.1f} & {r['misclass_auroc']:.1f} & {time_ms:.1f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")


def evaluate_on_dataset(X_tensor, targets, device, evidential_model, mc_model, ensemble_models, dataset_name):
    """Run all three methods on a dataset and collect results."""
    X = X_tensor.to(device)

    results = {}

    # Evidential
    pred, total_unc, epi, ale, conf = evaluate_evidential(evidential_model, X, device)
    metrics = compute_metrics(pred, targets, total_unc)
    t_mean, t_std = measure_inference_time(evaluate_evidential, evidential_model, X, device)
    metrics['time_mean'] = t_mean
    metrics['time_std'] = t_std
    metrics['mean_epi'] = float(epi.mean())
    metrics['std_epi'] = float(epi.std())
    metrics['mean_ale'] = float(ale.mean())
    metrics['std_ale'] = float(ale.std())
    results['Evidential IENet'] = metrics

    # MC Dropout (T=50)
    pred, total_unc, epi, ale, conf = evaluate_mc_dropout(mc_model, X, T=50)
    metrics = compute_metrics(pred, targets, total_unc)
    t_mean, t_std = measure_inference_time(evaluate_mc_dropout, mc_model, X, 50)
    metrics['time_mean'] = t_mean
    metrics['time_std'] = t_std
    metrics['mean_epi'] = float(epi.mean())
    metrics['std_epi'] = float(epi.std())
    metrics['mean_ale'] = float(ale.mean())
    metrics['std_ale'] = float(ale.std())
    results['MC Dropout (T=50)'] = metrics

    # Deep Ensemble (M=5)
    pred, total_unc, epi, ale, conf = evaluate_ensemble(ensemble_models, X)
    metrics = compute_metrics(pred, targets, total_unc)
    t_mean, t_std = measure_inference_time(evaluate_ensemble, ensemble_models, X)
    metrics['time_mean'] = t_mean
    metrics['time_std'] = t_std
    metrics['mean_epi'] = float(epi.mean())
    metrics['std_epi'] = float(epi.std())
    metrics['mean_ale'] = float(ale.mean())
    metrics['std_ale'] = float(ale.std())
    results['Deep Ensemble (M=5)'] = metrics

    print_comparison_table(results, dataset_name)
    print_latex_table(results, dataset_name)

    return results


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    print("Loading models...")
    evidential_model = load_evidential_model('weights/evidential_transformer.pth', device)
    mc_model = load_mc_dropout_model('weights/standard_mc_dropout_best.pth', device)
    ensemble_paths = [f'weights/ensemble_member_{i}_best.pth' for i in range(5)]
    ensemble_models = load_ensemble(ensemble_paths, device)

    # DS1 Test
    print("\n--- DS1 Test Set ---")
    ds1 = ImpactEchoDatasetClassifier(['data/X_test_860.npy'], y_path=['data/y_test.npy'], array_size=860)
    X_ds1 = torch.stack([ds1[i][0] for i in range(len(ds1))]).unsqueeze(1).float().to(device)
    y_ds1 = torch.tensor([ds1[i][1] for i in range(len(ds1))]).long()
    ds1_results = evaluate_on_dataset(X_ds1, y_ds1, device, evidential_model, mc_model, ensemble_models, "DS1 Test")

    # DS3 (domain shift)
    print("\n--- DS3 (Domain Shift - Overlay) ---")
    ds3 = ImpactEchoDatasetClassifier(['data/X_overlayed_860.npy'], y_path=['data/y_overlayed.npy'], array_size=860)
    X_ds3 = torch.stack([ds3[i][0] for i in range(len(ds3))]).unsqueeze(1).float().to(device)
    y_ds3 = torch.tensor([ds3[i][1] for i in range(len(ds3))]).long()
    ds3_results = evaluate_on_dataset(X_ds3, y_ds3, device, evidential_model, mc_model, ensemble_models, "DS3 Overlay")

    # Save all results
    torch.save({
        'ds1': ds1_results,
        'ds3': ds3_results,
    }, 'weights/uq_comparison_results.pth')
    print("\nResults saved to weights/uq_comparison_results.pth")


if __name__ == '__main__':
    main()
