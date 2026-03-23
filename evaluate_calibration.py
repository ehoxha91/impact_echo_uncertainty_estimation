"""
Calibration analysis for all three UQ methods.
Computes ECE, Brier score, and generates reliability diagrams.
"""
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataloaders.dataloader import ImpactEchoDatasetClassifier
from models.evidential_model import create_model
from models.standard_model import create_standard_model
from models.mc_dropout import mc_dropout_predict
from models.deep_ensemble import load_ensemble, ensemble_predict


def compute_ece(confidences, accuracies_per_sample, n_bins=10):
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            bin_accs.append(0)
            bin_confs.append((lo + hi) / 2)
            bin_counts.append(0)
            continue
        bin_acc = accuracies_per_sample[mask].mean()
        bin_conf = confidences[mask].mean()
        bin_count = mask.sum()
        ece += (bin_count / len(confidences)) * abs(bin_acc - bin_conf)
        bin_accs.append(bin_acc)
        bin_confs.append(bin_conf)
        bin_counts.append(bin_count)

    return ece, np.array(bin_accs), np.array(bin_confs), np.array(bin_counts)


def compute_brier_score(probs, targets):
    """Compute Brier score (lower is better)."""
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(targets)), targets] = 1
    return np.mean(np.sum((probs - one_hot) ** 2, axis=1))


def get_predictions_evidential(model, X, device):
    """Get probabilities and predictions from evidential model."""
    model.eval()
    with torch.no_grad():
        prob, _, _, _, confidence, _ = model.predict_with_uncertainty(X)
    prob = prob.squeeze(0) if prob.dim() == 3 else prob
    return prob.cpu().numpy(), confidence.cpu().numpy()


def get_predictions_mc_dropout(model, X, T=50):
    """Get probabilities and predictions from MC Dropout."""
    mean_prob, _, _, _, confidence = mc_dropout_predict(model, X, T=T)
    return mean_prob.cpu().numpy(), confidence.cpu().numpy()


def get_predictions_ensemble(models, X):
    """Get probabilities and predictions from Deep Ensemble."""
    mean_prob, _, _, _, confidence = ensemble_predict(models, X)
    return mean_prob.cpu().numpy(), confidence.cpu().numpy()


def plot_reliability_diagrams(results_dict, dataset_name, save_path):
    """Plot reliability diagrams for all methods side by side."""
    n_methods = len(results_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4.5), squeeze=False)

    for idx, (name, res) in enumerate(results_dict.items()):
        ax = axes[0][idx]
        bin_accs = res['bin_accs']
        bin_confs = res['bin_confs']
        bin_counts = res['bin_counts']
        n_bins = len(bin_accs)

        # Bar width
        width = 1.0 / n_bins
        positions = np.linspace(width / 2, 1 - width / 2, n_bins)

        # Accuracy bars
        ax.bar(positions, bin_accs, width=width * 0.85, alpha=0.7,
               color='#2196F3', edgecolor='#1565C0', linewidth=0.8, label='Accuracy')

        # Gap bars (calibration error)
        gaps = np.array(bin_confs) - np.array(bin_accs)
        gap_colors = ['#FF5252' if g > 0 else '#4CAF50' for g in gaps]
        for i, (pos, gap, bc) in enumerate(zip(positions, gaps, bin_counts)):
            if bc > 0:
                ax.bar(pos, gap, bottom=bin_accs[i], width=width * 0.85,
                       alpha=0.35, color=gap_colors[i], edgecolor='none')

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.2, alpha=0.6, label='Perfect')

        ax.set_xlabel('Confidence', fontsize=11)
        if idx == 0:
            ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title(f'{name}\nECE={res["ece"]:.3f}, Brier={res["brier"]:.3f}', fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Reliability Diagrams — {dataset_name}', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved reliability diagram: {save_path}")


def evaluate_calibration_on_dataset(X, targets_np, device, evidential_model, mc_model, ensemble_models,
                                     dataset_name, save_path, n_bins=10):
    """Evaluate calibration for all three methods on a dataset."""
    results = {}

    methods = {
        'Evidential IENet': lambda: get_predictions_evidential(evidential_model, X, device),
        'MC Dropout (T=50)': lambda: get_predictions_mc_dropout(mc_model, X, T=50),
        'Deep Ensemble (M=5)': lambda: get_predictions_ensemble(ensemble_models, X),
    }

    for name, predict_fn in methods.items():
        probs, confidences = predict_fn()
        pred_classes = probs.argmax(axis=1)
        correct = (pred_classes == targets_np).astype(float)

        ece, bin_accs, bin_confs, bin_counts = compute_ece(confidences, correct, n_bins=n_bins)
        brier = compute_brier_score(probs, targets_np)

        results[name] = {
            'ece': ece,
            'brier': brier,
            'bin_accs': bin_accs,
            'bin_confs': bin_confs,
            'bin_counts': bin_counts,
            'accuracy': correct.mean() * 100,
        }

        print(f"  {name}: ECE={ece:.4f}, Brier={brier:.4f}, Acc={correct.mean()*100:.1f}%")

    plot_reliability_diagrams(results, dataset_name, save_path)
    return results


def print_latex_calibration_table(ds1_results, ds3_results):
    """Print LaTeX table for calibration metrics."""
    print("\n% LaTeX calibration table")
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\caption{Calibration metrics for uncertainty quantification methods. ECE: Expected Calibration Error (lower is better). Brier: Brier Score (lower is better).}")
    print(r"\label{tab:calibration}")
    print(r"\begin{tabular}{l|cc|cc}")
    print(r"\toprule")
    print(r"& \multicolumn{2}{c|}{\textbf{DS1 Test}} & \multicolumn{2}{c}{\textbf{DS3 Overlay}} \\")
    print(r"\textbf{Method} & \textbf{ECE} $\downarrow$ & \textbf{Brier} $\downarrow$ & \textbf{ECE} $\downarrow$ & \textbf{Brier} $\downarrow$ \\")
    print(r"\midrule")

    for name in ds1_results:
        d1 = ds1_results[name]
        d3 = ds3_results[name]
        print(f"{name} & {d1['ece']:.3f} & {d1['brier']:.3f} & {d3['ece']:.3f} & {d3['brier']:.3f} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    print("Loading models...")
    evidential_model = create_model().to(device)
    ckpt = torch.load('weights/evidential_transformer.pth', map_location=device)
    evidential_model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    evidential_model.eval()

    mc_model = create_standard_model().to(device)
    ckpt = torch.load('weights/standard_mc_dropout_best.pth', map_location=device)
    mc_model.load_state_dict(ckpt['model_state_dict'])

    ensemble_paths = [f'weights/ensemble_member_{i}_best.pth' for i in range(5)]
    ensemble_models = load_ensemble(ensemble_paths, device)

    # DS1
    print("\n--- DS1 Test Set Calibration ---")
    ds1 = ImpactEchoDatasetClassifier(['data/X_test_860.npy'], y_path=['data/y_test.npy'], array_size=860)
    X_ds1 = torch.stack([ds1[i][0] for i in range(len(ds1))]).unsqueeze(1).float().to(device)
    y_ds1 = np.array([ds1[i][1] for i in range(len(ds1))])
    ds1_results = evaluate_calibration_on_dataset(
        X_ds1, y_ds1, device, evidential_model, mc_model, ensemble_models,
        "DS1 Test", "figures/reliability_diagram_ds1.pdf"
    )

    # DS3
    print("\n--- DS3 Overlay Calibration ---")
    ds3 = ImpactEchoDatasetClassifier(['data/X_overlayed_860.npy'], y_path=['data/y_overlayed.npy'], array_size=860)
    X_ds3 = torch.stack([ds3[i][0] for i in range(len(ds3))]).unsqueeze(1).float().to(device)
    y_ds3 = np.array([ds3[i][1] for i in range(len(ds3))])
    ds3_results = evaluate_calibration_on_dataset(
        X_ds3, y_ds3, device, evidential_model, mc_model, ensemble_models,
        "DS3 Overlay (Domain Shift)", "figures/reliability_diagram_ds3.pdf"
    )

    print_latex_calibration_table(ds1_results, ds3_results)

    # Save results
    torch.save({'ds1': ds1_results, 'ds3': ds3_results}, 'weights/calibration_results.pth')
    print("\nResults saved to weights/calibration_results.pth")


if __name__ == '__main__':
    main()
