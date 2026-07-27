"""
Max-Softmax Probability (MSP) baseline for misclassification / OOD detection.
Reference: Hendrycks & Gimpel, "A Baseline for Detecting Misclassified and
Out-of-Distribution Examples in Neural Networks", ICLR 2017.

Uses the same StandardIENet backbone as MC Dropout, but with a single forward
pass in eval() mode (dropout off). Uncertainty = 1 - max_k p_k.

Side-by-side with Evidential, MC Dropout, Deep Ensemble on DS1 + DS3.
"""
import warnings
warnings.filterwarnings("ignore")

import time
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from dataloaders.dataloader import ImpactEchoDatasetClassifier
from models.evidential_model import create_model
from models.standard_model import create_standard_model
from models.mc_dropout import mc_dropout_predict
from models.deep_ensemble import load_ensemble, ensemble_predict


def load_evidential_model(path, device):
    model = create_model().to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()
    return model


def load_standard_model(path, device):
    model = create_standard_model().to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()
    return model


def evaluate_msp(model, X):
    """Single forward pass, softmax, MSP as confidence. unc = 1 - max_k p_k."""
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = F.softmax(logits, dim=-1)
        confidence, pred = probs.max(dim=-1)
        unc = 1.0 - confidence
    return pred.cpu(), unc.cpu(), confidence.cpu(), probs.cpu()


def evaluate_evidential(model, X):
    model.eval()
    with torch.no_grad():
        prob, epistemic, aleatoric, total_unc, confidence, _ = model.predict_with_uncertainty(X)
    prob = prob.squeeze(0) if prob.dim() == 3 else prob
    total_unc = total_unc.squeeze(0) if total_unc.dim() == 2 else total_unc
    pred = prob.argmax(dim=-1)
    return pred.cpu(), total_unc.cpu(), confidence.cpu()


def evaluate_mc_dropout(model, X, T=50):
    mean_prob, total_unc, _, _, confidence = mc_dropout_predict(model, X, T=T)
    pred = mean_prob.argmax(dim=-1)
    return pred.cpu(), total_unc.cpu(), confidence.cpu()


def evaluate_ensemble(models, X):
    mean_prob, total_unc, _, _, confidence = ensemble_predict(models, X)
    pred = mean_prob.argmax(dim=-1)
    return pred.cpu(), total_unc.cpu(), confidence.cpu()


def compute_metrics(pred, targets, unc):
    y_true = targets.numpy()
    y_pred = pred.numpy()
    u = unc.numpy()

    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, zero_division=0) * 100
    rec = recall_score(y_true, y_pred, zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100

    defect_mask = y_true == 1
    nondef_mask = y_true == 0
    defect_acc = accuracy_score(y_true[defect_mask], y_pred[defect_mask]) * 100 if defect_mask.sum() else 0
    nondef_acc = accuracy_score(y_true[nondef_mask], y_pred[nondef_mask]) * 100 if nondef_mask.sum() else 0

    correct = (y_pred == y_true).astype(float)
    auroc = 0.0
    if len(np.unique(correct)) > 1:
        auroc = roc_auc_score(1 - correct, u) * 100

    return {
        'accuracy': acc,
        'defect_acc': defect_acc,
        'nondef_acc': nondef_acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'misclass_auroc': auroc,
        'mean_unc': float(u.mean()),
        'std_unc': float(u.std()),
    }


def measure_time(fn, *args, n_runs=10):
    fn(*args)  # warmup
    ts = []
    for _ in range(n_runs):
        t = time.time()
        fn(*args)
        ts.append(time.time() - t)
    return float(np.mean(ts)), float(np.std(ts))


def run_on_dataset(X, y, dataset_name, models, device):
    ev_model, mc_model, ens_models, std_model = models

    results = {}

    # MSP baseline (standard softmax classifier)
    pred, unc, _, _ = evaluate_msp(std_model, X)
    m = compute_metrics(pred, y, unc)
    t_mean, t_std = measure_time(lambda x: evaluate_msp(std_model, x), X)
    m['time_mean'] = t_mean; m['time_std'] = t_std
    results['Max-Softmax (MSP)'] = m

    # Evidential
    pred, unc, _ = evaluate_evidential(ev_model, X)
    m = compute_metrics(pred, y, unc)
    t_mean, t_std = measure_time(lambda x: evaluate_evidential(ev_model, x), X)
    m['time_mean'] = t_mean; m['time_std'] = t_std
    results['Evidential IENet'] = m

    # MC Dropout
    pred, unc, _ = evaluate_mc_dropout(mc_model, X, T=50)
    m = compute_metrics(pred, y, unc)
    t_mean, t_std = measure_time(lambda x: evaluate_mc_dropout(mc_model, x, 50), X)
    m['time_mean'] = t_mean; m['time_std'] = t_std
    results['MC Dropout (T=50)'] = m

    # Deep Ensemble
    pred, unc, _ = evaluate_ensemble(ens_models, X)
    m = compute_metrics(pred, y, unc)
    t_mean, t_std = measure_time(lambda x: evaluate_ensemble(ens_models, x), X)
    m['time_mean'] = t_mean; m['time_std'] = t_std
    results['Deep Ensemble (M=5)'] = m

    print(f"\n{'='*100}")
    print(f"UQ Method Comparison - {dataset_name}  (MSP baseline added)")
    print(f"{'='*100}")
    header = f"{'Method':<22} {'Acc%':>6} {'Def%':>6} {'NDef%':>6} {'Prec%':>6} {'Rec%':>6} {'F1%':>6} {'AUROC%':>7} {'Time(ms)':>10}"
    print(header)
    print("-" * 100)
    for name, r in results.items():
        tm = f"{r['time_mean']*1000:.1f}+-{r['time_std']*1000:.1f}"
        print(f"{name:<22} {r['accuracy']:>6.1f} {r['defect_acc']:>6.1f} {r['nondef_acc']:>6.1f} "
              f"{r['precision']:>6.1f} {r['recall']:>6.1f} {r['f1']:>6.1f} {r['misclass_auroc']:>7.1f} {tm:>10}")
    print("-" * 100)
    return results


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading models...")
    ev_model  = load_evidential_model('weights/evidential_transformer.pth', device)
    mc_model  = load_standard_model('weights/standard_mc_dropout_best.pth', device)
    std_model = load_standard_model('weights/standard_mc_dropout_best.pth', device)
    ens_paths = [f'weights/ensemble_member_{i}_best.pth' for i in range(5)]
    ens_models = load_ensemble(ens_paths, device)
    models = (ev_model, mc_model, ens_models, std_model)

    # DS1
    print("\n--- DS1 Test Set ---")
    ds1 = ImpactEchoDatasetClassifier(['data/X_test_860.npy'], y_path=['data/y_test.npy'], array_size=860)
    X1 = torch.stack([ds1[i][0] for i in range(len(ds1))]).unsqueeze(1).float().to(device)
    y1 = torch.tensor([ds1[i][1] for i in range(len(ds1))]).long()
    r1 = run_on_dataset(X1, y1, "DS1 Test", models, device)

    # DS3
    print("\n--- DS3 (Domain Shift - Overlay) ---")
    ds3 = ImpactEchoDatasetClassifier(['data/X_overlayed_860.npy'], y_path=['data/y_overlayed.npy'], array_size=860)
    X3 = torch.stack([ds3[i][0] for i in range(len(ds3))]).unsqueeze(1).float().to(device)
    y3 = torch.tensor([ds3[i][1] for i in range(len(ds3))]).long()
    r3 = run_on_dataset(X3, y3, "DS3 Overlay", models, device)

    torch.save({'ds1': r1, 'ds3': r3}, 'weights/uq_comparison_with_msp.pth')
    print("\nSaved to weights/uq_comparison_with_msp.pth")


if __name__ == '__main__':
    main()
