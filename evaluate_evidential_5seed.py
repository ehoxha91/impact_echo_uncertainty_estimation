"""
Evaluate Evidential IENet across 5 seeds on DS1 + DS3.
Reports mean +/- SD on the same metrics as Table 5 (tab:uq_comparison):
  Accuracy, Defect Acc, Non-Defect Acc, Precision, Misclass. AUROC, Time (ms).
"""
import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

from dataloaders.dataloader import ImpactEchoDatasetClassifier
from models.evidential_model import create_model


SEEDS = [42, 123, 456, 789, 1024]
CKPT = "weights/evidential_seed_{s}_best.pth"


def load(path, device):
    model = create_model().to(device)
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck.get("model_state_dict", ck))
    model.eval()
    return model


def evaluate(model, X, y):
    with torch.no_grad():
        prob, _, _, total_unc, _, _ = model.predict_with_uncertainty(X)
    prob = prob.squeeze(0) if prob.dim() == 3 else prob
    total_unc = total_unc.squeeze(0) if total_unc.dim() == 2 else total_unc
    pred = prob.argmax(dim=-1).cpu().numpy()
    unc = total_unc.cpu().numpy().squeeze()
    y_np = y.numpy()

    acc = accuracy_score(y_np, pred) * 100
    prec = precision_score(y_np, pred, zero_division=0) * 100
    def_mask = y_np == 1
    ndef_mask = y_np == 0
    def_acc = accuracy_score(y_np[def_mask], pred[def_mask]) * 100 if def_mask.sum() else 0
    ndef_acc = accuracy_score(y_np[ndef_mask], pred[ndef_mask]) * 100 if ndef_mask.sum() else 0
    correct = (pred == y_np).astype(float)
    auroc = roc_auc_score(1 - correct, unc) * 100 if len(np.unique(correct)) > 1 else 0.0
    return acc, def_acc, ndef_acc, prec, auroc


def time_ms(model, X, n=10):
    # warmup
    _ = model.predict_with_uncertainty(X)
    ts = []
    for _ in range(n):
        t0 = time.time()
        _ = model.predict_with_uncertainty(X)
        ts.append((time.time() - t0) * 1000)
    return np.mean(ts), np.std(ts)


def run_dataset(name, X_path, y_path, device):
    ds = ImpactEchoDatasetClassifier([X_path], y_path=[y_path], array_size=860)
    X = torch.stack([ds[i][0] for i in range(len(ds))]).unsqueeze(1).float().to(device)
    y = torch.tensor([ds[i][1] for i in range(len(ds))]).long()

    rows = []
    for s in SEEDS:
        m = load(CKPT.format(s=s), device)
        acc, da, nda, pr, au = evaluate(m, X, y)
        tm, ts = time_ms(m, X)
        rows.append([acc, da, nda, pr, au, tm])
        print(f"  seed={s:>4}  acc={acc:5.2f}  def={da:5.2f}  ndef={nda:5.2f}  "
              f"prec={pr:5.2f}  auroc={au:5.2f}  t={tm:.1f}+-{ts:.1f} ms")
    arr = np.array(rows)
    print(f"\n{name} 5-seed mean +/- SD:")
    labels = ["Acc", "Def", "NDef", "Prec", "AUROC", "Time(ms)"]
    for i, lab in enumerate(labels):
        print(f"  {lab:<9} {arr[:, i].mean():6.2f} +/- {arr[:, i].std():.2f}")
    return arr


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}\n")
    print("--- DS1 Test ---")
    a1 = run_dataset("DS1", "data/X_test_860.npy", "data/y_test.npy", device)
    print("\n--- DS3 Overlay ---")
    a3 = run_dataset("DS3", "data/X_overlayed_860.npy", "data/y_overlayed.npy", device)

    np.savez("weights/evidential_5seed_uq.npz", ds1=a1, ds3=a3)
    print("\nSaved: weights/evidential_5seed_uq.npz")


if __name__ == "__main__":
    main()
