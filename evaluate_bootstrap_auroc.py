"""
Bootstrap 95% CIs on Misclassification AUROC for UQ methods on DS1 + DS3.
Methods: Max-Softmax (MSP), Evidential IENet, MC Dropout (T=50), Deep Ensemble (M=5).
1000 resamples with replacement on the test set.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from dataloaders.dataloader import ImpactEchoDatasetClassifier
from models.evidential_model import create_model
from models.standard_model import create_standard_model
from models.mc_dropout import mc_dropout_predict
from models.deep_ensemble import load_ensemble, ensemble_predict


N_BOOT = 1000
RNG = np.random.default_rng(0)


def load_ev(path, device):
    m = create_model().to(device)
    ck = torch.load(path, map_location=device)
    m.load_state_dict(ck.get("model_state_dict", ck))
    m.eval()
    return m


def load_std(path, device):
    m = create_standard_model().to(device)
    ck = torch.load(path, map_location=device)
    m.load_state_dict(ck.get("model_state_dict", ck))
    m.eval()
    return m


def preds_msp(model, X):
    with torch.no_grad():
        probs = F.softmax(model(X), dim=-1)
    conf, pred = probs.max(dim=-1)
    return pred.cpu().numpy(), (1.0 - conf).cpu().numpy()


def preds_evidential(model, X):
    with torch.no_grad():
        prob, _, _, total_unc, _, _ = model.predict_with_uncertainty(X)
    prob = prob.squeeze(0) if prob.dim() == 3 else prob
    total_unc = total_unc.squeeze(0) if total_unc.dim() == 2 else total_unc
    return prob.argmax(dim=-1).cpu().numpy(), total_unc.cpu().numpy().squeeze()


def preds_mc(model, X):
    mean_prob, total_unc, _, _, _ = mc_dropout_predict(model, X, T=50)
    return mean_prob.argmax(dim=-1).cpu().numpy(), total_unc.cpu().numpy().squeeze()


def preds_ens(models, X):
    mean_prob, total_unc, _, _, _ = ensemble_predict(models, X)
    return mean_prob.argmax(dim=-1).cpu().numpy(), total_unc.cpu().numpy().squeeze()


def bootstrap_auroc(pred, unc, y, n=N_BOOT):
    correct = (pred == y).astype(int)
    wrong = 1 - correct
    if len(np.unique(wrong)) < 2:
        return float("nan"), float("nan"), float("nan")
    vals = []
    idx_all = np.arange(len(y))
    for _ in range(n):
        idx = RNG.choice(idx_all, size=len(y), replace=True)
        w = wrong[idx]
        u = unc[idx]
        if len(np.unique(w)) < 2:
            continue
        vals.append(roc_auc_score(w, u) * 100)
    arr = np.array(vals)
    return float(roc_auc_score(wrong, unc) * 100), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def run_on(name, X, y, ev, mc, std, ens):
    print(f"\n=== {name} ===")
    results = {}
    for tag, fn in [
        ("MSP", lambda: preds_msp(std, X)),
        ("Evidential", lambda: preds_evidential(ev, X)),
        ("MC Dropout", lambda: preds_mc(mc, X)),
        ("Deep Ensemble", lambda: preds_ens(ens, X)),
    ]:
        pred, unc = fn()
        point, lo, hi = bootstrap_auroc(pred, unc, y)
        results[tag] = (point, lo, hi)
        print(f"  {tag:<15} AUROC = {point:5.2f}  [{lo:5.2f}, {hi:5.2f}]")
    return results


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}   Bootstrap n={N_BOOT}")
    ev  = load_ev("weights/evidential_transformer.pth", device)
    mc  = load_std("weights/standard_mc_dropout_best.pth", device)
    std = load_std("weights/standard_mc_dropout_best.pth", device)
    ens = load_ensemble([f"weights/ensemble_member_{i}_best.pth" for i in range(5)], device)

    ds1 = ImpactEchoDatasetClassifier(["data/X_test_860.npy"], y_path=["data/y_test.npy"], array_size=860)
    X1 = torch.stack([ds1[i][0] for i in range(len(ds1))]).unsqueeze(1).float().to(device)
    y1 = np.array([ds1[i][1] for i in range(len(ds1))])
    r1 = run_on("DS1 Test", X1, y1, ev, mc, std, ens)

    ds3 = ImpactEchoDatasetClassifier(["data/X_overlayed_860.npy"], y_path=["data/y_overlayed.npy"], array_size=860)
    X3 = torch.stack([ds3[i][0] for i in range(len(ds3))]).unsqueeze(1).float().to(device)
    y3 = np.array([ds3[i][1] for i in range(len(ds3))])
    r3 = run_on("DS3 Overlay", X3, y3, ev, mc, std, ens)

    import json
    with open("weights/bootstrap_auroc.json", "w") as f:
        json.dump({"ds1": r1, "ds3": r3, "n_boot": N_BOOT}, f, indent=2)
    print("\nSaved: weights/bootstrap_auroc.json")


if __name__ == "__main__":
    main()
