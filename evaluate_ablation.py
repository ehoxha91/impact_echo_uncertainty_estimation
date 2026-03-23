"""
Ablation study on evidential loss function components.
Tests variants: Full, NLL-only, NLL+KL, NLL+Penalty.
"""
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from dataloaders.dataloader import ImpactEchoDatasetClassifierAug, ImpactEchoDatasetClassifier
from models.evidential_model import create_model, ImprovedEvidentialIENet
from losses.evidential_loss import dirichlet_kl_divergence


def ablation_loss(evidence, targets, epoch, use_kl=True, use_penalty=True,
                  annealing_coefficient=1.0, regularization_coefficient=0.5, penalty_weight=0.005):
    """Evidential loss with toggleable components."""
    num_classes = evidence.size(1)
    alphas = evidence + 1.0
    alpha_sum = torch.sum(alphas, dim=1, keepdim=True)
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()

    # NLL (always present)
    nll = -torch.sum(targets_one_hot * (torch.digamma(alphas) - torch.digamma(alpha_sum)), dim=1)

    loss = nll

    # KL divergence
    kl_div = torch.tensor(0.0, device=evidence.device)
    if use_kl:
        kl_div = dirichlet_kl_divergence(alphas)
        annealing_factor = min(1.0, annealing_coefficient * epoch / 100.0)
        loss = loss + annealing_factor * regularization_coefficient * kl_div

    # Evidence penalty
    evidence_penalty = torch.tensor(0.0, device=evidence.device)
    if use_penalty:
        incorrect_evidence = torch.sum(evidence * (1 - targets_one_hot), dim=1)
        evidence_penalty = torch.mean(F.relu(incorrect_evidence - 2.0))

    total_loss = torch.mean(loss) + penalty_weight * evidence_penalty
    return total_loss


def train_ablation_variant(name, device, X_path, y_path, use_kl=True, use_penalty=True,
                           reg_coeff=0.5, penalty_weight=0.005, seed=42):
    """Train one ablation variant."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    dataset = ImpactEchoDatasetClassifierAug(X_path, y_path=y_path, array_size=860, use_augmentation=False)
    val_size = int(len(dataset) * 0.3)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    y_data = np.load(y_path[0])
    y_data[y_data < 1] = 0
    y_data[y_data > 0] = 1
    class_counts = np.bincount(y_data.astype(int))
    total_samples = len(y_data)
    class_weights = torch.FloatTensor([total_samples / (2 * count) for count in class_counts]).to(device)

    model = create_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    save_path = f'weights/ablation_{name}_best.pth'
    best_val_accuracy = 0.0
    patience_counter = 0

    for epoch in range(100):
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            X = data[0].to(device, dtype=torch.float).unsqueeze(1)
            labels = data[1].to(device, dtype=torch.long)
            evidence, _ = model(X)
            evidence = evidence.squeeze(0)
            loss = ablation_loss(evidence, labels, epoch, use_kl=use_kl, use_penalty=use_penalty,
                                 regularization_coefficient=reg_coeff, penalty_weight=penalty_weight)
            loss = loss * torch.mean(class_weights[labels])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for data in val_loader:
                X = data[0].to(device, dtype=torch.float).unsqueeze(1)
                labels = data[1].to(device, dtype=torch.long)
                evidence, _ = model(X)
                evidence = evidence.squeeze(0)
                alphas = evidence + 1.0
                prob = alphas / alphas.sum(dim=1, keepdim=True)
                predicted = prob.argmax(dim=1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100.0 * val_correct / val_total
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            patience_counter = 0
            torch.save({'model_state_dict': model.state_dict()}, save_path)
        else:
            patience_counter += 1

        if patience_counter >= 25 or val_acc >= 99.5:
            break
        scheduler.step()

    print(f"  {name}: best val acc = {best_val_accuracy:.2f}%")
    return save_path


def evaluate_ablation(model_path, device, X_path, y_path):
    """Evaluate an ablation model on test set."""
    model = create_model().to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    ds = ImpactEchoDatasetClassifier(X_path, y_path=y_path, array_size=860)
    X = torch.stack([ds[i][0] for i in range(len(ds))]).unsqueeze(1).float().to(device)
    y = np.array([ds[i][1] for i in range(len(ds))])

    with torch.no_grad():
        prob, epistemic, aleatoric, total_unc, confidence, _ = model.predict_with_uncertainty(X)
    prob = prob.squeeze(0) if prob.dim() == 3 else prob
    total_unc = (total_unc.squeeze(0) if total_unc.dim() == 2 else total_unc).cpu().numpy()
    probs_np = prob.cpu().numpy()
    preds = probs_np.argmax(axis=1)

    acc = accuracy_score(y, preds) * 100
    f1 = f1_score(y, preds, zero_division=0) * 100
    defect_mask = y == 1
    defect_acc = accuracy_score(y[defect_mask], preds[defect_mask]) * 100

    correct = (preds == y).astype(float)
    misclass_auroc = 0.0
    if len(np.unique(correct)) > 1:
        misclass_auroc = roc_auc_score(1 - correct, total_unc) * 100

    mean_unc = float(total_unc.mean())
    epi = (epistemic.squeeze(0) if epistemic.dim() == 2 else epistemic).cpu().numpy()
    mean_epi = float(epi.mean())

    return {
        'accuracy': acc, 'defect_acc': defect_acc, 'f1': f1,
        'misclass_auroc': misclass_auroc, 'mean_unc': mean_unc, 'mean_epi': mean_epi,
    }


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    X_train = ['data/X_train_860.npy']
    y_train = ['data/y_train.npy']
    X_test = ['data/X_test_860.npy']
    y_test = ['data/y_test.npy']
    X_ds3 = ['data/X_overlayed_860.npy']
    y_ds3 = ['data/y_overlayed.npy']

    # Define ablation variants
    variants = [
        ('Full',            dict(use_kl=True,  use_penalty=True,  reg_coeff=0.5, penalty_weight=0.005)),
        ('NLL only',        dict(use_kl=False, use_penalty=False, reg_coeff=0.5, penalty_weight=0.005)),
        ('NLL + KL',        dict(use_kl=True,  use_penalty=False, reg_coeff=0.5, penalty_weight=0.005)),
        ('NLL + Penalty',   dict(use_kl=False, use_penalty=True,  reg_coeff=0.5, penalty_weight=0.005)),
        ('Full, beta=0.1',  dict(use_kl=True,  use_penalty=True,  reg_coeff=0.1, penalty_weight=0.005)),
        ('Full, beta=1.0',  dict(use_kl=True,  use_penalty=True,  reg_coeff=1.0, penalty_weight=0.005)),
    ]

    print("Training ablation variants...")
    paths = {}
    for name, kwargs in variants:
        path = train_ablation_variant(name.replace(', ', '_').replace(' ', '_').replace('=', ''),
                                       device, X_train, y_train, **kwargs)
        paths[name] = path

    # Evaluate
    print(f"\n{'='*90}")
    print(f"{'Variant':<22} | {'Acc':>5} {'DefAcc':>6} {'F1':>5} {'AUROC':>6} {'Unc':>6} {'Epi':>6} | {'Acc':>5} {'DefAcc':>6} {'F1':>5} {'AUROC':>6}")
    print(f"{'':22} | {'--------- DS1 Test ---------':^38} | {'------ DS3 Overlay ------':^30}")
    print("-" * 90)

    all_results = {}
    for name, _ in variants:
        ds1 = evaluate_ablation(paths[name], device, X_test, y_test)
        ds3 = evaluate_ablation(paths[name], device, X_ds3, y_ds3)
        all_results[name] = {'ds1': ds1, 'ds3': ds3}

        print(f"{name:<22} | {ds1['accuracy']:>5.1f} {ds1['defect_acc']:>6.1f} {ds1['f1']:>5.1f} "
              f"{ds1['misclass_auroc']:>6.1f} {ds1['mean_unc']:>6.3f} {ds1['mean_epi']:>6.3f} | "
              f"{ds3['accuracy']:>5.1f} {ds3['defect_acc']:>6.1f} {ds3['f1']:>5.1f} {ds3['misclass_auroc']:>6.1f}")

    # LaTeX table
    print("\n% LaTeX ablation table")
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\caption{Ablation study on evidential loss components. $\beta$: KL regularization coefficient. $\mathcal{L}_\text{pen}$: evidence penalty. AUROC: misclassification detection.}")
    print(r"\label{tab:ablation}")
    print(r"\begin{tabular}{l|ccc|ccc}")
    print(r"\toprule")
    print(r"& \multicolumn{3}{c|}{\textbf{DS1 Test}} & \multicolumn{3}{c}{\textbf{DS3 Overlay}} \\")
    print(r"\textbf{Loss Variant} & \textbf{Acc. (\%)} & \textbf{F1 (\%)} & \textbf{AUROC (\%)} & \textbf{Acc. (\%)} & \textbf{F1 (\%)} & \textbf{AUROC (\%)} \\")
    print(r"\midrule")
    for name, _ in variants:
        d1 = all_results[name]['ds1']
        d3 = all_results[name]['ds3']
        label = name
        if name == 'Full':
            label = r"Full ($\beta$=0.5)"
        elif name == 'NLL only':
            label = r"$\mathcal{L}_\text{NLL}$ only"
        elif name == 'NLL + KL':
            label = r"$\mathcal{L}_\text{NLL}$ + KL"
        elif name == 'NLL + Penalty':
            label = r"$\mathcal{L}_\text{NLL}$ + $\mathcal{L}_\text{pen}$"
        elif name == 'Full, beta=0.1':
            label = r"Full ($\beta$=0.1)"
        elif name == 'Full, beta=1.0':
            label = r"Full ($\beta$=1.0)"
        print(f"{label} & {d1['accuracy']:.1f} & {d1['f1']:.1f} & {d1['misclass_auroc']:.1f} "
              f"& {d3['accuracy']:.1f} & {d3['f1']:.1f} & {d3['misclass_auroc']:.1f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")

    torch.save(all_results, 'weights/ablation_results.pth')
    print("\nResults saved to weights/ablation_results.pth")


if __name__ == '__main__':
    main()
