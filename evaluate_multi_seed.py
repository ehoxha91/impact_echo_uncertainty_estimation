"""
Multi-seed evaluation for Evidential IENet.
Trains N models with different seeds and reports mean +/- std for all metrics.
"""
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from dataloaders.dataloader import ImpactEchoDatasetClassifierAug, ImpactEchoDatasetClassifier
from models.evidential_model import create_model, ImprovedEvidentialIENet
from losses.evidential_loss import evidential_loss


def train_one_seed(seed, device, X_path, y_path, epochs=100, batch_size=128,
                   learning_rate=0.0001, validation_split=0.3, patience=25):
    """Train evidential model with a given seed and return best model path."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    dataset = ImpactEchoDatasetClassifierAug(X_path, y_path=y_path, array_size=860, use_augmentation=False)
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    y_data = np.load(y_path[0])
    y_data[y_data < 1] = 0
    y_data[y_data > 0] = 1
    class_counts = np.bincount(y_data.astype(int))
    total_samples = len(y_data)
    class_weights = torch.FloatTensor([total_samples / (2 * count) for count in class_counts]).to(device)

    model = create_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    save_path = f'weights/evidential_seed_{seed}_best.pth'
    best_val_accuracy = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for data in train_loader:
            optimizer.zero_grad()
            X = data[0].to(device, dtype=torch.float).unsqueeze(1)
            labels = data[1].to(device, dtype=torch.long)
            evidence, _ = model(X)
            evidence = evidence.squeeze(0)
            loss, _, _, _ = evidential_loss(evidence, labels, epoch, 1.0, 0.5)
            if class_weights is not None:
                loss = loss * torch.mean(class_weights[labels])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            alphas = evidence + 1.0
            prob = alphas / alphas.sum(dim=1, keepdim=True)
            predicted = prob.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Validate
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
            torch.save({'model_state_dict': model.state_dict(), 'seed': seed,
                        'epoch': epoch, 'best_val_accuracy': best_val_accuracy}, save_path)
        else:
            patience_counter += 1

        if patience_counter >= patience or val_acc >= 99.5:
            break
        scheduler.step()

    print(f"  Seed {seed}: best val acc = {best_val_accuracy:.2f}%")
    return save_path


def evaluate_model(model_path, device, X_path, y_path):
    """Load a model and evaluate on test set, returning all metrics."""
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
    total_unc_np = (total_unc.squeeze(0) if total_unc.dim() == 2 else total_unc).cpu().numpy()
    probs_np = prob.cpu().numpy()
    preds = probs_np.argmax(axis=1)

    acc = accuracy_score(y, preds) * 100
    defect_mask = y == 1
    nondef_mask = y == 0
    defect_acc = accuracy_score(y[defect_mask], preds[defect_mask]) * 100
    nondef_acc = accuracy_score(y[nondef_mask], preds[nondef_mask]) * 100
    prec = precision_score(y, preds, zero_division=0) * 100
    rec = recall_score(y, preds, zero_division=0) * 100
    f1 = f1_score(y, preds, zero_division=0) * 100

    correct = (preds == y).astype(float)
    misclass_auroc = 0.0
    if len(np.unique(correct)) > 1:
        misclass_auroc = roc_auc_score(1 - correct, total_unc_np) * 100

    return {
        'accuracy': acc, 'defect_acc': defect_acc, 'nondef_acc': nondef_acc,
        'precision': prec, 'recall': rec, 'f1': f1, 'misclass_auroc': misclass_auroc,
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

    seeds = [42, 123, 456, 789, 1024]

    # Train
    print("Training evidential models with multiple seeds...")
    model_paths = []
    for seed in seeds:
        path = train_one_seed(seed, device, X_train, y_train)
        model_paths.append(path)

    # Evaluate on DS1
    print("\n--- DS1 Test Results (5 seeds) ---")
    ds1_results = []
    for path in model_paths:
        r = evaluate_model(path, device, X_test, y_test)
        ds1_results.append(r)

    # Evaluate on DS3
    print("\n--- DS3 Results (5 seeds) ---")
    ds3_results = []
    for path in model_paths:
        r = evaluate_model(path, device, X_ds3, y_ds3)
        ds3_results.append(r)

    # Print summary
    for dataset_name, results in [("DS1 Test", ds1_results), ("DS3 Overlay", ds3_results)]:
        print(f"\n{'='*70}")
        print(f"Evidential IENet — {dataset_name} (N=5 seeds)")
        print(f"{'='*70}")
        metrics = ['accuracy', 'defect_acc', 'nondef_acc', 'precision', 'recall', 'f1', 'misclass_auroc']
        labels = ['Overall Acc', 'Defect Acc', 'Non-Def Acc', 'Precision', 'Recall', 'F1', 'Misclass AUROC']
        for metric, label in zip(metrics, labels):
            vals = [r[metric] for r in results]
            print(f"  {label:>16}: {np.mean(vals):5.1f} +/- {np.std(vals):4.1f}%  (range: {np.min(vals):.1f}-{np.max(vals):.1f})")

        # Per-seed breakdown
        per_seed = [f"{r['accuracy']:.1f}" for r in results]
        print(f"\n  Per-seed accuracy: {per_seed}")

    # Save
    torch.save({'ds1': ds1_results, 'ds3': ds3_results, 'seeds': seeds},
               'weights/multi_seed_results.pth')
    print("\nResults saved to weights/multi_seed_results.pth")


if __name__ == '__main__':
    main()
