"""
Training script for baseline UQ models: Standard model (for MC Dropout) and Deep Ensemble members.
Uses the same hyperparameters as the evidential model for fair comparison.
"""
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split

from dataloaders.dataloader import ImpactEchoDatasetClassifierAug, ImpactEchoDatasetClassifier
from models.standard_model import create_standard_model
from utils.utils import save_model


def train_standard_model(seed, model_name, device, X_path, y_path, epochs=100,
                         batch_size=128, learning_rate=0.0001, validation_split=0.3, patience=25):
    """Train a single standard model with a given seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    # Load data
    dataset = ImpactEchoDatasetClassifierAug(X_path, y_path=y_path, array_size=860, use_augmentation=False)
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    # Class weights
    y_data = np.load(y_path[0])
    y_data[y_data < 1] = 0
    y_data[y_data > 0] = 1
    class_counts = np.bincount(y_data.astype(int))
    total_samples = len(y_data)
    class_weights = torch.FloatTensor([total_samples / (2 * count) for count in class_counts]).to(device)

    # Model, optimizer, scheduler
    model = create_standard_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    print(f"\nTraining {model_name} (seed={seed})")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_val_accuracy = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        correct, total, total_loss = 0, 0, 0.0
        for data in train_loader:
            optimizer.zero_grad()
            X = data[0].to(device, dtype=torch.float).unsqueeze(1)
            labels = data[1].to(device, dtype=torch.long)
            logits = model(X)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            predicted = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()

        train_acc = 100.0 * correct / total

        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for data in val_loader:
                X = data[0].to(device, dtype=torch.float).unsqueeze(1)
                labels = data[1].to(device, dtype=torch.long)
                logits = model(X)
                predicted = logits.argmax(dim=1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100.0 * val_correct / val_total

        if epoch % 10 == 0 or val_acc > best_val_accuracy:
            print(f"  Epoch {epoch:03d}: Train Acc={train_acc:.1f}%, Val Acc={val_acc:.1f}%")

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_accuracy': best_val_accuracy,
                'seed': seed,
            }, f'weights/{model_name}_best.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break
        if val_acc >= 99.5:
            break

        scheduler.step()

    print(f"  Best val accuracy: {best_val_accuracy:.2f}%")
    return best_val_accuracy


def main():
    X_path = ['data/X_train_860.npy']
    y_path = ['data/y_train.npy']
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train 1 model for MC Dropout
    print("=" * 60)
    print("Training standard model for MC Dropout")
    print("=" * 60)
    train_standard_model(
        seed=42, model_name='standard_mc_dropout',
        device=device, X_path=X_path, y_path=y_path
    )

    # Train 5 models for Deep Ensemble
    print("\n" + "=" * 60)
    print("Training Deep Ensemble (5 members)")
    print("=" * 60)
    ensemble_seeds = [42, 123, 456, 789, 1024]
    for i, seed in enumerate(ensemble_seeds):
        train_standard_model(
            seed=seed, model_name=f'ensemble_member_{i}',
            device=device, X_path=X_path, y_path=y_path
        )

    print("\nAll baseline models trained successfully.")
    print("Weights saved in weights/ directory.")


if __name__ == '__main__':
    main()
