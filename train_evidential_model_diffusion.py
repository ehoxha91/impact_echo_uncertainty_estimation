"""Baseline evidential training on real DS1 + 2000 diffusion-augmented samples.

Same model/loss/schedule as train_evidential_model.py. Train data is the
concatenation of data/X_train_860.npy and data/X_train_860_diffusion.npy
(with corresponding y files).
"""
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from dataloaders.dataloader import ImpactEchoDatasetClassifier, ImpactEchoDatasetClassifierAug
from models.evidential_model import create_model
from training.trainer import train_evidential_classifier, validate_evidential_classifier, evaluate_evidential_classifier


def main():
    X_path = ['data/X_train_860.npy', 'data/X_train_860_diffusion.npy']
    y_path = ['data/y_train.npy', 'data/y_train_diffusion.npy']

    epochs = 100
    model_name = 'evidential_transformer_diffusion'
    batch_size = 128
    learning_rate = 0.0001
    num_classes = 2
    validation_split = 0.3
    patience = 25

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = ImpactEchoDatasetClassifierAug(X_path, y_path=y_path, array_size=860, use_augmentation=False)
    print(f"Total number of samples: {len(dataset)}")

    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    y_all = np.concatenate([np.load(p) for p in y_path])
    y_all[y_all < 1] = 0
    y_all[y_all > 0] = 1
    class_counts = np.bincount(y_all.astype(int))
    total_samples = len(y_all)
    class_weights = torch.FloatTensor([total_samples / (2 * count) for count in class_counts]).to(device)
    print(f"Class distribution (binarized, real+diffusion): {class_counts}")
    print(f"Class weights: {class_weights}")

    model = create_model().to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    print('Training Evidential Deep Learning classifier (real + 2000 diffusion)...')
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    writer = SummaryWriter(log_dir=f'runs/{model_name}')
    best_val_accuracy = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_nll, train_kl_div, train_penalty, train_acc = train_evidential_classifier(
            model, train_dataloader, optimizer, device, epoch, class_weights=class_weights
        )
        avg_val_loss, val_accuracy = validate_evidential_classifier(model, val_dataloader, device, epoch)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f} (NLL: {train_nll:.4f}, KL: {train_kl_div:.4f}, Penalty: {train_penalty:.4f}), '
              f'Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, LR: {current_lr:.8f}')
        writer.add_scalar("Loss/train_total", train_loss, epoch)
        writer.add_scalar("Loss/val_total", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)

        if val_accuracy > best_val_accuracy:
            print(f"New Best: {val_accuracy:.2f}% (was {best_val_accuracy:.2f}%)")
            best_val_accuracy = val_accuracy
            patience_counter = 0
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_accuracy': best_val_accuracy, 'val_loss': avg_val_loss,
            }, f'weights/{model_name}_best.pth')
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping: no improvement for {patience} epochs")
            break
        if val_accuracy >= 99.5:
            break
        scheduler.step()

    print(f"\nTraining completed! Best val accuracy: {best_val_accuracy:.2f}%\n")

    checkpoint = torch.load(f'weights/{model_name}_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']}")

    print("\nEvaluating on real DS1 test set...")
    test_dataset = ImpactEchoDatasetClassifier(['data/X_test_860.npy'], y_path=['data/y_test.npy'], array_size=860)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    (accuracy, predictions, uncertainties, epistemic_unc,
     aleatoric_unc, confidences, targets, alphas) = evaluate_evidential_classifier(model, test_loader, device)

    print(f"\nTest Results:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Total Uncertainty: {uncertainties.mean():.4f} +/- {uncertainties.std():.4f}")
    print(f"  Epistemic: {epistemic_unc.mean():.4f} +/- {epistemic_unc.std():.4f}")
    print(f"  Aleatoric: {aleatoric_unc.mean():.4f} +/- {aleatoric_unc.std():.4f}")
    print(f"  Confidence: {confidences.mean():.4f} +/- {confidences.std():.4f}")

    pred_classes = torch.argmax(predictions.squeeze(0), dim=1)
    correct = (pred_classes == targets)
    print(f"\nUncertainty by Correctness:")
    print(f"  Correct   - Uncertainty: {uncertainties.squeeze(0)[correct].mean():.4f}, Confidence: {confidences[correct].mean():.4f}")
    print(f"  Incorrect - Uncertainty: {uncertainties.squeeze(0)[~correct].mean():.4f}, Confidence: {confidences[~correct].mean():.4f}")
    print(f"\nPer-class:")
    for i in range(num_classes):
        mask = targets == i
        cc = (pred_classes[mask] == targets[mask]).sum().item()
        tc = mask.sum().item()
        print(f"  Class {i}: {(100.0*cc/tc if tc else 0):.2f}% ({cc}/{tc})")

    torch.save({
        'predictions': predictions, 'uncertainties': uncertainties,
        'epistemic': epistemic_unc, 'aleatoric': aleatoric_unc,
        'confidences': confidences, 'targets': targets, 'alphas': alphas,
        'accuracy': accuracy, 'best_val_accuracy': best_val_accuracy,
    }, f'weights/{model_name}_results.pth')
    writer.close()


if __name__ == '__main__':
    main()
