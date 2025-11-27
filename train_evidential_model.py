"""
Main Training Script for Evidential Deep Learning Model
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
from utils.utils import save_model


def main():
    """Main training function"""
    # Configuration
    X_path = ['data/X_train_860.npy']
    y_path = ['data/y_train.npy']
    
    epochs = 100
    model_name = 'evidential_transformer'
    batch_size = 128
    learning_rate = 0.0001
    num_classes = 2
    validation_split = 0.3
    patience = 25
    
    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and prepare data (disable augmentation for speed)
    dataset = ImpactEchoDatasetClassifierAug(X_path, y_path=y_path, array_size=860, use_augmentation=False)
    print(f"Total number of samples: {len(dataset)}")
    
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    
    # Calculate class weights
    y_data = np.load(y_path[0])
    y_data[y_data < 1] = 0
    y_data[y_data > 0] = 1
    class_counts = np.bincount(y_data.astype(int))
    total_samples = len(y_data)
    class_weights = torch.FloatTensor([total_samples / (2 * count) for count in class_counts]).to(device)
    print(f"Class distribution: {class_counts}")
    print(f"Class weights: {class_weights}")
    
    # Initialize model
    model = create_model().to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    print('Training Full Evidential Deep Learning classifier...')
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize tensorboard and tracking variables
    writer = SummaryWriter()
    best_val_accuracy = 0.0
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        train_loss, train_nll, train_kl_div, train_penalty, train_acc = train_evidential_classifier(
            model, train_dataloader, optimizer, device, epoch, class_weights=class_weights
        )
        
        # Validate
        avg_val_loss, val_accuracy = validate_evidential_classifier(model, val_dataloader, device, epoch)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f} (NLL: {train_nll:.4f}, KL: {train_kl_div:.4f}, Penalty: {train_penalty:.4f}), '
              f'Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, LR: {current_lr:.8f}')
        
        # Log to tensorboard
        writer.add_scalar("Loss/train_total", train_loss, epoch)
        writer.add_scalar("Loss/train_nll", train_nll, epoch)
        writer.add_scalar("Loss/train_kl_divergence", train_kl_div, epoch)
        writer.add_scalar("Loss/train_evidence_penalty", train_penalty, epoch)
        writer.add_scalar("Loss/val_total", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)
        writer.add_scalar("Learning_Rate", current_lr, epoch)
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            print(f"🎯 New Best: {val_accuracy:.2f}% (was {best_val_accuracy:.2f}%)")
            best_val_accuracy = val_accuracy
            patience_counter = 0
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_accuracy': best_val_accuracy,
                'val_loss': avg_val_loss,
            }, f'weights/{model_name}_best.pth')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping: no improvement for {patience} epochs")
            break
        if val_accuracy >= 99.5:
            print(f"Stopping: near-perfect accuracy ({val_accuracy:.2f}%)")
            break
        
        scheduler.step()
    
    print(f"\n{'='*50}")
    print(f"Training completed! Best val accuracy: {best_val_accuracy:.2f}%")
    print(f"{'='*50}\n")
    
    # Load best model
    checkpoint = torch.load(f'weights/{model_name}_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    # Test evaluation
    print("\nEvaluating on test set...")
    test_dataset = ImpactEchoDatasetClassifier(['data/X_test_860.npy'], y_path=['data/y_test.npy'], array_size=860)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    (accuracy, predictions, uncertainties, epistemic_unc, 
     aleatoric_unc, confidences, targets, alphas) = evaluate_evidential_classifier(model, test_loader, device)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Total Uncertainty: {uncertainties.mean():.4f} ± {uncertainties.std():.4f}")
    print(f"  Epistemic: {epistemic_unc.mean():.4f} ± {epistemic_unc.std():.4f}")
    print(f"  Aleatoric: {aleatoric_unc.mean():.4f} ± {aleatoric_unc.std():.4f}")
    print(f"  Confidence: {confidences.mean():.4f} ± {confidences.std():.4f}")
    
    # Uncertainty analysis
    pred_classes = torch.argmax(predictions.squeeze(0), dim=1)
    correct = (pred_classes == targets)
    
    print(f"\nUncertainty by Correctness:")
    print(f"  Correct   - Uncertainty: {uncertainties.squeeze(0)[correct].mean():.4f}, Confidence: {confidences[correct].mean():.4f}")
    print(f"  Incorrect - Uncertainty: {uncertainties.squeeze(0)[~correct].mean():.4f}, Confidence: {confidences[~correct].mean():.4f}")
    
    # Per-class accuracy
    print(f"\nPer-class:")
    for i in range(num_classes):
        mask = targets == i
        correct_count = (pred_classes[mask] == targets[mask]).sum().item()
        total_count = mask.sum().item()
        acc = 100.0 * correct_count / total_count if total_count > 0 else 0
        print(f"  Class {i}: {acc:.2f}% ({correct_count}/{total_count})")
    
    # Save results
    results_path = f'weights/{model_name}_results.pth'
    torch.save({
        'predictions': predictions,
        'uncertainties': uncertainties,
        'epistemic': epistemic_unc,
        'aleatoric': aleatoric_unc,
        'confidences': confidences,
        'targets': targets,
        'alphas': alphas,
        'accuracy': accuracy,
        'best_val_accuracy': best_val_accuracy,
    }, results_path)
    
    print(f"\nResults saved to {results_path}")
    writer.close()


if __name__ == '__main__':
    main()