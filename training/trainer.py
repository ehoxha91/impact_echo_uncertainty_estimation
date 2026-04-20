"""
Training and Evaluation Functions for Evidential Models
"""
import torch
import torch.nn.functional as F
import tqdm
from losses.evidential_loss import evidential_loss


def train_evidential_classifier(model, dataloader, optimizer, device, epoch, class_weights=None):
    """Training function for full evidential learning"""
    model.train()
    total_loss = 0.0
    total_nll = 0.0
    total_kl = 0.0
    total_penalty = 0.0
    correct = 0
    total = 0
    
    for data in tqdm.tqdm(dataloader):
        optimizer.zero_grad()

        X = data[0].to(device, dtype=torch.float)
        labels = data[1].to(device, dtype=torch.int).long()
        X = X.view(X.size(0), 1, X.size(1))
        
        # Forward pass
        evidence, _ = model(X)
        evidence = evidence.squeeze(0)
        
        # Compute evidential loss
        loss, nll, kl_div, penalty = evidential_loss(
            evidence, labels, epoch, 
            annealing_coefficient=1.0, 
            regularization_coefficient=0.5
        )
        
        # Apply class weights
        if class_weights is not None:
            class_loss_weights = class_weights[labels]
            weighted_loss = loss * torch.mean(class_loss_weights)
            loss = weighted_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Calculate accuracy
        alphas = evidence + 1.0
        alpha_sum = torch.sum(alphas, dim=1, keepdim=True)
        prob = alphas / alpha_sum
        predicted = torch.argmax(prob, dim=1)
        
        total_loss += loss.item()
        total_nll += nll.item()
        total_kl += kl_div.item()
        total_penalty += penalty.item()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    avg_nll = total_nll / len(dataloader)
    avg_kl = total_kl / len(dataloader)
    avg_penalty = total_penalty / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, avg_nll, avg_kl, avg_penalty, accuracy


def evaluate_evidential_classifier(model, test_loader, device):
    """Evaluate evidential classifier with uncertainty analysis"""
    model.eval()
    correct = 0
    total = 0
    
    all_predictions = []
    all_uncertainties = []
    all_epistemic = []
    all_aleatoric = []
    all_confidences = []
    all_targets = []
    all_alphas = []
    
    with torch.no_grad():
        for data in test_loader:
            X = data[0].to(device, dtype=torch.float)
            labels = data[1].to(device, dtype=torch.int).long()
            X = X.view(X.size(0), 1, X.size(1))
            
            prob, epistemic, aleatoric, total_unc, confidence, alpha_sum = model.predict_with_uncertainty(X)
            predicted = torch.argmax(prob.squeeze(0), dim=1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.append(prob.cpu())
            all_uncertainties.append(total_unc.cpu())
            all_epistemic.append(epistemic.cpu())
            all_aleatoric.append(aleatoric.cpu())
            all_confidences.append(confidence.cpu())
            all_targets.append(labels.cpu())
            all_alphas.append(alpha_sum.cpu())
    
    accuracy = 100.0 * correct / total
    
    predictions = torch.cat(all_predictions, dim=0)
    uncertainties = torch.cat(all_uncertainties, dim=0)
    epistemic_unc = torch.cat(all_epistemic, dim=0)
    aleatoric_unc = torch.cat(all_aleatoric, dim=0)
    confidences = torch.cat(all_confidences, dim=0)
    targets = torch.cat(all_targets, dim=0)
    alphas = torch.cat(all_alphas, dim=0)
    
    return (accuracy, predictions, uncertainties, epistemic_unc, 
            aleatoric_unc, confidences, targets, alphas)


def validate_evidential_classifier(model, val_loader, device, epoch):
    """Validation function for evidential classifier"""
    model.eval()
    val_total_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for data in val_loader:
            X = data[0].to(device, dtype=torch.float)
            labels = data[1].to(device, dtype=torch.int).long()
            X = X.view(X.size(0), 1, X.size(1))
            
            evidence, _ = model(X)
            evidence = evidence.squeeze(0)
            
            val_loss, _, _, _ = evidential_loss(
                evidence, labels, epoch,
                annealing_coefficient=1.0,
                regularization_coefficient=0.5
            )
            
            # Calculate validation accuracy
            alphas = evidence + 1.0
            alpha_sum = torch.sum(alphas, dim=1, keepdim=True)
            prob = alphas / alpha_sum
            predicted = torch.argmax(prob, dim=1)
            
            val_total_loss += val_loss.item()
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    avg_val_loss = val_total_loss / len(val_loader)
    val_accuracy = 100.0 * val_correct / val_total
    
    return avg_val_loss, val_accuracy
