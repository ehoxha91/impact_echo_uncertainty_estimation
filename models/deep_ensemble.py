"""
Deep Ensemble inference wrapper for uncertainty estimation.
Averages predictions from M independently trained models.
"""
import torch
import torch.nn.functional as F
from models.standard_model import create_standard_model


def load_ensemble(model_paths, device, input_length=860, num_classes=2, dropout=0.1):
    """Load M pre-trained models as an ensemble."""
    models = []
    for path in model_paths:
        model = create_standard_model(input_length, num_classes, dropout)
        checkpoint = torch.load(path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        models.append(model)
    return models


def ensemble_predict(models, x):
    """
    Run inference across all ensemble members and aggregate.

    Returns:
        mean_prob: Mean softmax probabilities [batch, num_classes]
        total_uncertainty: Predictive entropy (total) [batch]
        epistemic_uncertainty: Mutual information (epistemic) [batch]
        aleatoric_uncertainty: Expected entropy (aleatoric) [batch]
        confidence: Max mean probability [batch]
    """
    all_probs = []
    with torch.no_grad():
        for model in models:
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs)

    # [M, batch, num_classes]
    stacked = torch.stack(all_probs, dim=0)

    # Mean prediction
    mean_prob = stacked.mean(dim=0)

    # Total uncertainty: entropy of mean prediction
    total_uncertainty = -torch.sum(mean_prob * torch.log(mean_prob + 1e-10), dim=-1)

    # Aleatoric uncertainty: mean of individual entropies
    individual_entropies = -torch.sum(stacked * torch.log(stacked + 1e-10), dim=-1)
    aleatoric_uncertainty = individual_entropies.mean(dim=0)

    # Epistemic uncertainty: mutual information
    epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty

    confidence = mean_prob.max(dim=-1)[0]

    return mean_prob, total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty, confidence
