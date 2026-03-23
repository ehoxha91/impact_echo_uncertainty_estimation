"""
MC Dropout inference wrapper for uncertainty estimation.
Enables dropout at test time and runs T stochastic forward passes.
"""
import torch
import torch.nn.functional as F


def enable_dropout(model):
    """Enable dropout layers during inference."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def mc_dropout_predict(model, x, T=50):
    """
    Run T stochastic forward passes with dropout enabled.

    Returns:
        mean_prob: Mean softmax probabilities [batch, num_classes]
        total_uncertainty: Predictive entropy (total) [batch]
        epistemic_uncertainty: Mutual information (epistemic) [batch]
        aleatoric_uncertainty: Expected entropy (aleatoric) [batch]
        confidence: Max mean probability [batch]
    """
    model.eval()
    enable_dropout(model)

    all_probs = []
    with torch.no_grad():
        for _ in range(T):
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs)

    # [T, batch, num_classes]
    stacked = torch.stack(all_probs, dim=0)

    # Mean prediction
    mean_prob = stacked.mean(dim=0)

    # Total uncertainty: entropy of mean prediction (predictive entropy)
    total_uncertainty = -torch.sum(mean_prob * torch.log(mean_prob + 1e-10), dim=-1)

    # Aleatoric uncertainty: mean of individual entropies (expected entropy)
    individual_entropies = -torch.sum(stacked * torch.log(stacked + 1e-10), dim=-1)
    aleatoric_uncertainty = individual_entropies.mean(dim=0)

    # Epistemic uncertainty: mutual information = total - aleatoric
    epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty

    confidence = mean_prob.max(dim=-1)[0]

    model.eval()  # restore full eval mode

    return mean_prob, total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty, confidence
