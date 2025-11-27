"""
Evidential Loss Functions for Uncertainty Estimation
"""
import torch
import torch.nn.functional as F


def dirichlet_kl_divergence(alphas, target_concentration=1.0):
    """
    Compute KL divergence between Dirichlet distributions.
    KL(Dir(alpha) || Dir(alpha_0)) where alpha_0 is uniform prior
    """
    num_classes = alphas.size(1)
    
    # Target uniform Dirichlet parameters
    target_alphas = torch.ones_like(alphas) * target_concentration
    
    # Sum of parameters
    alpha_sum = torch.sum(alphas, dim=1, keepdim=True)
    target_sum = torch.sum(target_alphas, dim=1, keepdim=True)
    
    # KL divergence computation
    kl_div = torch.lgamma(alpha_sum) - torch.lgamma(target_sum)
    kl_div -= torch.sum(torch.lgamma(alphas) - torch.lgamma(target_alphas), dim=1, keepdim=True)
    kl_div += torch.sum((alphas - target_alphas) * (torch.digamma(alphas) - torch.digamma(alpha_sum)), dim=1, keepdim=True)
    
    return kl_div.squeeze()


def evidential_loss(evidence, targets, epoch, annealing_coefficient=1.0, regularization_coefficient=0.5):
    """
    Complete evidential loss function with KL regularization
    
    Args:
        evidence: Evidence values from model (batch_size, num_classes)
        targets: Ground truth labels (batch_size,)
        epoch: Current epoch for annealing
        annealing_coefficient: Coefficient for KL annealing
        regularization_coefficient: Weight for KL regularization term
    
    Returns:
        total_loss: Combined loss value
        nll: Negative expected log-likelihood
        kl_div: KL divergence
        evidence_penalty: Evidence regularization penalty
    """
    num_classes = evidence.size(1)
    
    # Convert evidence to Dirichlet parameters
    alphas = evidence + 1.0
    alpha_sum = torch.sum(alphas, dim=1, keepdim=True)
    
    # One-hot encode targets
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
    
    # Expected log-likelihood
    expected_log_likelihood = -torch.sum(
        targets_one_hot * (torch.digamma(alphas) - torch.digamma(alpha_sum)), 
        dim=1
    )
    
    # KL divergence regularization
    kl_div = dirichlet_kl_divergence(alphas, target_concentration=1.0)
    
    # Annealing factor for KL term
    annealing_factor = min(1.0, annealing_coefficient * epoch / 100.0)
    
    # Total loss
    loss = expected_log_likelihood + annealing_factor * regularization_coefficient * kl_div
    
    # Evidence regularization to prevent overconfidence
    incorrect_evidence = torch.sum(evidence * (1 - targets_one_hot), dim=1)
    evidence_penalty = torch.mean(F.relu(incorrect_evidence - 2.0))
    
    total_loss = torch.mean(loss) + 0.005 * evidence_penalty
    
    return total_loss, torch.mean(-expected_log_likelihood), torch.mean(kl_div), evidence_penalty
