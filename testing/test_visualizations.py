"""
Visualization functions for evidential model testing.

This module contains all visualization and plotting functions used for
analyzing and displaying uncertainty quantification results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# Visualization configuration
PLOT_STYLE = 'default'
COLOR_PALETTE = "husl"
FIGURE_DPI = 300


def create_advanced_uncertainty_distribution_analysis(pred_probs, epistemic_unc, aleatoric_unc, 
                                                      total_unc, confidences, alphas, targets, 
                                                      correct_predictions, experiment_name, model_name):
    """
    Create advanced uncertainty distribution analysis plots.
    
    Args:
        pred_probs: Prediction probabilities
        epistemic_unc: Epistemic uncertainty values
        aleatoric_unc: Aleatoric uncertainty values
        total_unc: Total uncertainty values
        confidences: Confidence scores
        alphas: Evidence strength values
        targets: True labels
        correct_predictions: Boolean array of correct predictions
        experiment_name: Experiment name for saving
        model_name: Model name for saving
    """
    plt.style.use(PLOT_STYLE)
    sns.set_palette(COLOR_PALETTE)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Evidential Uncertainty Distribution Analysis', fontsize=18)
    
    # Plot 1: Epistemic vs Aleatoric scatter
    axes[0, 0].scatter(epistemic_unc[correct_predictions], aleatoric_unc[correct_predictions], 
                      alpha=0.7, c='green', label='Correct', s=40, edgecolors='black', linewidth=0.5)
    axes[0, 0].scatter(epistemic_unc[~correct_predictions], aleatoric_unc[~correct_predictions], 
                      alpha=0.7, c='red', label='Incorrect', s=40, edgecolors='black', linewidth=0.5)
    axes[0, 0].set_xlabel('Epistemic Uncertainty', fontsize=12)
    axes[0, 0].set_ylabel('Aleatoric Uncertainty', fontsize=12)
    axes[0, 0].set_title('Epistemic vs Aleatoric Uncertainty', fontsize=14)
    axes[0, 0].legend(frameon=True, fancybox=True, shadow=True)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Total Uncertainty Distribution
    axes[0, 1].hist(total_unc[correct_predictions], bins=25, alpha=0.7, 
                   color='green', label='Correct', density=True, edgecolor='black', linewidth=0.8)
    axes[0, 1].hist(total_unc[~correct_predictions], bins=25, alpha=0.7, 
                   color='red', label='Incorrect', density=True, edgecolor='black', linewidth=0.8)
    axes[0, 1].set_xlabel('Total Uncertainty', fontsize=12)
    axes[0, 1].set_ylabel('Density', fontsize=12)
    axes[0, 1].set_title('Total Uncertainty Distribution', fontsize=14)
    axes[0, 1].legend(frameon=True, fancybox=True, shadow=True)
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    
    # Plot 3: Confidence Distribution
    axes[1, 0].hist(confidences[correct_predictions], bins=25, alpha=0.7, 
                   color='green', label='Correct', density=True, edgecolor='black', linewidth=0.8)
    axes[1, 0].hist(confidences[~correct_predictions], bins=25, alpha=0.7, 
                   color='red', label='Incorrect', density=True, edgecolor='black', linewidth=0.8)
    axes[1, 0].set_xlabel('Confidence', fontsize=12)
    axes[1, 0].set_ylabel('Density', fontsize=12)
    axes[1, 0].set_title('Confidence Distribution', fontsize=14)
    axes[1, 0].legend(frameon=True, fancybox=True, shadow=True)
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Plot 4: Uncertainty vs Confidence scatter
    axes[1, 1].scatter(total_unc[correct_predictions], confidences[correct_predictions], 
                      alpha=0.7, c='green', label='Correct', s=40, edgecolors='black', linewidth=0.5)
    axes[1, 1].scatter(total_unc[~correct_predictions], confidences[~correct_predictions], 
                      alpha=0.7, c='red', label='Incorrect', s=40, edgecolors='black', linewidth=0.5)
    axes[1, 1].set_xlabel('Total Uncertainty', fontsize=12)
    axes[1, 1].set_ylabel('Confidence', fontsize=12)
    axes[1, 1].set_title('Uncertainty vs Confidence', fontsize=14)
    axes[1, 1].legend(frameon=True, fancybox=True, shadow=True)
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'results/{experiment_name}/{model_name}_uncertainty_distributions.png', 
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"✓ Uncertainty distribution analysis saved")


def create_advanced_uncertainty_correlations(epistemic_unc, aleatoric_unc, total_unc, 
                                            confidences, alphas, correct_predictions, 
                                            experiment_name, model_name):
    """
    Create correlation analysis for uncertainty metrics.
    
    Args:
        epistemic_unc: Epistemic uncertainty values
        aleatoric_unc: Aleatoric uncertainty values
        total_unc: Total uncertainty values
        confidences: Confidence scores
        alphas: Evidence strength values
        correct_predictions: Boolean array of correct predictions
        experiment_name: Experiment name for saving
        model_name: Model name for saving
    """
    print("\n=== Advanced Uncertainty Correlation Analysis ===")
    
    epistemic_flat = epistemic_unc.squeeze() if epistemic_unc.ndim > 1 else epistemic_unc
    aleatoric_flat = aleatoric_unc.squeeze() if aleatoric_unc.ndim > 1 else aleatoric_unc
    total_flat = total_unc.squeeze() if total_unc.ndim > 1 else total_unc
    alphas_flat = alphas.squeeze() if alphas.ndim > 1 else alphas
    
    corr_data = np.column_stack([
        epistemic_flat, aleatoric_flat, total_flat, confidences, alphas_flat
    ])
    corr_labels = ['Epistemic', 'Aleatoric', 'Total Unc.', 'Confidence', 'Evidence']
    
    corr_matrix = np.corrcoef(corr_data.T)
    
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.3f', cbar_kws={"shrink": .8},
                xticklabels=corr_labels, yticklabels=corr_labels)
    plt.title('Uncertainty Metrics Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'results/{experiment_name}/{model_name}_correlation_matrix.png', 
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Epistemic-Aleatoric Correlation: {np.corrcoef(epistemic_flat, aleatoric_flat)[0,1]:.4f}")
    print(f"Total-Confidence Correlation: {np.corrcoef(total_flat, confidences)[0,1]:.4f}")
    print(f"Evidence-Confidence Correlation: {np.corrcoef(alphas_flat, confidences)[0,1]:.4f}")
    
    uncertainty_ratio = epistemic_flat / (epistemic_flat + aleatoric_flat + 1e-8)
    
    print(f"\n=== Uncertainty Decomposition Analysis ===")
    print(f"Mean Epistemic/(Epistemic+Aleatoric) Ratio: {np.mean(uncertainty_ratio):.4f}")
    print(f"Samples with high epistemic dominance (>0.7): {np.sum(uncertainty_ratio > 0.7)}")
    print(f"Samples with high aleatoric dominance (<0.3): {np.sum(uncertainty_ratio < 0.3)}")
    
    if correct_predictions.sum() > 0 and (~correct_predictions).sum() > 0:
        correct_epistemic = epistemic_flat[correct_predictions]
        incorrect_epistemic = epistemic_flat[~correct_predictions]
        
        t_stat, p_value = stats.ttest_ind(correct_epistemic, incorrect_epistemic)
        print(f"\n=== Statistical Significance Analysis ===")
        print(f"T-test (Correct vs Incorrect Epistemic): t={t_stat:.4f}, p={p_value:.6f}")
        
        if p_value < 0.05:
            print("✓ Significant difference in epistemic uncertainty")
        else:
            print("✗ No significant difference in epistemic uncertainty")
    
    print("✓ Advanced correlation analysis complete")


def create_spatial_uncertainty_maps(pred_probs, epistemic_unc, aleatoric_unc, total_unc, 
                                   confidences, alphas, targets, dataset_name,
                                   experiment_name, model_name):
    """
    Create spatial uncertainty maps with exact spatial representations.
    
    Args:
        pred_probs: Prediction probabilities
        epistemic_unc: Epistemic uncertainty
        aleatoric_unc: Aleatoric uncertainty
        total_unc: Total uncertainty
        confidences: Confidence scores
        alphas: Evidence strength
        targets: True labels
        dataset_name: Name of dataset
        experiment_name: Experiment name
        model_name: Model name
    """
    from .test_core import get_spatial_shape, pad_or_truncate_to_shape
    
    n_samples = len(pred_probs)
    predictions = np.argmax(pred_probs, axis=1)
    
    print(f"\n🗺️  Creating spatial maps for {dataset_name} ({n_samples} samples)")
    
    shape = get_spatial_shape(n_samples, dataset_name)
    print(f"✅ Using spatial shape: {shape[0]}×{shape[1]}")
    
    expected_samples = shape[0] * shape[1]
    
    if n_samples != expected_samples:
        print(f"Adjusting from {n_samples} to {expected_samples} samples")
        data_arrays = [predictions, epistemic_unc, aleatoric_unc, total_unc, confidences, alphas]
        adjusted = pad_or_truncate_to_shape(data_arrays, shape)
        predictions, epistemic_unc, aleatoric_unc, total_unc, confidences, alphas = adjusted
    
    try:
        pred_map = predictions.reshape(shape)
        epistemic_map = epistemic_unc.reshape(shape)
        aleatoric_map = aleatoric_unc.reshape(shape)
        total_uncertainty_map = total_unc.reshape(shape)
        confidence_map = confidences.reshape(shape)
        evidence_map = alphas.reshape(shape)
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Classification and Uncertainty Results - {dataset_name}', fontsize=18)
        
        defect_prob = np.argmax(predictions, axis=1) == 0 if predictions.ndim > 1 else predictions
        non_defect_map = defect_prob.astype(float).reshape(shape) if hasattr(defect_prob, 'reshape') else pred_map
        
        # 1. Predictions map
        im1 = axes[0, 0].imshow(non_defect_map, cmap='gray', interpolation='gaussian', aspect='equal')
        axes[0, 0].set_title(f'Classification Map\n(Non-Defect Probability)', fontsize=12)
        axes[0, 0].set_xlabel('Spatial X Position')
        axes[0, 0].set_ylabel('Spatial Y Position')
        plt.colorbar(im1, ax=axes[0, 0], shrink=0.8, label='Non-Defect Probability')
        
        # 2. Epistemic uncertainty
        im2 = axes[0, 1].imshow(epistemic_map, cmap='coolwarm', interpolation='gaussian', aspect='equal')
        axes[0, 1].set_title(f'Epistemic Uncertainty\n(Model Uncertainty)', fontsize=12)
        axes[0, 1].set_xlabel('Spatial X Position')
        axes[0, 1].set_ylabel('Spatial Y Position')
        plt.colorbar(im2, ax=axes[0, 1], shrink=0.8, label='Epistemic Uncertainty')
        
        # 3. Aleatoric uncertainty
        im3 = axes[0, 2].imshow(aleatoric_map, cmap='Blues_r', interpolation='gaussian', aspect='equal')
        axes[0, 2].set_title(f'Aleatoric Uncertainty\n(Data Uncertainty)', fontsize=12)
        axes[0, 2].set_xlabel('Spatial X Position')
        axes[0, 2].set_ylabel('Spatial Y Position')
        plt.colorbar(im3, ax=axes[0, 2], shrink=0.8, label='Aleatoric Uncertainty')
        
        # 4. Total uncertainty
        im4 = axes[1, 0].imshow(total_uncertainty_map, cmap='plasma', interpolation='gaussian', aspect='equal')
        axes[1, 0].set_title(f'Total Uncertainty', fontsize=12)
        axes[1, 0].set_xlabel('Spatial X Position')
        axes[1, 0].set_ylabel('Spatial Y Position')
        plt.colorbar(im4, ax=axes[1, 0], shrink=0.8, label='Total Uncertainty')
        
        # 5. Confidence
        im5 = axes[1, 1].imshow(confidence_map, cmap='Spectral_r', interpolation='gaussian', aspect='equal')
        axes[1, 1].set_title(f'Prediction Confidence\n(Higher=Better)', fontsize=12)
        axes[1, 1].set_xlabel('Spatial X Position')
        axes[1, 1].set_ylabel('Spatial Y Position')
        plt.colorbar(im5, ax=axes[1, 1], shrink=0.8, label='Confidence')
        
        # 6. Evidence strength
        im6 = axes[1, 2].imshow(evidence_map, cmap='viridis', interpolation='gaussian', aspect='equal')
        axes[1, 2].set_title(f'Evidence Strength\n(Alpha Sum)', fontsize=12)
        axes[1, 2].set_xlabel('Spatial X Position')
        axes[1, 2].set_ylabel('Spatial Y Position')
        plt.colorbar(im6, ax=axes[1, 2], shrink=0.8, label='Evidence Strength')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        safe_dataset_name = dataset_name.lower().replace(" ", "_").replace("-", "_")
        plt.savefig(f'results/{experiment_name}/{model_name}_uncertainty_maps_{safe_dataset_name}.png', 
                    dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Spatial uncertainty maps saved for {dataset_name}")
        
    except ValueError as e:
        print(f"❌ Error creating spatial maps for {dataset_name}: {e}")


def create_individual_defect_maps(pred_probs, epistemic_unc, aleatoric_unc, total_unc, 
                                 confidences, alphas, dataset_name,
                                 experiment_name, model_name):
    """
    Create individual baseline-style defect maps for each uncertainty type.
    
    Args:
        pred_probs: Prediction probabilities
        epistemic_unc: Epistemic uncertainty
        aleatoric_unc: Aleatoric uncertainty
        total_unc: Total uncertainty
        confidences: Confidence scores
        alphas: Evidence strength
        dataset_name: Name of dataset
        experiment_name: Experiment name
        model_name: Model name
    """
    from .test_core import get_spatial_shape, pad_or_truncate_to_shape
    
    print(f"\n🗺️  Creating individual defect maps for {dataset_name}...")
    
    n_samples = len(pred_probs)
    predictions = np.argmax(pred_probs, axis=1)
    defect_prob = pred_probs[:, 1]
    
    shape = get_spatial_shape(n_samples, dataset_name)
    print(f"✅ Using spatial shape: {shape[0]}×{shape[1]} for individual maps")
    
    expected_samples = shape[0] * shape[1]
    
    if n_samples != expected_samples:
        data_arrays = [defect_prob, epistemic_unc, aleatoric_unc, total_unc, confidences, alphas]
        adjusted = pad_or_truncate_to_shape(data_arrays, shape)
        defect_prob, epistemic_unc, aleatoric_unc, total_unc, confidences, alphas = adjusted
    
    try:
        classification_map = defect_prob.reshape(shape)
        epistemic_map = epistemic_unc.reshape(shape)
        aleatoric_map = aleatoric_unc.reshape(shape)
        total_uncertainty_map = total_unc.reshape(shape)
        confidence_map = confidences.reshape(shape)
        evidence_map = alphas.reshape(shape)
        
        save_individual_defect_maps(
            classification_map, epistemic_map, aleatoric_map,
            total_uncertainty_map, confidence_map, evidence_map,
            dataset_name, shape, experiment_name, model_name
        )
        
        print(f"✅ All individual defect maps generated for {dataset_name}")
        
    except ValueError as e:
        print(f"❌ Error creating individual defect maps for {dataset_name}: {e}")


def save_individual_defect_maps(classification_map, epistemic_map, aleatoric_map, 
                               total_uncertainty_map, confidence_map, evidence_map, 
                               dataset_name, shape, experiment_name, model_name):
    """
    Save individual baseline-style defect maps.
    
    Args:
        classification_map: Classification probability map
        epistemic_map: Epistemic uncertainty map
        aleatoric_map: Aleatoric uncertainty map
        total_uncertainty_map: Total uncertainty map
        confidence_map: Confidence map
        evidence_map: Evidence strength map
        dataset_name: Name of dataset
        shape: Spatial shape tuple
        experiment_name: Experiment name
        model_name: Model name
    """
    safe_name = dataset_name.lower().replace(" ", "_").replace("-", "_")
    
    print(f"🎨 Generating individual defect maps for {dataset_name} (Shape: {shape[0]}×{shape[1]})...")
    
    aspect_ratio = shape[1] / shape[0]
    fig_width = 12
    fig_height = fig_width / aspect_ratio
    
    # Classification map
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(classification_map, cmap='Spectral_r', interpolation='gaussian', aspect=1.0)
    plt.colorbar(label='Defect Probability', shrink=0.8)
    plt.title(f'Defect Map - {dataset_name}', fontsize=14)
    plt.xlabel('Spatial X Position')
    plt.ylabel('Spatial Y Position')
    plt.savefig(f'results/{experiment_name}/{model_name}_model_{safe_name}.png', 
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Individual defect maps saved for {dataset_name}")
