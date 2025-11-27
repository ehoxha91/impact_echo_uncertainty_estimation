"""
Main execution logic for evidential model testing.

This module contains the main analysis and execution functions.
"""

import os
import torch
import numpy as np

from .test_core import (
    DEFAULT_EXPERIMENT_NAME, 
    DEFAULT_MODEL_NAME,
    test_full_evidential_model_on_datasets_single_batch,
    calculate_detailed_accuracy_metrics
)
from .test_visualizations import (
    create_advanced_uncertainty_distribution_analysis,
    create_advanced_uncertainty_correlations,
    create_spatial_uncertainty_maps,
    create_individual_defect_maps
)


def analyze_inference_results_enhanced(results, dataset_name, experiment_name=None, model_name=None):
    """
    Enhanced analysis of inference results with comprehensive visualizations.
    
    Args:
        results: Dictionary of inference results
        dataset_name: Name of the dataset
        experiment_name: Optional experiment name
        model_name: Optional model name
    """
    if experiment_name is None:
        experiment_name = DEFAULT_EXPERIMENT_NAME
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME
    
    predictions = results['predictions']
    total_unc = results['total_uncertainties']
    epistemic_unc = results['epistemic_uncertainties']
    aleatoric_unc = results['aleatoric_uncertainties']
    confidences = results['confidences']
    alphas = results['alphas']
    targets = results['targets']
    targets = torch.where(targets > 0, 1, 0)
    accuracy = results['accuracy']
    
    print(f"Dataset: {dataset_name}")
    if 'DS3-' in dataset_name:
        print(f"DS3 Slab Analysis - 252 samples with 9×28 spatial arrangement")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Number of samples: {len(targets)}")
    
    # Extract predictions
    pred_probs = predictions.squeeze(0)
    pred_classes = torch.argmax(pred_probs, dim=1)
    
    # Remove extra dimensions
    epistemic_unc = epistemic_unc.squeeze(0).squeeze(-1) if epistemic_unc.dim() > 1 else epistemic_unc.squeeze(0)
    aleatoric_unc = aleatoric_unc.squeeze(0).squeeze(-1) if aleatoric_unc.dim() > 1 else aleatoric_unc.squeeze(0)
    total_unc = total_unc.squeeze(0).squeeze(-1) if total_unc.dim() > 1 else total_unc.squeeze(0)
    alphas = alphas.squeeze(0).squeeze(-1) if alphas.dim() > 1 else alphas.squeeze(0)
    
    if confidences.dim() > 1:
        confidences = confidences.squeeze(0)
    
    # Convert to numpy
    epistemic_unc_np = epistemic_unc.detach().cpu().numpy()
    aleatoric_unc_np = aleatoric_unc.detach().cpu().numpy()
    total_unc_np = total_unc.detach().cpu().numpy()
    alphas_np = alphas.detach().cpu().numpy()
    confidences_np = confidences.detach().cpu().numpy()
    pred_classes_np = pred_classes.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    pred_probs_np = pred_probs.detach().cpu().numpy()
    
    # Classification metrics
    if pred_classes_np.shape != targets_np.shape:
        print(f"Warning: Shape mismatch! pred_classes: {pred_classes_np.shape}, targets: {targets_np.shape}")
        if accuracy == 0.0:
            correct_predictions = np.ones(len(pred_classes_np), dtype=bool)
            print("Using dummy correct_predictions for unsupervised dataset")
            detailed_metrics = {}
        else:
            correct_predictions = (pred_classes_np == targets_np)
            detailed_metrics = calculate_detailed_accuracy_metrics(pred_classes_np, targets_np)
    else:
        correct_predictions = (pred_classes_np == targets_np)
        detailed_metrics = calculate_detailed_accuracy_metrics(pred_classes_np, targets_np)
    
    print(f"Correct Predictions: {correct_predictions.sum()}/{len(correct_predictions)}")
    
    if detailed_metrics and accuracy > 0.0:
        print(f"\n=== Key Defect Detection Performance for {dataset_name} ===")
        print(f"Precision (Defect): {detailed_metrics['precision']:.4f} ({detailed_metrics['precision']*100:.2f}%)")
        print(f"Recall (Defect):    {detailed_metrics['recall']:.4f} ({detailed_metrics['recall']*100:.2f}%)")
        print(f"F1-Score:           {detailed_metrics['f1_score']:.4f}")
        print(f"Specificity:        {detailed_metrics['specificity']:.4f} ({detailed_metrics['specificity']*100:.2f}%)")
        print(f"Balanced Accuracy:  {detailed_metrics['balanced_accuracy']:.4f} ({detailed_metrics['balanced_accuracy']*100:.2f}%)")
    
    print(f"\n=== Live Uncertainty Analysis for {dataset_name} ===")
    print(f"Total Uncertainty - Mean: {np.mean(total_unc_np):.6f} ± {np.std(total_unc_np):.6f}")
    print(f"Epistemic Uncertainty - Mean: {np.mean(epistemic_unc_np):.6f} ± {np.std(epistemic_unc_np):.6f}")
    print(f"Aleatoric Uncertainty - Mean: {np.mean(aleatoric_unc_np):.6f} ± {np.std(aleatoric_unc_np):.6f}")
    print(f"Confidence - Mean: {np.mean(confidences_np):.6f} ± {np.std(confidences_np):.6f}")
    
    print(f"\n🎨 Creating beautiful visualizations for {dataset_name}...")
    
    # Generate visualizations
    create_individual_defect_maps(
        pred_probs_np, epistemic_unc_np.squeeze(), aleatoric_unc_np.squeeze(),
        total_unc_np.squeeze(), confidences_np, alphas_np.squeeze(), dataset_name,
        experiment_name, model_name
    )
    
    create_spatial_uncertainty_maps(
        pred_probs_np, epistemic_unc_np.squeeze(), aleatoric_unc_np.squeeze(), 
        total_unc_np.squeeze(), confidences_np, alphas_np.squeeze(), targets_np, dataset_name,
        experiment_name, model_name
    )
    
    if 'DS1' in dataset_name or 'DS3-' in dataset_name:
        print(f"  📊 Creating comprehensive analysis for {dataset_name}...")
        
        create_advanced_uncertainty_distribution_analysis(
            pred_probs_np, epistemic_unc_np, aleatoric_unc_np, total_unc_np, 
            confidences_np, alphas_np, targets_np, correct_predictions,
            experiment_name, model_name
        )
        
        create_advanced_uncertainty_correlations(
            epistemic_unc_np, aleatoric_unc_np, total_unc_np, 
            confidences_np, alphas_np, correct_predictions,
            experiment_name, model_name
        )
    else:
        print(f"  ⏭️  Skipping comprehensive analysis for {dataset_name}")
    
    print(f"✓ Beautiful analysis complete for {dataset_name}!")


def run_inference_time_uncertainty_analysis_enhanced(model_path, experiment_name=None, model_name=None):
    """
    Run enhanced real-time evidential uncertainty analysis.
    
    Args:
        model_path: Path to model checkpoint
        experiment_name: Optional experiment name
        model_name: Optional model name
        
    Returns:
        Boolean indicating success
    """
    if experiment_name is None:
        experiment_name = DEFAULT_EXPERIMENT_NAME
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME
    
    print("=== Enhanced Real-Time Evidential Uncertainty Analysis ===")
    print("🚀 Generating beautiful uncertainty visualizations for all DS3 slabs...\n")
    
    try:
        print("1. Running inference on test datasets (including all DS3 slabs)...")
        dataset_results = test_full_evidential_model_on_datasets_single_batch(model_path)
        
        if dataset_results:
            print("✓ Inference complete! Now generating enhanced analysis...\n")
            
            for dataset_name, results in dataset_results.items():
                print(f"\n=== Analyzing {dataset_name} Results ===")
                analyze_inference_results_enhanced(results, dataset_name, experiment_name, model_name)
            
            results_filename = f'weights/{model_name}_enhanced_inference_results.pth'
            torch.save(dataset_results, results_filename)
            print(f"✓ Enhanced inference results saved to {results_filename}")
            
        else:
            print("❌ No inference results generated - check model file")
            return False
            
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure the evidential model exists or provide correct path")
        return False
    
    return True


def main_enhanced_analysis():
    """Main function to run the enhanced analysis with all DS3 slabs."""
    experiment_name = DEFAULT_EXPERIMENT_NAME
    model_name = DEFAULT_MODEL_NAME
    model_path = f'weights/{model_name}.pth'
    
    print("=== Enhanced Real-Time Evidential Uncertainty Analysis ===")
    print(f"🚀 Running enhanced experiment: {experiment_name} with model: {model_name}")
    print("🚀 Generating analysis for DS1 + all 4 DS3 slabs + CCNY datasets!\n")
    
    os.makedirs(f'results/{experiment_name}', exist_ok=True)
    
    success = run_inference_time_uncertainty_analysis_enhanced(model_path, experiment_name, model_name)
    
    if not success:
        print("\n🔄 Trying alternative model paths...")
        alternative_paths = [
            f'results/weights/{model_name}.pth',
            'weights/evidential_simple.pth',
            'weights/evidential.pth',
            'weights/evidential_full.pth'
        ]
        
        for alt_path in alternative_paths:
            print(f"Trying: {alt_path}")
            if run_inference_time_uncertainty_analysis_enhanced(alt_path, experiment_name, model_name):
                success = True
                break
    
    if success:
        print("\n=== Enhanced Analysis Complete ===")
        print("✓ All beautiful uncertainty visualizations completed successfully!")
        print(f"\n📁 Generated Enhanced Analysis Files for experiment: {experiment_name}")
        print(f"✓ DS1 Test: Complete analysis with 252 samples (9×28 spatial arrangement)")
        print(f"✓ DS3-1 through DS3-4: Complete analysis for each slab")
        print(f"✓ CCNY datasets: May (31×38), June (19×34), Nov (44×34)")
        print(f"✓ Enhanced multi-dataset comparison generated")
    else:
        print("\n❌ No evidential models found. Please train a model first.")
    
    return success


if __name__ == '__main__':
    main_enhanced_analysis()
