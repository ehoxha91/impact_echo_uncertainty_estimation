"""
Testing module for evidential deep learning models.

This module provides comprehensive testing and visualization capabilities for
evidential deep learning models, including uncertainty quantification and
spatial analysis across multiple datasets.
"""

from .test_core import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_MODEL_NAME,
    DATASET_SHAPES,
    load_model_checkpoint,
    load_ds3_multi_slab_data_into_torch_tensor,
    ensure_binary_predictions,
    process_ds3_slab_with_dataloader,
    process_ds3_slab_in_batches,
    test_full_evidential_model_on_datasets_single_batch,
    calculate_detailed_accuracy_metrics,
    get_spatial_shape,
    pad_or_truncate_to_shape,
)

from .test_visualizations import (
    create_advanced_uncertainty_distribution_analysis,
    create_advanced_uncertainty_correlations,
    create_spatial_uncertainty_maps,
    create_individual_defect_maps,
    save_individual_defect_maps,
)

from .test_analysis import (
    analyze_inference_results_enhanced,
    run_inference_time_uncertainty_analysis_enhanced,
    main_enhanced_analysis,
)

__all__ = [
    # Constants
    'DEFAULT_EXPERIMENT_NAME',
    'DEFAULT_MODEL_NAME',
    'DATASET_SHAPES',
    
    # Core functions
    'load_model_checkpoint',
    'load_ds3_multi_slab_data_into_torch_tensor',
    'ensure_binary_predictions',
    'process_ds3_slab_with_dataloader',
    'process_ds3_slab_in_batches',
    'test_full_evidential_model_on_datasets_single_batch',
    'calculate_detailed_accuracy_metrics',
    'get_spatial_shape',
    'pad_or_truncate_to_shape',
    
    # Visualization functions
    'create_advanced_uncertainty_distribution_analysis',
    'create_advanced_uncertainty_correlations',
    'create_spatial_uncertainty_maps',
    'create_individual_defect_maps',
    'save_individual_defect_maps',
    
    # Analysis functions
    'analyze_inference_results_enhanced',
    'run_inference_time_uncertainty_analysis_enhanced',
    'main_enhanced_analysis',
]
