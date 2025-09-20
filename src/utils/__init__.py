"""
Utility functions for the nnef-dists package.
"""

# from .performance import measure_jit_runtime, measure_memory_usage, benchmark_model  # Not available
from .matrix_utils import vector_to_upper_triangular, upper_triangular_to_vector
from .training import TrainingProgress, add_timing_to_history, train_with_progress
from .exact_covariance import (
    compute_exact_covariance_matrix, 
    compute_covariance_for_eta,
    verify_covariance_properties,
    supports_exact_covariance
)
from .data_utils import (
    load_training_data,
    load_3d_gaussian_data,
    compute_ground_truth_3d_tril,
    list_data_files,
    load_latest_data,
    inspect_data,
    quick_load
)

__all__ = [
    # 'measure_jit_runtime',  # Not available
    # 'measure_memory_usage', 
    # 'benchmark_model',
    'vector_to_upper_triangular',
    'upper_triangular_to_vector',
    'TrainingProgress',
    'add_timing_to_history',
    'train_with_progress',
    'compute_exact_covariance_matrix',
    'compute_covariance_for_eta', 
    'verify_covariance_properties',
    'supports_exact_covariance',
    'load_training_data',
    'load_3d_gaussian_data',
    'compute_ground_truth_3d_tril',
    'list_data_files',
    'load_latest_data',
    'inspect_data',
    'quick_load'
]
