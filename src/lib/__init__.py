"""
src.lib - Common utilities library for training and inference scripts.

Provides reusable components for:
- Logging configuration
- System and GPU utilities
- Model type management and path handling
- Configuration dataclasses
- Dataset loading and preprocessing
- Checkpoint management
- ONNX conversion utilities
- Model conversion and export utilities
"""

# Model conversion utilities
# Logging utilities
# Checkpoint management
from .checkpoint import CheckpointManager, setup_checkpoint_arguments

# Configuration classes
from .config import (
    CNNConfig,
    HierCodeConfig,
    OptimizationConfig,
    QATConfig,
    RadicalRNNConfig,
    RNNConfig,
    ViTConfig,
)
from .conversion import (
    calculate_compression_ratio,
    calculate_model_size,
    dequantize_state_dict,
    format_export_filename,
    get_model_date,
    infer_num_classes_from_state_dict,
    load_model_checkpoint,
    load_num_classes_from_config,
    log_conversion_summary,
    quantize_model_int8,
    quantize_state_dict_int8,
    quantize_tensor_to_f16,
    quantize_tensor_to_q4,
    quantize_tensor_to_q8,
    save_model_with_metadata,
)

# Dataset utilities
from .dataset import (
    SimpleDataset,
    create_data_loaders,
    get_dataset_directory,
    load_chunked_dataset,
    load_dataset,
    prepare_dataset_and_loaders,
    split_dataset,
    verify_dataset,
)
from .logging_utils import setup_logger, suppress_warnings

# Model utilities
from .models import (
    SUPPORTED_MODEL_TYPES,
    TRAINING_STRUCTURE,
    generate_export_path,
    get_training_dir,
    infer_model_type,
    is_model_type_valid,
)

# ONNX utilities
from .onnx import (
    export_to_onnx,
    get_onnx_model_info,
    test_onnx_inference,
    validate_onnx_model,
)

# Optimization utilities
from .optimization import (
    get_optimizer,
    get_scheduler,
    load_config_from_json,
    save_config,
)

# System utilities
from .system import (
    check_gpu_availability,
    check_virtual_environment,
    get_system_info,
    verify_and_setup_gpu,
)

# Training utilities
from .training import (
    collect_training_metrics,
    create_results_directory,
    load_best_model_for_testing,
    save_best_model,
    save_training_history,
    save_training_results,
)

__all__ = [
    # Logging
    "setup_logger",
    "suppress_warnings",
    # System
    "verify_and_setup_gpu",
    "check_gpu_availability",
    "check_virtual_environment",
    "get_system_info",
    # Conversion
    "get_model_date",
    "dequantize_state_dict",
    "infer_num_classes_from_state_dict",
    "load_num_classes_from_config",
    "load_model_checkpoint",
    "calculate_model_size",
    "calculate_compression_ratio",
    "quantize_model_int8",
    "quantize_state_dict_int8",
    "quantize_tensor_to_q4",
    "quantize_tensor_to_q8",
    "quantize_tensor_to_f16",
    "save_model_with_metadata",
    "format_export_filename",
    "log_conversion_summary",
    # Models
    "SUPPORTED_MODEL_TYPES",
    "TRAINING_STRUCTURE",
    "infer_model_type",
    "generate_export_path",
    "get_training_dir",
    "is_model_type_valid",
    # Configs
    "OptimizationConfig",
    "CNNConfig",
    "RNNConfig",
    "QATConfig",
    "RadicalRNNConfig",
    "HierCodeConfig",
    "ViTConfig",
    # Dataset
    "get_dataset_directory",
    "load_dataset",
    "load_chunked_dataset",
    "split_dataset",
    "create_data_loaders",
    "prepare_dataset_and_loaders",
    "SimpleDataset",
    "verify_dataset",
    # Optimization
    "get_optimizer",
    "get_scheduler",
    "save_config",
    "load_config_from_json",
    # Training
    "save_best_model",
    "save_training_results",
    "save_training_history",
    "load_best_model_for_testing",
    "create_results_directory",
    "collect_training_metrics",
    # Checkpoint
    "CheckpointManager",
    "setup_checkpoint_arguments",
    # ONNX
    "export_to_onnx",
    "validate_onnx_model",
    "get_onnx_model_info",
    "test_onnx_inference",
]
