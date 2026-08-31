"""
src.lib - Common utilities library for training and inference scripts.

Provides reusable components for:
- Logging configuration
- System and GPU utilities
- Configuration dataclasses
- Checkpoint management
- Model conversion and export utilities
- Quantization utilities
- ONNX analysis and validation
"""

# Try to import each module, but don't fail if some are missing (for selective use)
try:
    from .checkpoint import CheckpointManager, setup_checkpoint_arguments
except ImportError:
    pass

try:
    from .config import (
        CNNConfig,
        HierCodeConfig,
        OptimizationConfig,
        QATConfig,
        RadicalRNNConfig,
        RNNConfig,
        ViTConfig,
    )
except ImportError:
    pass

try:
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
        quantize_model,
        quantize_model_int8,
        quantize_state_dict_int8,
        quantize_tensor_to_f16,
        quantize_tensor_to_q4,
        quantize_tensor_to_q8,
        save_model_with_metadata,
    )
except ImportError:
    pass

try:
    from .logging_utils import setup_logger, suppress_warnings
except ImportError:
    pass

try:
    from .onnx import (
        export_to_onnx,
        get_onnx_model_info,
        test_onnx_inference,
        validate_onnx_model,
    )
except ImportError:
    pass

try:
    from .optimization import (
        get_optimizer,
        get_scheduler,
        load_config_from_json,
        save_config,
    )
except ImportError:
    pass

try:
    from .system import (
        check_gpu_availability,
        check_virtual_environment,
        get_system_info,
        verify_and_setup_gpu,
    )
except ImportError:
    pass

try:
    from .training import (
        collect_training_metrics,
        create_results_directory,
        load_best_model_for_testing,
        save_best_model,
        save_training_history,
        save_training_results,
    )
except ImportError:
    pass

# New modules added in consolidation
try:
    from .character_mapping import (
        CharacterMappingGenerator,
        JISConverter,
    )
except ImportError:
    pass

try:
    from .metadata_generator import (
        ChunkMetadataGenerator,
        DatasetMetadataManager,
        RootMetadataGenerator,
    )
except ImportError:
    pass

try:
    from .model_export import ModelExporter
except ImportError:
    pass

try:
    from .onnx_analysis import (
        ONNXModelAnalyzer,
        PoolingComparisonAnalyzer,
    )
except ImportError:
    pass

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
    "quantize_model",
    "quantize_model_int8",
    "quantize_state_dict_int8",
    "quantize_tensor_to_q4",
    "quantize_tensor_to_q8",
    "quantize_tensor_to_f16",
    "format_export_filename",
    "log_conversion_summary",
    "save_model_with_metadata",
    # Configuration
    "CNNConfig",
    "RNNConfig",
    "ViTConfig",
    "QATConfig",
    "RadicalRNNConfig",
    "HierCodeConfig",
    "OptimizationConfig",
    # Checkpoint
    "CheckpointManager",
    "setup_checkpoint_arguments",
    # Optimization
    "get_optimizer",
    "get_scheduler",
    "load_config_from_json",
    "save_config",
    # Training
    "collect_training_metrics",
    "create_results_directory",
    "save_training_history",
    "save_training_results",
    "save_best_model",
    "load_best_model_for_testing",
    # ONNX
    "export_to_onnx",
    "get_onnx_model_info",
    "test_onnx_inference",
    "validate_onnx_model",
    # Character mapping
    "JISConverter",
    "CharacterMappingGenerator",
    # Metadata generation
    "ChunkMetadataGenerator",
    "RootMetadataGenerator",
    "DatasetMetadataManager",
    # Model export
    "ModelExporter",
    # ONNX analysis
    "ONNXModelAnalyzer",
    "PoolingComparisonAnalyzer",
]


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
