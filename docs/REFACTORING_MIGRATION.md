# Code Refactoring & Migration Guide

This document describes the consolidation refactoring that eliminated ~5,000 LOC of duplication by moving reusable patterns into `src/lib/` modules and creating unified CLI entry points.

**Summary**: 31 Python scripts consolidated into 6 reusable modules + 3 unified CLI entry points. All original functionality preserved with 387+ passing tests.

---

## 🎯 Overview: Old Scripts → New Modules

| Original Scripts                                                     | New Home                                    | Purpose                      |
| -------------------------------------------------------------------- | ------------------------------------------- | ---------------------------- |
| `generate_mapping.py`, `create_class_mapping.py`, etc. (3 scripts)   | `src/lib/character_mapping.py`              | JIS ↔ Unicode conversion     |
| `generate_chunk_metadata.py`, `create_dataset_metadata.py` (2)       | `src/lib/metadata_generator.py`             | Dataset metadata generation  |
| `quantize_model.py`, `quantize_to_4bit_bitsandbytes.py` (2)          | `src/lib/conversion.py`                     | All quantization formats     |
| `convert_to_onnx.py`, `convert_to_safetensors.py`, etc. (3)          | `src/lib/model_export.py`                   | Multi-format model export    |
| `inspect_onnx_model.py`, `pooling_comparison.py` (2)                 | `src/lib/onnx_analysis.py`                  | ONNX model inspection        |
| `train_cnn_model.py`, `train_rnn.py`, etc. (7 training scripts)      | `src/lib/base_trainer.py`                   | Training loop consolidation  |
| `preflight_check.py`, `verify_etl9g_setup.py` (2)                    | `src/lib/setup_verification.py`             | Environment verification     |

**New Unified CLI Entry Points**:

| New CLI Script           | Replaces                                                     | Purpose                              |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------ |
| `scripts/convert_model.py`  | `convert_to_gguf.py`, `convert_to_onnx.py`, etc. (5 scripts) | Unified model conversion & export    |
| `scripts/verify_setup.py`   | `preflight_check.py`, `verify_etl9g_setup.py` (2 scripts)    | Unified environment verification     |
| `scripts/analyze_model.py`  | `inspect_onnx_model.py`, `pooling_comparison.py` (3 scripts) | Unified ONNX model analysis          |

---

## 📦 Library Modules Reference

### 1. Character Mapping (`src/lib/character_mapping.py`)

**Purpose**: Consolidates JIS X 0208 ↔ Unicode character conversion across datasets.

**Replaces**: `generate_mapping.py`, `create_class_mapping.py`, `generate_complete_class_mapping.py`

#### Key Classes

**`JISConverter`** - Static methods for JIS code conversion
```python
from src.lib.character_mapping import JISConverter

# Convert JIS code to Unicode character
char = JISConverter.jis_to_unicode(0x2421)  # Returns '亜'

# Get character type (kanji, hiragana, katakana, etc.)
char_type = JISConverter.get_character_type('亜')

# Estimate stroke count
strokes = JISConverter.estimate_stroke_count('亜')
```

**`CharacterMappingGenerator`** - Generate bidirectional character mappings
```python
from src.lib.character_mapping import CharacterMappingGenerator

generator = CharacterMappingGenerator(output_dir="output")

# Generate mappings from dataset metadata
mapping = generator.generate_from_metadata(
    metadata_path="dataset/etl9g/metadata.json",
    include_strokes=True
)

# Generate complete mapping for all 43,427 classes
complete_mapping = generator.generate_complete_mapping(
    include_stroke_counts=True,
    output_dir="output"
)
```

#### Migration Example

**Before** (3 separate scripts):
```python
# generate_mapping.py
import json
jis_codes = [0x2421, 0x2422, 0x2423]
mapping = {}
for code in jis_codes:
    mapping[str(code)] = chr(code)
with open('mapping.json', 'w') as f:
    json.dump(mapping, f)

# create_class_mapping.py
class_to_char = {}
for i, jis_code in enumerate(jis_codes):
    class_to_char[i] = chr(jis_code)

# generate_complete_class_mapping.py
# ... similar duplication for 43,427 classes
```

**After** (single reusable module):
```python
from src.lib.character_mapping import CharacterMappingGenerator

gen = CharacterMappingGenerator(output_dir="output")
gen.generate_complete_mapping(include_stroke_counts=True)
```

---

### 2. Metadata Generator (`src/lib/metadata_generator.py`)

**Purpose**: Consolidates dataset metadata generation for all ETL variants.

**Replaces**: `generate_chunk_metadata.py`, `create_dataset_metadata.py`

#### Key Classes

**`ChunkMetadataGenerator`** - Generate metadata for dataset chunks
```python
from src.lib.metadata_generator import ChunkMetadataGenerator

gen = ChunkMetadataGenerator(data_dir="dataset/etl9g")

# Generate metadata for all chunks
metadata = gen.generate_for_dataset(
    dataset_name="etl9g",
    chunk_dir="dataset/etl9g/chunks"
)
```

**`RootMetadataGenerator`** - Generate root-level dataset metadata
```python
from src.lib.metadata_generator import RootMetadataGenerator

gen = RootMetadataGenerator(data_dir="dataset")

# Generate metadata for entire dataset
root_metadata = gen.create_root_metadata(
    output_path="dataset/metadata.json"
)
```

**`DatasetMetadataManager`** - Unified metadata management
```python
from src.lib.metadata_generator import DatasetMetadataManager

manager = DatasetMetadataManager(data_dir="dataset")

# Generate all metadata automatically
manager.generate_all()

# Check if dataset is properly validated
if manager.validate_dataset("etl9g"):
    print("✓ Dataset ready for training")
```

#### Migration Example

**Before**:
```bash
python generate_chunk_metadata.py --data-dir dataset/etl9g
python create_dataset_metadata.py --input dataset/etl9g
```

**After** (programmatic):
```python
from src.lib.metadata_generator import DatasetMetadataManager
manager = DatasetMetadataManager(data_dir="dataset")
manager.generate_all()
```

---

### 3. Model Conversion (`src/lib/conversion.py`)

**Purpose**: Unified quantization and format conversion API.

**Replaces**: `quantize_model.py`, `quantize_to_4bit_bitsandbytes.py`

#### Key Functions

**`quantize_model(model, quantization_format, device)`** - Unified quantization API
```python
from src.lib.conversion import quantize_model
import torch

model = torch.load("model.pth")

# INT8 Quantization (~4x compression)
quantized_model, metadata = quantize_model(
    model, 
    quantization_format="int8",
    device="cpu"
)

# 4-bit NF4 (~8x compression, best accuracy)
quantized_model, metadata = quantize_model(
    model,
    quantization_format="4bit_nf4",
    device="cuda"
)

# 4-bit FP4 (~8x compression, extreme size)
quantized_model, metadata = quantize_model(
    model,
    quantization_format="4bit_fp4",
    device="cuda"
)

# BFloat16 (~2x compression)
quantized_model, metadata = quantize_model(
    model,
    quantization_format="bfloat16",
    device="cpu"
)
```

#### Migration Example

**Before**:
```bash
# INT8 quantization (dedicated script)
python quantize_model.py --model-path model.pth --model-type cnn

# 4-bit quantization (separate script)
python quantize_to_4bit_bitsandbytes.py --model-path model.pth --device cuda
```

**After** (unified CLI):
```bash
# All quantization formats with single command
uv run python scripts/convert_model.py \
  --checkpoint model.pth \
  --quantization int8 \
  --format pytorch

uv run python scripts/convert_model.py \
  --checkpoint model.pth \
  --quantization 4bit_nf4 \
  --format onnx
```

---

### 4. Model Export (`src/lib/model_export.py`)

**Purpose**: Multi-format model export with automatic metadata generation.

**Replaces**: `convert_to_gguf.py`, `convert_to_onnx.py`, `convert_to_safetensors.py`

#### Key Class

**`ModelExporter`** - Unified export interface
```python
from src.lib.model_export import ModelExporter
import torch

model = torch.load("model.pth")
exporter = ModelExporter(model_type="cnn", num_classes=43427, image_size=64)

# Export to ONNX
export_path = exporter.export_onnx(
    model=model,
    output_path="model_float32.onnx",
    opset_version=13
)

# Export to SafeTensors (with quantization)
export_path = exporter.export_safetensors(
    model=model,
    output_path="model_int8.safetensors",
    quantization_format="int8"
)

# Export to GGUF (CPU inference)
export_path = exporter.export_gguf(
    model=model,
    output_path="model.gguf",
    quantization_format="int8"
)

# Unified export with automatic format detection
export_path = exporter.export(
    model=model,
    output_path="model.onnx",
    export_format="onnx",
    quantization_format="int8",
    save_metadata=True
)
```

#### Migration Example

**Before**:
```bash
# Separate scripts for each format
python convert_to_onnx.py --model model.pth
python convert_to_safetensors.py --model model.pth
python convert_to_gguf.py --model model.pth
```

**After** (unified CLI):
```bash
# Single CLI with format argument
uv run python scripts/convert_model.py \
  --checkpoint model.pth \
  --format onnx

uv run python scripts/convert_model.py \
  --checkpoint model.pth \
  --format safetensors

uv run python scripts/convert_model.py \
  --checkpoint model.pth \
  --format gguf
```

---

### 5. ONNX Analysis (`src/lib/onnx_analysis.py`)

**Purpose**: Comprehensive ONNX model inspection and comparison.

**Replaces**: `inspect_onnx_model.py`, `onnx_operations_comparison.py`, `pooling_comparison.py`

#### Key Classes

**`ONNXModelAnalyzer`** - ONNX model inspection
```python
from src.lib.onnx_analysis import ONNXModelAnalyzer

analyzer = ONNXModelAnalyzer("model.onnx")

# Get model structure summary
summary = analyzer.analyze()

# Print detailed analysis
analyzer.print_summary()

# Save analysis to JSON
analyzer.save_analysis("analysis.json")

# Get specific information
inputs = analyzer.get_input_info()
outputs = analyzer.get_output_info()
nodes = analyzer.get_node_info()
```

**`PoolingComparisonAnalyzer`** - Compare pooling implementations
```python
from src.lib.onnx_analysis import PoolingComparisonAnalyzer

# Static methods for pooling comparison
comparison = PoolingComparisonAnalyzer.compare_pooling_operations(
    model_path="model.onnx"
)

efficiency = PoolingComparisonAnalyzer.analyze_pooling_efficiency(
    model_path="model.onnx"
)
```

#### Migration Example

**Before**:
```bash
# Inspect model structure
python inspect_onnx_model.py --model model.onnx

# Compare pooling implementations
python pooling_comparison.py --model model.onnx

# Compare operations
python onnx_operations_comparison.py --model model.onnx
```

**After** (unified CLI):
```bash
# Single CLI with --inspect argument
uv run python scripts/analyze_model.py \
  --model model.onnx \
  --inspect structure

uv run python scripts/analyze_model.py \
  --model model.onnx \
  --inspect pooling

uv run python scripts/analyze_model.py \
  --model model.onnx \
  --inspect operations
```

---

### 6. Training Base Classes (`src/lib/base_trainer.py`)

**Purpose**: Unified training loop for all model architectures with checkpoint management.

**Replaces**: `train_cnn_model.py`, `train_rnn.py`, `train_vit.py`, `train_hiercode.py`, `train_qat.py`, `train_radical_rnn.py`, `train_hiercode_higita.py`

#### Key Classes

**`BaseModelTrainer`** - Abstract base class for all trainers
```python
from src.lib.base_trainer import BaseModelTrainer, setup_trainer_for_model
import torch

model = torch.load("model.pth")
train_loader = ...  # DataLoader
val_loader = ...    # DataLoader
optimizer = torch.optim.AdamW(model.parameters())

# Option 1: Use factory function (recommended)
trainer = setup_trainer_for_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    model_type="cnn",
    checkpoint_dir="checkpoints",
    num_classes=43427,
    image_size=64
)

# Option 2: Use architecture-specific trainer directly
from src.lib.base_trainer import CNNTrainer

trainer = CNNTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    checkpoint_dir="checkpoints",
    num_classes=43427,
    image_size=64
)

# Train model
history = trainer.train(
    num_epochs=30,
    early_stopping=True,
    save_best_model=True
)

# Access training metrics
print(f"Best validation accuracy: {trainer.best_val_accuracy}")
print(f"Training history: {history}")
```

**Supported Architecture Trainers**:
- `CNNTrainer` - CNN models (LightweightKanjiNet)
- `RNNTrainer` - RNN models (KanjiRNN, RadicalRNN)
- `ViTTrainer` - Vision Transformer (with mixed precision)
- `HierCodeTrainer` - HierCode models
- `QATTrainer` - Quantization Aware Training models
- `HierCodeHiGITATrainer` - HierCode with Hi-GITA enhancement

#### Migration Example

**Before** (7 separate training scripts):
```bash
# CNN training
python train_cnn_model.py --epochs 30 --batch-size 64

# RNN training  
python train_rnn.py --epochs 30 --model-type hybrid_cnn_rnn

# ViT training
python train_vit.py --epochs 40 --batch-size 256

# Hiercode training
python train_hiercode.py --epochs 30
```

**After** (single unified interface):
```bash
# All models use same entry point with --model-type
uv run python scripts/train.py cnn --epochs 30
uv run python scripts/train.py rnn --epochs 30 --model-type hybrid_cnn_rnn
uv run python scripts/train.py vit --epochs 40
uv run python scripts/train.py hiercode --epochs 30

# Or programmatically:
from src.lib.base_trainer import setup_trainer_for_model

trainer = setup_trainer_for_model(model, train_loader, val_loader, optimizer, model_type="cnn")
history = trainer.train(num_epochs=30)
```

---

### 7. Setup Verification (`src/lib/setup_verification.py`)

**Purpose**: Comprehensive environment and dataset validation.

**Replaces**: `preflight_check.py`, `verify_etl9g_setup.py`

#### Key Class

**`SetupVerifier`** - Unified environment verification
```python
from src.lib.setup_verification import SetupVerifier

verifier = SetupVerifier(data_dir="dataset", verbose=True)

# Check specific components
python_ok = verifier.verify_python_version()
venv_ok = verifier.verify_virtual_environment()
deps_ok = verifier.verify_dependencies()
gpu_info = verifier.verify_gpu_setup()
system_info = verifier.verify_system_resources()
dataset_info = verifier.verify_dataset_structure("dataset")

# Comprehensive check
results = verifier.run_all_checks(
    check_training_scripts=True,
    check_model=False
)

if results["all_ok"]:
    print("✓ Environment ready for training")
else:
    print("✗ Fix issues before training:", results["errors"])
```

#### Migration Example

**Before**:
```bash
# Check environment preflight
python preflight_check.py

# Check ETL9G dataset
python verify_etl9g_setup.py --data-dir dataset/etl9g
```

**After** (unified CLI):
```bash
# All checks with single command
uv run python scripts/verify_setup.py

# Specific checks
uv run python scripts/verify_setup.py --check dependencies
uv run python scripts/verify_setup.py --check gpu
uv run python scripts/verify_setup.py --check etl9g --data-dir dataset/etl9g
```

---

## 🔧 Unified CLI Entry Points

### 1. `scripts/convert_model.py` - Model Conversion

Replaces 5 old scripts: `convert_to_gguf.py`, `convert_to_onnx.py`, `convert_to_safetensors.py`, `quantize_model.py`, `quantize_to_4bit_bitsandbytes.py`

```bash
# Export to ONNX (float32)
uv run python scripts/convert_model.py \
  --checkpoint training/cnn/best_model.pth \
  --format onnx

# Export with INT8 quantization
uv run python scripts/convert_model.py \
  --checkpoint training/cnn/best_model.pth \
  --format onnx \
  --quantization int8

# Export 4-bit NF4 to ONNX
uv run python scripts/convert_model.py \
  --checkpoint training/rnn/best_model.pth \
  --format onnx \
  --quantization 4bit_nf4 \
  --model-type rnn

# Export to SafeTensors
uv run python scripts/convert_model.py \
  --checkpoint model.pth \
  --format safetensors \
  --quantization int8

# Export to GGUF (CPU inference)
uv run python scripts/convert_model.py \
  --checkpoint model.pth \
  --format gguf \
  --quantization int8

# List all options
uv run python scripts/convert_model.py --help
```

**Arguments**:
- `--checkpoint` (required): Path to PyTorch model checkpoint
- `--format`: pytorch|onnx|safetensors|gguf (default: onnx)
- `--quantization`: none|int8|4bit_nf4|4bit_fp4|bfloat16|float32 (default: none)
- `--output`: Custom output path (auto-generated if not specified)
- `--model-type`: cnn|rnn|vit|hiercode|qat|hiercode_higita (default: cnn)
- `--num-classes`: Number of output classes (default: 43427)
- `--image-size`: Input image size (default: 64)
- `--device`: cuda|cpu (default: auto-detect)
- `--save-metadata`: Save JSON metadata (default: True)
- `-v/--verbose`: Enable debug logging

---

### 2. `scripts/verify_setup.py` - Environment Verification

Replaces 2 old scripts: `preflight_check.py`, `verify_etl9g_setup.py`

```bash
# Run all checks (environment, dependencies, GPU, system, dataset)
uv run python scripts/verify_setup.py

# Check only environment setup
uv run python scripts/verify_setup.py --check environment

# Check only dependencies
uv run python scripts/verify_setup.py --check dependencies

# Check GPU/CUDA setup
uv run python scripts/verify_setup.py --check gpu

# Check system resources (RAM, disk)
uv run python scripts/verify_setup.py --check system

# Check dataset structure
uv run python scripts/verify_setup.py --check dataset --data-dir dataset

# Check ETL9G specific validation
uv run python scripts/verify_setup.py --check etl9g --data-dir dataset/etl9g

# Check training time estimation
uv run python scripts/verify_setup.py --check training-time

# Output as JSON (for scripting)
uv run python scripts/verify_setup.py --json

# Verbose mode
uv run python scripts/verify_setup.py -v

# List all options
uv run python scripts/verify_setup.py --help
```

**Arguments**:
- `--check`: all|environment|dependencies|gpu|system|dataset|etl9g|training-scripts|training-time (default: all)
- `--data-dir`: Path to data directory (default: dataset)
- `--include-training-scripts`: Include training scripts verification (default: False)
- `-v/--verbose`: Enable debug logging
- `--json`: Output results as JSON

---

### 3. `scripts/analyze_model.py` - ONNX Model Analysis

Replaces 3 old scripts: `inspect_onnx_model.py`, `onnx_operations_comparison.py`, `pooling_comparison.py`

```bash
# Analyze model structure (inputs, outputs, nodes)
uv run python scripts/analyze_model.py --model model.onnx --inspect structure

# Analyze operations and computational cost
uv run python scripts/analyze_model.py --model model.onnx --inspect operations

# Compare pooling implementations
uv run python scripts/analyze_model.py --model model.onnx --inspect pooling

# Comprehensive analysis
uv run python scripts/analyze_model.py --model model.onnx --inspect all

# Save analysis to JSON file
uv run python scripts/analyze_model.py \
  --model model.onnx \
  --inspect all \
  --output analysis.json

# Verbose output
uv run python scripts/analyze_model.py --model model.onnx -v

# List all options
uv run python scripts/analyze_model.py --help
```

**Arguments**:
- `--model` (required): Path to ONNX model file
- `--inspect`: structure|operations|pooling|all (default: structure)
- `--output`: Output file path for JSON results
- `-v/--verbose`: Enable verbose output

---

## 🔍 Finding Your Use Case

**Q: I was using `train_cnn_model.py` - what should I use now?**
A: Replace with `scripts/train.py cnn --epochs 30` (unified CLI with same arguments)

**Q: I was using `convert_to_onnx.py` - what's the new command?**
A: Use `scripts/convert_model.py --checkpoint model.pth --format onnx`

**Q: I was using `quantize_model.py` and `quantize_to_4bit_bitsandbytes.py` - consolidated how?**
A: Use `scripts/convert_model.py --checkpoint model.pth --quantization int8` or `--quantization 4bit_nf4`

**Q: I was using `preflight_check.py` and `verify_etl9g_setup.py` - how do I verify setup?**
A: Use `scripts/verify_setup.py` (unified, supports `--check environment`, `--check etl9g`, etc.)

**Q: I want to use the modules directly in Python code - how?**
A: Import from `src.lib.*`:
```python
from src.lib.character_mapping import JISConverter
from src.lib.conversion import quantize_model
from src.lib.base_trainer import setup_trainer_for_model
from src.lib.setup_verification import SetupVerifier
```

---

## 📊 Consolidation Summary

| Metric                         | Value  |
| ------------------------------ | ------ |
| Old scripts consolidated       | 31     |
| New reusable modules           | 7      |
| New CLI entry points           | 3      |
| Total duplication eliminated   | ~5,000 |
| Test coverage                  | 387+   |
| All tests passing              | ✓      |

---

## 🚀 Getting Started with the New Structure

### Option 1: CLI (Recommended for Users)
```bash
# Convert models
uv run python scripts/convert_model.py --checkpoint model.pth --format onnx

# Verify environment
uv run python scripts/verify_setup.py

# Analyze ONNX models
uv run python scripts/analyze_model.py --model model.onnx
```

### Option 2: Python API (Recommended for Developers)
```python
from src.lib.conversion import quantize_model
from src.lib.base_trainer import setup_trainer_for_model
from src.lib.setup_verification import SetupVerifier

# Quantize model
quantized, metadata = quantize_model(model, "int8", "cpu")

# Setup training
trainer = setup_trainer_for_model(model, train_loader, val_loader, optimizer, "cnn")
history = trainer.train(30)

# Verify environment
verifier = SetupVerifier()
if verifier.run_all_checks()["all_ok"]:
    print("Ready to train!")
```

---

## 📚 Additional Resources

- **[README.md](../README.md)** - Main project documentation
- **[PROJECT_DIARY.md](../PROJECT_DIARY.md)** - Complete project history
- **[RESEARCH.md](../RESEARCH.md)** - Research findings and architecture comparisons
- **[4BIT_QUANTIZATION_GUIDE.md](../4BIT_QUANTIZATION_GUIDE.md)** - Quantization technical details
- **Test Coverage** - See `tests/` for 387+ unit and integration tests
