# RNN Training Models: Comparison with CNN and Radical RNN

## Overview

This document compares the RNN-based kanji recognition approaches with the existing Plain CNN and Radical RNN implementations, documenting their differences, strengths, and use cases.

## Architecture Comparison

### 1. Plain CNN (`train_cnn_model.py`)

**Architecture Approach:**
- Direct spatial feature extraction using convolutional layers
- Fixed receptive fields that progressively increase with depth
- No sequence or temporal modeling
- Channel attention mechanisms (SENet-style) for feature recalibration

**Key Features:**
- Simple feedforward processing: Input → Conv Blocks → Classification Head
- Efficient for spatial pattern recognition in single images
- No recurrent connections or memory between positions

**Model Parameters:**
```
Typical: 2-5M for ETL9G (3036 classes), 12-25M for combined dataset (43427 classes)
```

**Strengths:**
- Fast inference (5-15ms per sample)
- Simple architecture, easy to optimize and deploy
- Proven baseline for image classification
- Lower memory footprint during inference

**Weaknesses:**
- Does not model stroke order or writing patterns
- Cannot capture sequential relationships
- Less efficient for variable-length radical decomposition
- No explicit representation of character components

---

### 2. Radical RNN (`train_radical_rnn.py`)

**Architecture Approach:**
- Decomposes kanji into radical (component) sequences
- Uses radical embeddings + LSTM for sequence processing
- Leverages linguistic structure of kanji characters
- Encodes 3000+ characters as combinations of 200-500 radicals

**Radical Decomposition Theory:**
```
Example: 明 (brightness) = 日 (sun) + 月 (moon)

Benefits:
1. Vocabulary reduction: 3000+ chars → 500 radicals
2. Zero-shot learning: New chars from known radicals
3. Improved generalization
4. More semantically meaningful features
```

**Model Architecture:**
```
Input Image → Radical Extraction → Radical Embedding Layer
  → Bi-directional LSTM → Classification Head
```

**Key Components:**
- `RadicalExtractor`: Converts images to radical sequences
- Embedding layer: Projects radical IDs to dense vectors
- LSTM with attention for temporal relationships
- Configurable encoding types (one_hot, binary_tree, learned)

**Model Parameters:**
```
Typical: 1-3M parameters (70-80% reduction vs CNN)
- Radical vocab: ~500 embeddings × 128 dims = 64K params
- LSTM: Moderate recurrent parameters
- Classification head: Flexible for class count
```

**Strengths:**
- Significant parameter reduction (80-90% fewer than CNN)
- Semantically meaningful representations via radicals
- Better generalization to new character variations
- Supports zero-shot learning with known radicals
- More aligned with linguistic structure
- Natural handling of radical variations

**Weaknesses:**
- Requires accurate radical decomposition
- Additional preprocessing step (radical extraction)
- Performance depends on radical database quality
- Limited ablation on radical encoding impact

**Training Configuration Example:**
```bash
uv run python scripts/train_radical_rnn.py \
  --data-dir dataset/etl9g \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --hidden-size 256 \
  --num-layers 2
```

---

### 3. RNN Models (This Migration)

Four distinct RNN architectures for different sequence representations:

#### 3a. Basic RNN (`basic_rnn`)

**Architecture:**
```
Input Image → Grid-based Spatial Sequence (8×8=64 elements)
  → Bidirectional LSTM → Mean Pooling → Classification Head
```

**Sequence Creation:**
- Divide image into 8×8 grid
- Extract features per cell: [mean intensity, std, max, min]
- Creates 64-element sequence of 4-dimensional features

**Model Parameters:** ~2-3M

**Characteristics:**
- Natural sequence ordering (left-to-right, top-to-bottom)
- Bidirectional LSTM captures spatial context
- Simple and interpretable
- Mean pooling for sequence aggregation

**Best For:**
- Learning spatial patterns as sequences
- Baseline RNN comparison
- Understanding reading flow patterns

---

#### 3b. Stroke-based RNN (`stroke_rnn`)

**Architecture:**
```
Input Image → Stroke Extraction
  → Stroke Embedding → Bi-directional LSTM
  → Multi-head Attention → Classification Head
```

**Stroke Representation:**
- Extract strokes using contour detection
- Features per stroke: [centroid_x, centroid_y, bbox_x, bbox_y, width, height, area, perimeter]
- 8 normalized features per stroke, max 30 strokes

**Model Components:**
- Linear embedding layer (stroke_features → hidden_size/2)
- Bi-directional LSTM (hidden_size=256, num_layers=2)
- Multi-head attention (8 heads) for stroke importance weighting
- Classification head with dropout

**Model Parameters:** ~3-4M

**Characteristics:**
- Models temporal/sequential stroke order
- Attention mechanism weights important strokes
- Packed sequences for efficient processing of variable lengths
- Max pooling after attention

**Best For:**
- Understanding writing patterns and stroke order
- Modeling temporal progression of character drawing
- Capturing stroke-level importance variations
- Learning from handwriting variations

**Research Background:**
- RAN (Residual Attention Networks) - considers spatial relationships
- Based on writing trajectory understanding
- Applicable to handwritten character recognition

---

#### 3c. Radical-based RNN (Alternative Radical Processing)

**Architecture:**
```
Input Image → Radical Extraction → Radical Embedding
  → Bi-directional LSTM → Mean Pooling → Classification Head
```

**Key Differences from `train_radical_rnn.py`:**
- Similar radical decomposition concept
- Alternative implementation in RNN framework
- Configurable radical vocabulary size (default: 500)
- Embedding dimension: 128 (vs inline in train_radical_rnn)
- Mean pooling with masking for variable-length sequences

**Model Parameters:** ~1-2M

**When to Use:**
- Comparative study vs Radical RNN implementation
- Testing radical sequence processing alternatives
- Ensemble with other approaches

---

#### 3d. Hybrid CNN-RNN (`hybrid_cnn_rnn`)

**Architecture:**
```
Input Image → CNN Backbone (4 blocks, 64→256 features)
  ├─ Global CNN Features → Projection → Global Features
  └─ Spatial Features (4×4 grid) → Bi-directional LSTM → Pooling
                                          ↓
                          Concatenate [Global + LSTM]
                                   → Classification Head
```

**CNN Backbone:**
- 4 convolutional blocks with batch normalization
- Block structure: Conv → BatchNorm → ReLU → MaxPool
- Final adaptive average pooling to 4×4 spatial grid
- Output channels: 32 → 64 → 128 → 256

**RNN Processing:**
- Treats spatial features as sequence elements
- Bi-directional LSTM: 256 cells, 2 layers
- Processes 16 spatial locations (4×4 grid) as sequence

**Feature Combination:**
- CNN global features: 512 dimensions (projected)
- LSTM pooled features: 512 dimensions (256×2 for bidirectional)
- Combined features: 1024 dimensions
- Classification MLP: 1024 → 256 → num_classes

**Model Parameters:** ~4-5M

**Characteristics:**
- Combines local spatial and global temporal information
- CNN extracts visual features, LSTM models spatial relationships
- Moderate parameter increase vs pure CNN or Radical RNN
- Balanced approach: spatial + temporal modeling

**Best For:**
- Tasks requiring both spatial and sequential understanding
- Capturing multi-scale patterns
- Ensemble learning (diverse architecture)
- Transfer learning from vision tasks

---

## Performance Characteristics

| Model | Parameters | Inference Speed | Memory | Accuracy* | Best For |
|-------|-----------|-----------------|--------|-----------|----------|
| **Plain CNN** | 2-5M | 5-15ms | Low | Baseline | Speed, simplicity |
| **Radical RNN** | 1-3M | 15-25ms | Medium | High | Generalization, efficiency |
| **Basic RNN** | 2-3M | 20-30ms | Medium | Medium | Spatial sequences |
| **Stroke RNN** | 3-4M | 25-35ms | Medium | Medium-High | Writing patterns |
| **Hybrid CNN-RNN** | 4-5M | 30-40ms | High | High | Balanced approach |

*Accuracy depends on training data and hyperparameters

---

## Use Case Selection Guide

### Choose **Plain CNN** when:
- ✓ Maximum speed is critical
- ✓ Minimal memory footprint required
- ✓ Simple deployment (ONNX, WASM)
- ✓ Baseline comparison needed

### Choose **Radical RNN** when:
- ✓ Parameter efficiency is important
- ✓ Semantic understanding valued
- ✓ Zero-shot learning capability needed
- ✓ Good radical database available
- ✓ Handling rare/complex kanji

### Choose **Basic RNN** when:
- ✓ Understanding spatial sequences
- ✓ Comparing RNN vs CNN approaches
- ✓ Baseline RNN implementation needed
- ✓ Research/educational purposes

### Choose **Stroke RNN** when:
- ✓ Modeling writing patterns important
- ✓ Handwritten character recognition
- ✓ Stroke order matters for domain
- ✓ Attention mechanism useful

### Choose **Hybrid CNN-RNN** when:
- ✓ Multi-scale features needed
- ✓ Combining visual and sequential understanding
- ✓ Moderate parameter budget available
- ✓ Ensemble with diverse architectures

---

## Implementation Differences

### Data Preprocessing

| Aspect | Plain CNN | Radical RNN | RNN Models |
|--------|----------|------------|-----------|
| **Input** | Raw image | Image | Image or sequence |
| **Preprocessing** | Normalization | + Radical extraction | + Sequence conversion |
| **Complexity** | Simple | Medium | Medium-High |

### Training Strategy

**Plain CNN:**
- Standard image classification training
- Data augmentation: noise, rotation, affine
- Channel attention helps with feature selection

**Radical RNN:**
- Two-stage: Radical extraction → RNN training
- Radical encoding crucial for performance
- Benefit from linguistic data quality

**RNN Models:**
- Sequence generation step required
- Model type determines preprocessing
- Variable sequence lengths handled by packing

### Checkpoint Management

All models use unified `CheckpointManager`:
- Automatic checkpoint creation
- Resume from latest checkpoint capability
- Training history tracking
- Best model selection

### Training Commands

```bash
# Plain CNN
uv run python scripts/train_cnn_model.py --data-dir dataset --epochs 50

# Radical RNN
uv run python scripts/train_radical_rnn.py --data-dir dataset --epochs 50

# RNN Models
uv run python scripts/train_rnn.py --data-dir dataset --model-type basic_rnn --epochs 50
uv run python scripts/train_rnn.py --data-dir dataset --model-type stroke_rnn --epochs 50
uv run python scripts/train_rnn.py --data-dir dataset --model-type radical_rnn --epochs 50
uv run python scripts/train_rnn.py --data-dir dataset --model-type hybrid_cnn_rnn --epochs 50
```

---

## Migration Notes

### Original Location
- Previous implementation: `scripts/rnn/` (subdirectory)
  - `train_rnn_model.py` - Training script
  - `rnn_model.py` - Model definitions
  - `evaluate_rnn_models.py` - Evaluation utilities
  - `deploy_rnn_model.py` - Inference tools
  - `data_processor.py` - Data utilities
  - `README.md` - Documentation

### New Location & Structure
- Consolidated: `scripts/train_rnn.py` (unified with CNN/Radical training)
- Follows project standards:
  - `model_training/rnn/checkpoints/` - Saved models
  - `model_training/rnn/results/` - Training metrics/plots
  - `model_training/rnn/config/` - Configuration files

### Key Improvements
1. **Consistency**: Aligned with `train_cnn_model.py` and `train_radical_rnn.py` style
2. **Checkpoint Management**: Uses unified `CheckpointManager`
3. **Configuration**: Leverages `optimization_config.py`
4. **Modularity**: Factory pattern for model creation
5. **Documentation**: Clear separation of concerns

---

## Recommendation for Multi-Model Evaluation

To compare all approaches:

```bash
# 1. Train Plain CNN (baseline)
uv run python scripts/train_cnn_model.py --data-dir dataset --epochs 50

# 2. Train Radical RNN (semantic approach)
uv run python scripts/train_radical_rnn.py --data-dir dataset --epochs 50

# 3. Train RNN variants (sequential approaches)
uv run python scripts/train_rnn.py --data-dir dataset --model-type basic_rnn --epochs 50
uv run python scripts/train_rnn.py --data-dir dataset --model-type stroke_rnn --epochs 50
uv run python scripts/train_rnn.py --data-dir dataset --model-type hybrid_cnn_rnn --epochs 50

# 4. Compare results
# - Accuracy: Which approach performs best?
# - Parameters: Which is most efficient?
# - Speed: Which has best inference performance?
# - Robustness: How do they handle edge cases?
```

---

## References

- **RAN (Residual Attention Networks)** - 2017 - Spatial attention mechanisms
- **DenseRAN** - 2018 - Densely connected attention
- **STAR (Spatial-Temporal Attention RNN)** - 2022
- **RSST (Radical + Stroke + Structural + Temporal)** - 2022
- **MegaHan97K** - 2025 - Large-scale kanji dataset with multilingual support

---

## Next Steps

1. **Evaluation**: Implement comprehensive benchmark comparing all 5 approaches
2. **Ensemble**: Combine predictions from multiple model types
3. **Optimization**: Quantization and pruning for each model type
4. **Deployment**: Generate ONNX versions for each approach
5. **Documentation**: Create per-model deployment guides
