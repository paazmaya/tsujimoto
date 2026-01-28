---
license: mit
language:
  - ja
tags:
  - image-classification
  - kanji-recognition
  - japanese
  - computer-vision
  - pytorch
  - etl9g
  - etl8g
  - etl7
  - etl6
  - cnn
  - rnn
  - hiercode
  - channel-attention
  - safetensors
  - onnx
  - quantization
  - int8
  - edge-deployment
datasets:
  - ETL9G
  - ETL8G
  - ETL7
  - ETL6
metrics:
  - accuracy
  - model-size
  - inference-speed
model-index:
  - name: ETL9G CNN Baseline
    results:
      - task:
          type: image-classification
          name: Image Classification
        dataset:
          type: ETL9G
          name: ETL Character Database 9G
          split: test
        metrics:
          - type: accuracy
            value: 97.18
            name: Validation Accuracy
          - type: accuracy
            value: 93.51
            name: Training Accuracy
  - name: HierCode (Production)
    results:
      - task:
          type: image-classification
          name: Image Classification
        dataset:
          type: ETL9G
          name: ETL Character Database 9G
          split: test
        metrics:
          - type: accuracy
            value: 95.56
            name: Validation Accuracy
          - type: accuracy
            value: 92.88
            name: Training Accuracy
          - type: model-size
            value: 1.67
            name: Model Size (MB) - Quantized ONNX
  - name: RNN (Best Accuracy)
    results:
      - task:
          type: image-classification
          name: Image Classification
        dataset:
          type: ETL9G
          name: ETL Character Database 9G
          split: test
        metrics:
          - type: accuracy
            value: 98.4
            name: Validation Accuracy
library_name: pytorch
pipeline_tag: image-classification
---

# ETL9G+ Kanji Recognition Models

A comprehensive collection of neural network architectures for Japanese kanji character recognition. Includes lightweight CNN baselines, high-accuracy RNN models, and production-ready HierCode with quantization support for edge deployment.

## Model Description

This project implements **multiple architectures** for Japanese kanji character recognition, trained on the **ETL Character Database** (ETL9G and expanded ETL6-9 dataset):

### Available Models

1. **CNN** (Baseline)
   - 5-layer CNN with SENet-style channel attention
   - **97.18% accuracy** on ETL9G
   - Fast inference, easy deployment
   - ~15 MB (PyTorch), 6.6 MB (SafeTensors)

2. **RNN** (Best Accuracy)
   - Bidirectional LSTM with radical decomposition
   - **98.4% accuracy** on ETL9G
   - Best for high-precision applications
   - ~23 MB model size

3. **HierCode** (Recommended for Production)
   - Hierarchical radical decomposition
   - **95.56% accuracy** with 82% size reduction
   - **1.67 MB quantized ONNX** - ideal for edge deployment
   - Supports INT8 quantization

4. **QAT** (Quantization-Aware Training)
   - Optimized for embedded deployment
   - 1.7 MB ultra-lightweight model
   - Real-time inference on edge devices

5. **ViT** (Vision Transformer)
   - Transformer-based architecture
   - Explored as alternative approach
   - Research phase

### Key Features

- **Multiple Architectures**: CNN (97.18%), RNN (98.4%), HierCode (95.56%), QAT, ViT
- **3,036-4,154 character classes**: JIS Level 1 Kanji (2,965) + Hiragana (71) + Optional Katakana/Symbols/Numerals
- **Expandable Dataset**: Single dataset (~607K ETL9G) or combined ETL6-9 (934K samples, +53% more training data)
- **Production-Ready Quantization**: INT8 (4.5x smaller), 4-bit NF4/FP4 (ultra-lightweight)
- **Lightweight ONNX**: 1.67 MB quantized model - suitable for web, mobile, edge deployment
- **Channel Attention**: SENet-style mechanisms for adaptive feature weighting (CNN variant)
- **High Accuracy**: 98.4% (RNN) / 97.18% (CNN) / 95.56% (HierCode)
- **Multi-Format Support**: PyTorch (.pth), SafeTensors, and ONNX formats
- **Cross-Platform**: Compatible with ONNX Runtime, TensorRT, CoreML, WebAssembly
- **Automatic Checkpoint System**: Crash-safe training with automatic resumption

### Architecture Details

**CNN Baseline** (5-layer architecture):

```
Input (64×64 grayscale) → Conv1 (1→32) → Conv2 (32→64) → Conv3 (64→128) + Attention
                       → Conv4 (128→256) + Attention → Conv5 (256→512) + Attention
                       → Global Average Pool → Classifier (3,036 classes)
```

**Channel Progression**: 1 → 32 → 64 → 128 → 256 → 512  
**Total Parameters**: 1,735,527 (~1.7M parameters)  
**Model Size**: ~14.7 MB (float32), 4.5 MB (INT8), 1.67 MB (quantized ONNX)

**HierCode Architecture**:

- Hierarchical radical decomposition
- Multi-level classification for kanji radicals
- Significantly smaller than CNN while maintaining competitive accuracy
- Native INT8 quantization support

**RNN Architecture**:

- Bidirectional LSTM processing
- Radical-based sequence modeling
- Achieves highest accuracy (98.4%)

## Intended Use

### Primary Use Cases

- **Japanese Text Recognition**: OCR systems processing handwritten or printed kanji
- **Educational Applications**: Kanji learning and practice tools with real-time feedback
- **Document Processing**: Automated Japanese document analysis and digitization
- **Mobile Applications**: On-device kanji recognition with edge models
- **Web Applications**: Browser-based Japanese text recognition (ONNX Runtime / WebAssembly)
- **Enterprise Systems**: Integration into document management and archival systems

### Model Selection by Use Case

| Use Case                       | Recommended Model      | Reason                             |
| ------------------------------ | ---------------------- | ---------------------------------- |
| Highest accuracy (laboratory)  | RNN                    | 98.4% accuracy                     |
| Balanced (standard deployment) | CNN                    | 97.18% accuracy, fast, well-tested |
| Edge/Mobile devices            | HierCode (ONNX)        | 1.67 MB, 95.56% accuracy           |
| Web browsers                   | HierCode (ONNX strict) | 1.67 MB, WebAssembly compatible    |
| Embedded systems               | QAT or HierCode INT8   | Ultra-lightweight, 1.7-2.1 MB      |

### Direct Use

Each model accepts **64×64 pixel grayscale images** of individual kanji characters and outputs classification scores across 3,036 possible characters (or 4,154 if trained on combined ETL6-9).

### Downstream Use

This model can be integrated into:

- OCR pipelines for Japanese text processing
- Educational software for kanji learning with immediate feedback
- Document digitization systems for Japanese archives
- Handwriting recognition applications (tablets, pens)
- Real-time stroke-order practice applications

## Training Data

### Dataset Options

#### ETL9G (Default - 607K samples)

- **Source**: National Institute of Advanced Industrial Science and Technology (AIST)
- **Total Samples**: 607,200 character images
- **Writers**: 4,000 different individuals
- **Character Classes**: 3,036 (2,965 JIS Level 1 Kanji + 71 Hiragana)
- **Image Format**: 128×127 pixels, 16 grayscale levels
- **Split**: 80% training, 20% validation

#### ETL6-9 Combined (Expanded - 934K samples, +53% more data)

- **Total Samples**: 934,622 character images
- **Character Classes**: ~4,154 (expanded character coverage)
- **Component Breakdown**:
  - ETL9G: 607,200 samples (2,965 kanji + 71 hiragana)
  - ETL8G: 152,960 samples (956 educational kanji variants)
  - ETL7: 16,800 samples (48 hiragana variants)
  - ETL6: 157,662 samples (114 katakana + numerals + symbols + ASCII)
- **Expected Accuracy Improvement**: +2-3% over ETL9G alone
- **Training Time**: 1.5-2.0x longer per epoch

**Dataset URL**: http://etlcdb.db.aist.go.jp/

### Preprocessing

- **Resize**: 128×127 → 64×64 pixels
- **Normalization**: Pixel values normalized to [-1, 1] range
- **Format**: Single-channel grayscale input

## Training Procedure

### Hyperparameters (Default - CNN Baseline)

- **Epochs**: 30 (with early stopping)
- **Batch Size**: 64
- **Learning Rate**: Adaptive (initial: 0.001, final: 2.54e-05)
- **Optimizer**: Adam with weight decay
- **Loss Function**: CrossEntropyLoss
- **Hardware**: NVIDIA GPU with CUDA 13.0+
- **Checkpoint System**: Automatic resumption on crash, keeps last 5 checkpoints

### Training Results (ETL9G - CNN)

| Metric                        | Value    |
| ----------------------------- | -------- |
| **Final Validation Accuracy** | 97.18%   |
| **Final Training Accuracy**   | 93.51%   |
| **Validation Loss**           | 1.512    |
| **Training Epochs**           | 27/30    |
| **Learning Rate (final)**     | 2.54e-05 |

### Training Results (ETL9G - RNN)

| Metric                        | Value |
| ----------------------------- | ----- |
| **Final Validation Accuracy** | 98.4% |
| **Final Training Accuracy**   | 97.2% |
| **Model Size**                | 23 MB |

### Training Results (ETL9G - HierCode)

| Metric                           | Value   |
| -------------------------------- | ------- |
| **Validation Accuracy**          | 95.56%  |
| **Training Accuracy**            | 92.88%  |
| **PyTorch Model Size**           | 9.56 MB |
| **Quantized INT8 Size**          | 2.10 MB |
| **Quantized ONNX Size**          | 1.67 MB |
| **Size Reduction (vs original)** | 82%     |

### Dataset Selection

All training scripts automatically select the best available dataset:

```
Priority Order (auto-detected):
1. combined_all_etl  ← 934K samples, 43,427 classes (recommended)
2. etl9g             ← 607K samples, 3,036 classes (default)
3. etl8g             ← 153K samples, 956 classes
4. etl7              ← 16.8K samples, 48 classes
5. etl6              ← 157K samples, 114 classes
```

## Evaluation

### Testing Data and Metrics

**Test Split**: 20% validation split from dataset (~121,440 samples from ETL9G, ~186,924 from combined ETL6-9)

**Evaluation Metrics**:

- **Accuracy**: Primary metric for multi-class character classification
- **Loss**: Cross-entropy loss for training monitoring
- **Model Size**: Comparison across format/quantization methods

### Performance Summary

| Model                       | Dataset | Accuracy | Model Size  | Inference Speed | Deployment          |
| --------------------------- | ------- | -------- | ----------- | --------------- | ------------------- |
| **CNN**                     | ETL9G   | 97.18%   | 15 MB       | ⚡⚡⚡ (Fast)   | Python/ONNX/Browser |
| **RNN**                     | ETL9G   | 98.4%    | 23 MB       | ⚡⚡ (Medium)   | Python/ONNX         |
| **HierCode**                | ETL9G   | 95.56%   | 9.56 MB     | ⚡⚡⚡          | Python/ONNX         |
| **HierCode INT8**           | ETL9G   | 95.56%   | 2.10 MB     | ⚡⚡⚡          | CPU/Edge            |
| **HierCode Quantized ONNX** | ETL9G   | 95.56%   | **1.67 MB** | ⚡⚡⚡⚡        | Edge/Mobile/Web     |

### Quantization Impact

- **INT8 (PyTorch)**: 4.5x size reduction, 99-100% accuracy retention
- **4-bit (BitsAndBytes)**: 3.8 MB runtime footprint, 95-98% accuracy retention, 2-4x faster inference
- **Quantized ONNX**: 1.67 MB for HierCode, compatible with WebAssembly and mobile platforms

## Model Formats

### Available Formats

| Format                     | Size        | Use Case              | Platforms              |
| -------------------------- | ----------- | --------------------- | ---------------------- |
| **PyTorch (.pth)**         | 9-23 MB     | Fine-tuning, research | Python, GPU            |
| **SafeTensors**            | 6.6-14.7 MB | Secure deployment     | Python, web            |
| **ONNX (Float32)**         | 6.86-23 MB  | Cross-platform        | CPU/GPU, mobile, web   |
| **ONNX (INT8)**            | 6.86-7 MB   | Optimized CPU         | CPU, embedded, edge    |
| **ONNX (Quantized 4-bit)** | **1.67 MB** | Ultra-lightweight     | Edge, mobile, IoT, web |

### ONNX Opset Support

All models use **ONNX opset 14** for broad compatibility.

### ONNX Backend Compatibility

| Backend                 | Support | Model Variant  | Notes              |
| ----------------------- | ------- | -------------- | ------------------ |
| **ONNX Runtime**        | ✅ Full | All variants   | CPU/GPU fallback   |
| **TensorRT**            | ✅ Full | Float32/INT8   | NVIDIA GPU only    |
| **CoreML**              | ✅ Full | All variants   | iOS/macOS          |
| **WebAssembly**         | ✅ Full | Strict variant | Browser-based      |
| **TVM**                 | ✅ Full | All variants   | Compiler framework |
| **Mobile (Android)**    | ✅ Full | Quantized      | NNAPI support      |
| **Mobile (iOS)**        | ✅ Full | Quantized      | CoreML             |
| **Edge (Raspberry Pi)** | ✅ Full | Quantized INT8 | ARM CPU            |
| **Edge (Jetson)**       | ✅ Full | All variants   | NVIDIA GPU         |

## Environmental Impact

- **Training Time**: ~2-3 hours on modern GPU
- **Model Size**: 6.62-14.7 MB (format dependent)
- **Inference Speed**: Optimized for real-time applications
- **Energy Efficiency**: Lightweight architecture suitable for edge deployment

## Technical Specifications

### Model Architecture

```python
class LightweightKanjiNet(nn.Module):
    # 5-layer CNN with channel attention
    # Depthwise separable convolutions
    # 3 SENet-style attention modules
    # Global average pooling
    # Final classifier layer (3,036 classes)
```

### Input Requirements

- **Format**: Single-channel (grayscale) image
- **Size**: 64×64 pixels
- **Data Type**: Float32
- **Range**: [-1.0, 1.0] (normalized)
- **Shape**: (batch_size, 1, 64, 64)

### Output

- **Format**: Logits (pre-softmax scores)
- **Shape**: (batch_size, 3036)
- **Classes**: 3,036 possible kanji/hiragana characters
- **Mapping**: Available in accompanying JSON files

## Usage Examples

### PyTorch

```python
import torch
from safetensors.torch import load_file

# Load model
model_weights = load_file("kanji_model_etl9g_64x64_3036classes.safetensors")
model.load_state_dict(model_weights)
model.eval()

# Inference
with torch.no_grad():
    output = model(input_tensor)  # shape: (batch, 3036)
    predicted_class = torch.argmax(output, dim=1)
```

### ONNX Runtime

```python
import onnxruntime as ort

# Load ONNX model
session = ort.InferenceSession("kanji_model_etl9g_64x64_3036classes_strict.onnx")

# Inference
outputs = session.run(None, {"input": input_array})
predictions = outputs[0]
```

## Limitations and Bias

### Known Limitations

1. **Single Character Only**: Designed for individual character recognition, not full text
2. **Fixed Input Size**: Requires 64×64 pixel input images
3. **Grayscale Only**: Does not process color images
4. **Writing Style**: Trained primarily on handwritten samples from ETL9G dataset
5. **Character Set**: Limited to JIS Level 1 Kanji + basic Hiragana

### Potential Biases

- **Writer Demographics**: Reflects the demographic distribution of ETL9G dataset writers
- **Writing Styles**: May be biased toward specific handwriting patterns in the training data
- **Character Frequency**: Some characters may be better recognized due to training data distribution

### Recommendations

- Pre-process images to match training data characteristics
- Consider ensemble methods for critical applications
- Validate performance on your specific use case
- Monitor for drift when deployed in production

## Environmental Impact & Carbon Footprint

### Model Efficiency Summary

- **Training Time**: 2-3 hours per model on modern GPU (varies by dataset)
- **Model Sizes**: 1.67-23 MB (format and quantization dependent)
- **Inference Speed**: Optimized for real-time applications
- **Energy Efficiency**: Lightweight architectures suitable for edge deployment

### CO2 Emissions from Training

**CNN Baseline Training (ETL9G)**:

- **Hardware**: NVIDIA GeForce RTX 4070 Ti
- **Duration**: 2.5 hours
- **Energy**: ~1.25 kWh
- **CO2 Emitted**: 0.594 kg CO2 (global average at 475g CO2/kWh)

**Breakdown by Region**:

- USA Grid (386g CO2/kWh): 0.483 kg CO2
- European Grid (276g CO2/kWh): 0.345 kg CO2
- With Renewable Energy (41g CO2/kWh): 0.051 kg CO2

**Methodology**:

This carbon footprint estimate is calculated using:

1. **System Specifications**:
   - CPU: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel (24 cores)
   - GPU: NVIDIA GeForce RTX 4070 Ti (12GB VRAM)
   - Training Duration: 2.5 hours
   - Total Power Consumption: 500 W

2. **Power Consumption**:
   - CPU (150W) + GPU (250W) + System overhead (100W)
   - Total Energy: 1.25 kWh

3. **Emission Factors**:
   - Global Average: 475g CO2/kWh
   - USA Grid: 386g CO2/kWh
   - European Grid: 276g CO2/kWh
   - Renewable Energy: 41g CO2/kWh

**Note**: These are estimates based on system specifications. For precise measurements, use tools like [CodeCarbon](https://codecarbon.io/) during actual training.

### Environmental Impact Comparison

| Reference Point          | CO2 Emissions |
| ------------------------ | ------------- |
| This Model Training      | 0.594 kg CO2  |
| Smartphone usage (1 day) | 0.30 kg CO2   |
| Car travel (3.5 km)      | 0.594 kg CO2  |
| Tree absorption (1 year) | 21.8 kg CO2   |

### Daily Inference Impact

For typical daily inference (10,000 images/day):

- **Energy**: 0.0014 kWh/day
- **CO2 Emissions**: 0.66 g CO2/day (global average)
- **Annual Impact**: 241 g CO2/year

### Recommendations for Reduced Impact

- **Use renewable energy**: Reduces emissions by ~91% (to 0.051 kg CO2)
- **Deploy quantized models**: Reduces inference energy by 3-4x
- **Use edge deployment**: Eliminates cloud infrastructure overhead
- **Model reuse**: Fine-tune existing models rather than training from scratch
- **Hyperparameter optimization**: Efficient training reduces epochs (30→27 in our case)

---

_Carbon footprint measured on November 20, 2025_

## Citation

### BibTeX

```bibtex
@software{tsujimoto_kanji_recognition_2025,
  title={Tsujimoto - Multi-Architecture Kanji Recognition Platform},
  author={Paazmaya, Jukka},
  year={2025},
  url={https://github.com/paazmaya/tsujimoto},
  note={CNN, RNN, HierCode architectures with quantization support},
  license={MIT}
}
```

### Dataset Citation

```bibtex
@dataset{etl_character_database,
  title={ETL Character Database (ETLCDB)},
  author={National Institute of Advanced Industrial Science and Technology (AIST)},
  year={1999-2005},
  url={http://etlcdb.db.aist.go.jp/},
  note={ETL1-9G: Comprehensive handwritten and printed character datasets}
}
```

### Related Work

This project builds on hierarchical character decomposition concepts:

- **HierCode Original**: For initial hierarchical radical decomposition framework
- **Recent Extensions** (2022-2025):
  - Hi-GITA (2505.24837): Hierarchical multi-granularity alignment
  - RZCR (2207.05842): Knowledge graph over radicals
  - STAR (2210.08490): Stroke + radical decompositions
  - MegaHan97K (2506.04807): Extended benchmark dataset

## Model Card Authors & Maintainers

- **Jukka Paasonen** - Project lead, model development, training
- See [GitHub repository](https://github.com/paazmaya/tsujimoto) for contributors

## Support & Contact

For questions, issues, or suggestions about this model:

- 📋 **GitHub Issues**: [Open an issue](https://github.com/paazmaya/tsujimoto/issues)
- 📧 **Email**: paazmaya@yahoo.com
- 🔗 **Repository**: [tsujimoto](https://github.com/paazmaya/tsujimoto)

---

## Model Card Metadata

| Property             | Value                                    |
| -------------------- | ---------------------------------------- |
| **Last Updated**     | November 20, 2025                        |
| **Model Version**    | v3.0                                     |
| **License**          | MIT                                      |
| **Task**             | Image Classification (Kanji Recognition) |
| **Framework**        | PyTorch + ONNX                           |
| **Input Size**       | 64×64 pixels (grayscale)                 |
| **Output Classes**   | 3,036 (or ~4,154 with combined ETL6-9)   |
| **Primary Language** | Python 3.11+                             |
| **Status**           | Production Ready                         |
