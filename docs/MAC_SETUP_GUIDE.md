# Mac Setup Guide for Tsujimoto (Japanese Kanji Recognition)

This guide provides step-by-step instructions for setting up and training the Tsujimoto project on macOS (Intel and Apple Silicon).

## Quick Summary

| Device                            | GPU Support | Speed                        | Recommended      |
| --------------------------------- | ----------- | ---------------------------- | ---------------- |
| **Intel Mac CPU**                 | ❌ No       | Slow (~10-15 min/epoch CNN)  | For testing only |
| **Apple Silicon (M1/M2/M3+) MPS** | ✅ Yes      | Fast (~2-3 min/epoch CNN)    | **Recommended**  |
| **Apple Silicon CPU-only**        | ❌ No       | Very slow (~10-15 min/epoch) | Not recommended  |

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation Steps](#installation-steps)
3. [Verification](#verification)
4. [Running Your First Training](#running-your-first-training)
5. [Performance Expectations](#performance-expectations)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **macOS**: 10.15+ (Catalina or later)
- **Python**: 3.11+ (check with `python3 --version`)
- **Memory**:
  - Minimum: 8GB RAM
  - Recommended: 16GB+ RAM
- **Disk Space**:
  - Installation: ~2GB (project + dependencies)
  - Datasets: ~7.5GB (combined_all_etl, the largest dataset)
  - Checkpoints: ~500MB (depends on training runs)

### Hardware Detection

To check your Mac:

```bash
# Check CPU type (Apple Silicon vs Intel)
uname -m
# Apple Silicon: arm64
# Intel: x86_64

# Check available RAM
vm_stat | grep "Pages free" | awk '{print ($3 * 4096 / 1024 / 1024 / 1024), "GB"}'

# Check available disk space
df -h /
```

---

## Installation Steps

### Step 1: Install Python 3.11+ (if not already installed)

**Option A: Using Homebrew (Recommended)**

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.11+
brew install python@3.11

# Verify installation
python3 --version  # Should be 3.11.x or higher
```

**Option B: Using macOS native installer**

Download from [python.org](https://www.python.org/downloads/macos/) and follow the installer.

### Step 2: Set Up a Virtual Environment

```bash
# Navigate to the project directory
cd /path/to/tsujimoto

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# You should see (venv) at the start of your terminal prompt
```

### Step 3: Install Dependencies

The project uses `uv` for fast dependency management, but you can also use `pip`.

**Option A: Using uv (Faster) - RECOMMENDED**

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv pip install -e .

# This will install PyTorch with CPU support (works on all Macs)
# and configure other dependencies automatically
```

**Option B: Using pip**

```bash
# Install PyTorch CPU-only (compatible with all Macs)
pip install torch torchvision

# Install the project
pip install -e .
```

**Option C: For Apple Silicon with GPU Support (MPS)**

```bash
# PyTorch CPU wheels work on Apple Silicon and support MPS
# No special installation needed - just use Option A or B
# MPS support is automatic when available
```

### Step 4: Download Datasets

```bash
# Navigate to project directory
cd /path/to/tsujimoto

# Download and prepare datasets automatically
python scripts/download_datasets.py --dataset combined_all_etl

# Or choose a smaller dataset to start with:
python scripts/download_datasets.py --dataset etl9g
python scripts/download_datasets.py --dataset etl6

# This will:
# - Download files from ETLCDB
# - Extract and process them
# - Create .npz chunks for efficient loading
# - Generate metadata
```

**Manual Download (if automated script fails):**

1. Visit [ETL Character Database](http://etlcdb.db.aist.go.jp/)
2. Download desired ETL files (ETL6, ETL7, ETL8G, ETL9G)
3. Extract to `ETL*/` directories in project root
4. Run: `python scripts/prepare_dataset.py`

---

## Verification

### Step 1: Verify Installation

```bash
# Activate your virtual environment
source venv/bin/activate

# Run setup verification
python scripts/verify_setup.py

# You should see output like:
# ✓ Python version: 3.11.x
# ✓ PyTorch installed
# ✓ CUDA available: False
# ✓ MPS available: True (if Apple Silicon)
# ✓ Found X dependencies: [list of packages]
# ✓ Datasets available: combined_all_etl (934K samples)
```

### Step 2: Test a Quick Training Run

```bash
# Train CNN for just 1 epoch to verify everything works
python -m src.lib.cli train-cnn \
    --epochs 1 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --data-dir ./dataset

# Expected output:
# Epoch 1/1: 100%|████████| 2304/2304 [01:45<00:00, 21.9 samples/s]
# Train Loss: 5.234, Train Acc: 0.0342, Val Loss: 5.198, Val Acc: 0.0412
# ✓ Training completed successfully
```

---

## Running Your First Training

### Basic Training Example (CNN)

```bash
# Activate virtual environment
source venv/bin/activate

# Train CNN with recommended Mac settings
python -m src.lib.cli train-cnn \
    --epochs 30 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --device auto \
    --data-dir ./dataset

# Estimated time: 10-15 hours on Apple Silicon MPS, 1-2 hours on GPU
```

### Custom Training Script

```python
# train_mac.py
from src.lib.config import TrainingConfig
from src.lib.base_trainer import CNNTrainer
from src.lib.datasets import load_dataset
import torch

# Load dataset
train_loader, val_loader = load_dataset(
    dataset_name="combined_all_etl",
    batch_size=64,
    num_workers=2  # Adjust based on your CPU cores
)

# Create model
from src.lib.model import create_lightweight_kanji_net
model = create_lightweight_kanji_net(43427)  # 43,427 classes

# Train with automatic device detection
config = TrainingConfig(
    device="auto",  # Auto-detects CUDA -> MPS -> CPU
    epochs=30,
    batch_size=64,
    learning_rate=0.001
)

trainer = CNNTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=config.device
)

history = trainer.train(num_epochs=config.epochs)
```

### Running with MPS (Apple Silicon Only)

```bash
# Explicitly use MPS if available
python -m src.lib.cli train-cnn \
    --device mps \
    --epochs 30 \
    --batch-size 64

# Check MPS utilization (while training in another terminal):
python -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available()}')"
```

---

## Performance Expectations

### Training Time per Epoch (CNN on combined_all_etl)

| Device                   | Time/Epoch | Relative Speed |
| ------------------------ | ---------- | -------------- |
| **Apple Silicon M1 MPS** | 2-3 min    | 5-7x faster    |
| **Apple Silicon CPU**    | 10-15 min  | 1x baseline    |
| **Intel Mac CPU**        | 12-18 min  | 0.8-1x         |
| **GPU (Linux RTX 4090)** | 20-30 sec  | 30-50x faster  |

### Memory Usage

| Model             | Min RAM | Recommended RAM |
| ----------------- | ------- | --------------- |
| CNN (Lightweight) | 4GB     | 8GB+            |
| RNN (KanjiRNN)    | 6GB     | 12GB+           |
| ViT               | 8GB     | 16GB+           |

### Hyperparameter Tips for Mac

```bash
# For Apple Silicon with 16GB RAM:
--batch-size 128        # Larger batch = faster training
--num-workers 4         # Match your CPU core count
--learning-rate 0.001   # Standard rate for CNNs
--epochs 30-50          # Full training

# For Intel Mac or limited RAM:
--batch-size 64         # Reduce batch size
--num-workers 2         # Fewer workers
--epochs 20-30          # Shorter training
```

---

## Troubleshooting

### Issue: "RuntimeError: Expected all tensors to be on the same device"

**Cause**: Model and data on different devices (especially when device="cpu" but MPS available)

**Solution**:

```bash
# Use explicit device setting
python scripts/train_modern.py --device cpu
# OR
python scripts/train_modern.py --device mps
# NOT
python scripts/train_modern.py --device auto  # if you're having issues
```

### Issue: "ModuleNotFoundError: No module named 'bitsandbytes'"

**Cause**: BitsAndBytes requires CUDA (not available on Mac)

**Solution**: Use INT8 or BFloat16 quantization instead:

```bash
python -m src.lib.cli train-cnn --quantization int8
python -m src.lib.cli train-cnn --quantization bfloat16
```

### Issue: "FileNotFoundError: dataset/combined_all_etl/..."

**Cause**: Datasets not downloaded

**Solution**:

```bash
# Download datasets
python scripts/download_datasets.py --dataset combined_all_etl

# Or manually:
# 1. Visit http://etlcdb.db.aist.go.jp/
# 2. Download ETL6, ETL7, ETL8G, ETL9G
# 3. Extract to project root
# 4. Run: python scripts/prepare_dataset.py
```

### Issue: "Killed: 9" (Process killed suddenly)

**Cause**: Out of memory (OOM)

**Solution**:

```bash
# Reduce batch size and enable gradient checkpointing
python -m src.lib.cli train-cnn \
    --batch-size 32 \
    --gradient-checkpointing \
    --epochs 20
```

### Issue: Training is very slow (10+ minutes per epoch on Apple Silicon)

**Cause**: Running on CPU instead of MPS

**Solution**:

```bash
# Check if MPS is being used
python -c "import torch; print(torch.backends.mps.is_available())"

# If False, update PyTorch:
pip install --upgrade torch

# If True, explicitly use it:
python -m src.lib.cli train-cnn --device mps
```

### Issue: "ImportError: torch.cuda.amp" or "CUDA-specific autocast"

**Cause**: Code tried to use CUDA when it's not available

**Solution**:

- This should be fixed in the latest version (device-aware mixed precision)
- If you see this, update the project:
  ```bash
  git pull origin main
  pip install -e .
  ```

### Issue: Virtual Environment Not Activating

**Solution**:

```bash
# Use full path to activate
source /full/path/to/tsujimoto/venv/bin/activate

# Verify activation (should show (venv) in prompt)
which python
# Should show: /path/to/tsujimoto/venv/bin/python
```

---

## Advanced Configuration

### Multi-GPU Training (Not Applicable on Mac)

Mac doesn't support multiple GPUs (MPS is unified memory), but distributed training on multiple Macs is possible with PyTorch Lightning.

### Checkpoint Management

```bash
# Save checkpoint manually
python -c "from src.lib.checkpoint import CheckpointManager; \
    cm = CheckpointManager('checkpoints/cnn', 'cnn'); \
    cm.save_checkpoint(model, optimizer, epoch=10, metrics={'acc': 0.95})"

# Resume from checkpoint
python -m src.lib.cli train-cnn \
    --checkpoint ./checkpoints/cnn/checkpoint_best.pt \
    --epochs 50
```

### Monitoring Training

```bash
# Use TensorBoard (if installed)
tensorboard --logdir ./runs

# Use custom logging
python scripts/train_modern.py --log-dir ./my_logs --log-level debug
```

---

## Next Steps

1. **Read the main README**: [README.md](README.md)
2. **Check Training Guide**: [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)
3. **Explore Model Architectures**: [RESEARCH.md](RESEARCH.md)
4. **See Platform Compatibility**: [PLATFORM_COMPATIBILITY.md](PLATFORM_COMPATIBILITY.md)

---

## Getting Help

- **Installation Issues**: Check [Troubleshooting](#troubleshooting)
- **Dataset Questions**: See [docs/DATASET_SETUP.md](docs/DATASET_SETUP.md)
- **Training Issues**: See [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)
- **Report Bugs**: [GitHub Issues](https://github.com/paazmaya/tsujimoto/issues)

---

## Mac-Specific Notes

### PyTorch on Mac

- **CPU wheels**: Compatible with both Intel and Apple Silicon
- **MPS (Metal Performance Shaders)**: Automatic GPU acceleration on Apple Silicon M1/M2/M3+
- **Automatic device selection**: Project uses `device="auto"` by default, which:
  1. Checks for CUDA (Linux/Windows with NVIDIA GPU)
  2. Checks for MPS (Apple Silicon)
  3. Falls back to CPU

### Performance Optimization Tips

1. **Use MPS** on Apple Silicon for 5-7x faster training
2. **Increase batch size** (128-256) if you have 16GB+ RAM
3. **Reduce image resolution** from 64x64 to 32x32 to speed up training
4. **Use smaller models** (CNN best on Mac, RNN slower, ViT slowest)
5. **Enable gradient checkpointing** to reduce memory usage

### Known Limitations on Mac

- ❌ No 4-bit quantization (BitsAndBytes requires CUDA)
- ✅ INT8 quantization works fine
- ✅ BFloat16 quantization works fine
- ❌ Multi-GPU training not applicable
- ✅ Single MPS device training fully supported

---

**Last Updated**: 2026-09-01  
**Tested on**: macOS 12.x, 13.x, 14.x with Python 3.11+, PyTorch 2.1+
