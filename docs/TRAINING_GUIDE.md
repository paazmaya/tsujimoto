# Training Guide

## Quick Start

### 1. Setup

```python
from src.lib.config import CNNConfig
from src.lib.lightning_trainer import create_trainer
from src.lib.logging_utils import setup_logger

logger = setup_logger(__name__)
```

### 2. Create Configuration

```python
config = CNNConfig(
    epochs=100,
    batch_size=64,
    learning_rate=1e-3,
)
```

### 3. Initialize Model & Trainer

```python
from models import LightweightCNNKanji

model = LightweightCNNKanji(num_classes=config.num_classes)
trainer = create_trainer(model, config, model_type='cnn')
```

### 4. Train Model

```python
history = trainer.train(
    train_loader,
    val_loader,
    num_epochs=config.epochs,
    early_stopping_patience=10
)
```

### 5. Load Best Model

```python
best_model = trainer.load_best_model()
```

## Advanced Features

### Distributed Training

```python
from src.lib.training_advanced import create_distributed_trainer

trainer = create_distributed_trainer(
    model,
    config,
    num_gpus=2
)
trainer.train(train_loader, val_loader)
```

### Experiment Tracking

```python
from src.lib.training_advanced import ExperimentTracker

tracker = ExperimentTracker("kanji-cnn-v1")
tracker.log_params(config.model_dump())

trainer.train(train_loader, val_loader)

tracker.log_metrics({"final_accuracy": 0.95})
tracker.save_model(model, "model.pt")
tracker.save_metadata()
```

### Model Optimization

```python
from src.lib.optimization_advanced import ModelOptimizer

optimizer = ModelOptimizer(model, device='cuda')

# Export to ONNX
optimizer.export_onnx("model.onnx", optimize=True)

# Quantize
quantized = optimizer.quantize(method="dynamic")

# Get compression stats
stats = optimizer.get_compression_stats()
print(f"Sparsity: {stats['sparsity']:.2%}")
```

## CLI Usage

### Training

```bash
uv run python scripts/train.py cnn \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 1e-3
```

### Validation

```bash
uv run python scripts/verify_setup.py
```

### Show Help

```bash
uv run python scripts/train.py cnn --help
```

## Tips & Best Practices

- Always verify your setup first: `uv run python scripts/verify_setup.py`
- Use early stopping to prevent overfitting
- Monitor Tensorboard logs during training
- Save model cards when uploading to Hub
- Track experiments with ExperimentTracker for reproducibility
- Quantize models for faster inference on CPU
