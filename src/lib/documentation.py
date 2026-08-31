"""
Documentation generation utilities for API docs, guides, and model cards.

This module provides tools for:
- Automatic API documentation generation
- Training guides and tutorials
- Model card generation
- Reproducibility documentation

Example Usage:
    >>> from src.lib.documentation import DocumentationGenerator, ModelCardGenerator
    >>> 
    >>> # Generate API docs
    >>> gen = DocumentationGenerator("docs/api")
    >>> gen.generate_api_docs()
    >>> 
    >>> # Generate model card
    >>> card_gen = ModelCardGenerator("CNN", config)
    >>> card = card_gen.generate()
    >>> card_gen.save(Path("models/cnn/MODEL_CARD.md"))
"""

import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional

from .logging_utils import setup_logger

logger = setup_logger(__name__)


class DocumentationGenerator:
    """Generate API documentation for the project."""

    def __init__(self, docs_dir: Optional[Path] = None):
        """
        Initialize documentation generator.

        Args:
            docs_dir: Directory for documentation
        """
        self.docs_dir = Path(docs_dir or "docs")
        self.docs_dir.mkdir(parents=True, exist_ok=True)

        self.api_dir = self.docs_dir / "api"
        self.api_dir.mkdir(exist_ok=True)

        logger.info(f"DocumentationGenerator initialized: {self.docs_dir}")

    def generate_module_docs(self, module_name: str, module: Any) -> str:
        """
        Generate documentation for a module.

        Args:
            module_name: Name of module
            module: Python module object

        Returns:
            Generated markdown documentation
        """
        docs = f"# {module_name}\n\n"

        # Module docstring
        if module.__doc__:
            docs += f"{module.__doc__}\n\n"

        # Classes
        classes = [
            (name, obj)
            for name, obj in inspect.getmembers(module, inspect.isclass)
            if not name.startswith("_")
        ]

        if classes:
            docs += "## Classes\n\n"
            for class_name, class_obj in classes:
                docs += self._generate_class_docs(class_name, class_obj)

        # Functions
        functions = [
            (name, obj)
            for name, obj in inspect.getmembers(module, inspect.isfunction)
            if not name.startswith("_")
        ]

        if functions:
            docs += "## Functions\n\n"
            for func_name, func_obj in functions:
                docs += self._generate_function_docs(func_name, func_obj)

        return docs

    def _generate_class_docs(self, class_name: str, class_obj: type) -> str:
        """Generate documentation for a class."""
        docs = f"### {class_name}\n\n"

        if class_obj.__doc__:
            docs += f"{class_obj.__doc__}\n\n"

        # Constructor
        init_sig = inspect.signature(class_obj.__init__)
        docs += f"**Constructor**: `{class_name}{init_sig}`\n\n"

        # Methods
        methods = [
            (name, method)
            for name, method in inspect.getmembers(class_obj, inspect.ismethod)
            if not name.startswith("_")
        ]

        if methods:
            docs += "**Methods**:\n\n"
            for method_name, method_obj in methods:
                sig = inspect.signature(method_obj)
                docs += f"- `{method_name}{sig}`\n"

        docs += "\n"
        return docs

    def _generate_function_docs(self, func_name: str, func_obj) -> str:
        """Generate documentation for a function."""
        sig = inspect.signature(func_obj)
        docs = f"### {func_name}\n\n"

        if func_obj.__doc__:
            docs += f"{func_obj.__doc__}\n\n"

        docs += f"**Signature**: `{func_name}{sig}`\n\n"
        return docs

    def generate_api_docs(self) -> None:
        """Generate API documentation for all modules."""
        modules_to_doc = [
            "src.lib.config",
            "src.lib.logging_utils",
            "src.lib.lightning_trainer",
            "src.lib.datasets",
            "src.lib.cli",
            "src.lib.hub",
            "src.lib.optimization_advanced",
            "src.lib.training_advanced",
        ]

        for module_name in modules_to_doc:
            try:
                # Import module
                parts = module_name.split(".")
                module = __import__(module_name, fromlist=[parts[-1]])

                # Generate docs
                docs = self.generate_module_docs(parts[-1], module)

                # Save to file
                doc_file = self.api_dir / f"{parts[-1]}.md"
                with open(doc_file, "w") as f:
                    f.write(docs)

                logger.info(f"Generated API docs: {doc_file}")

            except Exception as e:
                logger.warning(f"Could not generate docs for {module_name}: {e}")

    def generate_training_guide(self) -> str:
        """Generate training guide."""
        guide = """# Training Guide

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
python scripts/train_modern.py train \\
    --model cnn \\
    --epochs 100 \\
    --batch-size 64 \\
    --learning-rate 1e-3
```

### Validation

```bash
python scripts/train_modern.py validate
```

### Show Configuration

```bash
python scripts/train_modern.py show-config --model cnn
```

## Tips & Best Practices

- Always validate your setup first: `train_modern.py validate`
- Use early stopping to prevent overfitting
- Monitor Tensorboard logs during training
- Save model cards when uploading to Hub
- Track experiments with ExperimentTracker for reproducibility
- Quantize models for faster inference on CPU
"""

        # Save guide
        guide_file = self.docs_dir / "TRAINING_GUIDE.md"
        with open(guide_file, "w") as f:
            f.write(guide)

        logger.info(f"Generated training guide: {guide_file}")
        return guide


class ModelCardGenerator:
    """Generate model cards for documentation."""

    def __init__(
        self,
        model_name: str,
        config: Any,
    ):
        """
        Initialize model card generator.

        Args:
            model_name: Name of model
            config: Model configuration
        """
        self.model_name = model_name
        self.config = config

    def generate(
        self,
        metrics: Optional[Dict] = None,
        limitations: Optional[List[str]] = None,
    ) -> str:
        """
        Generate model card.

        Args:
            metrics: Model performance metrics
            limitations: Model limitations

        Returns:
            Generated model card markdown
        """
        card = f"""---
title: {self.model_name}
datasets:
  - etl9g
task_ids:
  - image-classification
language: en
license: mit
---

# {self.model_name}

## Model Details

### Model Description

- **Model Type**: {self.model_name}
- **Architecture**: Kanji Recognition Neural Network
- **Task**: Image Classification (Japanese Kanji Characters)
- **Dataset**: ETL9G

### Model Configuration

```python
{self._format_config()}
```

## Model Use

### Intended Use

This model is designed for recognizing Japanese kanji characters from images.

### Recommended Use Cases

- OCR systems for Japanese text
- Handwriting recognition
- Document processing
- Character classification tasks

### Not Recommended For

- Real-time video processing without optimization
- Deployment on extremely resource-constrained devices
- General image classification tasks (not trained for generic images)

## How to Use

### Python

```python
import torch
from src.lib.lightning_trainer import LightningTrainer

# Load model
model = torch.load("model.pt")

# Make predictions
with torch.no_grad():
    output = model(input_batch)
    predictions = output.argmax(dim=1)
```

### CLI

```bash
python scripts/train_modern.py train --model cnn \\
    --learning-rate 1e-3 \\
    --epochs 100
```

## Training Data

- **Dataset**: ETL9G (ETL Character Database)
- **Number of Classes**: {self.config.num_classes}
- **Image Size**: {self.config.image_size}x{self.config.image_size}
- **Data Augmentation**: {self.config.augment_enabled}
- **Train/Val/Test Split**: {self.config.val_split}/{self.config.test_split}

## Evaluation Results

"""

        if metrics:
            card += "### Performance Metrics\n\n"
            for key, value in metrics.items():
                if isinstance(value, float):
                    card += f"- **{key}**: {value:.4f}\n"
                else:
                    card += f"- **{key}**: {value}\n"
        else:
            card += "*(Metrics to be added after training)*\n\n"

        card += """
## Limitations and Biases

"""

        if limitations:
            for limitation in limitations:
                card += f"- {limitation}\n"
        else:
            card += """- Model trained only on ETL9G dataset
- May not generalize well to handwritten input
- Requires image preprocessing for best results
- Character recognition accuracy depends on image quality
"""

        card += """
## Ethical Considerations

- Model is designed for character recognition tasks
- No bias evaluation performed beyond dataset composition
- User is responsible for ethical use

## Caveats and Recommendations

- Always validate on your specific use case
- Consider character distribution in your data
- Monitor model performance over time
- Retrain periodically with new data

## Model Card Contact

For questions about this model, please open an issue on the project repository.

## References

- ETL9G Character Database
- PyTorch Lightning Documentation
- Kanji Character Recognition Literature

---

Generated with Kanji Recognition Modernization Framework
"""

        return card

    def _format_config(self) -> str:
        """Format configuration as Python code."""
        if hasattr(self.config, "model_dump"):
            import json

            config_dict = self.config.model_dump()
            return json.dumps(config_dict, indent=2)
        return str(self.config)

    def save(self, path: Path) -> None:
        """
        Save model card to file.

        Args:
            path: Path to save model card
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        card = self.generate()
        with open(path, "w") as f:
            f.write(card)

        logger.info(f"Saved model card: {path}")


def create_documentation_generator(docs_dir: Optional[Path] = None):
    """Factory function for documentation generator."""
    return DocumentationGenerator(docs_dir)


def create_model_card_generator(model_name: str, config: Any):
    """Factory function for model card generator."""
    return ModelCardGenerator(model_name, config)
