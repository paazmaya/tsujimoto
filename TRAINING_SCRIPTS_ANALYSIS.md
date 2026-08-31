# Training Scripts Analysis: Common Patterns & Base Trainer Extraction

## Executive Summary

Analysis of 7 training scripts reveals **highly redundant patterns** that could be consolidated into a reusable **Base Trainer Class**. This document identifies extractable patterns and proposes a refactoring strategy.

---

## 1. DEVICE INITIALIZATION & GPU SETUP

### Common Pattern Found

**All scripts use identical GPU setup:**

```python
from src.lib import verify_and_setup_gpu

device = verify_and_setup_gpu()
model = model.to(device)
```

**Scripts affected:** `train_cnn_model.py`, `train_rnn.py`, `train_vit.py`, `train_hiercode.py`, `train_qat.py`, `train_radical_rnn.py`, `train_hiercode_higita.py`

### Extractable Pattern: Device Manager

```python
class DeviceManager:
    @staticmethod
    def setup_and_move_model(model):
        device = verify_and_setup_gpu()
        model.to(device)
        return device, model
```

---

## 2. DATA LOADING PATTERNS

### Common Pattern #1: Auto-Detection

```python
# Pattern appears in ALL 7 scripts
data_dir_arg = getattr(args, "data_dir", "dataset")

if data_dir_arg == "dataset":
    data_path = get_dataset_directory()  # Auto-detect
else:
    data_path = Path(data_dir_arg)  # Use specified

data_dir = str(data_path)
logger.info(f"Using dataset from: {data_dir}")
```

### Common Pattern #2: Dataset Loading & Splitting

**CNN & HierCode Hi-GITA scripts:**
```python
prepare_dataset_and_loaders(
    data_dir=str(data_path),
    dataset_fn=create_tensor_dataset,
    batch_size=batch_size,
    sample_limit=sample_limit,
    logger=logger,
)
```

**RNN scripts:**
- Extract sequences (spatial, stroke, radical)
- Use specialized collate functions for variable-length sequences

**ViT & QAT & HierCode:**
- Use `load_chunked_dataset()` from `src.lib`
- Use `create_data_loaders()` helper

### Common Pattern #3: DataLoader Configuration

```python
# Consistent across all scripts
DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True/False,
    num_workers=4,           # ← Same everywhere
    pin_memory=True,         # ← Same everywhere
)
```

### Extractable Pattern: DataLoader Factory

```python
class DataLoaderFactory:
    @staticmethod
    def create_data_loaders(
        data_dir: str,
        batch_size: int,
        model_type: str,  # "cnn", "rnn", "vit", etc.
        sample_limit: Optional[int] = None
    ) -> Tuple[DataLoader, DataLoader]:
        """Unified data loading with auto-detection and preprocessing."""
        ...

    @staticmethod
    def get_auto_dataset_path():
        """Auto-detect dataset path using get_dataset_directory()."""
        ...
```

---

## 3. MODEL BUILDING PATTERNS

### Common Pattern #1: Model Architecture Declaration

```python
# All scripts follow this pattern:
model = ModelClass(
    num_classes=num_classes,
    image_size=image_size,
    ... model-specific args ...
)
model = model.to(device)
logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Common Pattern #2: Config Objects

**All scripts create config objects:**
- `CNNConfig` (train_cnn_model.py)
- `ViTConfig` (train_vit.py)
- `HierCodeConfig` (train_hiercode.py)
- `QATConfig` (train_qat.py)
- `RadicalRNNConfig` (train_radical_rnn.py)
- `HiGITAConfig` (train_hiercode_higita.py)

**Config usage pattern (universal):**
```python
config = ModelConfig(
    data_dir=data_dir,
    image_size=64,
    num_classes=num_classes,
    epochs=epochs,
    batch_size=batch_size,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    ... model-specific params ...
)
```

### Extractable Pattern: Config Base Class

```python
@dataclass
class BaseTrainerConfig:
    data_dir: str
    image_size: int = 64
    num_classes: int = 43427
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    model_dir: str = "training/model/config"
    results_dir: str = "training/model/results"
    checkpoint_dir: str = "training/model/checkpoints"
```

---

## 4. TRAINING LOOP PATTERNS

### Common Pattern #1: Epoch Loop Structure

**All scripts use identical structure:**

```python
for epoch in range(start_epoch, epochs):
    # Train
    train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
    
    # Validate
    val_loss, val_acc = self.validate(val_loader, criterion)
    
    # Update scheduler
    scheduler.step()
    
    # Save checkpoint
    checkpoint_manager.save_checkpoint(epoch, model, optimizer, scheduler, ...)
    
    # Track best model
    if val_acc > best_val_acc:
        torch.save(model.state_dict(), best_model_path)
        best_val_acc = val_acc
```

### Common Pattern #2: train_epoch() Implementation

**Identical pattern across all trainers:**

```python
def train_epoch(self, dataloader, optimizer, criterion, epoch: int):
    self.model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    with tqdm(dataloader, desc=f"Epoch {epoch} Train") as pbar:
        for batch in pbar:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Optional: Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100*correct/total:.1f}%"})
    
    return total_loss / len(dataloader), 100.0 * correct / total
```

**Variations:**
- ViT: Uses mixed precision (`with autocast()`) and GradScaler
- QAT: Same structure but model in fake-quantize mode
- All others: Standard FP32 training

### Common Pattern #3: Validation Implementation

```python
def validate(self, dataloader, criterion):
    self.model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    
    return total_loss / len(dataloader), 100.0 * correct / total
```

### Common Pattern #4: Early Stopping

```python
# CNN model
best_val_acc = 0
patience_counter = 0
max_patience = 15

if val_acc > best_val_acc:
    best_val_acc = val_acc
    patience_counter = 0
    # save best model
else:
    patience_counter += 1
    if patience_counter >= max_patience:
        logger.info(f"Early stopping at epoch {epoch + 1}")
        break
```

### Common Pattern #5: Loss Functions

```python
# Standard pattern
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Variations:
# - ViT: nn.CrossEntropyLoss() (no smoothing)
# - QAT: nn.CrossEntropyLoss()
# - HierCode Hi-GITA: nn.CrossEntropyLoss() + optional contrastive loss
```

### Common Pattern #6: Progress Logging

```python
# All scripts track same metrics
progress_log = {
    "epochs": [],
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "learning_rate": [],
}

# Logged each epoch
logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
```

### Extractable Pattern: Base Trainer Class

```python
class BaseTrainer:
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }
    
    def train_epoch(self, train_loader, optimizer, criterion, epoch):
        """Trainable, mixable with mixed_precision parameter."""
        ...
    
    def validate(self, val_loader, criterion):
        """Standard validation loop."""
        ...
    
    def train(self, train_loader, val_loader, optimizer, scheduler, 
              criterion, epochs, checkpoint_manager=None):
        """Full training loop with checkpointing and early stopping."""
        ...
    
    def save_metrics(self, epoch, metrics, save_path):
        """Save training metrics to JSON."""
        ...
```

---

## 5. CHECKPOINT & MODEL SAVING PATTERNS

### Common Pattern #1: CheckpointManager Usage

```python
# Pattern appears in ALL scripts
checkpoint_manager = CheckpointManager(checkpoint_dir, model_name)

# Save after each epoch
checkpoint_manager.save_checkpoint(
    epoch,
    model,
    optimizer,
    scheduler,
    metrics={"train_loss": train_loss, "val_accuracy": val_acc, ...},
    is_best=val_acc > best_val_acc,
)

# Resume from checkpoint
start_epoch, best_metrics = checkpoint_manager.load_checkpoint_for_training(
    model,
    optimizer,
    scheduler,
    device,
    resume_from=resume_from,
    args_no_checkpoint=no_checkpoint,
)
```

### Common Pattern #2: Best Model Tracking

```python
best_val_acc = 0
best_path = Path(results_dir) / "best_model.pth"

if val_acc > best_val_acc:
    torch.save(model.state_dict(), best_path)
    best_val_acc = val_acc
    logger.info(f"New best model saved! Accuracy: {best_val_acc:.2f}%")
```

### Common Pattern #3: Results Saving

```python
# All scripts save training history as JSON
progress_log = {
    "epochs": [...],
    "train_loss": [...],
    "train_acc": [...],
    "val_loss": [...],
    "val_acc": [...],
}

with open(results_dir / "training_progress.json", "w") as f:
    json.dump(progress_log, f, indent=2)
```

### Extractable Pattern: Checkpoint Manager (Already Exists!)

The `CheckpointManager` class from `src.lib` already encapsulates this pattern.

**Missing additions:**
```python
class CheckpointManager:
    def save_metrics(self, metrics, save_path):
        """Save metrics JSON."""
        ...
    
    def cleanup_old_checkpoints(self, keep_last_n=5):
        """Remove old checkpoint files."""
        ...
```

---

## 6. CLI ARGUMENT PATTERNS

### Common Pattern #1: Argument Definitions

All scripts use centralized `training_args.py` with:

```python
COMMON_ARGS = {
    "data_dir", "epochs", "batch_size", "learning_rate",
    "optimizer", "scheduler", "sample_limit", "resume_from",
    "no_checkpoint", "keep_last_n"
}

IMAGE_ARGS = {
    "image_size", "num_classes"
}

CHECKPOINT_ARGS = {
    "checkpoint_dir"
}

# Variant-specific:
RNN_ARGS, HIERCODE_ARGS, HIERCODE_HIGITA_ARGS, RADICAL_RNN_ARGS, etc.
```

### Common Pattern #2: add_variant_args_to_parser()

```python
# Used in ALL scripts
from scripts.training_args import add_variant_args_to_parser

parser = argparse.ArgumentParser(description="...")
add_variant_args_to_parser(parser, "cnn", checkpoint_dir_default="training/cnn/checkpoints")
args = parser.parse_args()
train_cnn(args)
```

### Common Pattern #3: Argument Extraction with Safe Defaults

```python
# All scripts use identical pattern
epochs = getattr(args, "epochs", 30)
batch_size = getattr(args, "batch_size", 64)
learning_rate = getattr(args, "learning_rate", 0.001)
# ... etc
```

### Common Pattern #4: Main Function Structure

```python
def main():
    parser = argparse.ArgumentParser(...)
    from scripts.training_args import add_variant_args_to_parser
    add_variant_args_to_parser(parser, "variant", checkpoint_dir_default="...")
    args = parser.parse_args()
    train_variant(args)

def train_variant(args):
    # Extract arguments with defaults
    # Setup logging and GPU
    # Load dataset
    # Create model and config
    # Initialize trainer
    # Train
    # Save results

if __name__ == "__main__":
    main()
```

### Extractable Pattern: Argument Parser Factory

```python
class ArgumentParserFactory:
    @staticmethod
    def create_parser(variant: str, description: str):
        """Create argument parser for a training variant."""
        parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        add_variant_args_to_parser(parser, variant)
        return parser
    
    @staticmethod
    def extract_args_with_defaults(args, arg_definitions):
        """Safely extract arguments with defaults."""
        return {
            name: getattr(args, name, defn.default)
            for name, defn in arg_definitions.items()
        }
```

---

## 7. OPTIMIZER & SCHEDULER PATTERNS

### Common Pattern #1: Unified Factory Functions

All scripts use identical pattern:

```python
from src.lib import get_optimizer, get_scheduler

# Create config object
config = ModelConfig(
    learning_rate=learning_rate,
    weight_decay=1e-4,
    epochs=epochs,
    optimizer="adamw",
    scheduler="cosine",
    scheduler_t_max=epochs,
    scheduler_eta_min=1e-6,
)

# Use factory functions
optimizer = get_optimizer(model, config)
scheduler = get_scheduler(optimizer, config)
```

### Optimizer Types

```python
# AdamW (default, used in 6/7 scripts)
optimizer = get_optimizer(model, config)  # type: adamw, lr: 0.001, weight_decay: 1e-4

# SGD (alternative in training_args.py)
```

### Scheduler Types

```python
# Cosine annealing (default, used in 6/7 scripts)
scheduler = get_scheduler(optimizer, config)  # type: cosine, T_max: epochs, eta_min: 1e-6

# Step scheduler (alternative in training_args.py)
```

### Extractable Pattern: Already Exists!

The `get_optimizer()` and `get_scheduler()` factory functions already encapsulate this.

---

## 8. LOGGING PATTERNS

### Common Pattern: Logger Setup

```python
# All 7 scripts use identical pattern
from src.lib import setup_logger

logger = setup_logger(__name__)

# Usage throughout
logger.info("Using dataset from: ...")
logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
logger.warning("⚠️ Warning message")
```

---

## 9. SPECIAL/UNIQUE PATTERNS

### Pattern #1: Mixed Precision Training (ViT only)

```python
# train_vit.py specific
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

scaler = GradScaler(device)

with autocast(device, dtype=torch.float16):
    outputs = model(images)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Pattern #2: Quantization-Aware Training (QAT only)

```python
# train_qat.py specific
import torch.quantization as tq

model.qconfig = tq.get_default_qat_qconfig(backend)
tq.prepare_qat(model, inplace=True)

# Training loop with FakeQuantize layers
# ... train ...

# After training:
tq.convert(model, inplace=True)
```

### Pattern #3: Variable-Length Sequence Handling (RNN scripts)

```python
# train_rnn.py and train_radical_rnn.py specific
def collate_fn(batch):
    # Pad variable-length sequences
    sequences = [item["sequences"] for item in batch]
    max_length = max(len(s) for s in sequences)
    padded = [
        torch.cat([s, torch.zeros(max_length - len(s), ...)]) 
        for s in sequences
    ]
    return torch.stack(padded)
```

### Pattern #4: Hierarchical Codebook Selection (HierCode)

```python
# train_hiercode.py specific
logits = self.codeword_selector(features)
selection = self.gumbel_softmax_sample(logits, k=num_active_codes, hard=True)
```

### Pattern #5: Contrastive Learning (HierCode Hi-GITA)

```python
# train_hiercode_higita.py specific
text_output = text_encoder(stroke_codes, radical_codes)
contrastive_loss = contrastive_loss_fn(visual_features, text_output)
total_loss = ce_loss + config.contrastive_weight * contrastive_loss
```

---

## 10. PROPOSED BASE TRAINER CLASS HIERARCHY

### Level 0: Base Class (Core Training Loop)

```python
class BaseTrainer:
    """Universal trainer with core training loop."""
    
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.history = {...}
    
    def train_epoch(self, train_loader, optimizer, criterion, epoch):
        """Standard training loop."""
        
    def validate(self, val_loader, criterion):
        """Standard validation loop."""
    
    def train(self, train_loader, val_loader, epochs, optimizer, scheduler, 
              criterion, checkpoint_manager=None, start_epoch=0):
        """Main training loop with checkpointing."""
    
    def save_progress(self, path):
        """Save training history."""
        
    def _setup_early_stopping(self, patience=15):
        """Setup early stopping mechanism."""
```

### Level 1: Specialized Trainers

```python
class CNNTrainer(BaseTrainer):
    """CNN-specific trainer."""
    pass  # Uses base implementation as-is

class ViTTrainer(BaseTrainer):
    """ViT trainer with mixed precision support."""
    
    def train_epoch(self, train_loader, optimizer, criterion, epoch):
        """Overridden with mixed precision (autocast + GradScaler)."""

class QATTrainer(BaseTrainer):
    """QAT trainer with quantization simulation."""
    
    def prepare_qat(self):
        """Prepare model for fake quantization."""
    
    def train_epoch(self, ...):
        """Training with fake quantization active."""

class RNNTrainer(BaseTrainer):
    """RNN trainer with sequence handling."""
    
    def train_epoch(self, train_loader, optimizer, criterion, epoch):
        """Handles variable-length sequences with pack_padded_sequence."""

class HierCodeTrainer(BaseTrainer):
    """HierCode trainer with codebook selection."""
    
    def train_epoch(self, train_loader, optimizer, criterion, epoch):
        """Trains with multi-hot codebook encoding."""

class HierCodeHiGITATrainer(BaseTrainer):
    """HierCode Hi-GITA trainer with contrastive learning."""
    
    def train_epoch(self, train_loader, optimizer, criterion, epoch):
        """Combines classification loss with contrastive loss."""
```

### Level 2: Factory Function

```python
class TrainerFactory:
    @staticmethod
    def create_trainer(model, config, device, trainer_type="base"):
        """Factory to instantiate appropriate trainer."""
        trainers = {
            "base": BaseTrainer,
            "cnn": CNNTrainer,
            "vit": ViTTrainer,
            "qat": QATTrainer,
            "rnn": RNNTrainer,
            "hiercode": HierCodeTrainer,
            "hiercode_higita": HierCodeHiGITATrainer,
        }
        return trainers[trainer_type](model, config, device)
```

---

## 11. REFACTORING ROADMAP

### Phase 1: Extract Common Infrastructure (Low Risk)

1. **BaseTrainerConfig dataclass**
   - Consolidate all config objects
   - Reduce from 7 config classes to 1 base + optional mixins

2. **BaseTrainer class**
   - Extract `train_epoch()`, `validate()`, `train()` from CNN
   - Move to `src/lib/trainer.py`
   - Reduce code duplication

3. **TrainerFactory**
   - Centralize trainer instantiation
   - Used by all training scripts

### Phase 2: Reduce Script Boilerplate (Medium Risk)

1. **Update train_*.py scripts**
   - Replace duplicated `train_epoch()` with inheritance
   - Remove redundant checkpoint loading
   - Simplify from 400-500 lines to 100-150 lines

2. **Create unified main() template**
   - Extract common argument parsing
   - Extract common data loading
   - Extract common logging setup

### Phase 3: Advanced Optimization (Higher Risk)

1. **Mixed-precision trainer mixin**
   - Extracted from ViT trainer
   - Reusable by other models

2. **Quantization trainer mixin**
   - Extracted from QAT trainer
   - Reusable for future quantized models

3. **Contrastive learning mixin**
   - Extracted from Hi-GITA trainer
   - Reusable for multi-task learning

---

## 12. CODE METRICS

### Current State (Duplicate Code)

| Component | Train CNN | Train RNN | Train ViT | Train HierCode | Train QAT | Train Radical | Train Hi-GITA | **TOTAL** |
|-----------|-----------|-----------|-----------|----------------|-----------|---------------|---------------|----------|
| GPU Setup | ~5 lines | ~5 lines | ~5 lines | ~5 lines | ~5 lines | ~5 lines | ~5 lines | **35** |
| Data Loading | ~15 lines | ~15 lines | ~15 lines | ~15 lines | ~15 lines | ~15 lines | ~15 lines | **105** |
| Model Creation | ~10 lines | ~10 lines | ~10 lines | ~10 lines | ~10 lines | ~10 lines | ~10 lines | **70** |
| train_epoch() | ~40 lines | ~40 lines | ~60 lines* | ~40 lines | ~40 lines | ~40 lines | ~50 lines | **310** |
| validate() | ~20 lines | ~20 lines | ~25 lines* | ~20 lines | ~20 lines | ~20 lines | ~20 lines | **145** |
| Main loop | ~30 lines | ~30 lines | ~30 lines | ~30 lines | ~30 lines | ~30 lines | ~30 lines | **210** |
| Checkpointing | ~15 lines | ~15 lines | ~15 lines | ~15 lines | ~15 lines | ~15 lines | ~15 lines | **105** |
| **SUBTOTAL** | **135** | **135** | **160** | **135** | **135** | **135** | **145** | **980** |
| Trainer class | 400-500 | 400-500 | 400-500 | 400-500 | 400-500 | 400-500 | 400-500 | **2,800-3,500** |
| **TOTAL LINES** | **500-600** | **500-600** | **600-700** | **500-600** | **500-600** | **500-600** | **500-600** | **3,500-4,300** |

**Duplication rate: ~75%**

### Proposed Refactored State (Inheritance-Based)

| Component | BaseTrainer | CNN | RNN | ViT | HierCode | QAT | Radical | Hi-GITA |
|-----------|-------------|-----|-----|-----|----------|-----|---------|---------|
| train_epoch() | 40 lines | 0 (inherit) | 30 (override) | 30 (override) | 30 (override) | 30 (override) | 30 (override) | 40 (override) |
| validate() | 20 lines | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Main loop | 30 lines | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Checkpointing | 15 lines | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Trainer class** | **105** | **50** | **80** | **100** | **80** | **100** | **80** | **110** |
| Main function | 100 lines | 50 | 50 | 50 | 50 | 50 | 50 | 50 |
| **TOTAL** | **~205** | **~150** | **~180** | **~200** | **~180** | **~200** | **~180** | **~210** |

**Total codebase: ~1,500 lines** (57% reduction from 3,500)

---

## 13. SUMMARY TABLE: Common Patterns

| Pattern | Found In | Extractable? | Implementation |
|---------|----------|-------------|-----------------|
| GPU setup | All 7 | ✅ Yes | Use `verify_and_setup_gpu()` |
| Dataset auto-detection | All 7 | ✅ Yes | Use `get_dataset_directory()` |
| DataLoader creation | All 7 | ✅ Yes | Unified factory with num_workers=4, pin_memory=True |
| train_epoch() | All 7 | ✅ Yes (with hooks) | BaseTrainer.train_epoch() + overrides for mixed-precision, QAT |
| validate() | All 7 | ✅ Yes | BaseTrainer.validate() (identical in all) |
| Main training loop | All 7 | ✅ Yes | BaseTrainer.train() with early stopping |
| Checkpoint management | All 7 | ✅ Yes | Use CheckpointManager (already exists) |
| Best model tracking | All 7 | ✅ Yes | CheckpointManager.save_best_model() |
| Metrics logging | All 7 | ✅ Yes | BaseTrainer.save_metrics() |
| Argument parsing | All 7 | ✅ Yes | Centralized training_args.py + factory |
| Optimizer creation | All 7 | ✅ Yes | Use `get_optimizer()` factory (already exists) |
| Scheduler creation | All 7 | ✅ Yes | Use `get_scheduler()` factory (already exists) |
| Mixed precision | ViT only | ✅ Yes (mixin) | TrainerMixin with autocast + GradScaler |
| QAT setup | QAT only | ✅ Yes (mixin) | TrainerMixin with prepare_qat() + convert() |
| Contrastive learning | Hi-GITA only | ✅ Yes (mixin) | TrainerMixin with auxiliary loss |
| Variable-length sequences | RNN, Radical-RNN | ✅ Yes (mixin) | TrainerMixin with pack_padded_sequence |

---

## 14. ACTIONABLE RECOMMENDATIONS

### Immediate Actions (Quick Wins)

1. **Create `src/lib/trainer.py`** with BaseTrainer class
   - Extract train_epoch, validate, train methods from train_cnn_model.py
   - Save ~500 lines of code across all scripts

2. **Create `src/lib/trainer_factory.py`**
   - Implement TrainerFactory for consistent instantiation
   - Used by all training scripts

3. **Enhance `CheckpointManager`**
   - Add `save_metrics()` method
   - Add `cleanup_old_checkpoints()` method

### Medium-Term Actions (Structural Improvements)

1. **Create trainer mixins** for specialized functionality
   - `MixedPrecisionMixin` (ViT-style training)
   - `QuantizationMixin` (QAT-style training)
   - `SequenceMixin` (RNN-style variable-length handling)
   - `ContrastiveMixin` (Hi-GITA-style auxiliary loss)

2. **Refactor train_*.py scripts**
   - Use inheritance instead of duplication
   - Reduce each script to ~200-250 lines (from 500-600)
   - Only keep model-specific logic

### Long-Term Actions (Architectural)

1. **Unified training CLI**
   - Single `train.py` script with model selection
   - All variants accessible via `python train.py --model-type cnn --epochs 30`

2. **Configuration system**
   - YAML-based configs for experiment management
   - Reproducible training runs
   - Hyperparameter sweep capability

---

## 15. RISK ASSESSMENT

### Low Risk (Safe to Implement)

- Extract GPU setup to function ✅
- Extract data loading to factory ✅
- Extract checkpoint management ✅
- Centralize argument parsing ✅

### Medium Risk (Needs Testing)

- Create BaseTrainer with train_epoch() override pattern
- Trainer mixins for specialized behavior
- Refactor existing scripts to use inheritance

### High Risk (Need Careful Implementation)

- Unified training CLI (need backward compatibility)
- Major structural refactoring (need comprehensive testing)

---

## Conclusion

**Key Finding:** 75% of training script code is duplicated across the 7 variants. By implementing a `BaseTrainer` class hierarchy with optional mixins, we can reduce total codebase from **3,500-4,300 lines to ~1,500 lines** while maintaining the same functionality and improving maintainability.

**Priority:** Extract BaseTrainer and refactor train_cnn_model.py first (safest choice), then progressively migrate other scripts.
