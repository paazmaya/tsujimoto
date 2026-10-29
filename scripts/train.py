#!/usr/bin/env python3
"""
Unified Training Entry Point for Kanji Recognition Models
==========================================================

Consolidates all training variants into a single entry point with Click subcommands.

Supported variants:
  - cnn         : Lightweight CNN model
  - hiercode    : HierCode (Hierarchical Codebook)
  - hiercode_higita : HierCode with Hi-GITA enhancement
  - rnn         : RNN-based model (5 variants: basic, stroke, simple_radical, hybrid_cnn, linguistic_radical)
  - vit         : Vision Transformer (ViT with T2T)
  - qat         : Quantization-Aware Training

Usage:
    python scripts/train.py cnn --epochs 30 --batch-size 64
    python scripts/train.py hiercode --codebook-total-size 1024
    python scripts/train.py rnn --model-type hybrid_cnn_rnn
    python scripts/train.py vit --epochs 40
    python scripts/train.py qat --epochs 25
    python scripts/train.py hiercode_higita --enable-higita-enhancement

For variant-specific help:
    python scripts/train.py cnn --help
    python scripts/train.py hiercode --help
    python scripts/train.py rnn --help
"""

import sys
from pathlib import Path

import click

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Optional imports for existing training variants
try:
    from scripts import (
        train_cnn_model,
        train_hiercode,
        train_hiercode_higita,
        train_qat,
        train_rnn,
        train_vit,
    )
except ImportError:
    # If individual training modules don't exist, create stub modules
    class StubModule:
        def __getattr__(self, name):
            def stub(*args, **kwargs):
                click.echo("Training module not configured. Please use train_modern.py instead.")

            return stub

    train_cnn_model = StubModule()
    train_rnn = StubModule()
    train_vit = StubModule()
    train_qat = StubModule()
    train_hiercode = StubModule()
    train_hiercode_higita = StubModule()

from scripts.training_args import COMMON_ARGS, IMAGE_ARGS

# ========== COMMON CLICK OPTIONS (DRY Principle) ==========


def add_common_options(func):
    """Decorator to add common training options to all variants.

    Dynamically built from COMMON_ARGS configuration.
    """
    # Apply decorators for all common arguments
    for arg_def in reversed(list(COMMON_ARGS.values())):
        arg_name = f"--{arg_def.name.replace('_', '-')}"
        if arg_def.is_flag:
            func = click.option(arg_name, is_flag=True, help=arg_def.help_text)(func)
        elif arg_def.choices:
            func = click.option(
                arg_name,
                type=click.Choice(arg_def.choices),
                default=arg_def.default,
                help=arg_def.help_text,
            )(func)
        else:
            func = click.option(
                arg_name,
                type=arg_def.arg_type,
                default=arg_def.default,
                help=arg_def.help_text,
            )(func)
    return func


def add_image_options(func):
    """Decorator to add image-related options.

    Dynamically built from IMAGE_ARGS configuration.
    """
    # Apply decorators in reverse order
    for arg_def in reversed(list(IMAGE_ARGS.values())):
        arg_name = f"--{arg_def.name.replace('_', '-')}"
        func = click.option(
            arg_name,
            type=arg_def.arg_type,
            default=arg_def.default,
            help=arg_def.help_text,
        )(func)
    return func


def add_checkpoint_dir_option(default_path: str):
    """Decorator factory to add checkpoint directory option with specific default."""

    def decorator(func):
        return click.option(
            "--checkpoint-dir",
            type=click.Path(),
            default=default_path,
            help="Checkpoint directory",
        )(func)

    return decorator


DATASET_CLASS_COUNTS = {
    "etl6": 114,
    "etl7": 48,
    "etl8g": 956,
    "etl9g": 3036,
    "combined_all_etl": 4154,
    "megahan97k": 97455,
    "dkds": 3831,
    "chronicles_ocr": 2400,
    "jawildtext": 3643,
    "mccd": 3500,
    "stroke_database": 1200,
}


def _resolve_dataset_and_loaders(
    dataset_name: str | None,
    data_dir: str,
    batch_size: int,
    num_classes_arg: int | None,
    sample_limit: int | None = None,
):
    """Resolve dataset name, build train/validation DataLoaders, and infer num_classes."""
    from torch.utils.data import DataLoader, Subset

    from src.lib.datasets import KanjiDatasetLoader, ResearchDatasetLoader

    if not dataset_name:
        # Auto-detect priority order in data_dir
        priority = ["combined_all_etl", "etl9g", "etl8g", "etl7", "etl6"]
        detected = "etl9g"
        base_path = Path(data_dir)
        for candidate in priority:
            if (base_path / candidate).exists():
                detected = candidate
                break
        dataset_name = detected

    etl_datasets = ["etl6", "etl7", "etl8g", "etl9g", "combined_all_etl"]
    default_classes = DATASET_CLASS_COUNTS.get(dataset_name, 3036)

    if dataset_name in etl_datasets:
        loader = KanjiDatasetLoader(
            dataset_name=dataset_name,
            cache_dir=data_dir,
            num_workers=0,
            offline=True,
        )
        num_classes = num_classes_arg or default_classes
        train_loader = loader.get_dataloader("train", batch_size=batch_size)
        val_loader = None
        val_split = None
        for split_name in ["validation", "val", "test"]:
            try:
                val_loader = loader.get_dataloader(split_name, batch_size=batch_size * 2)
                val_split = split_name
                break
            except Exception:  # noqa: BLE001, S110
                pass
    else:
        resolved_dir = Path("data/research_datasets") if data_dir == "dataset" else Path(data_dir)
        loader = ResearchDatasetLoader(
            dataset_name=dataset_name,
            data_dir=resolved_dir,
            num_workers=0,
        )
        num_classes = num_classes_arg or (getattr(loader, "num_classes", None) or default_classes)
        train_loader = loader.get_dataloader(batch_size=batch_size, split="train")
        if train_loader is None:
            raise RuntimeError(
                f"Failed to load research dataset '{dataset_name}' from {resolved_dir}"
            )
        val_loader = loader.get_dataloader(batch_size=batch_size * 2, split="test")
        val_split = "test" if val_loader is not None else None

    if val_loader is None:
        val_loader = train_loader

    if sample_limit is not None and sample_limit > 0:
        train_ds = train_loader.dataset
        limited_train = Subset(train_ds, list(range(min(len(train_ds), sample_limit))))
        train_loader = DataLoader(limited_train, batch_size=batch_size, shuffle=True)

    return dataset_name, train_loader, val_loader, val_split, num_classes


def _configure_trainer_options(
    model,
    trainer,
    epochs: int,
    learning_rate: float,
    optimizer_name: str = "adamw",
    scheduler_name: str = "cosine",
    weight_decay: float = 1e-4,
    resume_from: str | None = None,
    no_checkpoint: bool = False,
    keep_last_n: int = 5,
) -> None:
    """Apply optimizer, scheduler, and checkpoint preferences to a BaseModelTrainer."""
    import torch

    if optimizer_name.lower() == "sgd":
        trainer.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    else:
        trainer.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if scheduler_name.lower() == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            trainer.optimizer, step_size=max(1, epochs // 3), gamma=0.1
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, T_max=max(1, epochs)
        )
    trainer.set_scheduler(scheduler)

    if hasattr(trainer, "checkpoint_manager") and trainer.checkpoint_manager is not None:
        trainer.checkpoint_manager.keep_last_n = keep_last_n

    if resume_from and not no_checkpoint:
        trainer.load_checkpoint(resume_from)


@click.group(invoke_without_command=True)
@click.option(
    "--generate-metadata",
    is_flag=True,
    default=False,
    help="Generate dataset metadata and exit",
)
@click.pass_context
def train(ctx, generate_metadata: bool = False):
    """Unified training entry point for Kanji recognition models."""
    if generate_metadata:
        from src.lib.metadata_generator import DatasetMetadataManager

        manager = DatasetMetadataManager(base_dir="dataset")
        click.echo(f"Dataset metadata manager initialized for: {manager.base_dir}")
        return
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@train.command()
@add_common_options
@add_image_options
@add_checkpoint_dir_option("training/cnn/checkpoints")
def cnn(**kwargs):
    """Train Lightweight CNN model for Kanji recognition.

    Supports both ETL and research datasets:
    - ETL: etl6, etl7, etl8g, etl9g, combined_all_etl
    - Research: kanji_full, kanji_dataset_v3, kanji, megahan97k, dkds, etc.

    Examples:
        python scripts/train.py cnn --dataset kanji_full --epochs 30
        python scripts/train.py cnn --dataset etl9g --epochs 100
    """
    import traceback

    import torch

    from src.lib.base_trainer import setup_trainer_for_model
    from src.lib.models import LightweightKanjiNet

    # Extract parameters
    dataset_name_arg = kwargs.pop("dataset_name", None)
    epochs = kwargs.pop("epochs", 30)
    batch_size = kwargs.pop("batch_size", 32)
    learning_rate = kwargs.pop("learning_rate", 0.001)
    checkpoint_dir = kwargs.pop("checkpoint_dir")
    data_dir = kwargs.pop("data_dir", "dataset")
    image_size = kwargs.pop("image_size", 64)
    num_classes_arg = kwargs.pop("num_classes", None)
    optimizer_name = kwargs.pop("optimizer", "adamw")
    scheduler_name = kwargs.pop("scheduler", "cosine")
    sample_limit = kwargs.pop("sample_limit", None)
    resume_from = kwargs.pop("resume_from", None)
    no_checkpoint = kwargs.pop("no_checkpoint", False)
    keep_last_n = kwargs.pop("keep_last_n", 5)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset_name, train_loader, val_loader, val_split, num_classes = (
            _resolve_dataset_and_loaders(
                dataset_name_arg, data_dir, batch_size, num_classes_arg, sample_limit
            )
        )
        click.echo(f"Training CNN on dataset: {dataset_name}")
        click.echo(f"Using data directory: {data_dir}")
        click.echo(f"Using device: {device}")
        click.echo("  ✓ Loaded training data")
        if val_split:
            click.echo(f"  ✓ Loaded {val_split} data")

        # Create model
        click.echo("Creating CNN model...")
        model = LightweightKanjiNet(
            num_classes=num_classes,
            input_channels=1,
            image_size=image_size,
        ).to(device)

        click.echo(f"  ✓ Model created with {num_classes} classes")

        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )

        # Setup trainer
        trainer = setup_trainer_for_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=str(device),
            model_type="cnn",
            checkpoint_dir=checkpoint_dir,
            num_classes=num_classes,
            image_size=image_size,
        )
        _configure_trainer_options(
            model=model,
            trainer=trainer,
            epochs=epochs,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
            resume_from=resume_from,
            no_checkpoint=no_checkpoint,
            keep_last_n=keep_last_n,
        )

        # Train
        click.echo(f"Starting training for {epochs} epochs...")
        trainer.train(num_epochs=epochs)
        click.echo("✓ Training completed successfully!")
    except Exception as e:
        click.echo(f"✗ Training failed: {e}", err=True)
        traceback.print_exc()
        raise


@train.command()
@add_common_options
@add_image_options
@add_checkpoint_dir_option("training/hiercode/checkpoints")
@click.option(
    "--codebook-total-size",
    type=int,
    default=1024,
    help="Total codebook size",
)
@click.option("--codebook-dim", type=int, default=128, help="Codebook dimension")
@click.option("--hierarch-depth", type=int, default=10, help="Hierarchical tree depth")
@click.option("--multi-hot-k", type=int, default=5, help="Number of active codewords")
@click.option("--temperature", type=float, default=0.1, help="Gumbel-softmax temperature")
def hiercode(**kwargs):
    """Train HierCode (Hierarchical Codebook) model."""
    import torch

    from src.lib.base_trainer import setup_trainer_for_model
    from src.lib.models import HierCodeNet

    # Extract parameters
    dataset_name_arg = kwargs.pop("dataset_name", None)
    epochs = kwargs.pop("epochs", 30)
    batch_size = kwargs.pop("batch_size", 32)
    learning_rate = kwargs.pop("learning_rate", 0.001)
    checkpoint_dir = kwargs.pop("checkpoint_dir")
    data_dir = kwargs.pop("data_dir", "dataset")
    num_classes_arg = kwargs.pop("num_classes", None)
    image_size = kwargs.pop("image_size", 64)
    codebook_total_size = kwargs.pop("codebook_total_size", 1024)
    optimizer_name = kwargs.pop("optimizer", "adamw")
    scheduler_name = kwargs.pop("scheduler", "cosine")
    sample_limit = kwargs.pop("sample_limit", None)
    resume_from = kwargs.pop("resume_from", None)
    no_checkpoint = kwargs.pop("no_checkpoint", False)
    keep_last_n = kwargs.pop("keep_last_n", 5)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset_name, train_loader, val_loader, val_split, num_classes = (
            _resolve_dataset_and_loaders(
                dataset_name_arg, data_dir, batch_size, num_classes_arg, sample_limit
            )
        )
        click.echo(f"Training HierCode on dataset: {dataset_name}")
        click.echo(f"Using data directory: {data_dir}")
        click.echo(f"Using device: {device}")
        click.echo("  ✓ Loaded training data")
        if val_split:
            click.echo(f"  ✓ Loaded {val_split} data")

        # Create model
        click.echo("Creating HierCode model...")
        model = HierCodeNet(
            num_classes=num_classes,
            codebook_size=codebook_total_size,
            num_levels=3,
            input_channels=1,
            image_size=image_size,
        ).to(device)

        click.echo(f"  ✓ Model created with {num_classes} classes")

        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )

        # Setup trainer
        trainer = setup_trainer_for_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=str(device),
            model_type="hiercode",
            checkpoint_dir=checkpoint_dir,
            num_classes=num_classes,
            image_size=image_size,
        )
        _configure_trainer_options(
            model=model,
            trainer=trainer,
            epochs=epochs,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
            resume_from=resume_from,
            no_checkpoint=no_checkpoint,
            keep_last_n=keep_last_n,
        )

        # Train
        click.echo(f"Starting training for {epochs} epochs...")
        trainer.train(num_epochs=epochs)
        click.echo("✓ Training completed successfully!")
    except Exception as e:
        click.echo(f"✗ Training failed: {e}", err=True)
        import traceback

        traceback.print_exc()
        raise


@train.command()
@add_common_options
@add_image_options
@add_checkpoint_dir_option("training/hiercode_higita/checkpoints")
@click.option(
    "--enable-higita-enhancement",
    is_flag=True,
    help="Enable Hi-GITA enhancement",
)
def hiercode_higita(**kwargs):
    """Train HierCode with Hi-GITA enhancement."""
    import torch

    from src.lib.base_trainer import setup_trainer_for_model
    from src.lib.models import HierCodeHiGITA

    # Extract parameters
    dataset_name_arg = kwargs.pop("dataset_name", None)
    epochs = kwargs.pop("epochs", 30)
    batch_size = kwargs.pop("batch_size", 32)
    learning_rate = kwargs.pop("learning_rate", 0.001)
    checkpoint_dir = kwargs.pop("checkpoint_dir")
    data_dir = kwargs.pop("data_dir", "dataset")
    num_classes_arg = kwargs.pop("num_classes", None)
    image_size = kwargs.pop("image_size", 64)
    enable_higita = kwargs.pop("enable_higita_enhancement", False)
    optimizer_name = kwargs.pop("optimizer", "adamw")
    scheduler_name = kwargs.pop("scheduler", "cosine")
    sample_limit = kwargs.pop("sample_limit", None)
    resume_from = kwargs.pop("resume_from", None)
    no_checkpoint = kwargs.pop("no_checkpoint", False)
    keep_last_n = kwargs.pop("keep_last_n", 5)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset_name, train_loader, val_loader, val_split, num_classes = (
            _resolve_dataset_and_loaders(
                dataset_name_arg, data_dir, batch_size, num_classes_arg, sample_limit
            )
        )
        click.echo(f"Training HierCode-HiGITA on dataset: {dataset_name}")
        click.echo(f"Using data directory: {data_dir}")
        if enable_higita:
            click.echo("Hi-GITA enhancement enabled")
        click.echo(f"Using device: {device}")
        click.echo("  ✓ Loaded training data")
        if val_split:
            click.echo(f"  ✓ Loaded {val_split} data")

        # Create model
        click.echo("Creating HierCode-HiGITA model...")
        model = HierCodeHiGITA(
            num_classes=num_classes,
            codebook_size=1024,
            input_channels=1,
            image_size=image_size,
        ).to(device)

        click.echo(f"  ✓ Model created with {num_classes} classes")

        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )

        # Setup trainer
        trainer = setup_trainer_for_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=str(device),
            model_type="hiercode_higita",
            checkpoint_dir=checkpoint_dir,
            num_classes=num_classes,
            image_size=image_size,
        )
        _configure_trainer_options(
            model=model,
            trainer=trainer,
            epochs=epochs,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
            resume_from=resume_from,
            no_checkpoint=no_checkpoint,
            keep_last_n=keep_last_n,
        )

        # Train
        click.echo(f"Starting training for {epochs} epochs...")
        trainer.train(num_epochs=epochs)
        click.echo("✓ Training completed successfully!")
    except Exception as e:
        click.echo(f"✗ Training failed: {e}", err=True)
        import traceback

        traceback.print_exc()
        raise


@train.command()
@add_common_options
@add_image_options
@add_checkpoint_dir_option("training/rnn/checkpoints")
@click.option(
    "--model-type",
    type=click.Choice(
        [
            "basic_rnn",
            "stroke_rnn",
            "simple_radical_rnn",
            "radical_rnn",
            "hybrid_cnn_rnn",
            "linguistic_radical_rnn",
        ]
    ),
    default="hybrid_cnn_rnn",
    help="Type of RNN model (5 variants available)",
)
@click.option("--weight-decay", type=float, default=1e-4, help="Weight decay")
@click.option("--hidden-size", type=int, default=256, help="RNN hidden size")
@click.option("--num-layers", type=int, default=2, help="Number of RNN layers")
@click.option("--dropout", type=float, default=0.3, help="Dropout rate")
@click.option(
    "--rnn-type",
    type=click.Choice(["lstm", "gru"]),
    default="lstm",
    help="RNN cell type (for linguistic variant)",
)
def rnn(**kwargs):
    """
    Train RNN-based model for Kanji recognition.

    Available variants:
    - basic_rnn: Spatial grid scanning with LSTM
    - stroke_rnn: Stroke sequence processing
    - simple_radical_rnn: Simple radical decomposition (500 vocab)
    - hybrid_cnn_rnn: Combined CNN-RNN architecture (best accuracy)
    - linguistic_radical_rnn: Advanced radical decomposition (2000 vocab)
    """
    import torch

    from src.lib.base_trainer import setup_trainer_for_model
    from src.lib.models import KanjiRNN

    # Extract parameters
    dataset_name_arg = kwargs.pop("dataset_name", None)
    epochs = kwargs.pop("epochs", 30)
    batch_size = kwargs.pop("batch_size", 32)
    learning_rate = kwargs.pop("learning_rate", 0.001)
    checkpoint_dir = kwargs.pop("checkpoint_dir")
    data_dir = kwargs.pop("data_dir", "dataset")
    num_classes_arg = kwargs.pop("num_classes", None)
    hidden_size = kwargs.pop("hidden_size", 256)
    num_layers = kwargs.pop("num_layers", 2)
    rnn_type = kwargs.pop("rnn_type", "lstm")
    model_type = kwargs.pop("model_type", "hybrid_cnn_rnn")
    image_size = kwargs.pop("image_size", 64)
    weight_decay = kwargs.pop("weight_decay", 1e-4)
    optimizer_name = kwargs.pop("optimizer", "adamw")
    scheduler_name = kwargs.pop("scheduler", "cosine")
    sample_limit = kwargs.pop("sample_limit", None)
    resume_from = kwargs.pop("resume_from", None)
    no_checkpoint = kwargs.pop("no_checkpoint", False)
    keep_last_n = kwargs.pop("keep_last_n", 5)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset_name, train_loader, val_loader, val_split, num_classes = (
            _resolve_dataset_and_loaders(
                dataset_name_arg, data_dir, batch_size, num_classes_arg, sample_limit
            )
        )
        click.echo(f"Training RNN ({model_type}) on dataset: {dataset_name}")
        click.echo(f"Using data directory: {data_dir}")
        click.echo(f"Using device: {device}")
        click.echo("  ✓ Loaded training data")
        if val_split:
            click.echo(f"  ✓ Loaded {val_split} data")

        # Create model
        click.echo("Creating RNN model...")
        model = KanjiRNN(
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            rnn_type=rnn_type,
            input_channels=1,
            image_size=image_size,
        ).to(device)

        click.echo(f"  ✓ Model created ({model_type} variant)")

        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Setup trainer
        trainer = setup_trainer_for_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=str(device),
            model_type="rnn",
            checkpoint_dir=checkpoint_dir,
            num_classes=num_classes,
            image_size=image_size,
        )
        _configure_trainer_options(
            model=model,
            trainer=trainer,
            epochs=epochs,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
            weight_decay=weight_decay,
            resume_from=resume_from,
            no_checkpoint=no_checkpoint,
            keep_last_n=keep_last_n,
        )

        # Train
        click.echo(f"Starting training for {epochs} epochs...")
        trainer.train(num_epochs=epochs)
        click.echo("✓ Training completed successfully!")
    except Exception as e:
        click.echo(f"✗ Training failed: {e}", err=True)
        import traceback

        traceback.print_exc()
        raise


@train.command()
@add_common_options
@add_image_options
@add_checkpoint_dir_option("training/vit/checkpoints")
@click.option("--patch-size", type=int, default=8, help="Patch size for ViT")
@click.option("--embed-dim", type=int, default=256, help="Embedding dimension")
@click.option("--depth", type=int, default=12, help="Number of transformer blocks")
@click.option("--num-heads", type=int, default=8, help="Number of attention heads")
def vit(**kwargs):
    """Train Vision Transformer (ViT) model for Kanji recognition."""
    import torch

    from src.lib.base_trainer import setup_trainer_for_model
    from src.lib.models import KanjiViT

    # Extract parameters
    dataset_name_arg = kwargs.pop("dataset_name", None)
    epochs = kwargs.pop("epochs", 30)
    batch_size = kwargs.pop("batch_size", 32)
    learning_rate = kwargs.pop("learning_rate", 0.001)
    checkpoint_dir = kwargs.pop("checkpoint_dir")
    data_dir = kwargs.pop("data_dir", "dataset")
    num_classes_arg = kwargs.pop("num_classes", None)
    image_size = kwargs.pop("image_size", 64)
    patch_size = kwargs.pop("patch_size", 8)
    embed_dim = kwargs.pop("embed_dim", 256)
    depth = kwargs.pop("depth", 12)
    num_heads = kwargs.pop("num_heads", 8)
    optimizer_name = kwargs.pop("optimizer", "adamw")
    scheduler_name = kwargs.pop("scheduler", "cosine")
    sample_limit = kwargs.pop("sample_limit", None)
    resume_from = kwargs.pop("resume_from", None)
    no_checkpoint = kwargs.pop("no_checkpoint", False)
    keep_last_n = kwargs.pop("keep_last_n", 5)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset_name, train_loader, val_loader, val_split, num_classes = (
            _resolve_dataset_and_loaders(
                dataset_name_arg, data_dir, batch_size, num_classes_arg, sample_limit
            )
        )
        click.echo(f"Training Vision Transformer on dataset: {dataset_name}")
        click.echo(f"Using data directory: {data_dir}")
        click.echo(f"Using device: {device}")
        click.echo("  ✓ Loaded training data")
        if val_split:
            click.echo(f"  ✓ Loaded {val_split} data")

        # Create model
        click.echo("Creating Vision Transformer model...")
        model = KanjiViT(
            num_classes=num_classes,
            image_size=image_size,
            patch_size=patch_size,
            dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_dim=embed_dim * 4,
            input_channels=1,
        ).to(device)

        click.echo(f"  ✓ Model created with {num_classes} classes")

        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )

        # Setup trainer
        trainer = setup_trainer_for_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=str(device),
            model_type="vit",
            checkpoint_dir=checkpoint_dir,
            num_classes=num_classes,
            image_size=image_size,
        )
        _configure_trainer_options(
            model=model,
            trainer=trainer,
            epochs=epochs,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
            resume_from=resume_from,
            no_checkpoint=no_checkpoint,
            keep_last_n=keep_last_n,
        )

        # Train
        click.echo(f"Starting training for {epochs} epochs...")
        trainer.train(num_epochs=epochs)
        click.echo("✓ Training completed successfully!")
    except Exception as e:
        click.echo(f"✗ Training failed: {e}", err=True)
        import traceback

        traceback.print_exc()
        raise


@train.command()
@add_common_options
@add_image_options
@add_checkpoint_dir_option("training/qat/checkpoints")
def qat(**kwargs):
    """Train Quantization-Aware Training (QAT) model for Kanji recognition."""
    import torch

    from src.lib.base_trainer import setup_trainer_for_model
    from src.lib.models import QuantizableLightweightKanjiNet

    # Extract parameters
    dataset_name_arg = kwargs.pop("dataset_name", None)
    epochs = kwargs.pop("epochs", 30)
    batch_size = kwargs.pop("batch_size", 32)
    learning_rate = kwargs.pop("learning_rate", 0.001)
    checkpoint_dir = kwargs.pop("checkpoint_dir")
    data_dir = kwargs.pop("data_dir", "dataset")
    num_classes_arg = kwargs.pop("num_classes", None)
    image_size = kwargs.pop("image_size", 64)
    optimizer_name = kwargs.pop("optimizer", "adamw")
    scheduler_name = kwargs.pop("scheduler", "cosine")
    sample_limit = kwargs.pop("sample_limit", None)
    resume_from = kwargs.pop("resume_from", None)
    no_checkpoint = kwargs.pop("no_checkpoint", False)
    keep_last_n = kwargs.pop("keep_last_n", 5)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset_name, train_loader, val_loader, val_split, num_classes = (
            _resolve_dataset_and_loaders(
                dataset_name_arg, data_dir, batch_size, num_classes_arg, sample_limit
            )
        )
        click.echo(f"Training QAT model on dataset: {dataset_name}")
        click.echo(f"Using data directory: {data_dir}")
        click.echo(f"Using device: {device}")
        click.echo("  ✓ Loaded training data")
        if val_split:
            click.echo(f"  ✓ Loaded {val_split} data")

        # Create model
        click.echo("Creating Quantization-Aware Training model...")
        model = QuantizableLightweightKanjiNet(
            num_classes=num_classes,
            input_channels=1,
            image_size=image_size,
        ).to(device)

        click.echo(f"  ✓ Model created with {num_classes} classes")

        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )

        # Setup trainer
        trainer = setup_trainer_for_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=str(device),
            model_type="qat",
            checkpoint_dir=checkpoint_dir,
            num_classes=num_classes,
            image_size=image_size,
        )
        _configure_trainer_options(
            model=model,
            trainer=trainer,
            epochs=epochs,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
            resume_from=resume_from,
            no_checkpoint=no_checkpoint,
            keep_last_n=keep_last_n,
        )

        # Train with QAT enabled
        click.echo(f"Starting QAT training for {epochs} epochs...")
        trainer.train(num_epochs=epochs, qat_enabled=True)
        click.echo("✓ Training completed successfully!")
    except Exception as e:
        click.echo(f"✗ Training failed: {e}", err=True)
        import traceback

        traceback.print_exc()
        raise


# ============================================================================
# PHASE 1-6: Advanced Training Methods
# ============================================================================


@train.command()
@add_common_options
@add_image_options
@add_checkpoint_dir_option("training/glhpn/checkpoints")
@click.option("--embedding-dim", type=int, default=512, help="Embedding dimension")
@click.option("--top-k", type=int, default=100, help="Top-K candidates for retrieval")
@click.option("--num-ids-tokens", type=int, default=1024, help="Number of IDS tokens")
def glhpn(**kwargs):
    """Train GL-HPN (Global-Local Hierarchical Retrieval) for zero-shot character recognition.

    Implements: Cao et al., May 2026 - Zero-Shot Chinese Character Recognition via Global-Local Dual-Branch
    """
    import torch
    import torch.nn as nn

    from src.lib.base_trainer import CNNTrainer
    from src.lib.hierarchical_retrieval import create_glhpn_retriever
    from src.lib.models import LightweightKanjiNet

    dataset_name_arg = kwargs.pop("dataset_name", None)
    epochs = kwargs.pop("epochs", 30)
    batch_size = kwargs.pop("batch_size", 32)
    learning_rate = kwargs.pop("learning_rate", 0.001)
    checkpoint_dir = kwargs.pop("checkpoint_dir")
    data_dir = kwargs.pop("data_dir", "dataset")
    num_classes_arg = kwargs.pop("num_classes", None)
    image_size = kwargs.pop("image_size", 64)
    embedding_dim = kwargs.pop("embedding_dim", 512)
    top_k = kwargs.pop("top_k", 100)
    sample_limit = kwargs.pop("sample_limit", None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name, train_loader, val_loader, _, num_classes = _resolve_dataset_and_loaders(
        dataset_name_arg, data_dir, batch_size, num_classes_arg, sample_limit
    )

    click.echo(f"Training GL-HPN (embedding_dim={embedding_dim}, top_k={top_k}) on {dataset_name}")

    class _GLHPNClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = LightweightKanjiNet(
                num_classes=embedding_dim, input_channels=1, image_size=image_size
            )
            self.retriever = create_glhpn_retriever(
                backbone_output_dim=embedding_dim,
                embedding_dim=embedding_dim,
                top_k_candidates=top_k,
            )
            self.head = nn.Linear(embedding_dim, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feat = self.backbone(x)
            img_emb, _ = self.retriever.global_encoder(feat)
            return self.head(img_emb)

    model = _GLHPNClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    trainer = CNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=str(device),
        checkpoint_dir=checkpoint_dir,
        model_type="glhpn",
        num_classes=num_classes,
        image_size=image_size,
    )
    trainer.train(num_epochs=epochs)
    click.echo("✓ GL-HPN training completed successfully!")


@train.command()
@add_common_options
@add_image_options
@add_checkpoint_dir_option("training/dtrnet/checkpoints")
@click.option("--text-hidden-dim", type=int, default=1024, help="Text decoder hidden dimension")
@click.option(
    "--radical-hidden-dim", type=int, default=512, help="Radical decoder hidden dimension"
)
@click.option("--use-igca", is_flag=True, help="Enable IDS-Guided Confidence Adjustment")
def dtrnet(**kwargs):
    """Train DTRNet (Dual Text-Radical Decoding) for handwritten character verification.

    Implements: Li et al., August 2026 - Dual Text-Radical Decoding for Handwritten Text
    """
    import torch
    import torch.nn as nn

    from src.lib.base_trainer import CNNTrainer
    from src.lib.dual_decoder import DTRNetModule
    from src.lib.models import LightweightKanjiNet

    dataset_name_arg = kwargs.pop("dataset_name", None)
    epochs = kwargs.pop("epochs", 30)
    batch_size = kwargs.pop("batch_size", 32)
    learning_rate = kwargs.pop("learning_rate", 0.001)
    checkpoint_dir = kwargs.pop("checkpoint_dir")
    data_dir = kwargs.pop("data_dir", "dataset")
    num_classes_arg = kwargs.pop("num_classes", None)
    image_size = kwargs.pop("image_size", 64)
    text_hidden_dim = kwargs.pop("text_hidden_dim", 1024)
    radical_hidden_dim = kwargs.pop("radical_hidden_dim", 512)
    use_igca = kwargs.pop("use_igca", False)
    sample_limit = kwargs.pop("sample_limit", None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name, train_loader, val_loader, _, num_classes = _resolve_dataset_and_loaders(
        dataset_name_arg, data_dir, batch_size, num_classes_arg, sample_limit
    )

    click.echo(f"Training DTRNet (use_igca={use_igca}) on {dataset_name}")

    class _DTRNetClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = LightweightKanjiNet(
                num_classes=512, input_channels=1, image_size=image_size
            )
            self.dtrnet = DTRNetModule(
                input_dim=512,
                text_hidden_dim=text_hidden_dim,
                radical_hidden_dim=radical_hidden_dim,
                num_character_classes=num_classes,
                use_igca=use_igca,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feat = self.backbone(x).unsqueeze(1)
            out = self.dtrnet(feat)
            return out["adjusted_logits"].squeeze(1)

    model = _DTRNetClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    trainer = CNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=str(device),
        checkpoint_dir=checkpoint_dir,
        model_type="dtrnet",
        num_classes=num_classes,
        image_size=image_size,
    )
    trainer.train(num_epochs=epochs)
    click.echo("✓ DTRNet training completed successfully!")


@train.command()
@add_common_options
@add_image_options
@add_checkpoint_dir_option("training/degradation/checkpoints")
@click.option(
    "--degradation-types",
    type=str,
    default="blur,stain,contrast,seal",
    help="Comma-separated degradation types",
)
@click.option("--severity-min", type=float, default=0.2, help="Minimum degradation severity")
@click.option("--severity-max", type=float, default=0.8, help="Maximum degradation severity")
@click.option(
    "--restoration-enabled", is_flag=True, help="Enable document restoration preprocessing"
)
def degradation(**kwargs):
    """Train with Degradation-Aware augmentation for robustness to real-world document degradation.

    Implements: Ju et al., 2025-2026 - Degraded Kuzushiji Documents with Seals (DKDS)
    """
    import torch
    import torch.nn as nn

    from src.lib.base_trainer import CNNTrainer
    from src.lib.degradation import create_degradation_pipeline
    from src.lib.models import LightweightKanjiNet
    from src.lib.restoration import create_restoration_preprocessor

    dataset_name_arg = kwargs.pop("dataset_name", None)
    epochs = kwargs.pop("epochs", 30)
    batch_size = kwargs.pop("batch_size", 32)
    learning_rate = kwargs.pop("learning_rate", 0.001)
    checkpoint_dir = kwargs.pop("checkpoint_dir")
    data_dir = kwargs.pop("data_dir", "dataset")
    num_classes_arg = kwargs.pop("num_classes", None)
    image_size = kwargs.pop("image_size", 64)
    degradation_types_str = kwargs.pop("degradation_types", "blur,stain,contrast,seal")
    severity_min = kwargs.pop("severity_min", 0.2)
    severity_max = kwargs.pop("severity_max", 0.8)
    restoration_enabled = kwargs.pop("restoration_enabled", False)
    sample_limit = kwargs.pop("sample_limit", None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name, train_loader, val_loader, _, num_classes = _resolve_dataset_and_loaders(
        dataset_name_arg, data_dir, batch_size, num_classes_arg, sample_limit
    )

    deg_types = [t.strip() for t in degradation_types_str.split(",") if t.strip()]
    click.echo(f"Training Degradation-Aware model ({deg_types}) on {dataset_name}")

    class _DegradationAwareModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.deg_pipeline = create_degradation_pipeline(
                degradation_types=deg_types,
                severity_range=(severity_min, severity_max),
            )
            self.restoration = create_restoration_preprocessor() if restoration_enabled else None
            self.backbone = LightweightKanjiNet(
                num_classes=num_classes, input_channels=1, image_size=image_size
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.training:
                x = self.deg_pipeline(x)
                if self.restoration is not None:
                    x = self.restoration(x)["restored"]
            return self.backbone(x)

    model = _DegradationAwareModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    trainer = CNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=str(device),
        checkpoint_dir=checkpoint_dir,
        model_type="degradation",
        num_classes=num_classes,
        image_size=image_size,
    )
    trainer.train(num_epochs=epochs)
    click.echo("✓ Degradation-Aware training completed successfully!")


@train.command()
@add_common_options
@add_image_options
@add_checkpoint_dir_option("training/trajectory/checkpoints")
@click.option("--max-strokes", type=int, default=20, help="Maximum number of strokes per character")
@click.option(
    "--trajectory-embedding-dim", type=int, default=128, help="Trajectory embedding dimension"
)
@click.option("--use-hybrid", is_flag=True, help="Enable hybrid image+trajectory fusion")
def trajectory(**kwargs):
    """Train with Online Handwriting Trajectory data for writer-aware modeling.

    Implements: Xu et al., September 2025 - A Stroke-Level Large-Scale Database of Chinese Character Handwriting
    """
    import torch
    import torch.nn as nn

    from src.lib.base_trainer import CNNTrainer
    from src.lib.models import LightweightKanjiNet
    from src.lib.trajectory_processing import HybridTrajectoryVisionModel

    dataset_name_arg = kwargs.pop("dataset_name", None)
    epochs = kwargs.pop("epochs", 30)
    batch_size = kwargs.pop("batch_size", 32)
    learning_rate = kwargs.pop("learning_rate", 0.001)
    checkpoint_dir = kwargs.pop("checkpoint_dir")
    data_dir = kwargs.pop("data_dir", "dataset")
    num_classes_arg = kwargs.pop("num_classes", None)
    image_size = kwargs.pop("image_size", 64)
    traj_dim = kwargs.pop("trajectory_embedding_dim", 128)
    use_hybrid = kwargs.pop("use_hybrid", False)
    sample_limit = kwargs.pop("sample_limit", None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name, train_loader, val_loader, _, num_classes = _resolve_dataset_and_loaders(
        dataset_name_arg, data_dir, batch_size, num_classes_arg, sample_limit
    )

    click.echo(f"Training Trajectory model (use_hybrid={use_hybrid}) on {dataset_name}")

    class _TrajectoryWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = LightweightKanjiNet(
                num_classes=512, input_channels=1, image_size=image_size
            )
            self.traj_proj = nn.Linear(512, traj_dim)
            self.hybrid = HybridTrajectoryVisionModel(
                image_feature_dim=512,
                trajectory_embedding_dim=traj_dim,
                num_classes=num_classes,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            img_feat = self.backbone(x)
            traj_emb = self.traj_proj(img_feat) if use_hybrid else None
            return self.hybrid(image_features=img_feat, trajectory_embedding=traj_emb)["logits"]

    model = _TrajectoryWrapper().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    trainer = CNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=str(device),
        checkpoint_dir=checkpoint_dir,
        model_type="trajectory",
        num_classes=num_classes,
        image_size=image_size,
    )
    trainer.train(num_epochs=epochs)
    click.echo("✓ Trajectory training completed successfully!")


@train.command()
@add_common_options
@add_image_options
@add_checkpoint_dir_option("training/multigranular/checkpoints")
@click.option("--stroke-loss-weight", type=float, default=0.25, help="Weight for stroke-level loss")
@click.option(
    "--radical-loss-weight", type=float, default=0.35, help="Weight for radical-level loss"
)
@click.option(
    "--character-loss-weight", type=float, default=0.40, help="Weight for character-level loss"
)
@click.option(
    "--consistency-weight", type=float, default=0.1, help="Weight for cross-level consistency"
)
def multigranular(**kwargs):
    """Train with Multi-Granular Contrastive Learning for hierarchical character understanding.

    Implements: Zhu et al., May 2025 - Hi-GITA: Hierarchical Multi-Granularity Image-Text Aligning
    """
    import torch
    import torch.nn as nn

    from src.lib.base_trainer import CNNTrainer
    from src.lib.granular_encoders import MultiGranularConfig, create_multigranular_encoders
    from src.lib.models import LightweightKanjiNet

    dataset_name_arg = kwargs.pop("dataset_name", None)
    epochs = kwargs.pop("epochs", 30)
    batch_size = kwargs.pop("batch_size", 32)
    learning_rate = kwargs.pop("learning_rate", 0.001)
    checkpoint_dir = kwargs.pop("checkpoint_dir")
    data_dir = kwargs.pop("data_dir", "dataset")
    num_classes_arg = kwargs.pop("num_classes", None)
    image_size = kwargs.pop("image_size", 64)
    stroke_w = kwargs.pop("stroke_loss_weight", 0.25)
    radical_w = kwargs.pop("radical_loss_weight", 0.35)
    char_w = kwargs.pop("character_loss_weight", 0.40)
    cons_w = kwargs.pop("consistency_weight", 0.1)
    sample_limit = kwargs.pop("sample_limit", None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name, train_loader, val_loader, _, num_classes = _resolve_dataset_and_loaders(
        dataset_name_arg, data_dir, batch_size, num_classes_arg, sample_limit
    )

    click.echo(f"Training Multi-Granular model on {dataset_name}")

    mg_config = MultiGranularConfig(
        num_character_classes=num_classes,
        stroke_loss_weight=stroke_w,
        radical_loss_weight=radical_w,
        character_loss_weight=char_w,
        consistency_loss_weight=cons_w,
    )

    class _MultiGranularClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = LightweightKanjiNet(
                num_classes=512, input_channels=1, image_size=image_size
            )
            self.encoders = nn.ModuleDict(create_multigranular_encoders(mg_config))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feat = self.backbone(x)
            return self.encoders["character"](feat)["logits"]

    model = _MultiGranularClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    trainer = CNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=str(device),
        checkpoint_dir=checkpoint_dir,
        model_type="multigranular",
        num_classes=num_classes,
        image_size=image_size,
    )
    trainer.train(num_epochs=epochs)
    click.echo("✓ Multi-Granular training completed successfully!")


@train.command(name="restoration-pipeline")
@add_common_options
@add_image_options
@add_checkpoint_dir_option("training/restoration_pipeline/checkpoints")
@click.option(
    "--training-strategy",
    type=click.Choice(["end_to_end", "staged", "alternating"]),
    default="end_to_end",
    help="Training strategy",
)
@click.option("--detection-weight", type=float, default=0.2, help="Weight for detection loss")
@click.option("--restoration-weight", type=float, default=0.3, help="Weight for restoration loss")
@click.option(
    "--classification-weight", type=float, default=0.5, help="Weight for classification loss"
)
def restoration_pipeline(**kwargs):
    """Train with End-to-End Restoration Pipeline for degraded document processing.

    Implements: Ju et al., February 2026 - Restoration-Guided Detection-Restoration-Recognition Pipeline
    """
    import torch
    import torch.nn as nn

    from src.lib.base_trainer import CNNTrainer
    from src.lib.models import LightweightKanjiNet
    from src.lib.restoration_pipeline import PipelineConfig, create_pipeline

    dataset_name_arg = kwargs.pop("dataset_name", None)
    epochs = kwargs.pop("epochs", 30)
    batch_size = kwargs.pop("batch_size", 32)
    learning_rate = kwargs.pop("learning_rate", 0.001)
    checkpoint_dir = kwargs.pop("checkpoint_dir")
    data_dir = kwargs.pop("data_dir", "dataset")
    num_classes_arg = kwargs.pop("num_classes", None)
    image_size = kwargs.pop("image_size", 64)
    training_strategy = kwargs.pop("training_strategy", "end_to_end")
    det_w = kwargs.pop("detection_weight", 0.2)
    rest_w = kwargs.pop("restoration_weight", 0.3)
    cls_w = kwargs.pop("classification_weight", 0.5)
    sample_limit = kwargs.pop("sample_limit", None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name, train_loader, val_loader, _, num_classes = _resolve_dataset_and_loaders(
        dataset_name_arg, data_dir, batch_size, num_classes_arg, sample_limit
    )

    click.echo(f"Training Restoration Pipeline (strategy={training_strategy}) on {dataset_name}")

    pipe_config = PipelineConfig(
        training_strategy=training_strategy,
        detection_loss_weight=det_w,
        restoration_loss_weight=rest_w,
        classification_loss_weight=cls_w,
    )

    class _RestorationPipelineClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            backbone = LightweightKanjiNet(num_classes=512, input_channels=3, image_size=image_size)
            self.pipeline, _ = create_pipeline(backbone=backbone, config=pipe_config)
            self.pipeline.classifier.classifier[-1] = nn.Linear(256, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            return self.pipeline(x)["logits"]

    model = _RestorationPipelineClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    trainer = CNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=str(device),
        checkpoint_dir=checkpoint_dir,
        model_type="restoration_pipeline",
        num_classes=num_classes,
        image_size=image_size,
    )
    trainer.train(num_epochs=epochs)
    click.echo("✓ Restoration Pipeline training completed successfully!")


def _call_variant_main(variant_module, click_kwargs):
    """Convert Click options to arguments namespace and call variant's training function.

    Uses a registry pattern to dynamically dispatch to the correct training function,
    making it easy to add new variants without modifying this function.
    """

    # Create a namespace-like object from Click kwargs
    class Args:
        pass

    args = Args()
    for key, value in click_kwargs.items():
        setattr(args, key, value)

    # Registry of training functions by module
    # Maps module to its training function name
    training_functions = {
        "train_cnn_model": "train_cnn",
        "train_hiercode": "train_hiercode",
        "train_rnn": "train_rnn",
        "train_vit": "train_vit",
        "train_qat": "train_qat",
        "train_hiercode_higita": "train_hiercode_higita",
    }

    # Get the module name from the imported module
    module_name = variant_module.__name__.split(".")[-1]

    # Get the function name from the registry
    func_name = training_functions.get(module_name)

    if func_name and hasattr(variant_module, func_name):
        # Call the training function dynamically
        getattr(variant_module, func_name)(args)
    else:
        # Fallback to main() for compatibility
        sys.argv = [sys.argv[0]]
        for key, value in click_kwargs.items():
            if value is None or (isinstance(value, bool) and not value):
                continue
            arg_name = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                sys.argv.append(arg_name)
            else:
                sys.argv.append(arg_name)
                sys.argv.append(str(value))

        variant_module.main()


if __name__ == "__main__":
    train()
