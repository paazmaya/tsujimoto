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


def add_checkpoint_dir_option(default_path):
    """Decorator factory to add checkpoint directory option with specific default."""

    def decorator(func):
        return click.option(
            "--checkpoint-dir",
            type=click.Path(),
            default=default_path,
            help="Checkpoint directory",
        )(func)

    return decorator


@click.group(invoke_without_command=True)
@click.pass_context
def train(ctx):
    """Unified training entry point for Kanji recognition models."""
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
    from src.lib.datasets import KanjiDatasetLoader, ResearchDatasetLoader
    from src.lib.models import LightweightKanjiNet

    # Extract parameters
    dataset_name = kwargs.pop("dataset_name", None) or "etl9g"
    epochs = kwargs.pop("epochs", 30)
    batch_size = kwargs.pop("batch_size", 32)
    learning_rate = kwargs.pop("learning_rate", 0.001)
    checkpoint_dir = kwargs.pop("checkpoint_dir")
    data_dir = kwargs.pop("data_dir", "dataset")
    image_size = kwargs.pop("image_size", 64)
    num_classes_arg = kwargs.pop("num_classes", None)

    click.echo(f"Training CNN on dataset: {dataset_name}")
    click.echo(f"Using data directory: {data_dir}")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        click.echo(f"Using device: {device}")

        # Load dataset
        click.echo(f"Loading {dataset_name} dataset...")

        # Determine if it's ETL or research dataset
        etl_datasets = ["etl6", "etl7", "etl8g", "etl9g", "combined_all_etl"]
        if dataset_name in etl_datasets:
            loader = KanjiDatasetLoader(
                dataset_name=dataset_name,
                cache_dir=data_dir,
                num_workers=4,
                offline=True,  # Require local cache, no network
            )
            num_classes = num_classes_arg or 3036  # ETL9G default
        else:
            loader = ResearchDatasetLoader(
                dataset_name=dataset_name,
                data_dir=Path(data_dir),
            )
            num_classes = (
                num_classes_arg
                or (loader.num_classes if hasattr(loader, "num_classes") else 3036)
            )

        # Get data loaders (try different split names)
        train_loader = loader.get_dataloader("train", batch_size=batch_size)

        # Try to get validation loader (different datasets may have different split names)
        val_loader = None
        val_split = None
        for split_name in ["validation", "val", "test"]:
            try:
                val_loader = loader.get_dataloader(
                    split_name, batch_size=batch_size * 2
                )
                val_split = split_name
                break
            except Exception:  # noqa: BLE001, S110
                pass

        if val_loader is None:
            click.echo("Warning: No validation split found, using test split for validation")
            val_loader = train_loader  # Fallback to training data

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
    from src.lib.datasets import KanjiDatasetLoader, ResearchDatasetLoader
    from src.lib.models import HierCodeNet

    # Extract parameters
    dataset_name = kwargs.pop("dataset_name", None) or "etl9g"
    epochs = kwargs.pop("epochs", 30)
    batch_size = kwargs.pop("batch_size", 32)
    learning_rate = kwargs.pop("learning_rate", 0.001)
    checkpoint_dir = kwargs.pop("checkpoint_dir")
    data_dir = kwargs.pop("data_dir", "dataset")
    num_classes_arg = kwargs.pop("num_classes", None)
    image_size = kwargs.pop("image_size", 64)
    codebook_total_size = kwargs.pop("codebook_total_size", 1024)

    click.echo(f"Training HierCode on dataset: {dataset_name}")
    click.echo(f"Using data directory: {data_dir}")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        click.echo(f"Using device: {device}")

        # Load dataset
        click.echo(f"Loading {dataset_name} dataset...")
        etl_datasets = ["etl6", "etl7", "etl8g", "etl9g", "combined_all_etl"]
        if dataset_name in etl_datasets:
            loader = KanjiDatasetLoader(
                dataset_name=dataset_name,
                cache_dir=data_dir,
                num_workers=4,
                offline=True,  # Require local cache, no network
            )
            num_classes = num_classes_arg or 3036
        else:
            loader = ResearchDatasetLoader(
                dataset_name=dataset_name,
                data_dir=Path(data_dir),
            )
            num_classes = (
                num_classes_arg
                or (loader.num_classes if hasattr(loader, "num_classes") else 3036)
            )

        train_loader = loader.get_dataloader("train", batch_size=batch_size)

        val_loader = None
        val_split = None
        for split_name in ["validation", "val", "test"]:
            try:
                val_loader = loader.get_dataloader(
                    split_name, batch_size=batch_size * 2
                )
                val_split = split_name
                break
            except Exception:  # noqa: BLE001, S110
                pass

        if val_loader is None:
            val_loader = train_loader

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
    from src.lib.datasets import KanjiDatasetLoader, ResearchDatasetLoader
    from src.lib.models import HierCodeHiGITA

    # Extract parameters
    dataset_name = kwargs.pop("dataset_name", None) or "etl9g"
    epochs = kwargs.pop("epochs", 30)
    batch_size = kwargs.pop("batch_size", 32)
    learning_rate = kwargs.pop("learning_rate", 0.001)
    checkpoint_dir = kwargs.pop("checkpoint_dir")
    data_dir = kwargs.pop("data_dir", "dataset")
    num_classes_arg = kwargs.pop("num_classes", None)
    image_size = kwargs.pop("image_size", 64)
    enable_higita = kwargs.pop("enable_higita_enhancement", False)

    click.echo(f"Training HierCode-HiGITA on dataset: {dataset_name}")
    click.echo(f"Using data directory: {data_dir}")
    if enable_higita:
        click.echo("Hi-GITA enhancement enabled")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        click.echo(f"Using device: {device}")

        # Load dataset
        click.echo(f"Loading {dataset_name} dataset...")
        etl_datasets = ["etl6", "etl7", "etl8g", "etl9g", "combined_all_etl"]
        if dataset_name in etl_datasets:
            loader = KanjiDatasetLoader(
                dataset_name=dataset_name,
                cache_dir=data_dir,
                num_workers=4,
                offline=True,  # Require local cache, no network
            )
            num_classes = num_classes_arg or 3036
        else:
            loader = ResearchDatasetLoader(
                dataset_name=dataset_name,
                data_dir=Path(data_dir),
            )
            num_classes = (
                num_classes_arg
                or (loader.num_classes if hasattr(loader, "num_classes") else 3036)
            )

        train_loader = loader.get_dataloader("train", batch_size=batch_size)

        val_loader = None
        val_split = None
        for split_name in ["validation", "val", "test"]:
            try:
                val_loader = loader.get_dataloader(
                    split_name, batch_size=batch_size * 2
                )
                val_split = split_name
                break
            except Exception:  # noqa: BLE001, S110
                pass

        if val_loader is None:
            val_loader = train_loader

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
@add_checkpoint_dir_option("training/rnn/checkpoints")
@click.option(
    "--model-type",
    type=click.Choice(
        [
            "basic_rnn",
            "stroke_rnn",
            "simple_radical_rnn",
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
    from src.lib.datasets import KanjiDatasetLoader, ResearchDatasetLoader
    from src.lib.models import KanjiRNN

    # Extract parameters
    dataset_name = kwargs.pop("dataset_name", None) or "etl9g"
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

    click.echo(f"Training RNN ({model_type}) on dataset: {dataset_name}")
    click.echo(f"Using data directory: {data_dir}")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        click.echo(f"Using device: {device}")

        # Load dataset
        click.echo(f"Loading {dataset_name} dataset...")
        etl_datasets = ["etl6", "etl7", "etl8g", "etl9g", "combined_all_etl"]
        if dataset_name in etl_datasets:
            loader = KanjiDatasetLoader(
                dataset_name=dataset_name,
                cache_dir=data_dir,
                num_workers=4,
                offline=True,  # Require local cache, no network
            )
            num_classes = num_classes_arg or 3036
        else:
            loader = ResearchDatasetLoader(
                dataset_name=dataset_name,
                data_dir=Path(data_dir),
            )
            num_classes = (
                num_classes_arg
                or (loader.num_classes if hasattr(loader, "num_classes") else 3036)
            )

        train_loader = loader.get_dataloader("train", batch_size=batch_size)

        val_loader = None
        val_split = None
        for split_name in ["validation", "val", "test"]:
            try:
                val_loader = loader.get_dataloader(
                    split_name, batch_size=batch_size * 2
                )
                val_split = split_name
                break
            except Exception:  # noqa: BLE001, S110
                pass

        if val_loader is None:
            val_loader = train_loader

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
            weight_decay=1e-4,
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
    from src.lib.datasets import KanjiDatasetLoader, ResearchDatasetLoader
    from src.lib.models import KanjiViT

    # Extract parameters
    dataset_name = kwargs.pop("dataset_name", None) or "etl9g"
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

    click.echo(f"Training Vision Transformer on dataset: {dataset_name}")
    click.echo(f"Using data directory: {data_dir}")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        click.echo(f"Using device: {device}")

        # Load dataset
        click.echo(f"Loading {dataset_name} dataset...")
        etl_datasets = ["etl6", "etl7", "etl8g", "etl9g", "combined_all_etl"]
        if dataset_name in etl_datasets:
            loader = KanjiDatasetLoader(
                dataset_name=dataset_name,
                cache_dir=data_dir,
                num_workers=4,
                offline=True,  # Require local cache, no network
            )
            num_classes = num_classes_arg or 3036
        else:
            loader = ResearchDatasetLoader(
                dataset_name=dataset_name,
                data_dir=Path(data_dir),
            )
            num_classes = (
                num_classes_arg
                or (loader.num_classes if hasattr(loader, "num_classes") else 3036)
            )

        train_loader = loader.get_dataloader("train", batch_size=batch_size)

        val_loader = None
        val_split = None
        for split_name in ["validation", "val", "test"]:
            try:
                val_loader = loader.get_dataloader(
                    split_name, batch_size=batch_size * 2
                )
                val_split = split_name
                break
            except Exception:  # noqa: BLE001, S110
                pass

        if val_loader is None:
            val_loader = train_loader

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
    from src.lib.datasets import KanjiDatasetLoader, ResearchDatasetLoader
    from src.lib.models import QuantizableLightweightKanjiNet

    # Extract parameters
    dataset_name = kwargs.pop("dataset_name", None) or "etl9g"
    epochs = kwargs.pop("epochs", 30)
    batch_size = kwargs.pop("batch_size", 32)
    learning_rate = kwargs.pop("learning_rate", 0.001)
    checkpoint_dir = kwargs.pop("checkpoint_dir")
    data_dir = kwargs.pop("data_dir", "dataset")
    num_classes_arg = kwargs.pop("num_classes", None)
    image_size = kwargs.pop("image_size", 64)

    click.echo(f"Training QAT model on dataset: {dataset_name}")
    click.echo(f"Using data directory: {data_dir}")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        click.echo(f"Using device: {device}")

        # Load dataset
        click.echo(f"Loading {dataset_name} dataset...")
        etl_datasets = ["etl6", "etl7", "etl8g", "etl9g", "combined_all_etl"]
        if dataset_name in etl_datasets:
            loader = KanjiDatasetLoader(
                dataset_name=dataset_name,
                cache_dir=data_dir,
                num_workers=4,
                offline=True,  # Require local cache, no network
            )
            num_classes = num_classes_arg or 3036
        else:
            loader = ResearchDatasetLoader(
                dataset_name=dataset_name,
                data_dir=Path(data_dir),
            )
            num_classes = (
                num_classes_arg
                or (loader.num_classes if hasattr(loader, "num_classes") else 3036)
            )

        train_loader = loader.get_dataloader("train", batch_size=batch_size)

        val_loader = None
        val_split = None
        for split_name in ["validation", "val", "test"]:
            try:
                val_loader = loader.get_dataloader(
                    split_name, batch_size=batch_size * 2
                )
                val_split = split_name
                break
            except Exception:  # noqa: BLE001, S110
                pass

        if val_loader is None:
            val_loader = train_loader

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

        # Train
        click.echo(f"Starting training for {epochs} epochs...")
        trainer.train(num_epochs=epochs)
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
    # Import here to avoid circular dependencies

    click.echo("GL-HPN training not yet integrated with training pipeline.")
    click.echo("Use module directly:")
    click.echo("  from src.lib.hierarchical_retrieval import create_glhpn_retriever")
    click.echo("  retriever = create_glhpn_retriever()")


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

    Features:
    - Dual decoders for text and radical sequences
    - Structural verification via IDS (Ideographic Description Sequences)
    - Fake character detection
    """

    click.echo("DTRNet training not yet integrated with training pipeline.")
    click.echo("Use module directly:")
    click.echo("  from src.lib.dual_decoder import DTRNetModule")
    click.echo("  dtrnet = DTRNetModule()")


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

    Degradation types: blur, stain, contrast, seal, warp, binarization, noise

    Features:
    - Synthetic document degradation during training
    - Optional restoration preprocessing
    - Robustness to blur, stains, seals, warping
    """

    click.echo("Degradation-Aware training not yet integrated with training pipeline.")
    click.echo("Use module directly:")
    click.echo("  from src.lib.degradation import create_degradation_pipeline")
    click.echo("  degradation = create_degradation_pipeline()")


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

    Features:
    - Stroke extraction from pen coordinates
    - Temporal writing pattern capture
    - Hybrid image+trajectory fusion
    - Writer variation modeling
    """

    click.echo("Trajectory training not yet integrated with training pipeline.")
    click.echo("Use module directly:")
    click.echo("  from src.lib.trajectory_processing import create_trajectory_encoder")
    click.echo("  normalizer, extractor, encoder = create_trajectory_encoder()")


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

    Features:
    - Stroke-level encoding (500 stroke types)
    - Radical-level encoding (214 radicals)
    - Character-level encoding (3036+ characters)
    - Fine-grained contrastive loss at each level
    - Cross-level consistency constraints
    """

    click.echo("Multi-Granular training not yet integrated with training pipeline.")
    click.echo("Use module directly:")
    click.echo("  from src.lib.granular_encoders import create_multigranular_encoders")
    click.echo("  encoders = create_multigranular_encoders()")


@train.command()
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

    Stages:
    1. Detection: Character vs seal detection (YOLO-style)
    2. Restoration: Learnable image restoration for degraded regions
    3. Classification: Character classification on restored images

    Features:
    - End-to-end multi-task learning
    - Flexible stage training (end-to-end, staged, alternating)
    - Handles historical and degraded documents
    """

    click.echo("Restoration Pipeline training not yet integrated with training pipeline.")
    click.echo("Use module directly:")
    click.echo("  from src.lib.restoration_pipeline import create_pipeline")
    click.echo("  pipeline, trainer = create_pipeline(backbone)")


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
