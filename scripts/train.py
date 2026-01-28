#!/usr/bin/env python3
"""
Unified Training Entry Point for Kanji Recognition Models
==========================================================

Consolidates all training variants into a single entry point with Click subcommands.

Supported variants:
  - cnn         : Lightweight CNN model
  - hiercode    : HierCode (Hierarchical Codebook)
  - hiercode_higita : HierCode with Hi-GITA enhancement
  - rnn         : RNN-based model (with multiple sub-types)
  - radical_rnn : Radical RNN (radical-aware RNN)
  - vit         : Vision Transformer (ViT with T2T)
  - qat         : Quantization-Aware Training

Usage:
    python scripts/train.py cnn --epochs 30 --batch-size 64
    python scripts/train.py hiercode --codebook-total-size 1024
    python scripts/train.py rnn --model-type hybrid_cnn_rnn
    python scripts/train.py vit --epochs 40
    python scripts/train.py qat --epochs 25
    python scripts/train.py radical_rnn --epochs 35
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

from scripts import (
    train_cnn_model,
    train_hiercode,
    train_hiercode_higita,
    train_qat,
    train_radical_rnn,
    train_rnn,
    train_vit,
)
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
    """Train Lightweight CNN model for Kanji recognition."""
    _call_variant_main(train_cnn_model, kwargs)


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
    _call_variant_main(train_hiercode, kwargs)


@train.command()
@add_common_options
@add_checkpoint_dir_option("training/hiercode_higita/checkpoints")
@click.option(
    "--enable-higita-enhancement",
    is_flag=True,
    help="Enable Hi-GITA enhancement",
)
def hiercode_higita(**kwargs):
    """Train HierCode with Hi-GITA enhancement."""
    _call_variant_main(train_hiercode_higita, kwargs)


@train.command()
@add_common_options
@add_checkpoint_dir_option("training/rnn/checkpoints")
@click.option(
    "--model-type",
    type=click.Choice(["basic_rnn", "stroke_rnn", "radical_rnn", "hybrid_cnn_rnn"]),
    default="hybrid_cnn_rnn",
    help="Type of RNN model",
)
@click.option("--weight-decay", type=float, default=1e-4, help="Weight decay")
@click.option("--hidden-size", type=int, default=256, help="RNN hidden size")
@click.option("--num-layers", type=int, default=2, help="Number of RNN layers")
@click.option("--dropout", type=float, default=0.3, help="Dropout rate")
def rnn(**kwargs):
    """Train RNN-based model for Kanji recognition."""
    _call_variant_main(train_rnn, kwargs)


@train.command()
@add_common_options
@add_checkpoint_dir_option("training/radical_rnn/checkpoints")
@click.option("--num-radicals", type=int, default=214, help="Number of radicals")
def radical_rnn(**kwargs):
    """Train Radical RNN model for Kanji recognition."""
    _call_variant_main(train_radical_rnn, kwargs)


@train.command()
@add_common_options
@add_image_options
@add_checkpoint_dir_option("training/vit/checkpoints")
@click.option("--patch-size", type=int, default=4, help="Patch size for ViT")
@click.option("--embed-dim", type=int, default=192, help="Embedding dimension")
@click.option("--depth", type=int, default=12, help="Number of transformer blocks")
@click.option("--num-heads", type=int, default=3, help="Number of attention heads")
def vit(**kwargs):
    """Train Vision Transformer (ViT) model for Kanji recognition."""
    _call_variant_main(train_vit, kwargs)


@train.command()
@add_common_options
@add_image_options
@add_checkpoint_dir_option("training/qat/checkpoints")
def qat(**kwargs):
    """Train Quantization-Aware Training (QAT) model for Kanji recognition."""
    _call_variant_main(train_qat, kwargs)


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
        "train_radical_rnn": "train_radical_rnn",
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
