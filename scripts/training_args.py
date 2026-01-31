#!/usr/bin/env python3
"""
Centralized Training Arguments Configuration
=============================================

Single source of truth for all training arguments. Used by:
1. Click CLI decorators in train.py
2. Argparse parsers in each train_*.py script

This eliminates duplication and ensures consistency across all variants.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ArgumentDefinition:
    """Define a single argument with all its properties."""

    name: str  # Argument name (e.g., "epochs")
    arg_type: type  # Python type (int, float, str, bool)
    default: Optional[object] = None  # Default value
    help_text: str = ""  # Help text
    choices: Optional[list] = None  # For choice arguments
    is_flag: bool = False  # For boolean flags


# ========== COMMON ARGUMENTS (All Variants) ==========
COMMON_ARGS = {
    "data_dir": ArgumentDefinition(
        name="data_dir",
        arg_type=str,
        default="dataset",
        help_text="Dataset directory (auto-detects best available if using default)",
    ),
    "epochs": ArgumentDefinition(
        name="epochs",
        arg_type=int,
        default=30,
        help_text="Number of training epochs",
    ),
    "batch_size": ArgumentDefinition(
        name="batch_size",
        arg_type=int,
        default=32,
        help_text="Batch size",
    ),
    "learning_rate": ArgumentDefinition(
        name="learning_rate",
        arg_type=float,
        default=0.001,
        help_text="Learning rate",
    ),
    "optimizer": ArgumentDefinition(
        name="optimizer",
        arg_type=str,
        default="adamw",
        choices=["adamw", "sgd"],
        help_text="Optimizer",
    ),
    "scheduler": ArgumentDefinition(
        name="scheduler",
        arg_type=str,
        default="cosine",
        choices=["cosine", "step"],
        help_text="Learning rate scheduler",
    ),
    "sample_limit": ArgumentDefinition(
        name="sample_limit",
        arg_type=int,
        default=None,
        help_text="Limit samples for testing",
    ),
    "resume_from": ArgumentDefinition(
        name="resume_from",
        arg_type=str,
        default=None,
        help_text="Resume from checkpoint path",
    ),
    "no_checkpoint": ArgumentDefinition(
        name="no_checkpoint",
        arg_type=bool,
        default=False,
        is_flag=True,
        help_text="Start fresh training (ignore existing checkpoints)",
    ),
    "keep_last_n": ArgumentDefinition(
        name="keep_last_n",
        arg_type=int,
        default=5,
        help_text="Keep last N checkpoints",
    ),
}

# ========== IMAGE ARGUMENTS (CNN, HierCode, ViT, QAT) ==========
IMAGE_ARGS = {
    "image_size": ArgumentDefinition(
        name="image_size",
        arg_type=int,
        default=64,
        help_text="Input image size",
    ),
    "num_classes": ArgumentDefinition(
        name="num_classes",
        arg_type=int,
        default=43427,
        help_text="Number of character classes",
    ),
}

# ========== CHECKPOINT ARGUMENT (All Variants) ==========
CHECKPOINT_ARGS = {
    "checkpoint_dir": ArgumentDefinition(
        name="checkpoint_dir",
        arg_type=str,
        help_text="Checkpoint directory",
    ),
}

# ========== VARIANT-SPECIFIC ARGUMENTS ==========

RNN_ARGS = {
    "model_type": ArgumentDefinition(
        name="model_type",
        arg_type=str,
        default="hybrid_cnn_rnn",
        choices=["basic_rnn", "stroke_rnn", "radical_rnn", "hybrid_cnn_rnn"],
        help_text="Type of RNN model",
    ),
    "weight_decay": ArgumentDefinition(
        name="weight_decay",
        arg_type=float,
        default=1e-4,
        help_text="Weight decay",
    ),
    "hidden_size": ArgumentDefinition(
        name="hidden_size",
        arg_type=int,
        default=256,
        help_text="RNN hidden size",
    ),
    "num_layers": ArgumentDefinition(
        name="num_layers",
        arg_type=int,
        default=2,
        help_text="Number of RNN layers",
    ),
    "dropout": ArgumentDefinition(
        name="dropout",
        arg_type=float,
        default=0.3,
        help_text="Dropout rate",
    ),
}

HIERCODE_ARGS = {
    "codebook_total_size": ArgumentDefinition(
        name="codebook_total_size",
        arg_type=int,
        default=1024,
        help_text="Total codebook size",
    ),
    "codebook_dim": ArgumentDefinition(
        name="codebook_dim",
        arg_type=int,
        default=128,
        help_text="Codebook dimension",
    ),
    "hierarch_depth": ArgumentDefinition(
        name="hierarch_depth",
        arg_type=int,
        default=10,
        help_text="Hierarchical tree depth",
    ),
    "multi_hot_k": ArgumentDefinition(
        name="multi_hot_k",
        arg_type=int,
        default=5,
        help_text="Number of active codewords",
    ),
    "temperature": ArgumentDefinition(
        name="temperature",
        arg_type=float,
        default=0.1,
        help_text="Gumbel-softmax temperature",
    ),
}

HIERCODE_HIGITA_ARGS = {
    "enable_higita_enhancement": ArgumentDefinition(
        name="enable_higita_enhancement",
        arg_type=bool,
        default=False,
        is_flag=True,
        help_text="Enable Hi-GITA enhancement",
    ),
}

RADICAL_RNN_ARGS = {
    "num_radicals": ArgumentDefinition(
        name="num_radicals",
        arg_type=int,
        default=214,
        help_text="Number of radicals",
    ),
}

VIT_ARGS = {
    "patch_size": ArgumentDefinition(
        name="patch_size",
        arg_type=int,
        default=4,
        help_text="Patch size for ViT",
    ),
    "embed_dim": ArgumentDefinition(
        name="embed_dim",
        arg_type=int,
        default=192,
        help_text="Embedding dimension",
    ),
    "depth": ArgumentDefinition(
        name="depth",
        arg_type=int,
        default=12,
        help_text="Number of transformer blocks",
    ),
    "num_heads": ArgumentDefinition(
        name="num_heads",
        arg_type=int,
        default=3,
        help_text="Number of attention heads",
    ),
}

# ========== VARIANT ARGUMENT SETS ==========

VARIANT_ARGS = {
    "cnn": {
        "common": COMMON_ARGS,
        "image": IMAGE_ARGS,
        "checkpoint": CHECKPOINT_ARGS,
        "specific": {},
    },
    "hiercode": {
        "common": COMMON_ARGS,
        "image": IMAGE_ARGS,
        "checkpoint": CHECKPOINT_ARGS,
        "specific": HIERCODE_ARGS,
    },
    "hiercode_higita": {
        "common": COMMON_ARGS,
        "checkpoint": CHECKPOINT_ARGS,
        "specific": HIERCODE_HIGITA_ARGS,
    },
    "rnn": {
        "common": COMMON_ARGS,
        "checkpoint": CHECKPOINT_ARGS,
        "specific": RNN_ARGS,
    },
    "radical_rnn": {
        "common": COMMON_ARGS,
        "checkpoint": CHECKPOINT_ARGS,
        "specific": RADICAL_RNN_ARGS,
    },
    "vit": {
        "common": COMMON_ARGS,
        "image": IMAGE_ARGS,
        "checkpoint": CHECKPOINT_ARGS,
        "specific": VIT_ARGS,
    },
    "qat": {
        "common": COMMON_ARGS,
        "image": IMAGE_ARGS,
        "checkpoint": CHECKPOINT_ARGS,
        "specific": {},
    },
}


def get_variant_args(variant_name: str) -> dict:
    """Get all arguments for a specific variant.

    Args:
        variant_name: Name of the variant (e.g., "cnn", "rnn")

    Returns:
        Dictionary of all arguments for the variant
    """
    if variant_name not in VARIANT_ARGS:
        raise ValueError(f"Unknown variant: {variant_name}")

    variant = VARIANT_ARGS[variant_name]
    all_args = {}

    # Merge all argument categories
    for category in ["common", "image", "checkpoint", "specific"]:
        if category in variant:
            all_args.update(variant[category])

    return all_args


def add_variant_args_to_parser(parser, variant_name: str, checkpoint_dir_default: str) -> None:
    """Add all arguments for a variant to an argparse parser.

    Args:
        parser: argparse.ArgumentParser instance
        variant_name: Name of the variant
        checkpoint_dir_default: Default checkpoint directory path
    """
    args = get_variant_args(variant_name)

    for arg_def in args.values():
        # Skip checkpoint-dir - we'll add it with the variant-specific default
        if arg_def.name == "checkpoint_dir":
            continue

        # Build argument name for argparse
        arg_name = f"--{arg_def.name.replace('_', '-')}"

        if arg_def.is_flag:
            parser.add_argument(arg_name, action="store_true", help=arg_def.help_text)
        elif arg_def.choices:
            parser.add_argument(
                arg_name,
                type=arg_def.arg_type,
                default=arg_def.default,
                choices=arg_def.choices,
                help=arg_def.help_text,
            )
        else:
            parser.add_argument(
                arg_name,
                type=arg_def.arg_type,
                default=arg_def.default,
                help=arg_def.help_text,
            )

    # Add checkpoint-dir with variant-specific default
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=checkpoint_dir_default,
        help="Checkpoint directory",
    )
