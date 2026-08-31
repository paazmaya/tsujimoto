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
    _call_variant_main(train_rnn, kwargs)


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
