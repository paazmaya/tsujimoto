"""
Advanced ML features: distributed training, experiment tracking, and model registry.

This module integrates:
- Distributed Data Parallel (DDP) training with Lightning
- Experiment tracking with metadata logging
- Model registry for version control
- Hyperparameter optimization utilities

Example Usage:
    >>> from src.lib.training_advanced import DistributedTrainer, ExperimentTracker
    >>> 
    >>> # Setup distributed training
    >>> trainer = DistributedTrainer(model, config, num_gpus=2)
    >>> trainer.train(train_loader, val_loader)
    >>> 
    >>> # Track experiments
    >>> tracker = ExperimentTracker("kanji-recognition")
    >>> tracker.log_params(config.model_dump())
    >>> trainer.train(train_loader, val_loader)
    >>> tracker.log_metrics({"final_accuracy": 0.95})
    >>> tracker.save_model(model, "model.pt")
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

from .lightning_trainer import LightningTrainer
from .logging_utils import setup_logger

logger = setup_logger(__name__)


class DistributedTrainer(LightningTrainer):
    """
    Extended trainer with distributed training (DDP) support.

    Automatically manages multi-GPU training with Lightning's DDP strategy.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Any,
        num_gpus: int = 1,
        num_nodes: int = 1,
        checkpoint_dir: Optional[str] = None,
        model_type: str = "generic",
    ):
        """
        Initialize distributed trainer.

        Args:
            model: PyTorch model
            config: Training configuration
            num_gpus: Number of GPUs per node
            num_nodes: Number of nodes in cluster
            checkpoint_dir: Checkpoint directory
            model_type: Type of model
        """
        super().__init__(model, config, checkpoint_dir, model_type)

        self.num_gpus = num_gpus
        self.num_nodes = num_nodes
        self.distributed = num_gpus > 1 or num_nodes > 1

        logger.info(
            f"DistributedTrainer initialized "
            f"(gpus={num_gpus}, nodes={num_nodes}, distributed={self.distributed})"
        )

    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: Optional[int] = None,
        early_stopping_patience: int = 10,
    ) -> Dict:
        """
        Train with distributed strategy if applicable.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_epochs: Number of epochs
            early_stopping_patience: Early stopping patience

        Returns:
            Training history
        """
        num_epochs = num_epochs or self.config.epochs

        logger.info(
            f"Starting distributed training "
            f"(gpus={self.num_gpus}, nodes={self.num_nodes})"
        )

        # Setup DDP strategy if distributed
        strategy = None
        if self.distributed:
            strategy = DDPStrategy(
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )
            logger.info(f"Using DDPStrategy for distributed training")

        # Create trainer with DDP
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            accelerator="gpu" if self.num_gpus > 0 else "cpu",
            devices=self.num_gpus,
            num_nodes=self.num_nodes,
            strategy=strategy,
            callbacks=[
                pl.callbacks.ModelCheckpoint(
                    dirpath=str(self.checkpoint_dir),
                    filename="epoch_{epoch:03d}-val_acc_{val_accuracy:.3f}",
                    monitor="val_accuracy",
                    mode="max",
                    save_top_k=3,
                ),
                pl.callbacks.EarlyStopping(
                    monitor="val_accuracy",
                    patience=early_stopping_patience,
                    mode="max",
                ),
            ],
            logger=pl.loggers.TensorBoardLogger(
                save_dir=str(self.checkpoint_dir.parent),
                name=self.model_type,
            ),
            enable_progress_bar=True,
            log_every_n_steps=10,
        )

        trainer.fit(
            self.lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

        logger.info("Distributed training complete")
        return self.history


class ExperimentTracker:
    """
    Track experiments with parameters, metrics, and artifacts.

    Provides experiment metadata management and model registry.
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_dir: Optional[Path] = None,
    ):
        """
        Initialize experiment tracker.

        Args:
            experiment_name: Name of experiment
            tracking_dir: Directory for tracking data
        """
        self.experiment_name = experiment_name
        self.tracking_dir = Path(tracking_dir or "experiments") / experiment_name
        self.tracking_dir.mkdir(parents=True, exist_ok=True)

        self.metadata: Dict[str, Any] = {
            "experiment_name": experiment_name,
            "params": {},
            "metrics": {},
            "artifacts": {},
        }

        self.model_dir = self.tracking_dir / "models"
        self.model_dir.mkdir(exist_ok=True)

        logger.info(f"ExperimentTracker initialized: {experiment_name}")

    def log_params(self, params: Dict) -> None:
        """
        Log experiment parameters.

        Args:
            params: Dictionary of parameters
        """
        self.metadata["params"].update(params)
        logger.info(f"Logged {len(params)} parameters")

    def log_metrics(self, metrics: Dict, step: Optional[int] = None) -> None:
        """
        Log experiment metrics.

        Args:
            metrics: Dictionary of metrics
            step: Optional training step
        """
        if step is not None:
            key = f"step_{step}"
        else:
            key = "final"

        self.metadata["metrics"][key] = metrics
        logger.info(f"Logged metrics for {key}: {metrics}")

    def log_artifact(self, artifact_path: Path, artifact_name: Optional[str] = None) -> None:
        """
        Log an artifact (file) to experiment.

        Args:
            artifact_path: Path to artifact file
            artifact_name: Optional name for artifact
        """
        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            logger.warning(f"Artifact not found: {artifact_path}")
            return

        artifact_name = artifact_name or artifact_path.name
        self.metadata["artifacts"][artifact_name] = str(artifact_path)

        logger.info(f"Logged artifact: {artifact_name}")

    def save_model(
        self,
        model: nn.Module,
        model_name: str,
        metadata: Optional[Dict] = None,
    ) -> Path:
        """
        Save model to experiment directory.

        Args:
            model: PyTorch model
            model_name: Name for saved model
            metadata: Optional model metadata

        Returns:
            Path to saved model
        """
        model_path = self.model_dir / model_name
        torch.save(model.state_dict(), model_path)

        # Save metadata if provided
        if metadata:
            meta_path = model_path.parent / f"{model_path.stem}_metadata.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Saved model: {model_path}")
        return model_path

    def load_model(self, model: nn.Module, model_name: str) -> nn.Module:
        """
        Load model from experiment directory.

        Args:
            model: PyTorch model (to load weights into)
            model_name: Name of saved model

        Returns:
            Model with loaded weights
        """
        model_path = self.model_dir / model_name

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

        logger.info(f"Loaded model: {model_path}")
        return model

    def save_metadata(self) -> Path:
        """
        Save experiment metadata to JSON.

        Returns:
            Path to metadata file
        """
        metadata_path = self.tracking_dir / "metadata.json"

        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)

        logger.info(f"Saved experiment metadata: {metadata_path}")
        return metadata_path

    def get_summary(self) -> Dict:
        """
        Get experiment summary.

        Returns:
            Dictionary with experiment summary
        """
        return {
            "experiment_name": self.experiment_name,
            "tracking_dir": str(self.tracking_dir),
            "num_params": len(self.metadata["params"]),
            "num_metrics": sum(
                len(v) if isinstance(v, dict) else 1
                for v in self.metadata["metrics"].values()
            ),
            "num_artifacts": len(self.metadata["artifacts"]),
            "models": list((self.model_dir).glob("*.pt")),
        }


class ModelRegistry:
    """
    Simple model registry for tracking trained models.

    Manages model versions and metadata.
    """

    def __init__(self, registry_dir: Optional[Path] = None):
        """
        Initialize model registry.

        Args:
            registry_dir: Directory for model registry
        """
        self.registry_dir = Path(registry_dir or "model_registry")
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        self.registry_file = self.registry_dir / "registry.json"
        self.registry = self._load_registry()

        logger.info(f"ModelRegistry initialized: {self.registry_dir}")

    def _load_registry(self) -> Dict:
        """Load registry from disk."""
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                return json.load(f)
        return {}

    def register_model(
        self,
        model_name: str,
        model_path: Path,
        version: str,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Register a model in the registry.

        Args:
            model_name: Name of model
            model_path: Path to model file
            version: Version string
            metadata: Optional model metadata

        Returns:
            Registry entry
        """
        if model_name not in self.registry:
            self.registry[model_name] = {"versions": {}}

        entry = {
            "path": str(model_path),
            "version": version,
            "metadata": metadata or {},
        }

        self.registry[model_name]["versions"][version] = entry
        self.registry[model_name]["latest"] = version

        self._save_registry()

        logger.info(f"Registered model: {model_name} v{version}")
        return entry

    def get_model_path(self, model_name: str, version: str = "latest") -> Path:
        """
        Get path to a registered model.

        Args:
            model_name: Name of model
            version: Version (default: latest)

        Returns:
            Path to model file
        """
        if model_name not in self.registry:
            raise ValueError(f"Model not found: {model_name}")

        if version == "latest":
            version = self.registry[model_name]["latest"]

        entry = self.registry[model_name]["versions"].get(version)
        if not entry:
            raise ValueError(f"Version not found: {model_name} v{version}")

        return Path(entry["path"])

    def list_models(self) -> Dict:
        """
        List all registered models.

        Returns:
            Dictionary of model names and versions
        """
        return {
            name: list(data["versions"].keys()) for name, data in self.registry.items()
        }

    def _save_registry(self) -> None:
        """Save registry to disk."""
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2)


def create_distributed_trainer(
    model: nn.Module,
    config: Any,
    num_gpus: int = 1,
    model_type: str = "generic",
) -> DistributedTrainer:
    """
    Factory function for distributed trainer.

    Args:
        model: PyTorch model
        config: Training configuration
        num_gpus: Number of GPUs
        model_type: Type of model

    Returns:
        DistributedTrainer instance
    """
    return DistributedTrainer(
        model, config, num_gpus=num_gpus, model_type=model_type
    )


def create_experiment_tracker(
    experiment_name: str,
    tracking_dir: Optional[Path] = None,
) -> ExperimentTracker:
    """
    Factory function for experiment tracker.

    Args:
        experiment_name: Name of experiment
        tracking_dir: Tracking directory

    Returns:
        ExperimentTracker instance
    """
    return ExperimentTracker(experiment_name, tracking_dir)
