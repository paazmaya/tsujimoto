"""
Hugging Face Hub integration utilities.

This module provides a clean interface for uploading, downloading, and
managing models on the Hugging Face Hub.

Features:
- Model uploading with versioning
- Model downloading with caching
- Repository management
- Model card generation
- Token handling

Example Usage:
    >>> from src.lib.hub import HubModelManager
    >>> 
    >>> manager = HubModelManager(repo_id="username/kanji-recognition")
    >>> 
    >>> # Upload a model
    >>> manager.upload_model(
    ...     model_path="checkpoints/best_model.pt",
    ...     model_type="cnn",
    ...     metrics={"accuracy": 0.95}
    ... )
    >>> 
    >>> # Download a model
    >>> model_path = manager.download_model(
    ...     revision="main",
    ...     cache_dir="./models"
    ... )
"""

from pathlib import Path
from typing import Dict, Optional

from huggingface_hub import (
    HfApi,
    create_repo,
    delete_repo,
    model_info,
    upload_file,
)
import torch

from .logging_utils import setup_logger

logger = setup_logger(__name__)


class HubModelManager:
    """
    Manages model uploads, downloads, and metadata on Hugging Face Hub.

    Handles:
    - Model uploading with versioning
    - Repository creation and management
    - Model card generation
    - Token-based authentication
    """

    def __init__(
        self,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False,
    ):
        """
        Initialize the Hub manager.

        Args:
            repo_id: Repository ID (username/repo_name)
            token: Hugging Face token (uses env var if not provided)
            private: Whether repository should be private
        """
        self.repo_id = repo_id
        self.token = token
        self.private = private
        self.api = HfApi()

        logger.info(
            f"HubModelManager initialized for {repo_id} "
            f"(private={private})"
        )

    def create_repo(
        self,
        repo_type: str = "model",
        exist_ok: bool = True,
    ) -> str:
        """
        Create a new repository on the Hub.

        Args:
            repo_type: Type of repo (model, dataset, space)
            exist_ok: If True, don't raise error if repo exists

        Returns:
            Repository URL
        """
        logger.info(f"Creating repository {self.repo_id}")

        try:
            repo_url = create_repo(
                repo_id=self.repo_id,
                repo_type=repo_type,
                private=self.private,
                token=self.token,
                exist_ok=exist_ok,
            )

            logger.info(f"✓ Repository created: {repo_url}")
            return repo_url

        except Exception as e:
            logger.error(f"Failed to create repository: {e}")
            raise

    def upload_model(
        self,
        model_path: Path,
        model_type: str = "pytorch",
        commit_message: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Upload a model file to the Hub.

        Args:
            model_path: Path to model file
            model_type: Type of model (pytorch, onnx, safetensors)
            commit_message: Custom commit message
            metadata: Model metadata (accuracy, loss, etc.)

        Returns:
            Commit URL
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Uploading {model_path.name} to {self.repo_id}")

        commit_msg = commit_message or f"Upload {model_type} model"
        if metadata:
            commit_msg += f" - {metadata}"

        try:
            file_url = upload_file(
                path_or_fileobj=str(model_path),
                path_in_repo=model_path.name,
                repo_id=self.repo_id,
                token=self.token,
                commit_message=commit_msg,
            )

            logger.info(f"✓ Model uploaded: {file_url}")
            return file_url

        except Exception as e:
            logger.error(f"Failed to upload model: {e}")
            raise

    def download_model(
        self,
        filename: str = "pytorch_model.bin",
        revision: str = "main",
        cache_dir: Optional[Path] = None,
    ) -> Path:
        """
        Download a model from the Hub.

        Args:
            filename: Name of model file
            revision: Hub revision (branch, tag, commit)
            cache_dir: Local cache directory

        Returns:
            Path to downloaded model file
        """
        logger.info(
            f"Downloading {filename} from {self.repo_id} "
            f"(revision={revision})"
        )

        try:
            from huggingface_hub import hf_hub_download

            file_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                revision=revision,
                cache_dir=str(cache_dir) if cache_dir else None,
                token=self.token,
            )

            logger.info(f"✓ Model downloaded: {file_path}")
            return Path(file_path)

        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise

    def get_model_info(self) -> Dict:
        """
        Get metadata about the model repository.

        Returns:
            Dictionary with model information
        """
        logger.info(f"Fetching info for {self.repo_id}")

        try:
            info = model_info(self.repo_id, token=self.token)

            metadata = {
                "repo_id": self.repo_id,
                "private": info.private,
                "downloads": info.downloads,
                "last_modified": str(info.last_modified),
                "siblings": len(info.siblings) if info.siblings else 0,
            }

            logger.info(f"✓ Model info fetched")
            return metadata

        except Exception as e:
            logger.error(f"Failed to fetch model info: {e}")
            raise

    def delete_repo(self) -> bool:
        """
        Delete the repository.

        Returns:
            True if successful
        """
        logger.warning(f"Deleting repository {self.repo_id}")

        try:
            delete_repo(
                repo_id=self.repo_id,
                token=self.token,
            )

            logger.info(f"✓ Repository deleted")
            return True

        except Exception as e:
            logger.error(f"Failed to delete repository: {e}")
            raise

    def generate_model_card(
        self,
        model_name: str,
        description: str,
        metrics: Optional[Dict] = None,
    ) -> str:
        """
        Generate a model card (README) for the Hub.

        Args:
            model_name: Name of the model
            description: Model description
            metrics: Model performance metrics

        Returns:
            Generated model card content
        """
        logger.info(f"Generating model card for {model_name}")

        model_card = f"""---
language: en
license: mit
---

# {model_name}

## Description
{description}

## Model Details
- **Repository**: {self.repo_id}
- **Model Type**: Kanji Recognition
- **Task**: Image Classification

## Training Data
- **Dataset**: ETL9G
- **Languages**: Japanese

## Evaluation Results
"""

        if metrics:
            model_card += "\n### Performance Metrics\n"
            for key, value in metrics.items():
                model_card += f"- **{key}**: {value}\n"

        model_card += f"""
## Usage
```python
import torch
from huggingface_hub import hf_hub_download

# Download model
model_path = hf_hub_download(
    repo_id="{self.repo_id}",
    filename="pytorch_model.bin"
)

# Load and use
model = torch.jit.load(model_path)
```

## Model Limitations
- Trained on ETL9G dataset
- Best performance on similar image distributions
- See paper for detailed evaluation

## License
MIT

"""

        logger.info(f"✓ Model card generated")
        return model_card

    def push_model_card(self, content: str):
        """
        Push model card to the Hub.

        Args:
            content: Model card content (markdown)
        """
        logger.info(f"Pushing model card to {self.repo_id}")

        try:
            upload_file(
                path_or_fileobj=content.encode(),
                path_in_repo="README.md",
                repo_id=self.repo_id,
                token=self.token,
                commit_message="Update model card",
            )

            logger.info("✓ Model card pushed")

        except Exception as e:
            logger.error(f"Failed to push model card: {e}")
            raise


def create_hub_manager(
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
) -> HubModelManager:
    """
    Factory function to create a Hub manager.

    Args:
        repo_id: Repository ID (username/repo_name)
        token: Hugging Face token
        private: Whether repository should be private

    Returns:
        HubModelManager instance
    """
    return HubModelManager(
        repo_id=repo_id,
        token=token,
        private=private,
    )
