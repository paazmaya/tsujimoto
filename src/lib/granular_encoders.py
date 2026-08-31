"""
Phase 5: Multi-Granular Contrastive Learning

Implements hierarchical multi-level alignment of visual and textual representations
at stroke, radical, and character levels with contrastive losses.

Based on: Zero-Shot Chinese Character Recognition with Hierarchical Multi-Granularity
Image-Text Aligning (Hi-GITA, Zhu et al., May 2025)

Key contributions:
- Stroke-level: Individual stroke recognition and alignment
- Radical-level: Component/radical recognition and text encoding
- Character-level: Full character recognition with descriptions
- Multi-level contrastive loss: Fine-grained per-level alignment
- Hierarchical composition: Characters from radicals, radicals from strokes
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class MultiGranularConfig:
    """Configuration for multi-granular learning."""

    stroke_embedding_dim: int = 128
    radical_embedding_dim: int = 256
    character_embedding_dim: int = 512
    num_stroke_classes: int = 500  # Number of distinct stroke types
    num_radical_classes: int = 214  # Traditional radical count
    num_character_classes: int = 3036

    # Loss weights
    stroke_loss_weight: float = 0.25
    radical_loss_weight: float = 0.35
    character_loss_weight: float = 0.40
    consistency_loss_weight: float = 0.1

    # Contrastive settings
    temperature: float = 0.07
    use_text_encoder: bool = True


class StrokeEncoder(nn.Module):
    """
    Encodes individual strokes to embeddings.

    Stroke is the basic unit: individual curved segments in handwriting.
    """

    def __init__(
        self,
        input_dim: int = 256,  # From CNN feature extraction
        embedding_dim: int = 128,
        num_classes: int = 500,
    ):
        """
        Initialize stroke encoder.

        Args:
            input_dim: Input feature dimension
            embedding_dim: Output embedding dimension
            num_classes: Number of stroke types to classify
        """
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Feature processing
        self.feature_fc = nn.Sequential(
            nn.Linear(input_dim, embedding_dim * 2),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Embedding layer (normalized)
        self.embedding_fc = nn.Linear(embedding_dim * 2, embedding_dim)

        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        """
        Encode stroke features.

        Args:
            features: (batch, input_dim) stroke features

        Returns:
            Dictionary with:
                - embedding: (batch, embedding_dim) normalized embeddings
                - logits: (batch, num_classes) classification logits
        """
        # Process features
        hidden = self.feature_fc(features)  # (batch, embedding_dim*2)

        # Generate embedding
        embedding = self.embedding_fc(hidden)  # (batch, embedding_dim)
        embedding = F.normalize(embedding, p=2, dim=1)

        # Classification
        logits = self.classifier(embedding)

        return {
            "embedding": embedding,
            "logits": logits,
            "hidden": hidden,
        }


class RadicalEncoder(nn.Module):
    """
    Encodes radical/component sequences to embeddings.

    Radical is a recurring component in characters (e.g., heart radical).
    Composed of multiple strokes.
    """

    def __init__(
        self,
        input_dim: int = 256,  # CNN features
        embedding_dim: int = 256,
        num_classes: int = 214,
        num_strokes_per_radical: int = 8,  # Expected strokes
    ):
        """
        Initialize radical encoder.

        Args:
            input_dim: Input feature dimension
            embedding_dim: Output embedding dimension
            num_classes: Number of radical types
            num_strokes_per_radical: Expected strokes per radical
        """
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_strokes_per_radical = num_strokes_per_radical

        # Process input features
        self.feature_fc = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Attention over strokes (if provided)
        self.stroke_attention = nn.MultiheadAttention(
            embedding_dim,
            num_heads=4,
            batch_first=True,
        )

        # Aggregate strokes to radical
        self.aggregate_fc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
        )

        # Embedding layer
        self.embedding_fc = nn.Linear(embedding_dim, embedding_dim)

        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(
        self,
        features: Tensor,
        stroke_embeddings: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Encode radical features.

        Args:
            features: (batch, input_dim) radical features
            stroke_embeddings: Optional (batch, num_strokes, embedding_dim) stroke embeddings

        Returns:
            Dictionary with:
                - embedding: (batch, embedding_dim) normalized radical embeddings
                - logits: (batch, num_classes) classification logits
        """
        # Process input
        hidden = self.feature_fc(features)  # (batch, embedding_dim)

        # Use stroke attention if available
        if stroke_embeddings is not None and stroke_embeddings.shape[1] > 0:
            # Expand hidden to query shape
            query = hidden.unsqueeze(1)  # (batch, 1, embedding_dim)

            # Attend to strokes
            attn_out, attn_weights = self.stroke_attention(
                query, stroke_embeddings, stroke_embeddings
            )

            # Combine with original hidden
            combined = attn_out.squeeze(1) + hidden  # (batch, embedding_dim)
        else:
            combined = hidden

        # Aggregate to radical embedding
        aggregated = self.aggregate_fc(combined)

        # Generate embedding
        embedding = self.embedding_fc(aggregated)
        embedding = F.normalize(embedding, p=2, dim=1)

        # Classification
        logits = self.classifier(embedding)

        return {
            "embedding": embedding,
            "logits": logits,
            "hidden": aggregated,
        }


class CharacterEncoder(nn.Module):
    """
    Encodes full character images to embeddings.

    Character is composed of one or more radicals.
    """

    def __init__(
        self,
        input_dim: int = 512,  # CNN features
        embedding_dim: int = 512,
        num_classes: int = 3036,
    ):
        """
        Initialize character encoder.

        Args:
            input_dim: Input feature dimension
            embedding_dim: Output embedding dimension
            num_classes: Number of character classes
        """
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Feature processing
        self.feature_fc = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )

        # Embedding layer (normalized)
        self.embedding_fc = nn.Linear(embedding_dim, embedding_dim)

        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        """
        Encode character features.

        Args:
            features: (batch, input_dim) character features

        Returns:
            Dictionary with:
                - embedding: (batch, embedding_dim) normalized character embeddings
                - logits: (batch, num_classes) classification logits
        """
        # Process features
        hidden = self.feature_fc(features)  # (batch, embedding_dim)

        # Generate embedding
        embedding = self.embedding_fc(hidden)
        embedding = F.normalize(embedding, p=2, dim=1)

        # Classification
        logits = self.classifier(embedding)

        return {
            "embedding": embedding,
            "logits": logits,
            "hidden": hidden,
        }


class TextEncoder(nn.Module):
    """
    Encodes textual descriptions at multiple granularities.

    - Stroke descriptions: individual stroke names
    - Radical descriptions: radical names and meanings
    - Character descriptions: full descriptions with usage context
    """

    def __init__(
        self,
        vocab_size: int = 5000,
        embedding_dim: int = 512,
        max_seq_len: int = 128,
    ):
        """
        Initialize text encoder.

        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            max_seq_len: Maximum sequence length
        """
        super().__init__()

        self.embedding_dim = embedding_dim

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, embedding_dim)

        # Positional encoding
        self.register_buffer("pos_encoding", self._create_pos_encoding(max_seq_len, embedding_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Output projection
        self.output_fc = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, token_ids: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Encode text.

        Args:
            token_ids: (batch, seq_len) token indices
            mask: Optional (batch, seq_len) attention mask

        Returns:
            embeddings: (batch, embedding_dim) text embeddings
        """
        # Embed tokens
        x = self.token_embed(token_ids)  # (batch, seq_len, embedding_dim)

        # Add positional encoding
        seq_len = token_ids.shape[1]
        x = x + self.pos_encoding[:seq_len]

        # Transformer encoder
        x = self.transformer(x, src_key_padding_mask=mask)  # (batch, seq_len, embedding_dim)

        # Pool: use mean of all tokens
        if mask is not None:
            x = x * (~mask).unsqueeze(-1).float()
            x = x.sum(dim=1) / (~mask).sum(dim=1, keepdim=True).float()
        else:
            x = x.mean(dim=1)

        # Project
        embeddings = self.output_fc(x)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    @staticmethod
    def _create_pos_encoding(max_len: int, d_model: int) -> Tensor:
        """Create positional encoding."""
        pos = torch.arange(max_len).unsqueeze(1)
        dim_indices = torch.arange(0, d_model, 2)

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos / (10000 ** (dim_indices / d_model)))
        pe[:, 1::2] = torch.cos(pos / (10000 ** (dim_indices / d_model)))

        return pe


class MultiGranularityFusionModule(nn.Module):
    """
    Fuses multi-level representations into unified character representation.

    Ensures consistency across stroke → radical → character hierarchy.
    """

    def __init__(
        self,
        stroke_dim: int = 128,
        radical_dim: int = 256,
        character_dim: int = 512,
    ):
        """
        Initialize fusion module.

        Args:
            stroke_dim: Stroke embedding dimension
            radical_dim: Radical embedding dimension
            character_dim: Character embedding dimension
        """
        super().__init__()

        # Projections to common dimension
        self.stroke_proj = nn.Linear(stroke_dim, character_dim)
        self.radical_proj = nn.Linear(radical_dim, character_dim)
        self.character_proj = nn.Identity()  # Already in character_dim

        # Fusion weights (learnable)
        self.fusion_weights = nn.Parameter(torch.tensor([0.25, 0.35, 0.40]))

    def forward(
        self,
        stroke_emb: Optional[Tensor] = None,
        radical_emb: Optional[Tensor] = None,
        character_emb: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Fuse multi-level embeddings.

        Args:
            stroke_emb: (batch, stroke_dim) or None
            radical_emb: (batch, radical_dim) or None
            character_emb: (batch, character_dim) or None

        Returns:
            fused: (batch, character_dim) fused representation
        """
        components = []
        weights = []

        if stroke_emb is not None:
            components.append(self.stroke_proj(stroke_emb))
            weights.append(self.fusion_weights[0])

        if radical_emb is not None:
            components.append(self.radical_proj(radical_emb))
            weights.append(self.fusion_weights[1])

        if character_emb is not None:
            components.append(self.character_proj(character_emb))
            weights.append(self.fusion_weights[2])

        if not components:
            raise ValueError("At least one embedding must be provided")

        # Normalize weights
        weights = torch.tensor(weights, device=components[0].device)
        weights = weights / weights.sum()

        # Weighted sum
        fused = sum(w * c for w, c in zip(weights, components))
        fused = F.normalize(fused, p=2, dim=1)

        return fused


class FinegrainedDecoupledContrastiveLoss(nn.Module):
    """
    Fine-grained decoupled contrastive loss at each granularity level.

    Each level (stroke, radical, character) has independent contrastive loss,
    with optional alignment between levels.
    """

    def __init__(self, temperature: float = 0.07):
        """
        Initialize loss.

        Args:
            temperature: Temperature for softmax
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        image_emb: Tensor,
        text_emb: Tensor,
        level_name: str = "character",
    ) -> Tensor:
        """
        Compute contrastive loss (NT-Xent).

        Args:
            image_emb: (batch, embedding_dim) image embeddings
            text_emb: (batch, embedding_dim) text embeddings
            level_name: Name of granularity level (for logging)

        Returns:
            loss: Scalar contrastive loss
        """
        # Normalize
        image_emb = F.normalize(image_emb, p=2, dim=1)
        text_emb = F.normalize(text_emb, p=2, dim=1)

        # Concatenate image and text embeddings
        z = torch.cat([image_emb, text_emb], dim=0)  # (2*batch, embedding_dim)

        batch_size = image_emb.shape[0]

        # Similarity matrix
        similarity = torch.matmul(z, z.T) / self.temperature  # (2*batch, 2*batch)

        # Labels: positive pairs are (image[i], text[i])
        labels = torch.arange(batch_size, device=image_emb.device)
        labels = torch.cat([labels, labels], dim=0)

        # Mask out self-similarities and within-modality similarities
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=image_emb.device)
        mask[:batch_size, :batch_size] = True
        mask[batch_size:, batch_size:] = True

        similarity[mask] = -float("inf")

        # Contrastive loss
        targets = torch.cat(
            [
                torch.arange(batch_size, 2 * batch_size, device=image_emb.device),
                torch.arange(batch_size, device=image_emb.device),
            ],
            dim=0,
        )

        loss = F.cross_entropy(similarity, targets)

        return loss


class CrossLevelConsistencyLoss(nn.Module):
    """
    Ensures consistency across hierarchy levels.

    Stroke embeddings should be consistent with radicals they compose,
    and radicals should be consistent with characters they compose.
    """

    def __init__(self):
        """Initialize consistency loss."""
        super().__init__()

    def forward(
        self,
        stroke_emb: Tensor,
        radical_emb: Tensor,
        character_emb: Tensor,
    ) -> Tensor:
        """
        Compute hierarchical consistency loss.

        Args:
            stroke_emb: (batch, stroke_dim) stroke embeddings
            radical_emb: (batch, radical_dim) radical embeddings
            character_emb: (batch, character_dim) character embeddings

        Returns:
            loss: Consistency loss
        """
        # Normalize embeddings
        stroke_emb = F.normalize(stroke_emb, p=2, dim=1)
        radical_emb = F.normalize(radical_emb, p=2, dim=1)
        character_emb = F.normalize(character_emb, p=2, dim=1)

        # Project to common dimension for comparison
        # stroke_proj = F.normalize(F.linear(stroke_emb, torch.eye(stroke_emb.shape[1])), p=2, dim=1)

        # KL divergence between adjacent levels
        # (simplified: use cosine distance as proxy)
        stroke_to_radical_dist = 1 - torch.cosine_similarity(stroke_emb, radical_emb)
        radical_to_character_dist = 1 - torch.cosine_similarity(radical_emb, character_emb)

        loss = stroke_to_radical_dist.mean() + radical_to_character_dist.mean()

        return loss


# ==================== Integration ====================


def create_multigranular_encoders(
    config: MultiGranularConfig = None,
) -> Dict[str, nn.Module]:
    """
    Factory function to create all multi-granular encoders.

    Args:
        config: MultiGranularConfig or None for defaults

    Returns:
        Dictionary of initialized encoder modules
    """
    if config is None:
        config = MultiGranularConfig()

    encoders = {
        "stroke": StrokeEncoder(
            embedding_dim=config.stroke_embedding_dim,
            num_classes=config.num_stroke_classes,
        ),
        "radical": RadicalEncoder(
            embedding_dim=config.radical_embedding_dim,
            num_classes=config.num_radical_classes,
        ),
        "character": CharacterEncoder(
            embedding_dim=config.character_embedding_dim,
            num_classes=config.num_character_classes,
        ),
        "text": TextEncoder(
            embedding_dim=config.character_embedding_dim,
        ),
        "fusion": MultiGranularityFusionModule(
            stroke_dim=config.stroke_embedding_dim,
            radical_dim=config.radical_embedding_dim,
            character_dim=config.character_embedding_dim,
        ),
        "contrastive_loss": FinegrainedDecoupledContrastiveLoss(),
        "consistency_loss": CrossLevelConsistencyLoss(),
    }

    return encoders
