"""
Hi-GITA Enhancement for HierCode
=================================

Implements learnings from "Zero-Shot Chinese Character Recognition with
Hierarchical Multi-Granularity Image-Text Aligning" (arXiv:2505.24837v1, May 2025).

Hi-GITA extends HierCode's radical-aware approach with:
1. Multi-granularity image encoders (strokes → radicals → character)
2. Multi-granularity text encoders (stroke/radical sequences)
3. Contrastive image-text alignment
4. Fine-grained decoupled fusion modules

Key Innovation: Hierarchical representation at THREE levels instead of TWO:
  - Level 0: Strokes (individual brush components)
  - Level 1: Radicals (character building blocks)
  - Level 2: Full character (holistic semantic)

Each level is processed independently then fused hierarchically.

Author: Enhancement for tsujimoto
Based on: Hi-GITA paper (2505.24837v1)
Date: November 17, 2025
"""

import math
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

# Add parent directory to path to import src/lib
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib import setup_logger

logger = setup_logger(__name__)


class StrokeEncoder(nn.Module):
    """
    Stroke-level image encoder: Processes localized stroke patterns.

    Hi-GITA Innovation: Extracts fine-grained stroke features from character images.
    Instead of processing full 64x64 image, extracts 8x8 patches (64 patches total),
    treating each as a potential stroke component.

    Architecture:
    - Lightweight CNN for local stroke recognition
    - Attention-weighted patch aggregation
    - Output: stroke-level embeddings (Ns x D_s)
    """

    def __init__(self, input_channels: int = 1, output_dim: int = 128, num_patches: int = 64):
        super().__init__()
        logger.debug(f"🔧 Initializing StrokeEncoder (patches={num_patches}, dim={output_dim})")
        self.output_dim = output_dim
        self.num_patches = num_patches
        self.patch_size = int(math.sqrt(64 // num_patches)) * 8  # ~8x8 for 64x64

        # Lightweight stroke feature extractor
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Patch-wise attention
        self.patch_attention = nn.Sequential(
            nn.Linear(32 * 8 * 8, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

        # Project to embedding dimension
        self.fc = nn.Linear(32 * 8 * 8, output_dim)

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image: (B, 1, 64, 64) grayscale character image

        Returns:
            stroke_features: (B, Ns, D_s) stroke-level embeddings
            stroke_attention: (B, Ns) attention weights for strokes
        """
        logger.debug("→ StrokeEncoder forward pass")
        # Extract patches (strokes)
        patches = F.unfold(image, kernel_size=8, stride=8)  # (B, 64, 64)
        patches = patches.permute(0, 2, 1)  # (B, 64, 64)

        # Process through CNN
        image_features = self.conv1(image)
        image_features = F.relu(image_features)
        image_features = self.conv2(image_features)
        image_features = F.relu(image_features)  # (B, 32, 64, 64)

        # Extract patch-wise features
        patch_features = F.unfold(image_features, kernel_size=8, stride=8)  # (B, 2048, 64)
        patch_features = patch_features.permute(0, 2, 1)  # (B, 64, 2048)

        # Compute attention weights
        stroke_attention = self.patch_attention(patch_features)  # (B, 64, 1)
        stroke_attention = stroke_attention.squeeze(-1)  # (B, 64)

        # Project to embedding dimension
        stroke_features = self.fc(patch_features)  # (B, 64, D_s)

        return stroke_features, stroke_attention


class RadicalEncoder(nn.Module):
    """
    Radical-level image encoder: Processes character radical components.

    Hi-GITA Innovation: Uses hierarchical decomposition to identify radical regions.
    Combines strokes into higher-level radical components through learned grouping.

    Architecture:
    - Aggregates stroke features into radical features
    - Learns radical-specific attention masks
    - Output: radical-level embeddings (Nr x D_r)
    """

    def __init__(self, stroke_dim: int = 128, radical_dim: int = 256, num_radicals: int = 16):
        super().__init__()
        logger.debug(f"🔧 Initializing RadicalEncoder (radicals={num_radicals}, dim={radical_dim})")
        self.stroke_dim = stroke_dim
        self.radical_dim = radical_dim
        self.num_radicals = num_radicals

        # Stroke-to-radical aggregation
        self.stroke_to_radical = nn.Linear(stroke_dim, radical_dim)

        # Radical attention mechanism
        self.radical_attention = nn.Sequential(
            nn.Linear(radical_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

        # Radical grouping (learnable assignment)
        self.stroke_radical_assignment = nn.Parameter(
            torch.randn(64, num_radicals) / math.sqrt(num_radicals)
        )

    def forward(
        self, stroke_features: torch.Tensor, stroke_attention: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            stroke_features: (B, Ns, D_s) stroke-level embeddings
            stroke_attention: (B, Ns) stroke attention weights

        Returns:
            radical_features: (B, Nr, D_r) radical-level embeddings
            radical_attention: (B, Nr) radical attention weights
        """
        logger.debug("→ RadicalEncoder forward pass")
        # Soft assignment of strokes to radicals
        assignment_weights = F.softmax(self.stroke_radical_assignment, dim=0)  # (Ns, Nr)

        # Aggregate strokes into radicals (weighted by stroke attention)
        stroke_attention_expanded = stroke_attention.unsqueeze(-1)  # (B, Ns, 1)
        weighted_strokes = stroke_features * stroke_attention_expanded  # (B, Ns, D_s)

        # Project to radical dimension
        radical_candidates = self.stroke_to_radical(weighted_strokes)  # (B, Ns, D_r)

        # Aggregate to radicals: (B, Ns, D_r) @ (Ns, Nr) -> (B, Nr, D_r)
        radical_features = torch.matmul(
            assignment_weights.t().unsqueeze(0),  # (1, Nr, Ns)
            radical_candidates,  # (B, Ns, D_r)
        )  # (B, Nr, D_r)

        # Compute radical attention
        radical_attention = self.radical_attention(radical_features)  # (B, Nr, 1)
        radical_attention = radical_attention.squeeze(-1)  # (B, Nr)

        return radical_features, radical_attention


class CharacterEncoder(nn.Module):
    """
    Character-level image encoder: Processes full character holistically.

    Hi-GITA Innovation: Top-level aggregation of all radicals into character embedding.
    Combines radical information with global context.

    Architecture:
    - Global average pooling + attention
    - Radical fusion
    - Output: character-level embedding (1 x D_c)
    """

    def __init__(self, radical_dim: int = 256, character_dim: int = 512):
        super().__init__()
        logger.debug(f"🔧 Initializing CharacterEncoder (dim={character_dim})")
        self.radical_dim = radical_dim
        self.character_dim = character_dim

        # Global character encoder
        self.global_fc = nn.Sequential(
            nn.Linear(radical_dim, character_dim // 2),
            nn.ReLU(),
            nn.Linear(character_dim // 2, character_dim),
        )

        # Radical-to-character attention
        self.radical_to_character_attention = nn.Sequential(
            nn.Linear(radical_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(
        self, radical_features: torch.Tensor, radical_attention: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            radical_features: (B, Nr, D_r) radical-level embeddings
            radical_attention: (B, Nr) radical attention weights

        Returns:
            character_features: (B, 1, D_c) character-level embedding
            fused_attention: (B, Nr) fused attention map
        """
        logger.debug("→ CharacterEncoder forward pass")
        # Compute attention weights for radical aggregation
        radical_weights = self.radical_to_character_attention(radical_features)  # (B, Nr, 1)
        radical_weights = radical_weights.squeeze(-1)  # (B, Nr)

        # Combine with radical attention
        fused_attention = radical_attention * radical_weights  # (B, Nr)
        fused_attention = fused_attention / (fused_attention.sum(dim=1, keepdim=True) + 1e-8)

        # Weighted aggregation of radicals
        character_features = torch.sum(
            radical_features * fused_attention.unsqueeze(-1), dim=1, keepdim=True
        )  # (B, 1, D_r)

        # Project to character dimension
        character_features = self.global_fc(character_features)  # (B, 1, D_c)

        return character_features, fused_attention


class MultiGranularityImageEncoder(nn.Module):
    """
    Multi-granularity Image Encoder: Three-level hierarchy

    Hi-GITA Core Component: Processes character images at THREE levels:
    1. Stroke level (fine-grained local features)
    2. Radical level (intermediate semantic units)
    3. Character level (holistic semantic)

    Each level is processed independently then fused hierarchically.
    This enables multi-level contrastive learning.
    """

    def __init__(
        self,
        stroke_dim: int = 128,
        radical_dim: int = 256,
        character_dim: int = 512,
        num_radicals: int = 16,
    ):
        super().__init__()
        logger.info("🏗️  Building MultiGranularityImageEncoder")
        self.stroke_encoder = StrokeEncoder(output_dim=stroke_dim)
        self.radical_encoder = RadicalEncoder(stroke_dim, radical_dim, num_radicals)
        self.character_encoder = CharacterEncoder(radical_dim, character_dim)

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            image: (B, 1, 64, 64) character image

        Returns:
            dict with keys:
            - 'stroke': (B, 64, stroke_dim)
            - 'stroke_attention': (B, 64)
            - 'radical': (B, num_radicals, radical_dim)
            - 'radical_attention': (B, num_radicals)
            - 'character': (B, 1, character_dim)
            - 'character_attention': (B, num_radicals) - fused radical attention
        """
        logger.debug("→ MultiGranularityImageEncoder forward pass (3-level hierarchy)")
        # Level 0: Stroke encoding
        stroke_features, stroke_attention = self.stroke_encoder(image)

        # Level 1: Radical encoding
        radical_features, radical_attention = self.radical_encoder(
            stroke_features, stroke_attention
        )

        # Level 2: Character encoding
        character_features, character_attention = self.character_encoder(
            radical_features, radical_attention
        )

        logger.debug(
            f"   ✓ Stroke: {stroke_features.shape}, Radical: {radical_features.shape}, Character: {character_features.shape}"
        )

        return {
            "stroke": stroke_features,
            "stroke_attention": stroke_attention,
            "radical": radical_features,
            "radical_attention": radical_attention,
            "character": character_features,
            "character_attention": character_attention,
        }


class TextStrokeEncoder(nn.Module):
    """
    Text Stroke Encoder: Encodes stroke stroke sequences from text representation.

    Hi-GITA Text Component: Converts structured text (stroke codes) into embeddings.
    Each stroke is represented as a one-hot encoded vector indicating stroke type.
    """

    def __init__(self, num_strokes: int = 20, stroke_dim: int = 128):
        super().__init__()
        logger.debug(f"🔧 Initializing TextStrokeEncoder (strokes={num_strokes}, dim={stroke_dim})")
        self.stroke_embedding = nn.Embedding(num_strokes, stroke_dim)
        self.stroke_encoder_rnn = nn.GRU(
            input_size=stroke_dim, hidden_size=stroke_dim, batch_first=True, bidirectional=True
        )
        self.stroke_projection = nn.Linear(stroke_dim * 2, stroke_dim)

    def forward(self, stroke_codes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            stroke_codes: (B, max_strokes) stroke indices

        Returns:
            stroke_embeddings: (B, max_strokes, stroke_dim)
            stroke_importance: (B, max_strokes) importance weights
        """
        # Embed strokes
        embeddings = self.stroke_embedding(stroke_codes)  # (B, max_strokes, stroke_dim)

        # Encode with GRU
        encoded, _ = self.stroke_encoder_rnn(embeddings)  # (B, max_strokes, 2*stroke_dim)

        # Project back to stroke dimension
        projected = self.stroke_projection(encoded)  # (B, max_strokes, stroke_dim)

        # Compute importance via attention
        attention_logits = torch.sum(projected, dim=-1)  # (B, max_strokes)
        importance = torch.sigmoid(attention_logits)  # (B, max_strokes)

        return projected, importance


class TextRadicalEncoder(nn.Module):
    """
    Text Radical Encoder: Encodes radical sequences from text representation.

    Hi-GITA Text Component: Converts radical codes into embeddings.
    Represents higher-level semantic units (radicals).
    """

    def __init__(self, num_radicals: int = 214, radical_dim: int = 256):
        super().__init__()
        logger.debug(
            f"🔧 Initializing TextRadicalEncoder (radicals={num_radicals}, dim={radical_dim})"
        )
        self.radical_embedding = nn.Embedding(num_radicals, radical_dim)
        self.radical_encoder_rnn = nn.GRU(
            input_size=radical_dim, hidden_size=radical_dim, batch_first=True, bidirectional=True
        )
        self.radical_projection = nn.Linear(radical_dim * 2, radical_dim)

    def forward(self, radical_codes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            radical_codes: (B, num_radicals_in_char) radical indices

        Returns:
            radical_embeddings: (B, num_radicals_in_char, radical_dim)
            radical_importance: (B, num_radicals_in_char)
        """
        embeddings = self.radical_embedding(radical_codes)
        encoded, _ = self.radical_encoder_rnn(embeddings)
        projected = self.radical_projection(encoded)
        importance = torch.sigmoid(torch.sum(projected, dim=-1))

        return projected, importance


class TextCharacterEncoder(nn.Module):
    """
    Text Character Encoder: Holistic character representation from text.

    Hi-GITA Text Component: Final character-level semantic encoding.
    Combines all radical information into final character embedding.
    """

    def __init__(self, radical_dim: int = 256, character_dim: int = 512):
        super().__init__()
        logger.debug(f"🔧 Initializing TextCharacterEncoder (dim={character_dim})")
        self.character_fc = nn.Sequential(
            nn.Linear(radical_dim, character_dim // 2),
            nn.ReLU(),
            nn.Linear(character_dim // 2, character_dim),
        )

    def forward(
        self, radical_features: torch.Tensor, radical_importance: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            radical_features: (B, num_radicals, radical_dim)
            radical_importance: (B, num_radicals)

        Returns:
            character_embedding: (B, character_dim)
        """
        # Weighted average of radicals
        character_representation = torch.sum(
            radical_features * radical_importance.unsqueeze(-1), dim=1
        )

        return self.character_fc(character_representation)


class MultiGranularityTextEncoder(nn.Module):
    """
    Multi-granularity Text Encoder: Three-level hierarchy for text input

    Hi-GITA Core Component: Processes character descriptions at THREE levels:
    1. Stroke level (fine-grained stroke sequence)
    2. Radical level (intermediate radical sequence)
    3. Character level (holistic semantic)
    """

    def __init__(
        self,
        num_strokes: int = 20,
        num_radicals: int = 214,
        stroke_dim: int = 128,
        radical_dim: int = 256,
        character_dim: int = 512,
    ):
        super().__init__()
        logger.info("🏗️  Building MultiGranularityTextEncoder")
        self.stroke_encoder = TextStrokeEncoder(num_strokes, stroke_dim)
        self.radical_encoder = TextRadicalEncoder(num_radicals, radical_dim)
        self.character_encoder = TextCharacterEncoder(radical_dim, character_dim)

    def forward(
        self, stroke_codes: torch.Tensor, radical_codes: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            stroke_codes: (B, max_strokes) stroke sequence
            radical_codes: (B, num_radicals) radical sequence

        Returns:
            dict with keys:
            - 'stroke': (B, max_strokes, stroke_dim)
            - 'stroke_importance': (B, max_strokes)
            - 'radical': (B, num_radicals, radical_dim)
            - 'radical_importance': (B, num_radicals)
            - 'character': (B, character_dim)
        """
        logger.debug("→ MultiGranularityTextEncoder forward pass (3-level hierarchy)")
        stroke_features, stroke_importance = self.stroke_encoder(stroke_codes)
        radical_features, radical_importance = self.radical_encoder(radical_codes)
        character_features = self.character_encoder(radical_features, radical_importance)

        return {
            "stroke": stroke_features,
            "stroke_importance": stroke_importance,
            "radical": radical_features,
            "radical_importance": radical_importance,
            "character": character_features,
        }


class FineGrainedContrastiveLoss(nn.Module):
    """
    Fine-Grained Decoupled Contrastive Loss

    Hi-GITA Innovation: Contrastive loss at MULTIPLE granularity levels:
    1. Stroke level: Align stroke features from image and text
    2. Radical level: Align radical features from image and text
    3. Character level: Align character features from image and text

    Each level has independent contrastive learning objective.
    This enables learning at multiple semantic granularities simultaneously.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        weight_stroke: float = 0.3,
        weight_radical: float = 0.5,
        weight_character: float = 0.2,
    ):
        super().__init__()
        logger.info("🏗️  Building FineGrainedContrastiveLoss")
        logger.debug(
            f"   Weights - Stroke: {weight_stroke}, Radical: {weight_radical}, Character: {weight_character}"
        )
        self.temperature = temperature
        self.weight_stroke = weight_stroke
        self.weight_radical = weight_radical
        self.weight_character = weight_character

    def contrastive_loss(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss between image and text features.

        Args:
            image_features: (B, D) image embeddings
            text_features: (B, D) text embeddings

        Returns:
            loss: scalar contrastive loss
        """
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Cosine similarity matrix
        logits = torch.mm(image_features, text_features.t()) / self.temperature  # (B, B)

        # Labels: diagonal elements are positives (i-th image with i-th text)
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)

        # Cross-entropy loss
        loss_img_to_text = F.cross_entropy(logits, labels)
        loss_text_to_img = F.cross_entropy(logits.t(), labels)

        return (loss_img_to_text + loss_text_to_img) / 2

    def forward(
        self, image_outputs: Dict[str, torch.Tensor], text_outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            image_outputs: dict from MultiGranularityImageEncoder
            text_outputs: dict from MultiGranularityTextEncoder

        Returns:
            dict with keys:
            - 'stroke_loss': contrastive loss at stroke level
            - 'radical_loss': contrastive loss at radical level
            - 'character_loss': contrastive loss at character level
            - 'total_loss': weighted sum of all losses
        """
        logger.debug("→ Computing fine-grained contrastive losses")
        # Stroke-level contrastive loss
        # Need to aggregate multiple strokes into single vector
        image_strokes = image_outputs["stroke"].mean(dim=1)  # (B, stroke_dim)
        text_strokes = text_outputs["stroke"].mean(dim=1)  # (B, stroke_dim)
        stroke_loss = self.contrastive_loss(image_strokes, text_strokes)

        # Radical-level contrastive loss
        image_radicals = image_outputs["radical"].mean(dim=1)  # (B, radical_dim)
        text_radicals = text_outputs["radical"].mean(dim=1)  # (B, radical_dim)
        radical_loss = self.contrastive_loss(image_radicals, text_radicals)

        # Character-level contrastive loss
        image_characters = image_outputs["character"].squeeze(1)  # (B, character_dim)
        text_characters = text_outputs["character"]  # (B, character_dim)
        character_loss = self.contrastive_loss(image_characters, text_characters)

        # Total loss with weights
        total_loss = (
            self.weight_stroke * stroke_loss
            + self.weight_radical * radical_loss
            + self.weight_character * character_loss
        )

        logger.debug(
            f"   Losses - Stroke: {stroke_loss.item():.4f}, Radical: {radical_loss.item():.4f}, Character: {character_loss.item():.4f}, Total: {total_loss.item():.4f}"
        )

        return {
            "stroke_loss": stroke_loss,
            "radical_loss": radical_loss,
            "character_loss": character_loss,
            "total_loss": total_loss,
        }


class HierCodeWithHiGITA(nn.Module):
    """
    Enhanced HierCode with Hi-GITA learnings

    Combines HierCode's hierarchical codebook with Hi-GITA's multi-granularity
    image-text alignment approach.

    Key Features:
    1. Optional multi-granularity encoding (can be enabled/disabled)
    2. Contrastive learning at multiple levels
    3. Fine-grained fusion modules
    4. Backward compatible with standard HierCode
    """

    def __init__(
        self,
        num_classes: int = 3036,
        use_higita_enhancement: bool = True,
        stroke_dim: int = 128,
        radical_dim: int = 256,
        character_dim: int = 512,
    ):
        super().__init__()
        logger.info("=" * 70)
        logger.info("🏗️  INITIALIZING HIERCODE WITH HI-GITA ENHANCEMENT")
        logger.info("=" * 70)
        logger.info(f"Classes: {num_classes}, Hi-GITA: {use_higita_enhancement}")
        self.num_classes = num_classes
        self.use_higita_enhancement = use_higita_enhancement

        if use_higita_enhancement:
            # Hi-GITA enhanced image encoder
            self.image_encoder = MultiGranularityImageEncoder(
                stroke_dim=stroke_dim, radical_dim=radical_dim, character_dim=character_dim
            )

            # Classification head
            self.classifier = nn.Linear(character_dim, num_classes)
            logger.info("✓ Hi-GITA enhancement ENABLED")
            logger.info(
                f"  Stroke dim: {stroke_dim}, Radical dim: {radical_dim}, Character dim: {character_dim}"
            )
        else:
            # Standard HierCode (minimal image encoder)
            self.image_encoder = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Linear(256, num_classes)
            logger.info("✓ Standard HierCode (Hi-GITA disabled)")
        logger.info("=" * 70)

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            image: (B, 1, 64, 64)

        Returns:
            if use_higita_enhancement:
                dict with multi-granularity outputs and logits
            else:
                dict with standard HierCode output
        """
        if self.use_higita_enhancement:
            # Get multi-granularity features
            features = self.image_encoder(image)

            # Classification from character-level features
            character_features = features["character"].squeeze(1)
            logits = self.classifier(character_features)
            logger.debug(
                f"→ HierCode forward (Hi-GITA mode): image {image.shape} → logits {logits.shape}"
            )

            return {
                "logits": logits,
                "character_features": character_features,
                "features": features,  # All granularities
            }
        else:
            # Standard HierCode path
            features = self.image_encoder(image)
            features = features.view(features.size(0), -1)
            logits = self.classifier(features)
            logger.debug(
                f"→ HierCode forward (standard mode): image {image.shape} → logits {logits.shape}"
            )

            return {
                "logits": logits,
                "features": features,
            }


if __name__ == "__main__":
    # Test Hi-GITA enhancement

    logger.info("=" * 70)
    logger.info("TESTING HI-GITA ENHANCEMENT WITH HIERCODE")
    logger.info("=" * 70)

    # Create model with Hi-GITA enhancement
    logger.info("\n1️⃣  Creating HierCode model with Hi-GITA...")
    model = HierCodeWithHiGITA(num_classes=3036, use_higita_enhancement=True)

    # Test image input
    batch_size = 4
    image = torch.randn(batch_size, 1, 64, 64)
    logger.info(f"   Input shape: {image.shape}")

    # Forward pass
    logger.info("\n2️⃣  Testing image encoder forward pass...")
    output = model(image)
    logger.info(f"   ✓ Output logits: {output['logits'].shape}")
    logger.info(f"   ✓ Character features: {output['character_features'].shape}")

    # Test text encoder
    logger.info("\n3️⃣  Creating text encoder and testing forward pass...")
    text_encoder = MultiGranularityTextEncoder()
    stroke_codes = torch.randint(0, 20, (batch_size, 15))
    radical_codes = torch.randint(0, 214, (batch_size, 10))
    logger.info(f"   Stroke codes: {stroke_codes.shape}, Radical codes: {radical_codes.shape}")

    text_output = text_encoder(stroke_codes, radical_codes)
    logger.info(f"   ✓ Text stroke output: {text_output['stroke'].shape}")
    logger.info(f"   ✓ Text radical output: {text_output['radical'].shape}")
    logger.info(f"   ✓ Text character output: {text_output['character'].shape}")

    # Test contrastive loss
    logger.info("\n4️⃣  Testing contrastive loss computation...")
    contrastive = FineGrainedContrastiveLoss()
    losses = contrastive(output["features"], text_output)
    logger.info(f"   ✓ Stroke loss: {losses['stroke_loss'].item():.4f}")
    logger.info(f"   ✓ Radical loss: {losses['radical_loss'].item():.4f}")
    logger.info(f"   ✓ Character loss: {losses['character_loss'].item():.4f}")
    logger.info(f"   ✓ Total loss: {losses['total_loss'].item():.4f}")

    logger.info("\n" + "=" * 70)
    logger.info("✅ HI-GITA ENHANCEMENT TEST COMPLETE!")
    logger.info("=" * 70)
