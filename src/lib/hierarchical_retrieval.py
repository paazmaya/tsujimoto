"""
Phase 1: GL-HPN (Global-Local Hierarchical Perception Network) for Zero-Shot Retrieval

Implements efficient coarse-to-fine retrieval strategy for zero-shot character recognition.
Global branch: Fast similarity matching on whole-character embeddings
Local branch: Detailed patch-token interaction for fine-grained comparison

Based on: Zero-Shot Chinese Character Recognition via Global-Local Dual-Branch
Alignment and Hierarchical Inference (Cao, Xu, & Diao, May 2026)

Key contributions:
- Separates global coarse retrieval from local fine-grained matching
- Structure-filtering masks suppress non-visual IDS operators
- Parameter-free multiplicative fusion combines scores
- Reduces inference cost while maintaining accuracy
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

logger = logging.getLogger(__name__)


class GlobalBranchEncoder(nn.Module):
    """
    Global branch: Fast whole-character encoder for coarse retrieval.

    Learns global embeddings for complete character images and IDS representations.
    Used for efficient top-K candidate retrieval before local re-ranking.
    """

    def __init__(
        self,
        input_dim: int = 2048,  # From backbone features
        embedding_dim: int = 512,
        num_ids_tokens: int = 1024,
    ):
        """
        Initialize global encoder.

        Args:
            input_dim: Dimension of input features from backbone
            embedding_dim: Dimension of learned embeddings
            num_ids_tokens: Number of IDS operator tokens
        """
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_ids_tokens = num_ids_tokens

        # Image embedding branch
        self.image_fc = nn.Sequential(
            nn.Linear(input_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

        # IDS embedding branch (encodes structural information)
        self.ids_embedding = nn.Embedding(num_ids_tokens, embedding_dim)
        self.ids_fc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(
        self,
        image_features: Tensor,
        ids_tokens: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Encode image and IDS to global embeddings.

        Args:
            image_features: (batch, input_dim) image features from backbone
            ids_tokens: (batch, seq_len) IDS token indices, or None for images-only

        Returns:
            image_embedding: (batch, embedding_dim) normalized image embeddings
            ids_embedding: (batch, embedding_dim) normalized IDS embeddings or zeros
        """
        # Global image embedding
        image_emb = self.image_fc(image_features)
        image_emb = F.normalize(image_emb, p=2, dim=1)

        # Global IDS embedding
        if ids_tokens is not None:
            ids_emb = self.ids_embedding(ids_tokens)
            ids_emb = ids_emb.mean(dim=1)  # Pool over sequence
            ids_emb = self.ids_fc(ids_emb)
            ids_emb = F.normalize(ids_emb, p=2, dim=1)
        else:
            ids_emb = torch.zeros_like(image_emb)

        return image_emb, ids_emb


class LocalBranchEncoder(nn.Module):
    """
    Local branch: Fine-grained patch-token encoder for detailed comparison.

    Processes image patches and IDS tokens to capture component-level differences
    that a single global embedding might miss.
    """

    def __init__(
        self,
        patch_dim: int = 256,
        num_patches: int = 16,
        embedding_dim: int = 512,
        num_heads: int = 8,
    ):
        """
        Initialize local encoder.

        Args:
            patch_dim: Dimension of each image patch
            num_patches: Number of patches to extract
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
        """
        super().__init__()

        self.patch_dim = patch_dim
        self.num_patches = num_patches
        self.embedding_dim = embedding_dim

        # Patch embedding layer
        self.patch_embed = nn.Linear(patch_dim, embedding_dim)

        # Cross-attention between image patches and IDS tokens
        self.self_attention = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            dropout=0.1,
            batch_first=True,
        )

        # Patch-level output layer
        self.output_fc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(
        self,
        patches: Tensor,
        ids_tokens: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Encode patches with optional IDS guidance.

        Args:
            patches: (batch, num_patches, patch_dim) image patches
            ids_tokens: (batch, seq_len, embedding_dim) IDS embeddings

        Returns:
            local_features: (batch, num_patches, embedding_dim) local representations
        """
        # Embed patches
        patch_emb = self.patch_embed(patches)  # (batch, num_patches, embedding_dim)

        # Apply self-attention
        if ids_tokens is not None:
            # Cross-attention with IDS tokens as context
            attn_out, _ = self.self_attention(patch_emb, ids_tokens, ids_tokens)
        else:
            # Self-attention on patches only
            attn_out, _ = self.self_attention(patch_emb, patch_emb, patch_emb)

        # Process output
        local_features = self.output_fc(attn_out)

        return local_features


class StructureFilteringMask(nn.Module):
    """
    Filters IDS operators to suppress those without clear visual correspondence.

    IDS (Ideographic Description Sequences) encode character structure, but not all
    operators have direct visual counterparts. This module learns to suppress
    operators that don't meaningfully contribute to visual recognition.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        num_ids_tokens: int = 1024,
    ):
        """
        Initialize structure filtering mask.

        Args:
            embedding_dim: Dimension of embeddings
            num_ids_tokens: Number of IDS tokens
        """
        super().__init__()

        # Learn per-token visual importance scores
        self.token_importance = nn.Embedding(num_ids_tokens, 1)

        # Score refinement network
        self.score_refiner = nn.Sequential(
            nn.Linear(embedding_dim + 1, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

    def forward(
        self,
        ids_tokens: Tensor,
        ids_embeddings: Tensor,
    ) -> Tensor:
        """
        Compute filtering mask for IDS tokens.

        Args:
            ids_tokens: (batch, seq_len) token indices
            ids_embeddings: (batch, seq_len, embedding_dim) embeddings

        Returns:
            mask: (batch, seq_len) importance scores in [0, 1]
        """
        batch_size, seq_len, embedding_dim = ids_embeddings.shape

        # Get base importance scores
        token_scores = self.token_importance(ids_tokens)  # (batch, seq_len, 1)
        token_scores = torch.sigmoid(token_scores)  # Normalize to [0, 1]

        # Refine scores using embeddings
        combined = torch.cat([ids_embeddings, token_scores], dim=2)  # (batch, seq_len, emb+1)
        refined_scores = self.score_refiner(combined)  # (batch, seq_len, 1)
        refined_scores = torch.sigmoid(refined_scores)

        # Combine base and refined scores
        mask = (token_scores + refined_scores) / 2

        return mask.squeeze(2)  # (batch, seq_len)


class CoarseToFineRetriever(nn.Module):
    """
    Orchestrates coarse-to-fine retrieval: global top-K → local re-ranking.

    Stage 1 (coarse): Use global embeddings for fast nearest-neighbor search
    Stage 2 (fine): Apply local fine-grained matching to top-K candidates
    """

    def __init__(
        self,
        global_encoder: GlobalBranchEncoder,
        local_encoder: LocalBranchEncoder,
        structure_mask: StructureFilteringMask,
        top_k_candidates: int = 100,
        embedding_dim: int = 512,
    ):
        """
        Initialize retriever.

        Args:
            global_encoder: GlobalBranchEncoder instance
            local_encoder: LocalBranchEncoder instance
            structure_mask: StructureFilteringMask instance
            top_k_candidates: Number of candidates to re-rank locally
            embedding_dim: Embedding dimension
        """
        super().__init__()

        self.global_encoder = global_encoder
        self.local_encoder = local_encoder
        self.structure_mask = structure_mask
        self.top_k_candidates = top_k_candidates
        self.embedding_dim = embedding_dim

        # Optional: Use FAISS for efficient similarity search (if available)
        try:
            import importlib.util

            self.faiss_available = importlib.util.find_spec("faiss") is not None
        except ImportError:
            self.faiss_available = False
            logger.warning("FAISS not available, using torch similarity search")

    def forward(
        self,
        query_image_features: Tensor,
        query_ids_tokens: Optional[Tensor] = None,
        candidate_embeddings: Optional[Tensor] = None,
        top_k: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Retrieve top characters using coarse-to-fine strategy.

        Args:
            query_image_features: (batch, input_dim) query image features
            query_ids_tokens: (batch, seq_len) query IDS tokens
            candidate_embeddings: (num_candidates, embedding_dim) pre-computed candidate embeddings
            top_k: Number of top candidates to return (default: top_k_candidates)

        Returns:
            top_indices: (batch, top_k) indices of retrieved characters
            scores: (batch, top_k) retrieval confidence scores
        """
        top_k = top_k or self.top_k_candidates

        if candidate_embeddings is None:
            raise ValueError("candidate_embeddings must be provided")

        # Stage 1: Coarse retrieval with global embeddings
        query_image_emb, query_ids_emb = self.global_encoder(query_image_features, query_ids_tokens)

        # Combine global embeddings
        query_global = query_image_emb + query_ids_emb  # Simple addition
        query_global = F.normalize(query_global, p=2, dim=1)

        # Compute similarity with all candidates
        similarities = torch.matmul(query_global, candidate_embeddings.T)  # (batch, num_candidates)

        # Get top-K candidates
        top_scores, top_indices = torch.topk(
            similarities, k=min(top_k * 2, candidate_embeddings.shape[0]), dim=1
        )

        # Stage 2: Fine-grained re-ranking (simplified - in practice would use local patches)
        # For now, apply structure-aware filtering if IDS tokens available
        if query_ids_tokens is not None:
            mask = self.structure_mask(
                query_ids_tokens, self.global_encoder.ids_embedding(query_ids_tokens)
            )
            mask = mask.mean(dim=1, keepdim=True)  # Average mask across sequence
            top_scores = top_scores * (0.7 + 0.3 * mask)  # Apply soft weighting

        # Return final top-K after re-ranking
        final_top_scores, reranked_indices = torch.topk(top_scores, k=top_k, dim=1)
        final_top_indices = torch.gather(top_indices, 1, reranked_indices)

        return final_top_indices, final_top_scores


class MultiplicativeFusion(nn.Module):
    """
    Parameter-free fusion of global and local scores.

    Combines normalized global and local posterior probabilities
    using element-wise multiplication (multiplicative fusion).
    """

    def __init__(self, fusion_strategy: str = "multiplicative"):
        """
        Initialize fusion module.

        Args:
            fusion_strategy: 'multiplicative' (multiply), 'additive' (add), or 'attention' (learned)
        """
        super().__init__()
        self.fusion_strategy = fusion_strategy

        if fusion_strategy == "attention":
            # Learnable fusion weights
            self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        global_scores: Tensor,
        local_scores: Tensor,
    ) -> Tensor:
        """
        Fuse global and local scores.

        Args:
            global_scores: (batch, num_candidates) global similarity scores
            local_scores: (batch, num_candidates) local similarity scores

        Returns:
            fused_scores: (batch, num_candidates) combined scores
        """
        # Normalize scores to probabilities
        global_probs = F.softmax(global_scores, dim=1)
        local_probs = F.softmax(local_scores, dim=1)

        if self.fusion_strategy == "multiplicative":
            # Element-wise multiplication
            fused = global_probs * local_probs
            # Normalize result
            fused = fused / (fused.sum(dim=1, keepdim=True) + 1e-8)
        elif self.fusion_strategy == "additive":
            fused = (global_probs + local_probs) / 2
        elif self.fusion_strategy == "attention":
            fused = self.fusion_weight * global_probs + (1 - self.fusion_weight) * local_probs
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        return fused


# ==================== Integration with existing training ====================


def create_glhpn_retriever(
    backbone_output_dim: int = 2048,
    embedding_dim: int = 512,
    patch_dim: int = 256,
    num_patches: int = 16,
    top_k_candidates: int = 100,
) -> CoarseToFineRetriever:
    """
    Factory function to create a GL-HPN retriever.

    Args:
        backbone_output_dim: Output dimension from model backbone
        embedding_dim: Embedding dimension for encoders
        patch_dim: Dimension of image patches
        num_patches: Number of patches
        top_k_candidates: Number of candidates for re-ranking

    Returns:
        Configured CoarseToFineRetriever instance
    """
    global_encoder = GlobalBranchEncoder(
        input_dim=backbone_output_dim,
        embedding_dim=embedding_dim,
    )

    local_encoder = LocalBranchEncoder(
        patch_dim=patch_dim,
        num_patches=num_patches,
        embedding_dim=embedding_dim,
    )

    structure_mask = StructureFilteringMask(
        embedding_dim=embedding_dim,
    )

    retriever = CoarseToFineRetriever(
        global_encoder=global_encoder,
        local_encoder=local_encoder,
        structure_mask=structure_mask,
        top_k_candidates=top_k_candidates,
        embedding_dim=embedding_dim,
    )

    return retriever
