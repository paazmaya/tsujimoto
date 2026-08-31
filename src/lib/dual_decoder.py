"""
Phase 2: DTRNet (Dual Text-Radical Decoding) for Structural Verification

Implements dual-decoder architecture: text branch transcribes characters while
radical branch validates structural plausibility via IDS (Ideographic Description Sequences).

Based on: DTRNet: Dual Text-Radical Decoding for Handwritten Chinese Text Recognition
with Faked Character Detection (Li, Zhu, & Huang, August 2026)

Key contributions:
- Text decoder: Context-aware line-level transcription
- Radical decoder: Predicts legal IDS as independent structural evidence
- IDS-Guided Confidence Adjustment (IGCA): Refines predictions using structure agreement
- Fake character detection: Flags predictions where text and radical disagree with lexicon
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

logger = logging.getLogger(__name__)


class TextDecoder(nn.Module):
    """
    Text branch: Line-level character transcription.

    Standard sequence-to-sequence decoder for translating image sequences
    to character sequences, with optional character-level predictions.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 1024,
        num_classes: int = 3036,  # Number of Kanji characters
        decoder_type: str = "gru",  # 'gru' or 'lstm'
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """
        Initialize text decoder.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            num_classes: Number of output character classes
            decoder_type: Type of RNN ('gru' or 'lstm')
            num_layers: Number of RNN layers
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.decoder_type = decoder_type

        # Input projection
        self.input_fc = nn.Linear(input_dim, hidden_dim)

        # RNN backbone
        rnn_class = nn.GRU if decoder_type == "gru" else nn.LSTM
        self.rnn = rnn_class(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

        # Output classification layer
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: Tensor) -> Tensor:
        """
        Decode sequence of image features to character predictions.

        Args:
            features: (batch, seq_len, input_dim) sequence of features

        Returns:
            logits: (batch, seq_len, num_classes) character predictions
        """
        # Project input
        x = self.input_fc(features)  # (batch, seq_len, hidden_dim)

        # Process with RNN
        rnn_out, _ = self.rnn(x)  # (batch, seq_len, hidden_dim*2)

        # Classify each position
        logits = self.output_fc(rnn_out)  # (batch, seq_len, num_classes)

        return logits


class RadicalDecoder(nn.Module):
    """
    Radical branch: Predicts valid IDS (Ideographic Description Sequences).

    Independently predicts the structural components and their relationships
    as independent evidence. Can flag impossible character structures.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        num_ids_tokens: int = 64,  # Number of distinct IDS operators
        max_ids_length: int = 8,  # Max sequence length for IDS
        decoder_type: str = "gru",
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """
        Initialize radical decoder.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            num_ids_tokens: Number of unique IDS operators
            max_ids_length: Maximum IDS sequence length
            decoder_type: Type of RNN ('gru' or 'lstm')
            num_layers: Number of RNN layers
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_ids_tokens = num_ids_tokens
        self.max_ids_length = max_ids_length

        # Input projection
        self.input_fc = nn.Linear(input_dim, hidden_dim)

        # RNN backbone
        rnn_class = nn.GRU if decoder_type == "gru" else nn.LSTM
        self.rnn = rnn_class(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

        # IDS sequence prediction heads (multi-step)
        self.ids_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_ids_tokens),
                )
                for _ in range(max_ids_length)
            ]
        )

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Decode features to IDS predictions.

        Args:
            features: (batch, seq_len, input_dim) or (batch, input_dim) features

        Returns:
            ids_logits: (batch, max_ids_length, num_ids_tokens) IDS predictions
            ids_confidence: (batch,) confidence scores
        """
        # Handle both sequence and single features
        if features.dim() == 2:
            features = features.unsqueeze(1)  # Add sequence dimension if needed

        # Project input
        x = self.input_fc(features)  # (batch, seq_len, hidden_dim)

        # Process with RNN
        rnn_out, _ = self.rnn(x)  # (batch, seq_len, hidden_dim*2)

        # Pool sequence dimension (take last output or mean)
        pooled = rnn_out[:, -1, :]  # (batch, hidden_dim*2)

        # Predict IDS sequence
        ids_logits_list = []
        for head in self.ids_heads:
            ids_logits_list.append(head(pooled))

        ids_logits = torch.stack(ids_logits_list, dim=1)  # (batch, max_ids_length, num_ids_tokens)

        # Compute confidence as max probability across all IDS positions
        ids_probs = F.softmax(ids_logits, dim=2)  # (batch, max_ids_length, num_ids_tokens)
        ids_confidence = ids_probs.max(dim=2)[0].mean(dim=1)  # (batch,)

        return ids_logits, ids_confidence


class IDSValidator(nn.Module):
    """
    Validates predicted IDS sequences against lexicon.

    Maintains a database of valid character IDS and checks if
    predicted structure is plausible.
    """

    def __init__(self):
        """Initialize validator with empty lexicon."""
        super().__init__()
        # IDS lexicon: character_id -> valid IDS token sequences
        self.ids_lexicon: Dict[int, List[int]] = {}

    def add_character_ids(self, character_id: int, ids_sequence: List[int]):
        """
        Add IDS entry to lexicon.

        Args:
            character_id: Character class ID
            ids_sequence: IDS operator sequence
        """
        self.ids_lexicon[character_id] = ids_sequence

    def validate_ids(
        self,
        character_id: int,
        predicted_ids: Tensor,
    ) -> float:
        """
        Check if predicted IDS matches lexicon entry.

        Args:
            character_id: Predicted character ID
            predicted_ids: (max_ids_length,) predicted IDS tokens

        Returns:
            validity_score: 0.0 (invalid) to 1.0 (valid)
        """
        if character_id not in self.ids_lexicon:
            return 0.5  # Neutral score for unknown characters

        valid_ids = self.ids_lexicon[character_id]
        predicted_ids_np = predicted_ids.argmax(dim=-1).cpu().numpy()

        # Check if predicted IDS matches (allowing some flexibility)
        matches = sum(1 for p, v in zip(predicted_ids_np, valid_ids) if p == v)

        # Validity score based on match ratio
        validity_score = matches / max(len(valid_ids), 1)
        return float(validity_score)


class IDSGuidedConfidenceAdjustment(nn.Module):
    """
    IGCA: Refines character predictions using structural evidence.

    Adjusts text decoder confidence when structure agreement is low,
    effectively downweighting predictions that violate structural constraints.
    """

    def __init__(self, temperature: float = 0.5):
        """
        Initialize IGCA.

        Args:
            temperature: Temperature for softening confidence adjustments
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        text_logits: Tensor,
        ids_confidence: Tensor,
        structure_agreement: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Adjust text predictions using structural confidence.

        Args:
            text_logits: (batch, num_classes) character logits from text decoder
            ids_confidence: (batch,) IDS prediction confidence
            structure_agreement: (batch,) agreement between text and radical (0-1)

        Returns:
            adjusted_logits: (batch, num_classes) refined logits
        """
        # Default to using IDS confidence directly
        if structure_agreement is None:
            structure_agreement = ids_confidence

        # Expand for broadcasting
        adjustment = structure_agreement.view(-1, 1)  # (batch, 1)

        # Soft adjustment: interpolate between original and zero logits
        # High structure agreement → keep original logits
        # Low structure agreement → reduce confidence
        adjusted = adjustment * text_logits + (1 - adjustment) * text_logits * self.temperature

        return adjusted


class FakeCharacterDetector(nn.Module):
    """
    Detects fake/malformed character predictions where text and radical disagree.

    Compares text decoder output with radical decoder output and IDS lexicon
    to identify suspicious predictions.
    """

    def __init__(
        self,
        text_num_classes: int = 3036,
        threshold: float = 0.7,
    ):
        """
        Initialize detector.

        Args:
            text_num_classes: Number of character classes
            threshold: Confidence threshold for flagging as suspicious
        """
        super().__init__()
        self.text_num_classes = text_num_classes
        self.threshold = threshold

    def forward(
        self,
        text_logits: Tensor,
        ids_confidence: Tensor,
        structure_agreement: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Detect fake characters.

        Args:
            text_logits: (batch, num_classes) text predictions
            ids_confidence: (batch,) IDS confidence
            structure_agreement: (batch,) agreement scores (0-1)

        Returns:
            is_fake: (batch,) binary flags (1 = suspicious, 0 = normal)
            suspicion_score: (batch,) continuous suspicion scores [0, 1]
        """
        # Text confidence
        text_probs = F.softmax(text_logits, dim=1)
        text_confidence = text_probs.max(dim=1)[0]  # (batch,)

        # Suspicion score: high if text and radical disagree
        # OR if either is low confidence
        suspicion_score = 1.0 - (text_confidence * ids_confidence * structure_agreement)

        # Flag as fake if suspicion exceeds threshold
        is_fake = (suspicion_score > self.threshold).float()

        return is_fake, suspicion_score


# ==================== Combined DTRNet Module ====================


class DTRNetModule(nn.Module):
    """
    Complete DTRNet: Orchestrates text + radical decoders with structural verification.
    """

    def __init__(
        self,
        input_dim: int = 512,
        num_character_classes: int = 3036,
        num_ids_tokens: int = 64,
        text_decoder_type: str = "gru",
        radical_decoder_type: str = "gru",
        structure_agreement_weight: float = 0.3,
        use_igca: bool = True,
    ):
        """
        Initialize complete DTRNet.

        Args:
            input_dim: Input feature dimension
            num_character_classes: Number of character classes
            num_ids_tokens: Number of IDS operator types
            text_decoder_type: Text decoder type
            radical_decoder_type: Radical decoder type
            structure_agreement_weight: Weight for structure agreement loss
            use_igca: Whether to use IDS-Guided Confidence Adjustment
        """
        super().__init__()

        self.text_decoder = TextDecoder(
            input_dim=input_dim,
            num_classes=num_character_classes,
            decoder_type=text_decoder_type,
        )

        self.radical_decoder = RadicalDecoder(
            input_dim=input_dim,
            num_ids_tokens=num_ids_tokens,
            decoder_type=radical_decoder_type,
        )

        self.ids_validator = IDSValidator()

        self.igca = IDSGuidedConfidenceAdjustment() if use_igca else None

        self.fake_detector = FakeCharacterDetector(text_num_classes=num_character_classes)

        self.structure_agreement_weight = structure_agreement_weight

    def forward(
        self,
        features: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Forward pass through DTRNet.

        Args:
            features: (batch, seq_len, input_dim) or (batch, input_dim) features

        Returns:
            Dictionary with:
                - text_logits: (batch, seq_len, num_classes)
                - ids_logits: (batch, max_ids_length, num_ids_tokens)
                - adjusted_logits: (batch, seq_len, num_classes) if IGCA enabled
                - is_fake: (batch, seq_len) fake character flags
                - suspicion_score: (batch, seq_len) suspicion scores
        """
        # Text decoding
        text_logits = self.text_decoder(features)  # (batch, seq_len, num_classes)

        # Radical decoding
        ids_logits, ids_confidence = self.radical_decoder(features)  # (batch, max_ids_length, ...)

        # Structural agreement (simplified: just use IDS confidence)
        # In practice, would compute agreement between text predictions and IDS validity
        if features.dim() == 2:
            seq_len = 1
        else:
            seq_len = features.shape[1]

        structure_agreement = ids_confidence.unsqueeze(1).expand(-1, seq_len)  # (batch, seq_len)

        # Apply IGCA if enabled
        if self.igca is not None:
            adjusted_logits = self.igca(
                text_logits.view(-1, text_logits.shape[-1]),
                ids_confidence.unsqueeze(1).expand(-1, seq_len).reshape(-1),
                structure_agreement.reshape(-1),
            )
            adjusted_logits = adjusted_logits.view(*text_logits.shape)
        else:
            adjusted_logits = text_logits

        # Detect fake characters
        text_probs_flat = F.softmax(text_logits.view(-1, text_logits.shape[-1]), dim=1)

        is_fake, suspicion_score = self.fake_detector(
            text_probs_flat,
            ids_confidence.unsqueeze(1).expand(-1, seq_len).reshape(-1),
            structure_agreement.reshape(-1),
        )

        is_fake = is_fake.view(*text_logits.shape[:2])
        suspicion_score = suspicion_score.view(*text_logits.shape[:2])

        return {
            "text_logits": text_logits,
            "ids_logits": ids_logits,
            "adjusted_logits": adjusted_logits,
            "is_fake": is_fake,
            "suspicion_score": suspicion_score,
            "ids_confidence": ids_confidence,
            "structure_agreement": structure_agreement,
        }
