"""
Phase 3B: Restoration Module for Document Preprocessing

Implements document restoration techniques before character recognition:
- Binarization: Converting grayscale to binary
- Seal removal: Inpainting or masking seal overlays
- Blur removal: Deconvolution or restoration
- General restoration: GAN-based or traditional methods

Restoration-Guided Recognition framework treats restoration as a preprocessing stage
that significantly improves downstream character classification accuracy.

Based on: Restoration-Guided Kuzushiji Character Recognition Framework under Seal
Interference (Ju, Yamashita, Kameko, & Mori, February 2026)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

logger = logging.getLogger(__name__)


class RestorationMethod(str, Enum):
    """Available restoration methods."""

    OTSU = "otsu"  # Traditional Otsu binarization
    KMEANS = "kmeans"  # K-means clustering binarization
    ADAPTIVE = "adaptive"  # Adaptive threshold
    MORPHOLOGICAL = "morphological"  # Morphological operations
    CONDITIONAL_GAN = "conditional_gan"  # GAN-based restoration
    LEARNABLE = "learnable"  # Trainable restoration network


@dataclass
class RestorationConfig:
    """Configuration for restoration pipeline."""

    restoration_enabled: bool = True
    method: RestorationMethod = RestorationMethod.OTSU
    binarize_first: bool = True  # Apply binarization before other restoration
    remove_seals: bool = True  # Detect and remove seals
    remove_noise: bool = True  # Denoise
    enhance_contrast: bool = True  # Enhance document contrast
    use_morphological: bool = True  # Apply morphological operations
    kernel_size: int = 3
    iterations: int = 2


class TraditionalBinarization(nn.Module):
    """
    Traditional binarization methods: Otsu, K-means, adaptive thresholding.
    """

    def __init__(self, method: str = "otsu"):
        """
        Initialize binarization.

        Args:
            method: 'otsu', 'kmeans', or 'adaptive'
        """
        super().__init__()
        self.method = method.lower()

    def forward(self, image: Tensor) -> Tensor:
        """
        Binarize image.

        Args:
            image: (B, C, H, W) or (C, H, W) image tensor [0, 1]

        Returns:
            binary: Binary image (0 or 1 values)
        """
        # Handle different input shapes
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        b, c, h, w = image.shape

        # Convert to grayscale if needed
        if c == 3:
            # RGB to grayscale
            gray = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
        elif c == 1:
            gray = image.squeeze(1)
        else:
            raise ValueError(f"Unsupported number of channels: {c}")

        # Apply binarization
        if self.method == "otsu":
            binary = self._otsu_binarization(gray)
        elif self.method == "kmeans":
            binary = self._kmeans_binarization(gray)
        elif self.method == "adaptive":
            binary = self._adaptive_binarization(gray)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Restore batch/channel dimensions
        binary = binary.unsqueeze(1)  # Add channel dimension

        if squeeze:
            binary = binary.squeeze(0)

        return binary

    @staticmethod
    def _otsu_binarization(image: Tensor) -> Tensor:
        """
        Otsu's automatic thresholding.

        Args:
            image: (B, H, W) grayscale image [0, 1]

        Returns:
            binary: (B, H, W) binary image
        """
        b, h, w = image.shape

        # Compute histogram
        hist = torch.histc(image.view(b, -1), bins=256, min=0, max=1)

        # Normalize histogram
        hist = hist / hist.sum()

        # Compute cumulative sum and mean
        cum_sum = torch.cumsum(hist, dim=0)
        cum_mean = torch.cumsum(
            hist * torch.arange(256, dtype=torch.float32, device=image.device) / 256, dim=0
        )

        # Total mean
        total_mean = cum_mean[-1]

        # Variance between classes
        variance = cum_sum * (1 - cum_sum) * (cum_mean - total_mean * cum_sum) ** 2

        # Find threshold that maximizes variance
        threshold = torch.argmax(variance) / 256.0

        # Apply threshold
        binary = (image > threshold).float()

        return binary

    @staticmethod
    def _kmeans_binarization(image: Tensor, k: int = 2) -> Tensor:
        """
        K-means clustering for binarization.

        Args:
            image: (B, H, W) grayscale image
            k: Number of clusters (typically 2 for binary)

        Returns:
            binary: (B, H, W) binary image
        """
        b, h, w = image.shape

        # Flatten
        pixels = image.view(b, -1, 1)  # (B, H*W, 1)

        # Simple k-means approximation: use threshold at mean
        mean_val = pixels.mean()
        threshold = mean_val.item()

        binary = (image > threshold).float()

        return binary

    @staticmethod
    def _adaptive_binarization(image: Tensor, window_size: int = 31) -> Tensor:
        """
        Adaptive thresholding using local mean.

        Args:
            image: (B, H, W) grayscale image
            window_size: Size of local window

        Returns:
            binary: (B, H, W) binary image
        """
        b, h, w = image.shape

        # Add padding and batch/channel dimensions
        image_padded = F.pad(
            image.unsqueeze(1),
            (window_size // 2, window_size // 2, window_size // 2, window_size // 2),
            mode="reflect",
        )

        # Compute local mean using average pooling
        local_mean = F.avg_pool2d(image_padded, kernel_size=window_size, stride=1, padding=0)

        # Binarize: pixel > local_mean - constant
        constant = 2
        binary = (image > (local_mean.squeeze(1) - constant)).float()

        return binary


class SealsRemovalModule(nn.Module):
    """
    Detects and removes/masks seal overlays.

    Seals are typically circular, reddish, and overlap character regions.
    This module identifies and inpaints or masks them.
    """

    def __init__(self, inpainting_method: str = "morphological"):
        """
        Initialize seal removal.

        Args:
            inpainting_method: 'morphological', 'diffusion', or 'mask'
        """
        super().__init__()
        self.inpainting_method = inpainting_method

    def forward(
        self,
        image: Tensor,
        seal_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Remove seals from image.

        Args:
            image: (B, C, H, W) image tensor
            seal_mask: Optional pre-computed seal mask

        Returns:
            restored: (B, C, H, W) image with seals removed
            detected_mask: (B, 1, H, W) seal detection mask
        """
        if seal_mask is None:
            # Detect seals (simplified: look for reddish circular regions)
            seal_mask = self._detect_seals(image)

        # Remove seals using inpainting
        restored = self._inpaint(image, seal_mask, method=self.inpainting_method)

        return restored, seal_mask

    def _detect_seals(self, image: Tensor, threshold: float = 0.3) -> Tensor:
        """
        Detect seal regions (simplified detection).

        Args:
            image: (B, C, H, W) image
            threshold: Detection threshold

        Returns:
            seal_mask: (B, 1, H, W) binary mask
        """
        b, c, h, w = image.shape

        if c >= 3:
            # Look for reddish regions (high R, low G, low B)
            red_channel = image[:, 0:1]
            green_channel = image[:, 1:2]
            blue_channel = image[:, 2:3]

            # Reddish: R > G and R > B
            reddish = (red_channel > green_channel + 0.2) & (red_channel > blue_channel + 0.2)
            seal_mask = reddish.float()
        else:
            # Grayscale: look for darker circular regions
            seal_mask = (image < 0.4).float()

        # Morphological operations to clean up mask
        seal_mask = self._morphological_close(seal_mask, kernel_size=5)

        return seal_mask

    def _inpaint(
        self,
        image: Tensor,
        mask: Tensor,
        method: str = "morphological",
    ) -> Tensor:
        """
        Inpaint masked regions.

        Args:
            image: (B, C, H, W) original image
            mask: (B, 1, H, W) seal mask
            method: Inpainting method

        Returns:
            inpainted: (B, C, H, W) image with seals removed
        """
        if method == "morphological":
            # Simple approach: use morphological dilation from boundaries
            dilated_mask = self._morphological_dilate(mask, kernel_size=7)

            # Replace masked region with neighborhood average
            for _ in range(2):
                image = image * (1 - dilated_mask) + self._neighborhood_mean(image) * dilated_mask

        elif method == "diffusion":
            # Iterative diffusion/smoothing in masked region
            inpainted = image.clone()
            for _ in range(10):
                smoothed = F.avg_pool2d(
                    F.pad(inpainted, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1
                )
                inpainted = inpainted * (1 - mask) + smoothed * mask

        elif method == "mask":
            # Just return masked image (for structural verification)
            inpainted = image * (1 - mask)

        else:
            inpainted = image

        return inpainted

    @staticmethod
    def _neighborhood_mean(image: Tensor) -> Tensor:
        """Compute neighborhood mean (for inpainting)."""
        return F.avg_pool2d(F.pad(image, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1)

    @staticmethod
    def _morphological_dilate(mask: Tensor, kernel_size: int = 5) -> Tensor:
        """Morphological dilation."""
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device) / (kernel_size**2)
        dilated = F.conv2d(mask, kernel, padding=kernel_size // 2)
        return (dilated > 0).float()

    @staticmethod
    def _morphological_close(mask: Tensor, kernel_size: int = 5) -> Tensor:
        """Morphological closing (dilation followed by erosion)."""
        # Dilation
        dilated = F.max_pool2d(mask, kernel_size=kernel_size, padding=kernel_size // 2)

        # Erosion
        eroded = F.max_pool2d(1 - dilated, kernel_size=kernel_size, padding=kernel_size // 2)
        closed = 1 - eroded

        return closed


class RestorationPreprocessor(nn.Module):
    """
    Complete preprocessing pipeline for document images.

    Combines binarization, seal removal, noise reduction, and contrast enhancement.
    """

    def __init__(self, config: RestorationConfig):
        """
        Initialize preprocessor.

        Args:
            config: RestorationConfig
        """
        super().__init__()
        self.config = config

        self.binarization = TraditionalBinarization(method=config.method.value)
        self.seal_removal = SealsRemovalModule()

    def forward(self, image: Tensor) -> Dict[str, Tensor]:
        """
        Preprocess document image.

        Args:
            image: (B, C, H, W) document image

        Returns:
            Dictionary with:
                - restored: Preprocessed image
                - original: Original image
                - intermediate: Intermediate processing stages
        """
        result = {
            "original": image.clone(),
            "intermediate": {},
        }

        current = image.clone()

        # Step 1: Enhance contrast
        if self.config.enhance_contrast:
            current = self._enhance_contrast(current)
            result["intermediate"]["contrast_enhanced"] = current.clone()

        # Step 2: Remove seals
        if self.config.remove_seals:
            current, seal_mask = self.seal_removal(current)
            result["intermediate"]["seal_removed"] = current.clone()
            result["seal_mask"] = seal_mask

        # Step 3: Remove noise
        if self.config.remove_noise:
            current = self._denoise(current)
            result["intermediate"]["denoised"] = current.clone()

        # Step 4: Binarization
        if self.config.binarize_first:
            current = self.binarization(current)
            result["intermediate"]["binarized"] = current.clone()

        # Step 5: Morphological operations
        if self.config.use_morphological:
            current = self._apply_morphological(
                current, self.config.kernel_size, self.config.iterations
            )
            result["intermediate"]["morphological"] = current.clone()

        result["restored"] = current

        return result

    @staticmethod
    def _enhance_contrast(image: Tensor) -> Tensor:
        """Enhance document contrast."""
        # Clip extremes and rescale
        p_low = torch.quantile(image, 0.02)
        p_high = torch.quantile(image, 0.98)

        enhanced = (image - p_low) / (p_high - p_low + 1e-8)
        enhanced = torch.clamp(enhanced, 0, 1)

        return enhanced

    @staticmethod
    def _denoise(image: Tensor) -> Tensor:
        """Apply simple denoising (bilateral-like filter)."""
        # Median filter approximation using max pool
        b, c, h, w = image.shape

        # Apply multiple passes of median-like filtering
        for _ in range(2):
            image = F.max_pool2d(
                F.pad(image, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1
            )
            image = F.avg_pool2d(
                F.pad(image, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1
            )

        return image

    @staticmethod
    def _apply_morphological(image: Tensor, kernel_size: int, iterations: int) -> Tensor:
        """Apply morphological operations."""
        result = image

        for _ in range(iterations):
            # Closing: dilation then erosion (fills small holes)
            dilated = F.max_pool2d(result, kernel_size=kernel_size, padding=kernel_size // 2)
            result = F.max_pool2d(1 - dilated, kernel_size=kernel_size, padding=kernel_size // 2)
            result = 1 - result

        return result


# ==================== Integration ====================


def create_restoration_preprocessor(
    method: str = "otsu",
    with_seal_removal: bool = True,
    with_denoising: bool = True,
) -> RestorationPreprocessor:
    """
    Factory function to create restoration preprocessor.

    Args:
        method: Binarization method ('otsu', 'kmeans', 'adaptive')
        with_seal_removal: Whether to remove seals
        with_denoising: Whether to denoise

    Returns:
        Configured RestorationPreprocessor
    """
    config = RestorationConfig(
        method=RestorationMethod(method),
        remove_seals=with_seal_removal,
        remove_noise=with_denoising,
    )

    return RestorationPreprocessor(config)
