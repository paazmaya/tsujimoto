"""
Phase 3: Degradation-Aware Training with Synthetic Degradation Pipeline

Implements synthetic document degradation for robust model training.
Mimics real-world document conditions: blur, stains, contrast shifts, seals, warping.

Based on: DKDS: A Benchmark Dataset of Degraded Kuzushiji Documents with Seals
for Detection and Binarization (Ju et al., 2025-2026)

Key contributions:
- Realistic degradation pipeline: blur, stains, contrast, seals, warping
- Configurable severity levels
- On-the-fly augmentation during training
- Separate degradation handling for train/val/test
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

try:
    import albumentations  # noqa: F401

    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

try:
    import kornia as K  # noqa: N812

    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False

logger = logging.getLogger(__name__)


class DegradationType(str, Enum):
    """Types of image degradation to apply."""

    BLUR = "blur"
    STAIN = "stain"
    CONTRAST = "contrast"
    SEAL = "seal"
    WARPING = "warping"
    BINARIZATION = "binarization"
    NOISE = "noise"


@dataclass
class DegradationConfig:
    """Configuration for degradation pipeline."""

    degradation_enabled: bool = True
    degradation_types: List[DegradationType] = None
    severity_min: float = 0.2  # Minimum degradation intensity
    severity_max: float = 0.8  # Maximum degradation intensity
    probability: float = 0.7  # Probability of applying degradation
    apply_to_train: bool = True
    apply_to_val: bool = False
    apply_to_test: bool = True
    seed: int = 42

    def __post_init__(self):
        if self.degradation_types is None:
            self.degradation_types = [
                DegradationType.BLUR,
                DegradationType.STAIN,
                DegradationType.CONTRAST,
                DegradationType.SEAL,
            ]


class DegradationPipeline(nn.Module):
    """
    Main degradation pipeline: applies multiple degradation types.

    Each degradation is independently applied with configurable severity.
    """

    def __init__(self, config: DegradationConfig):
        """
        Initialize degradation pipeline.

        Args:
            config: DegradationConfig with settings
        """
        super().__init__()
        self.config = config

        # Initialize degradation modules
        self.degradations = nn.ModuleDict()

        if DegradationType.BLUR in config.degradation_types:
            self.degradations["blur"] = GaussianBlur()
        if DegradationType.STAIN in config.degradation_types:
            self.degradations["stain"] = StainDegradation()
        if DegradationType.CONTRAST in config.degradation_types:
            self.degradations["contrast"] = ContrastShift()
        if DegradationType.SEAL in config.degradation_types:
            self.degradations["seal"] = SealOverlay()
        if DegradationType.WARPING in config.degradation_types:
            self.degradations["warping"] = WarpingDistortion()
        if DegradationType.BINARIZATION in config.degradation_types:
            self.degradations["binarization"] = BinarizationNoise()
        if DegradationType.NOISE in config.degradation_types:
            self.degradations["noise"] = AdditiveNoise()

    def forward(
        self,
        image: Tensor,
        severity: Optional[float] = None,
        degradation_type: Optional[str] = None,
    ) -> Tensor:
        """
        Apply degradation to image.

        Args:
            image: (C, H, W) or (B, C, H, W) image tensor
            severity: Optional fixed severity (0-1), otherwise random
            degradation_type: Optional specific degradation type, otherwise random

        Returns:
            degraded_image: Same shape as input
        """
        # Skip if disabled or random chance
        if not self.config.degradation_enabled or torch.rand(1).item() > self.config.probability:
            return image

        # Set random severity if not provided
        if severity is None:
            severity = np.random.uniform(self.config.severity_min, self.config.severity_max)

        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_out = True
        else:
            squeeze_out = False

        # Select degradation type
        if degradation_type is None:
            degradation_type = np.random.choice(list(self.degradations.keys()))

        if degradation_type not in self.degradations:
            logger.warning(f"Degradation type {degradation_type} not available")
            return image.squeeze(0) if squeeze_out else image

        # Apply degradation
        degraded = self.degradations[degradation_type](image, severity=severity)

        # Remove batch dimension if it was added
        if squeeze_out:
            degraded = degraded.squeeze(0)

        return degraded


class GaussianBlur(nn.Module):
    """Applies Gaussian blur with variable kernel size."""

    def __init__(self):
        super().__init__()

    def forward(self, image: Tensor, severity: float = 0.5) -> Tensor:
        """
        Apply Gaussian blur.

        Args:
            image: (B, C, H, W) image tensor
            severity: Blur intensity (0-1), controls kernel size

        Returns:
            blurred: Same shape as input
        """
        # Map severity to kernel size
        kernel_size = int(3 + severity * 7)  # Range: 3-10
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

        # Sigma increases with severity
        sigma = 0.5 + severity * 2.0

        if KORNIA_AVAILABLE:
            # Use kornia for differentiable blur
            return K.filters.gaussian_blur2d(image, (kernel_size, kernel_size), (sigma, sigma))
        else:
            # Fallback: use torch convolution
            # Create Gaussian kernel
            kernel = self._create_gaussian_kernel(kernel_size, sigma)
            kernel = kernel.to(image.device).to(image.dtype)

            # Apply blur to each channel
            if image.shape[1] == 1:
                # Grayscale
                blurred = F.conv2d(
                    image, kernel.unsqueeze(0).unsqueeze(0), padding=kernel_size // 2
                )
            else:
                # RGB: apply same kernel to each channel
                kernel_expanded = kernel.unsqueeze(0).unsqueeze(0).repeat(image.shape[1], 1, 1, 1)
                blurred = F.conv2d(
                    image, kernel_expanded, groups=image.shape[1], padding=kernel_size // 2
                )

            return blurred

    @staticmethod
    def _create_gaussian_kernel(size: int, sigma: float) -> Tensor:
        """Create 2D Gaussian kernel."""
        x = torch.arange(size).float() - size // 2
        gauss = torch.exp(-x.pow(2.0) / (2 * sigma**2))
        kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)
        return kernel / kernel.sum()


class StainDegradation(nn.Module):
    """Simulates document stains and spots."""

    def __init__(self):
        super().__init__()

    def forward(self, image: Tensor, severity: float = 0.5) -> Tensor:
        """
        Add stain artifacts.

        Args:
            image: (B, C, H, W) image tensor
            severity: Stain intensity (0-1)

        Returns:
            stained: Same shape as input
        """
        b, c, h, w = image.shape
        device = image.device

        # Number of stains increases with severity
        num_stains = int(severity * 10) + 1

        stained = image.clone()

        for _ in range(num_stains):
            # Random stain position and size
            x = torch.randint(0, w, (1,)).item()
            y = torch.randint(0, h, (1,)).item()

            # Stain size (radius)
            radius = int(severity * 30) + 5

            # Create stain mask (circular or irregular)
            yy, xx = torch.meshgrid(
                torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
            )
            dist = torch.sqrt((xx - x) ** 2 + (yy - y) ** 2)

            # Stain color: darker than background (simulating ink)
            stain_color = 0.3 * severity

            # Apply stain with smooth edges
            mask_smooth = torch.exp(-dist / (radius / 2))
            mask_smooth = mask_smooth.unsqueeze(0).unsqueeze(0)

            stained = stained * (1 - mask_smooth * severity) + stain_color * mask_smooth * severity

        return torch.clamp(stained, 0, 1)


class ContrastShift(nn.Module):
    """Adjusts contrast and brightness."""

    def __init__(self):
        super().__init__()

    def forward(self, image: Tensor, severity: float = 0.5) -> Tensor:
        """
        Apply contrast and brightness shift.

        Args:
            image: (B, C, H, W) image tensor
            severity: Intensity (0-1)

        Returns:
            adjusted: Same shape as input
        """
        # Random brightness shift
        brightness = np.random.uniform(-severity * 0.3, severity * 0.3)

        # Random contrast shift
        contrast = np.random.uniform(1 - severity * 0.3, 1 + severity * 0.3)

        # Apply: output = contrast * (input - 0.5) + 0.5 + brightness
        adjusted = contrast * (image - 0.5) + 0.5 + brightness

        return torch.clamp(adjusted, 0, 1)


class SealOverlay(nn.Module):
    """Adds seal-like circular overlays (common in Asian documents)."""

    def __init__(self):
        super().__init__()

    def forward(self, image: Tensor, severity: float = 0.5) -> Tensor:
        """
        Overlay seal patterns.

        Args:
            image: (B, C, H, W) image tensor
            severity: Number and size of seals

        Returns:
            sealed: Same shape as input
        """
        b, c, h, w = image.shape
        device = image.device

        sealed = image.clone()

        # Number of seals
        num_seals = int(severity * 3) + 1

        for _ in range(num_seals):
            # Random seal position
            cx = torch.randint(w // 4, 3 * w // 4, (1,)).item()
            cy = torch.randint(h // 4, 3 * h // 4, (1,)).item()

            # Seal size
            radius = int(severity * 50) + 20

            # Create circular seal mask
            yy, xx = torch.meshgrid(
                torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
            )
            dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

            # Seal is typically red/dark
            seal_mask = torch.exp(-dist / (radius / 3))
            seal_mask = seal_mask.unsqueeze(0).unsqueeze(0)

            # Seal color: reddish tint
            seal_color = torch.tensor([0.8, 0.2, 0.2], device=device).view(1, 3, 1, 1)

            # Blend seal
            sealed = (
                sealed * (1 - seal_mask * severity * 0.5) + seal_color * seal_mask * severity * 0.5
            )

        return torch.clamp(sealed, 0, 1)


class WarpingDistortion(nn.Module):
    """Applies perspective and grid distortions."""

    def __init__(self):
        super().__init__()

    def forward(self, image: Tensor, severity: float = 0.5) -> Tensor:
        """
        Apply warping distortion.

        Args:
            image: (B, C, H, W) image tensor
            severity: Distortion intensity (0-1)

        Returns:
            warped: Same shape as input
        """
        if KORNIA_AVAILABLE:
            # Use kornia for differentiable warping
            b, c, h, w = image.shape

            # Create random perspective transformation
            strength = severity * 0.15

            # Random corner offsets
            pts1 = torch.tensor(
                [[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]],
                dtype=torch.float32,
                device=image.device,
            )

            pts2 = pts1 + torch.randn(4, 2, device=image.device) * strength * max(h, w)

            # Compute perspective transform
            m = K.geometry.transform.get_perspective_transform(pts1.unsqueeze(0), pts2.unsqueeze(0))

            # Apply warp
            warped = K.geometry.transform.warp_perspective(image, m, (h, w))
            return warped
        else:
            # Fallback: apply simple grid distortion
            b, c, h, w = image.shape
            grid = torch.nn.functional.affine_grid(
                torch.eye(2, 3, device=image.device).unsqueeze(0), [b, c, h, w]
            )

            # Perturb grid
            if severity > 0:
                grid = grid + torch.randn_like(grid) * severity * 0.05

            warped = torch.nn.functional.grid_sample(image, grid, align_corners=True)
            return warped


class BinarizationNoise(nn.Module):
    """Simulates binarization errors."""

    def __init__(self):
        super().__init__()

    def forward(self, image: Tensor, severity: float = 0.5) -> Tensor:
        """
        Apply binarization noise.

        Args:
            image: (B, C, H, W) image tensor
            severity: Noise level (0-1)

        Returns:
            noisy: Same shape as input
        """
        # Simple binarization: convert to binary then add noise
        b, c, h, w = image.shape

        # Binarize
        threshold = np.random.uniform(0.3, 0.7)
        binary = (image > threshold).float()

        # Add noise: flip random pixels
        noise_prob = severity * 0.2
        noise_mask = torch.rand_like(binary) < noise_prob

        noisy = binary.clone()
        noisy[noise_mask] = 1 - noisy[noise_mask]

        return noisy


class AdditiveNoise(nn.Module):
    """Adds Gaussian noise."""

    def __init__(self):
        super().__init__()

    def forward(self, image: Tensor, severity: float = 0.5) -> Tensor:
        """
        Add Gaussian noise.

        Args:
            image: (B, C, H, W) image tensor
            severity: Noise standard deviation

        Returns:
            noisy: Same shape as input
        """
        noise_std = severity * 0.1
        noise = torch.randn_like(image) * noise_std
        noisy = image + noise
        return torch.clamp(noisy, 0, 1)


# ==================== Dataset Integration ====================


def create_degradation_pipeline(
    degradation_types: List[str] = None,
    severity_range: Tuple[float, float] = (0.2, 0.8),
) -> DegradationPipeline:
    """
    Factory function to create degradation pipeline.

    Args:
        degradation_types: List of degradation type names
        severity_range: (min_severity, max_severity) tuple

    Returns:
        Configured DegradationPipeline
    """
    if degradation_types is None:
        degradation_types = ["blur", "stain", "contrast", "seal"]

    config = DegradationConfig(
        degradation_types=[DegradationType(t) for t in degradation_types],
        severity_min=severity_range[0],
        severity_max=severity_range[1],
    )

    return DegradationPipeline(config)
