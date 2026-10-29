"""
Comprehensive test suite for Phase 5-8 training methods.

Tests all 6 advanced training approaches:
1. GL-HPN: Global-Local Hierarchical Retrieval
2. DTRNet: Dual Text-Radical Decoding
3. Degradation-Aware: Synthetic degradation pipeline
4. Trajectory: Online handwriting trajectory training
5. Multi-Granular: Hierarchical contrastive learning
6. Restoration Pipeline: Detection→Restoration→Classification
"""

import pytest
import torch
import torch.nn as nn

from src.lib.degradation import (
    ContrastShift,
    DegradationConfig,
    DegradationPipeline,
    DegradationType,
    GaussianBlur,
    StainDegradation,
)
from src.lib.dual_decoder import (
    DTRNetModule,
    FakeCharacterDetector,
    RadicalDecoder,
    TextDecoder,
)
from src.lib.granular_encoders import (
    FinegrainedDecoupledContrastiveLoss,
    RadicalEncoder,
    StrokeEncoder,
    TextEncoder,
)
from src.lib.hierarchical_retrieval import (
    GlobalBranchEncoder,
    LocalBranchEncoder,
    create_glhpn_retriever,
)
from src.lib.restoration_pipeline import (
    PipelineConfig,
    RestorationHead,
    SimpleYOLODetector,
    create_pipeline,
)
from src.lib.trajectory_processing import (
    HybridTrajectoryVisionModel,
    StrokeExtractor,
    StrokeRNNEncoder,
    TrajectoryConfig,
    TrajectoryNormalizer,
)

# ============================================================================
# PHASE 1: GL-HPN Tests
# ============================================================================


class TestGLHPN:
    """Test GL-HPN global-local hierarchical retrieval."""

    def test_global_branch_encoder(self):
        """Test global branch encoding."""
        encoder = GlobalBranchEncoder(
            input_dim=2048,
            embedding_dim=512,
            num_ids_tokens=1024,
        )

        batch_size = 4
        image_features = torch.randn(batch_size, 2048)
        ids_tokens = torch.randint(0, 1024, (batch_size, 8))

        image_emb, ids_emb = encoder(image_features, ids_tokens)

        assert image_emb.shape == (batch_size, 512)
        assert ids_emb.shape == (batch_size, 512)
        assert torch.allclose(torch.norm(image_emb, dim=1), torch.ones(batch_size), atol=1e-5)

    def test_local_branch_encoder(self):
        """Test local branch encoding."""
        encoder = LocalBranchEncoder(
            patch_dim=256,
            num_patches=16,
            embedding_dim=512,
        )

        batch_size = 4
        patches = torch.randn(batch_size, 16, 256)

        local_features = encoder(patches)

        assert local_features.shape == (batch_size, 16, 512)

    def test_coarse_to_fine_retriever(self):
        """Test full retriever pipeline."""
        retriever = create_glhpn_retriever(
            embedding_dim=512,
            top_k_candidates=100,
        )

        batch_size = 4
        num_candidates = 1000

        query_features = torch.randn(batch_size, 2048)
        candidate_embeddings = torch.randn(num_candidates, 512)
        candidate_embeddings = torch.nn.functional.normalize(candidate_embeddings, p=2, dim=1)

        indices, scores = retriever(
            query_features,
            candidate_embeddings=candidate_embeddings,
            top_k=10,
        )

        assert indices.shape == (batch_size, 10)
        assert scores.shape == (batch_size, 10)


# ============================================================================
# PHASE 2: DTRNet Tests
# ============================================================================


class TestDTRNet:
    """Test DTRNet dual text-radical decoding."""

    def test_text_decoder(self):
        """Test text decoder."""
        decoder = TextDecoder(
            input_dim=512,
            hidden_dim=1024,
            num_classes=3036,
        )

        batch_size = 4
        seq_len = 20
        features = torch.randn(batch_size, seq_len, 512)

        logits = decoder(features)

        assert logits.shape == (batch_size, seq_len, 3036)

    def test_radical_decoder(self):
        """Test radical decoder."""
        decoder = RadicalDecoder(
            input_dim=512,
            num_ids_tokens=64,
            max_ids_length=8,
        )

        batch_size = 4
        features = torch.randn(batch_size, 512)

        ids_logits, ids_confidence = decoder(features)

        assert ids_logits.shape == (batch_size, 8, 64)
        assert ids_confidence.shape == (batch_size,)
        assert (ids_confidence >= 0).all() and (ids_confidence <= 1).all()

    def test_fake_character_detector(self):
        """Test fake character detection."""
        detector = FakeCharacterDetector(text_num_classes=3036)

        batch_size = 4
        text_logits = torch.randn(batch_size, 3036)
        ids_confidence = torch.rand(batch_size)
        structure_agreement = torch.rand(batch_size)

        is_fake, suspicion = detector(text_logits, ids_confidence, structure_agreement)

        assert is_fake.shape == (batch_size,)
        assert suspicion.shape == (batch_size,)
        assert (suspicion >= 0).all() and (suspicion <= 1).all()

    def test_dtrnet_module(self):
        """Test complete DTRNet module."""
        module = DTRNetModule(
            input_dim=512,
            num_character_classes=3036,
        )

        batch_size = 4
        seq_len = 20
        features = torch.randn(batch_size, seq_len, 512)

        outputs = module(features)

        assert "text_logits" in outputs
        assert "ids_logits" in outputs
        assert "is_fake" in outputs
        assert outputs["text_logits"].shape == (batch_size, seq_len, 3036)


# ============================================================================
# PHASE 3: Degradation Tests
# ============================================================================


class TestDegradation:
    """Test degradation pipeline."""

    def test_gaussian_blur(self):
        """Test Gaussian blur degradation."""
        blur = GaussianBlur()

        batch_size = 2
        image = torch.rand(batch_size, 3, 64, 64)

        blurred = blur(image, severity=0.5)

        assert blurred.shape == image.shape
        assert (blurred >= 0).all() and (blurred <= 1).all()

    def test_stain_degradation(self):
        """Test stain degradation."""
        stain = StainDegradation()

        image = torch.rand(1, 3, 64, 64)
        degraded = stain(image, severity=0.5)

        assert degraded.shape == image.shape
        assert (degraded >= 0).all() and (degraded <= 1).all()

    def test_contrast_shift(self):
        """Test contrast shift."""
        contrast = ContrastShift()

        image = torch.rand(2, 3, 64, 64)
        adjusted = contrast(image, severity=0.5)

        assert adjusted.shape == image.shape

    def test_full_pipeline(self):
        """Test complete degradation pipeline."""
        config = DegradationConfig(
            degradation_types=[
                DegradationType.BLUR,
                DegradationType.STAIN,
                DegradationType.CONTRAST,
            ]
        )
        pipeline = DegradationPipeline(config)

        image = torch.rand(2, 3, 128, 128)
        degraded = pipeline(image, severity=0.6)

        assert degraded.shape == image.shape


# ============================================================================
# PHASE 4: Trajectory Tests
# ============================================================================


class TestTrajectory:
    """Test online handwriting trajectory training."""

    def test_trajectory_normalizer(self):
        """Test trajectory normalization."""
        config = TrajectoryConfig()
        normalizer = TrajectoryNormalizer(config)

        # Create sample coordinates: (num_points, 4): x, y, pressure, timestamp
        num_points = 50
        coordinates = torch.randn(num_points, 4)
        coordinates[:, 2:] = torch.abs(coordinates[:, 2:])  # Ensure positive
        coordinates[:, 3] = torch.cumsum(torch.abs(coordinates[:, 3]), dim=0)  # Cumulative time

        normalized = normalizer(coordinates)

        assert normalized.shape[0] == num_points
        assert normalized.shape[1] >= 3  # At least x, y, speed

    def test_stroke_extractor(self):
        """Test stroke extraction."""
        extractor = StrokeExtractor()

        # Create coordinates for 3 strokes
        coordinates = torch.randn(100, 2)

        strokes, stroke_ids = extractor(coordinates)

        assert len(strokes) > 0
        assert stroke_ids.shape == (100,)

    def test_stroke_rnn_encoder(self):
        """Test stroke RNN encoder."""
        encoder = StrokeRNNEncoder(
            input_dim=4,
            embedding_dim=128,
            hidden_dim=256,
        )

        # Create sample strokes
        strokes = [torch.randn(30, 4) for _ in range(5)]

        stroke_embs, char_emb = encoder(strokes, max_strokes=10)

        assert stroke_embs.shape == (5, 128)
        assert char_emb.shape == (128,)

    def test_hybrid_trajectory_vision_model(self):
        """Test hybrid trajectory-vision model."""
        model = HybridTrajectoryVisionModel(
            image_feature_dim=512,
            trajectory_embedding_dim=128,
            num_classes=3036,
        )

        image_feat = torch.randn(4, 512)
        traj_emb = torch.randn(4, 128)

        outputs = model(image_feat, traj_emb)

        assert "logits" in outputs
        assert outputs["logits"].shape == (4, 3036)


# ============================================================================
# PHASE 5: Multi-Granular Tests
# ============================================================================


class TestMultiGranular:
    """Test multi-granular contrastive learning."""

    def test_stroke_encoder(self):
        """Test stroke encoder."""
        encoder = StrokeEncoder(
            input_dim=256,
            embedding_dim=128,
            num_classes=500,
        )

        features = torch.randn(4, 256)
        outputs = encoder(features)

        assert outputs["embedding"].shape == (4, 128)
        assert outputs["logits"].shape == (4, 500)

    def test_radical_encoder(self):
        """Test radical encoder."""
        encoder = RadicalEncoder(
            input_dim=256,
            embedding_dim=256,
            num_classes=214,
        )

        features = torch.randn(4, 256)
        outputs = encoder(features)

        assert outputs["embedding"].shape == (4, 256)
        assert outputs["logits"].shape == (4, 214)

    def test_text_encoder(self):
        """Test text encoder."""
        encoder = TextEncoder(
            vocab_size=5000,
            embedding_dim=512,
        )

        token_ids = torch.randint(0, 5000, (4, 32))
        embeddings = encoder(token_ids)

        assert embeddings.shape == (4, 512)

    def test_contrastive_loss(self):
        """Test contrastive loss."""
        loss_fn = FinegrainedDecoupledContrastiveLoss(temperature=0.07)

        image_emb = torch.randn(4, 512)
        text_emb = torch.randn(4, 512)

        loss = loss_fn(image_emb, text_emb)

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_cross_level_consistency_loss(self):
        """Test CrossLevelConsistencyLoss projects across 128 -> 256 -> 512 dimensions."""
        from src.lib.granular_encoders import (
            CrossLevelConsistencyLoss,
            MultiGranularityFusionModule,
        )

        consistency = CrossLevelConsistencyLoss(stroke_dim=128, radical_dim=256, character_dim=512)
        stroke_emb = torch.randn(4, 128, requires_grad=True)
        radical_emb = torch.randn(4, 256, requires_grad=True)
        char_emb = torch.randn(4, 512, requires_grad=True)

        loss = consistency(stroke_emb, radical_emb, char_emb)
        assert loss.ndim == 0
        assert loss.item() >= 0
        loss.backward()
        assert stroke_emb.grad is not None

        fusion = MultiGranularityFusionModule(stroke_dim=128, radical_dim=256, character_dim=512)
        fused_out = fusion(stroke_emb.detach(), radical_emb.detach(), char_emb.detach())
        assert fused_out.shape == (4, 512)

    def test_ids_validator_and_igca(self):
        """Test IDSValidator and IDSGuidedConfidenceAdjustment from dual_decoder."""
        from src.lib.dual_decoder import IDSGuidedConfidenceAdjustment, IDSValidator

        validator = IDSValidator()
        validator.add_character_ids(0, [0, 10, 11])
        pred_logits = torch.zeros(3, 16)
        pred_logits[0, 0] = 5.0
        pred_logits[1, 10] = 5.0
        pred_logits[2, 11] = 5.0
        conf = validator.validate_ids(0, pred_logits)
        assert isinstance(conf, float)
        assert conf == 1.0

        igca = IDSGuidedConfidenceAdjustment()
        logits = torch.randn(4, 20)
        ids_conf = torch.tensor([0.9, 0.2, 0.8, 0.5])
        agreement = torch.tensor([0.85, 0.1, 0.9, 0.4])
        adjusted = igca(logits, ids_conf, agreement)
        assert adjusted.shape == (4, 20)

    def test_restoration_preprocessor_all_methods(self):
        """Test TraditionalBinarization (otsu, kmeans, adaptive) and RestorationPreprocessor."""
        from src.lib.restoration import (
            RestorationPreprocessor,
            TraditionalBinarization,
            create_restoration_preprocessor,
        )

        img = torch.rand(2, 3, 32, 32)
        for method in ["otsu", "kmeans", "adaptive"]:
            binarizer = TraditionalBinarization(method=method)
            out = binarizer(img)
            assert out.shape == (2, 1, 32, 32)
            assert ((out == 0) | (out == 1)).all()

        preprocessor = create_restoration_preprocessor(
            method="otsu", with_seal_removal=True, with_denoising=True
        )
        assert isinstance(preprocessor, RestorationPreprocessor)
        result = preprocessor(img)
        assert "restored" in result
        assert "seal_mask" in result
        assert result["restored"].shape[0] == 2


class TestLightningModulesAllPhases:
    """Test all Lightning modules in src/lib/lightning_module.py."""

    def test_kanji_recognition_lightning_module(self):
        """Test base KanjiRecognitionLightningModule training, validation, and optimizer setup."""
        from src.lib.config import CNNConfig
        from src.lib.lightning_module import KanjiRecognitionLightningModule
        from src.lib.models import LightweightKanjiNet

        cfg = CNNConfig(
            num_classes=10, epochs=5, batch_size=4, optimizer="adamw", scheduler="cosine"
        )
        model = LightweightKanjiNet(num_classes=10)
        pl_mod = KanjiRecognitionLightningModule(model=model, config=cfg)

        images = torch.randn(4, 1, 64, 64)
        labels = torch.randint(0, 10, (4,))
        loss = pl_mod.training_step((images, labels), 0)
        assert loss.ndim == 0 and loss.item() > 0
        pl_mod.validation_step((images, labels), 0)
        pl_mod.test_step((images, labels), 0)

        opt_cfg = pl_mod.configure_optimizers()
        assert "optimizer" in opt_cfg
        assert "lr_scheduler" in opt_cfg

    def test_phase_1_to_6_lightning_modules(self):
        """Test GLHPN, DTRNet, DegradationAware, Trajectory, MultiGranular, and RestorationPipeline Lightning modules."""
        from src.lib.config import (
            DegradationAwareConfig,
            DTRNetConfig,
            GLHPNConfig,
            MultiGranularConfig,
            RestorationPipelineConfig,
            TrajectoryConfig,
        )
        from src.lib.degradation import create_degradation_pipeline
        from src.lib.dual_decoder import DTRNetModule
        from src.lib.granular_encoders import (
            FinegrainedDecoupledContrastiveLoss,
            create_multigranular_encoders,
        )
        from src.lib.hierarchical_retrieval import create_glhpn_retriever
        from src.lib.lightning_module import (
            DegradationAwareLightningModule,
            DTRNetLightningModule,
            GLHPNLightningModule,
            MultiGranularLightningModule,
            RestorationPipelineLightningModule,
            TrajectoryLightningModule,
        )
        from src.lib.models import LightweightKanjiNet
        from src.lib.restoration import create_restoration_preprocessor
        from src.lib.restoration_pipeline import create_pipeline
        from src.lib.trajectory_processing import HybridTrajectoryVisionModel

        # Phase 1: GLHPN
        glhpn_cfg = GLHPNConfig(embedding_dim=64, top_k_candidates=10, epochs=2)
        backbone = LightweightKanjiNet(num_classes=10)
        retriever = create_glhpn_retriever(
            backbone_output_dim=10, embedding_dim=64, top_k_candidates=5
        )
        m1 = GLHPNLightningModule(backbone, retriever, glhpn_cfg)
        loss1 = m1.training_step((torch.randn(2, 1, 64, 64), torch.randint(0, 10, (2,))), 0)
        assert loss1.item() > 0
        assert "optimizer" in m1.configure_optimizers()

        # Phase 2: DTRNet
        dtr_cfg = DTRNetConfig(text_hidden_dim=64, radical_hidden_dim=32, epochs=2)
        dtrnet = DTRNetModule(
            input_dim=32, text_hidden_dim=64, radical_hidden_dim=32, num_character_classes=10
        )
        m2 = DTRNetLightningModule(dtrnet, dtr_cfg)
        loss2 = m2.training_step((torch.randn(2, 4, 32), torch.randint(0, 10, (2,))), 0)
        assert loss2.item() > 0

        # Phase 3: DegradationAware
        deg_cfg = DegradationAwareConfig(restoration_enabled=True, epochs=2)
        m3 = DegradationAwareLightningModule(
            LightweightKanjiNet(num_classes=10),
            create_degradation_pipeline(degradation_types=["blur"]),
            create_restoration_preprocessor(),
            deg_cfg,
        )
        loss3 = m3.training_step((torch.rand(2, 1, 64, 64), torch.randint(0, 10, (2,))), 0)
        assert loss3.item() > 0

        # Phase 4: Trajectory
        traj_cfg = TrajectoryConfig(max_strokes=10, trajectory_embedding_dim=32, epochs=2)
        hybrid = HybridTrajectoryVisionModel(
            image_feature_dim=512, trajectory_embedding_dim=32, num_classes=10
        )
        m4 = TrajectoryLightningModule(hybrid, traj_cfg)
        loss4 = m4.training_step(
            (torch.randn(2, 1, 32, 32), torch.randn(2, 32), torch.randint(0, 10, (2,))), 0
        )
        assert loss4.item() > 0

        # Phase 5: MultiGranular
        mg_cfg = MultiGranularConfig(epochs=2)
        encoders = create_multigranular_encoders()
        m5 = MultiGranularLightningModule(encoders, FinegrainedDecoupledContrastiveLoss(), mg_cfg)
        loss5 = m5.training_step(
            (
                torch.randn(2, 256),
                torch.randn(2, 256),
                torch.randn(2, 512),
                torch.randint(0, 100, (2, 8)),
                torch.randint(0, 10, (2,)),
            ),
            0,
        )
        assert loss5.item() > 0

        # Phase 6: RestorationPipeline
        rp_cfg = RestorationPipelineConfig(epochs=2)
        rgb_backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 512),
        )
        pipe, pipe_trainer = create_pipeline(rgb_backbone)
        m6 = RestorationPipelineLightningModule(pipe, pipe_trainer, rp_cfg)
        det_out = pipe.detector(torch.rand(2, 3, 64, 64))
        loss6 = m6.training_step(
            (
                torch.rand(2, 3, 64, 64),
                torch.zeros_like(det_out),
                torch.rand(2, 3, 64, 64),
                torch.randint(0, 10, (2,)),
            ),
            0,
        )
        assert loss6.item() > 0


# ============================================================================
# PHASE 6: Restoration Pipeline Tests
# ============================================================================


class TestRestorationPipeline:
    """Test end-to-end restoration pipeline."""

    def test_detector(self):
        """Test YOLO-style detector."""
        detector = SimpleYOLODetector(num_classes=2, num_anchors=3)

        image = torch.randn(2, 3, 256, 256)
        detections = detector(image)

        assert detections.shape[0] == 2
        assert detections.shape[2] == 7  # 5 + 2 classes

    def test_restoration_head(self):
        """Test restoration head."""
        restorer = RestorationHead(input_channels=3)

        image = torch.rand(2, 3, 128, 128)
        restored = restorer(image)

        assert restored.shape == image.shape
        assert (restored >= 0).all() and (restored <= 1).all()

    def test_full_pipeline(self):
        """Test complete pipeline."""
        # Create simple backbone
        backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 512),
        )

        config = PipelineConfig()
        pipeline, trainer = create_pipeline(backbone, config)

        image = torch.rand(2, 3, 128, 128)
        outputs = pipeline(image, return_intermediate=True)

        assert "logits" in outputs
        assert outputs["logits"].shape == (2, 3036)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for all methods."""

    def test_all_configs_load(self):
        """Test that all configs load without errors."""
        from src.lib.config import (
            DegradationAwareConfig,
            DTRNetConfig,
            GLHPNConfig,
            MultiGranularConfig,
            RestorationPipelineConfig,
            TrajectoryConfig,
        )

        configs = [
            GLHPNConfig(),
            DTRNetConfig(),
            DegradationAwareConfig(),
            TrajectoryConfig(),
            MultiGranularConfig(),
            RestorationPipelineConfig(),
        ]

        for cfg in configs:
            assert cfg.to_dict() is not None

    def test_device_compatibility(self):
        """Test GPU/CPU device handling."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        encoder = GlobalBranchEncoder()
        encoder = encoder.to(device)

        features = torch.randn(2, 2048, device=device)
        image_emb, ids_emb = encoder(features)

        assert image_emb.device == device
        assert ids_emb.device == device


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
