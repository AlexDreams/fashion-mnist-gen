import pytest
import torch

from fashion_mnist_gen.metrics import (
    fashion_kid,
    feature_diversity,
    relative_feature_coverage,
)


def test_fashion_kid_distinguishes_identical_and_shifted_features() -> None:
    features = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    )

    same_distribution = fashion_kid(features, features)
    shifted_distribution = fashion_kid(features, features + 2)

    assert torch.isfinite(torch.tensor(same_distribution))
    assert same_distribution < shifted_distribution


def test_feature_diversity_is_zero_for_identical_vectors() -> None:
    features = torch.ones(4, 3)

    assert feature_diversity(features) == pytest.approx(0.0, abs=1e-6)


def test_relative_feature_coverage_recognizes_exact_real_samples() -> None:
    real = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    assert relative_feature_coverage(real, real) == 1.0


def test_pixel_diversity_detects_duplicates() -> None:
    from fashion_mnist_gen.metrics import pixel_diversity

    varied = torch.rand(16, 1, 28, 28, generator=torch.Generator().manual_seed(0))
    duplicated = varied[:1].repeat(16, 1, 1, 1)

    assert pixel_diversity(duplicated) == pytest.approx(0.0, abs=1e-4)
    assert pixel_diversity(varied) > 1.0


def test_class_coverage_balanced_and_collapsed() -> None:
    from fashion_mnist_gen.metrics import class_coverage

    balanced = class_coverage(torch.arange(10).repeat(5), num_classes=10)
    collapsed = class_coverage(torch.zeros(50, dtype=torch.long), num_classes=10)

    assert balanced["coverage"] == 1.0
    assert balanced["normalized_entropy"] == pytest.approx(1.0)
    assert collapsed["coverage"] == pytest.approx(0.1)
    assert collapsed["normalized_entropy"] == pytest.approx(0.0)


def test_score_sample_run_flags_collapsed_class() -> None:
    from evaluate_protocol import score_sample_run
    from fashion_mnist_gen.metrics import pixel_diversity

    generator = torch.Generator().manual_seed(1)
    real_images = [torch.rand(8, 1, 28, 28, generator=generator) for _ in range(10)]
    generated_images = [images.clone() for images in real_images]
    generated_images[3] = real_images[3][:1].repeat(8, 1, 1, 1)
    features = [torch.randn(8, 4, generator=generator) for _ in range(10)]
    predictions = [torch.full((8,), class_id, dtype=torch.long) for class_id in range(10)]

    result = score_sample_run(
        features,
        features,
        predictions,
        generated_images=generated_images,
        real_pixel_diversity=[pixel_diversity(images) for images in real_images],
    )

    assert result["class_fidelity"] == 1.0
    assert result["class_coverage"] == 1.0
    assert result["collapsed_classes"] == ["Dress"]
    assert result["collapsed_class_count"] == 1


def test_parse_diffusion_config() -> None:
    from evaluate_protocol import diffusion_key, parse_diffusion_config

    config = parse_diffusion_config("DDIM:50:1.5")

    assert config == {"method": "ddim", "steps": 50, "guidance_scale": 1.5}
    assert diffusion_key(config) == "ddpm_ddim50_g1.5"
    with pytest.raises(ValueError):
        parse_diffusion_config("ddim:50")
