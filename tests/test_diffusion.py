import pytest
import torch

from fashion_mnist_gen.diffusion import (
    ConditionalDDPM,
    ConditionalUNet,
    sinusoidal_embedding,
)


def test_sinusoidal_embedding_has_requested_shape() -> None:
    embedding = sinusoidal_embedding(torch.tensor([0, 1, 999]), dimension=7)

    assert embedding.shape == (3, 7)
    assert torch.isfinite(embedding).all()


def test_conditional_unet_predicts_noise_for_28_by_28_images() -> None:
    model = ConditionalUNet(base_channels=16)
    images = torch.randn(2, 1, 28, 28)
    timesteps = torch.tensor([1, 9])
    labels = torch.tensor([2, 10])

    prediction = model(images, timesteps, labels)

    assert prediction.shape == images.shape


def test_diffusion_training_loss_backpropagates() -> None:
    diffusion = ConditionalDDPM(num_timesteps=20, base_channels=16)
    images = torch.rand(2, 1, 28, 28) * 2 - 1
    labels = torch.tensor([1, 8])

    loss = diffusion.training_loss(images, labels, generator=torch.Generator().manual_seed(7))
    loss.backward()

    assert torch.isfinite(loss)
    assert diffusion.model.output.weight.grad is not None


def test_ddpm_posterior_mean_matches_closed_form_coefficients() -> None:
    diffusion = ConditionalDDPM(
        num_timesteps=3,
        beta_start=0.1,
        beta_end=0.3,
        base_channels=16,
    )
    noisy_images = torch.full((1, 1, 2, 2), 2.0)
    predicted_clean = torch.full_like(noisy_images, 4.0)
    timesteps = torch.tensor([1])
    beta = diffusion.betas[1]
    alpha = diffusion.alphas[1]
    alpha_bar = diffusion.alpha_bars[1]
    alpha_bar_previous = diffusion.alpha_bars_previous[1]
    expected = (
        beta * alpha_bar_previous.sqrt() / (1 - alpha_bar) * predicted_clean
        + alpha.sqrt()
        * (1 - alpha_bar_previous)
        / (1 - alpha_bar)
        * noisy_images
    )

    posterior = diffusion.posterior_mean(noisy_images, predicted_clean, timesteps)

    assert torch.allclose(posterior, expected)


def test_ddim_sampling_is_reproducible_and_returns_images_in_unit_range() -> None:
    diffusion = ConditionalDDPM(num_timesteps=20, base_channels=16)
    labels = torch.tensor([0, 9])
    first_generator = torch.Generator().manual_seed(12)
    second_generator = torch.Generator().manual_seed(12)

    first = diffusion.sample(
        labels, num_inference_steps=5, generator=first_generator
    )
    second = diffusion.sample(
        labels, num_inference_steps=5, generator=second_generator
    )

    assert first.shape == (2, 1, 28, 28)
    assert torch.equal(first, second)
    assert torch.all((first >= 0) & (first <= 1))


def test_ddpm_requires_every_timestep() -> None:
    diffusion = ConditionalDDPM(num_timesteps=20, base_channels=16)

    with pytest.raises(ValueError, match="every configured timestep"):
        diffusion.sample(
            torch.tensor([0]), method="ddpm", num_inference_steps=5
        )


def test_ddpm_ancestral_sampler_runs_full_schedule() -> None:
    diffusion = ConditionalDDPM(num_timesteps=4, base_channels=16)

    images = diffusion.sample(
        torch.tensor([3]),
        method="ddpm",
        num_inference_steps=4,
        generator=torch.Generator().manual_seed(4),
    )

    assert images.shape == (1, 1, 28, 28)
    assert torch.isfinite(images).all()


def test_classifier_free_guidance_runs_with_ddim() -> None:
    diffusion = ConditionalDDPM(num_timesteps=10, base_channels=16)

    images = diffusion.sample(
        torch.tensor([2]),
        num_inference_steps=3,
        guidance_scale=1.5,
        generator=torch.Generator().manual_seed(5),
    )

    assert images.shape == (1, 1, 28, 28)
    assert torch.isfinite(images).all()
