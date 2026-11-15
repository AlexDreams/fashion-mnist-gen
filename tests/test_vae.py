import pytest
import torch

from fashion_mnist_gen.vae import ConditionalVAE, vae_loss


def test_conditional_vae_forward_returns_reconstruction_and_latent_parameters() -> None:
    model = ConditionalVAE(latent_dim=8)
    images = torch.rand(4, 1, 28, 28)
    labels = torch.tensor([0, 1, 8, 9])

    reconstructions, mean, log_variance = model(images, labels)

    assert reconstructions.shape == images.shape
    assert mean.shape == (4, 8)
    assert log_variance.shape == (4, 8)
    assert torch.all((reconstructions >= 0) & (reconstructions <= 1))


def test_conditional_vae_samples_requested_classes() -> None:
    model = ConditionalVAE(latent_dim=8)
    labels = torch.tensor([2, 5, 9])
    first_generator = torch.Generator().manual_seed(7)
    second_generator = torch.Generator().manual_seed(7)

    samples = model.sample(labels, generator=first_generator)
    repeated_samples = model.sample(labels, generator=second_generator)

    assert samples.shape == (3, 1, 28, 28)
    assert torch.all((samples >= 0) & (samples <= 1))
    assert torch.equal(samples, repeated_samples)


def test_vae_loss_is_finite_and_backpropagates() -> None:
    model = ConditionalVAE(latent_dim=4)
    images = torch.rand(2, 1, 28, 28)
    labels = torch.tensor([1, 7])
    reconstructions, mean, log_variance = model(images, labels)

    loss, reconstruction, kl = vae_loss(
        reconstructions, images, mean, log_variance
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert reconstruction.item() > 0
    assert kl.item() >= 0
    assert model.mean.weight.grad is not None


def test_vae_loss_rejects_negative_beta() -> None:
    with pytest.raises(ValueError, match="beta must be non-negative"):
        vae_loss(
            torch.zeros(1, 1, 28, 28),
            torch.zeros(1, 1, 28, 28),
            torch.zeros(1, 4),
            torch.zeros(1, 4),
            beta=-1,
        )
