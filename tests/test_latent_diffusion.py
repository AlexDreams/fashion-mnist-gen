import pytest
import torch

from fashion_mnist_gen.latent_diffusion import LatentDiffusion
from fashion_mnist_gen.vae import ConditionalVAE


def make_model(timesteps: int = 20) -> LatentDiffusion:
    torch.manual_seed(0)
    vae = ConditionalVAE(latent_dim=4, hidden_dim=16)
    return LatentDiffusion(
        vae, torch.zeros(4), torch.ones(4), num_timesteps=timesteps,
        hidden_dim=32, num_blocks=2,
    )


def test_latent_diffusion_samples_images_with_both_samplers() -> None:
    model = make_model()
    labels = torch.tensor([0, 3, 9])

    ddim = model.sample(labels, method="ddim", num_inference_steps=5, guidance_scale=2.0)
    ddpm = model.sample(labels, method="ddpm", num_inference_steps=20, guidance_scale=1.0)

    for images in (ddim, ddpm):
        assert images.shape == (3, 1, 28, 28)
        assert torch.all((images >= 0) & (images <= 1))


def test_latent_diffusion_sampling_is_reproducible() -> None:
    model = make_model()
    labels = torch.tensor([1, 2])
    first = model.sample(labels, generator=torch.Generator().manual_seed(5),
                         num_inference_steps=5)
    second = model.sample(labels, generator=torch.Generator().manual_seed(5),
                          num_inference_steps=5)

    assert torch.equal(first, second)


def test_latent_diffusion_trains_only_the_denoiser() -> None:
    model = make_model()
    loss = model.training_loss(torch.randn(8, 4), torch.randint(0, 10, (8,)))
    loss.backward()

    assert torch.isfinite(loss)
    assert model.model.output.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.vae.parameters())


def test_latent_diffusion_checkpoint_round_trip() -> None:
    model = make_model()
    restored = LatentDiffusion.from_checkpoint(model.checkpoint_payload())
    labels = torch.tensor([4])

    expected = model.sample(labels, generator=torch.Generator().manual_seed(1),
                            num_inference_steps=4)
    actual = restored.sample(labels, generator=torch.Generator().manual_seed(1),
                             num_inference_steps=4)

    assert torch.allclose(expected, actual)


def test_latent_diffusion_rejects_partial_ddpm() -> None:
    with pytest.raises(ValueError, match="every configured timestep"):
        make_model().sample(torch.tensor([0]), method="ddpm", num_inference_steps=5)
