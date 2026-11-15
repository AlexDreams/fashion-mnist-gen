import torch

from fashion_mnist_gen.diffusion import ConditionalDDPM
from fashion_mnist_gen.gan import ConditionalGenerator
from fashion_mnist_gen.latent_diffusion import LatentDiffusion
from fashion_mnist_gen.vae import ConditionalVAE
from fashion_mnist_gen import visualize


def test_forward_noising_goes_from_image_to_noise():
    images = torch.rand(2, 1, 28, 28)
    steps = visualize.forward_noising(images, [0, 999], generator=torch.Generator().manual_seed(0))
    assert len(steps) == 2 and steps[0].shape == images.shape
    assert (steps[0] - images).abs().max() < 0.05
    assert (steps[1] - images).abs().mean() > 0.2


def test_ddim_trajectory_is_recorded_and_ends_on_output():
    model = ConditionalDDPM(num_timesteps=20, base_channels=8)
    trajectory = []
    images = model.sample(torch.tensor([0, 3]), num_inference_steps=5, trajectory=trajectory,
                          generator=torch.Generator().manual_seed(0))
    assert [step["timestep"] for step in trajectory] == sorted(
        {step["timestep"] for step in trajectory}, reverse=True)
    assert len(trajectory) == 5 and trajectory[0]["noisy"].shape == images.shape
    assert len(visualize.pick_steps(trajectory, 3)) == 3


def test_latent_trajectory_is_decoded_to_images():
    vae = ConditionalVAE()
    ldm = LatentDiffusion(vae, torch.zeros(16), torch.ones(16), num_timesteps=20,
                          hidden_dim=32, num_blocks=1)
    trajectory = []
    ldm.sample(torch.tensor([1]), num_inference_steps=4, trajectory=trajectory)
    assert len(trajectory) == 4 and trajectory[-1]["clean"].shape == (1, 1, 28, 28)


def test_vae_latent_views():
    vae = ConditionalVAE().eval()
    codes = visualize.latent_codes(vae, torch.rand(50, 1, 28, 28), torch.randint(10, (50,)))
    assert codes.shape == (50, 16)
    assert visualize.project_2d(codes).shape == (50, 2)
    assert visualize.latent_map(vae, codes, 3, size=4).shape == (16, 1, 28, 28)
    assert visualize.interpolate(vae, "vae", 2, 1, 2, steps=6).shape == (6, 1, 28, 28)


def test_gan_views():
    generator = ConditionalGenerator().eval()
    path = visualize.interpolate(generator, "gan", 0, 1, 2, steps=5)
    assert path.shape == (5, 1, 28, 28)
    grid = visualize.same_noise_all_classes(generator, "gan", seed=0, rows=2)
    assert grid.shape == (20, 1, 28, 28)
