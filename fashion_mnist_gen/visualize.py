"""Helpers to show what each model does (requested after the progress review).

* diffusion: forward noising q(x_t | x_0) and the reverse trajectory
  (noisy x_t and predicted x_0 at selected steps);
* VAE: 2-D projection of the latent codes and decoded latent maps;
* GAN: interpolation in noise space and "same noise, every class" grids.
"""

from typing import Any, Sequence

import torch


def linear_alpha_bars(
    num_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02
) -> torch.Tensor:
    """Cumulative products of the linear schedule used by our diffusion models."""
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    return torch.cumprod(1 - betas, dim=0)


def forward_noising(
    images: torch.Tensor,
    timesteps: Sequence[int],
    alpha_bars: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> list[torch.Tensor]:
    """Noise ``images`` (in [0, 1]) to each timestep with the SAME noise draw."""
    alpha_bars = linear_alpha_bars() if alpha_bars is None else alpha_bars.cpu()
    clean = images.cpu() * 2 - 1
    noise = torch.randn(clean.shape, generator=generator)
    noisy = []
    for timestep in timesteps:
        alpha_bar = alpha_bars[timestep]
        x_t = alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise
        noisy.append(((x_t.clamp(-1, 1) + 1) / 2))
    return noisy


def pick_steps(trajectory: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """``count`` entries evenly spaced along a trajectory, first and last included."""
    if not trajectory:
        return []
    count = max(1, min(count, len(trajectory)))
    if count == 1:
        return [trajectory[-1]]
    indices = torch.linspace(0, len(trajectory) - 1, count).round().long().tolist()
    return [trajectory[index] for index in indices]


@torch.no_grad()
def latent_codes(
    vae: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor, batch_size: int = 512
) -> torch.Tensor:
    """Encoder means for each image (the VAE latent codes)."""
    device = next(vae.parameters()).device
    codes = []
    for start in range(0, len(images), batch_size):
        mean, _ = vae.encode(
            images[start:start + batch_size].to(device),
            labels[start:start + batch_size].to(device),
        )
        codes.append(mean.cpu())
    return torch.cat(codes)


def pca_basis(codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean, two principal directions (columns) and their standard deviations."""
    mean = codes.mean(0)
    centered = codes - mean
    _, singular_values, components = torch.linalg.svd(centered, full_matrices=False)
    std = singular_values[:2] / (len(codes) - 1) ** 0.5
    return mean, components[:2].T, std


def project_2d(codes: torch.Tensor, method: str = "pca", seed: int = 0) -> torch.Tensor:
    """2-D coordinates of latent codes, by PCA or t-SNE."""
    if method == "pca":
        mean, directions, _ = pca_basis(codes)
        return (codes - mean) @ directions
    if method == "tsne":
        from sklearn.manifold import TSNE

        coordinates = TSNE(n_components=2, init="pca", random_state=seed).fit_transform(
            codes.numpy()
        )
        return torch.from_numpy(coordinates).float()
    raise ValueError("method must be 'pca' or 'tsne'")


@torch.no_grad()
def latent_map(
    vae: torch.nn.Module, codes: torch.Tensor, class_id: int, size: int = 9, span: float = 2.5
) -> torch.Tensor:
    """Decode a size x size grid along the two main latent directions.

    Rows follow direction 2, columns direction 1, from -span to +span standard
    deviations around the mean code. Returns (size * size, 1, 28, 28).
    """
    mean, directions, std = pca_basis(codes)
    steps = torch.linspace(-span, span, size)
    grid = torch.stack([
        mean + a * std[0] * directions[:, 0] + b * std[1] * directions[:, 1]
        for b in steps for a in steps
    ])
    device = next(vae.parameters()).device
    labels = torch.full((len(grid),), class_id, dtype=torch.long, device=device)
    return vae.decode(grid.to(device), labels).cpu()


def slerp(start: torch.Tensor, end: torch.Tensor, fractions: torch.Tensor) -> torch.Tensor:
    """Spherical interpolation, suited to Gaussian noise vectors."""
    cosine = torch.nn.functional.cosine_similarity(start, end, dim=-1).clamp(-1, 1)
    angle = torch.arccos(cosine)
    if angle.abs() < 1e-4:
        return torch.stack([(1 - f) * start + f * end for f in fractions])
    return torch.stack([
        (torch.sin((1 - f) * angle) * start + torch.sin(f * angle) * end) / torch.sin(angle)
        for f in fractions
    ])


@torch.no_grad()
def interpolate(
    model: torch.nn.Module, kind: str, class_id: int, seed_a: int, seed_b: int, steps: int = 10
) -> torch.Tensor:
    """Walk from the input of seed_a to the input of seed_b (VAE latent or GAN noise)."""
    device = next(model.parameters()).device
    size = model.latent_dim if kind == "vae" else model.noise_dim
    start = torch.randn(size, generator=torch.Generator().manual_seed(seed_a))
    end = torch.randn(size, generator=torch.Generator().manual_seed(seed_b))
    path = slerp(start, end, torch.linspace(0, 1, steps)).to(device)
    labels = torch.full((steps,), class_id, dtype=torch.long, device=device)
    images = model.decode(path, labels) if kind == "vae" else model(path, labels)
    return images.cpu()


@torch.no_grad()
def same_noise_all_classes(
    model: torch.nn.Module, kind: str, seed: int, rows: int = 4, num_classes: int = 10
) -> torch.Tensor:
    """rows x num_classes grid: each row shares one input, each column is a class."""
    device = next(model.parameters()).device
    size = model.latent_dim if kind == "vae" else model.noise_dim
    noise = torch.randn(rows, size, generator=torch.Generator().manual_seed(seed))
    noise = noise.repeat_interleave(num_classes, dim=0).to(device)
    labels = torch.arange(num_classes, device=device).repeat(rows)
    images = model.decode(noise, labels) if kind == "vae" else model(noise, labels)
    return images.cpu()
