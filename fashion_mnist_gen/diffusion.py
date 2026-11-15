"""Class-conditional DDPM with explicit DDPM and DDIM sampling loops."""

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal_embedding(timesteps: torch.Tensor, dimension: int) -> torch.Tensor:
    """Encode integer diffusion steps using sinusoidal features."""
    if timesteps.ndim != 1:
        raise ValueError("timesteps must have shape (batch,)")
    if dimension < 2:
        raise ValueError("embedding dimension must be at least two")

    half_dimension = dimension // 2
    frequencies = torch.exp(
        -math.log(10_000)
        * torch.arange(half_dimension, device=timesteps.device, dtype=torch.float32)
        / max(half_dimension - 1, 1)
    )
    phases = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat((phases.sin(), phases.cos()), dim=1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ConditionalResidualBlock(nn.Module):
    """Residual convolution block modulated by a timestep/class embedding."""

    def __init__(self, input_channels: int, output_channels: int, embedding_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, input_channels)
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, output_channels)
        self.embedding = nn.Linear(embedding_dim, output_channels * 2)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv2d(input_channels, output_channels, 1)
        )

    def forward(self, inputs: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(inputs)))
        scale, shift = self.embedding(F.silu(embedding)).chunk(2, dim=1)
        hidden = self.norm2(hidden)
        hidden = hidden * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv2(F.silu(hidden))
        return hidden + self.skip(inputs)


class ConditionalUNet(nn.Module):
    """Compact U-Net predicting the noise added to 28 x 28 images."""

    def __init__(
        self,
        num_classes: int = 10,
        base_channels: int = 32,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        if base_channels < 8 or base_channels % 8:
            raise ValueError("base_channels must be a positive multiple of eight")
        if embedding_dim < 2:
            raise ValueError("embedding_dim must be at least two")

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.class_embedding = nn.Embedding(num_classes + 1, embedding_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.SiLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )
        self.input = nn.Conv2d(1, base_channels, 3, padding=1)
        self.down_block = ConditionalResidualBlock(
            base_channels, base_channels, embedding_dim
        )
        self.downsample = nn.Conv2d(
            base_channels, base_channels * 2, 3, stride=2, padding=1
        )
        self.middle1 = ConditionalResidualBlock(
            base_channels * 2, base_channels * 2, embedding_dim
        )
        self.middle2 = ConditionalResidualBlock(
            base_channels * 2, base_channels * 2, embedding_dim
        )
        self.upsample = nn.Conv2d(base_channels * 2, base_channels, 3, padding=1)
        self.up_block = ConditionalResidualBlock(
            base_channels * 2, base_channels, embedding_dim
        )
        self.output_norm = nn.GroupNorm(8, base_channels)
        self.output = nn.Conv2d(base_channels, 1, 3, padding=1)

    def forward(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_images.ndim != 4 or noisy_images.shape[1:] != (1, 28, 28):
            raise ValueError("noisy_images must have shape (batch, 1, 28, 28)")
        batch_size = noisy_images.shape[0]
        if timesteps.shape != (batch_size,):
            raise ValueError("timesteps must have shape (batch,)")
        if labels.shape != (batch_size,) or labels.dtype != torch.long:
            raise ValueError("labels must be a torch.long tensor with shape (batch,)")
        embedding = self.time_projection(
            sinusoidal_embedding(timesteps, self.embedding_dim)
        )
        embedding = embedding + self.class_embedding(labels)

        skip = self.down_block(self.input(noisy_images), embedding)
        hidden = self.downsample(skip)
        hidden = self.middle1(hidden, embedding)
        hidden = self.middle2(hidden, embedding)
        hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="nearest")
        hidden = self.upsample(hidden)
        hidden = self.up_block(torch.cat((hidden, skip), dim=1), embedding)
        return self.output(F.silu(self.output_norm(hidden)))


class ConditionalDDPM(nn.Module):
    """DDPM noise schedule, training objective, and DDPM/DDIM samplers."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        num_classes: int = 10,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        if num_timesteps < 2:
            raise ValueError("num_timesteps must be at least two")
        if not 0 < beta_start < beta_end < 1:
            raise ValueError("betas must satisfy 0 < beta_start < beta_end < 1")

        self.num_timesteps = num_timesteps
        self.num_classes = num_classes
        self.model = ConditionalUNet(
            num_classes=num_classes,
            base_channels=base_channels,
        )
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_previous = F.pad(alpha_bars[:-1], (1, 0), value=1.0)
        posterior_variance = (
            betas * (1 - alpha_bars_previous) / (1 - alpha_bars)
        )
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_previous", alpha_bars_previous)
        self.register_buffer("posterior_variance", posterior_variance)

    def add_noise(
        self,
        images: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Sample x_t from q(x_t | x_0) for each image in a batch."""
        if images.shape != noise.shape:
            raise ValueError("images and noise must have matching shapes")
        if timesteps.shape != (images.shape[0],):
            raise ValueError("timesteps must have shape (batch,)")
        if torch.any((timesteps < 0) | (timesteps >= self.num_timesteps)):
            raise ValueError("timesteps are outside the configured schedule")

        alpha_bar = self.alpha_bars[timesteps].view(-1, 1, 1, 1)
        return alpha_bar.sqrt() * images + (1 - alpha_bar).sqrt() * noise

    def posterior_mean(
        self,
        noisy_images: torch.Tensor,
        predicted_clean: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the exact q(x_{t-1} | x_t, x_0) posterior mean."""
        if noisy_images.shape != predicted_clean.shape:
            raise ValueError("noisy_images and predicted_clean must have matching shapes")
        if timesteps.shape != (noisy_images.shape[0],):
            raise ValueError("timesteps must have shape (batch,)")

        shape = (-1, 1, 1, 1)
        beta = self.betas[timesteps].view(shape)
        alpha = self.alphas[timesteps].view(shape)
        alpha_bar = self.alpha_bars[timesteps].view(shape)
        alpha_bar_previous = self.alpha_bars_previous[timesteps].view(shape)
        coefficient_clean = beta * alpha_bar_previous.sqrt() / (1 - alpha_bar)
        coefficient_noisy = (
            alpha.sqrt() * (1 - alpha_bar_previous) / (1 - alpha_bar)
        )
        return coefficient_clean * predicted_clean + coefficient_noisy * noisy_images

    def training_loss(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        class_dropout_probability: float = 0.1,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Predict the sampled noise, randomly dropping class labels for CFG."""
        if not 0 <= class_dropout_probability <= 1:
            raise ValueError("class_dropout_probability must be between zero and one")
        if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
            raise ValueError("images must have shape (batch, 1, 28, 28)")
        if labels.shape != (images.shape[0],) or labels.dtype != torch.long:
            raise ValueError("labels must be a torch.long tensor with shape (batch,)")

        batch_size = images.shape[0]
        timesteps = torch.randint(
            self.num_timesteps, (batch_size,), device=images.device, generator=generator
        )
        noise = torch.randn(
            images.shape,
            device=images.device,
            dtype=images.dtype,
            generator=generator,
        )
        noisy_images = self.add_noise(images, timesteps, noise)
        if class_dropout_probability:
            dropped = torch.rand(
                batch_size, device=images.device, generator=generator
            ) < class_dropout_probability
            conditioned_labels = labels.masked_fill(dropped, self.num_classes)
        else:
            conditioned_labels = labels
        predicted_noise = self.model(noisy_images, timesteps, conditioned_labels)
        return F.mse_loss(predicted_noise, noise)

    @torch.inference_mode()
    def sample(
        self,
        labels: torch.Tensor,
        method: str = "ddim",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        generator: torch.Generator | None = None,
        trajectory: list[dict[str, Any]] | None = None,
    ) -> torch.Tensor:
        """Generate images using a complete DDPM or accelerated DDIM loop.

        If ``trajectory`` is a list, one entry per reverse step is appended:
        the timestep, the current noisy image x_t and the predicted clean
        image x_0, both in [0, 1] on the CPU (used for visualisation).
        """
        if labels.ndim != 1 or labels.dtype != torch.long:
            raise ValueError("labels must be a one-dimensional torch.long tensor")
        if torch.any((labels < 0) | (labels >= self.num_classes)):
            raise ValueError("sampling labels must be valid class indices")
        if method not in {"ddpm", "ddim"}:
            raise ValueError("method must be 'ddpm' or 'ddim'")
        if not 1 <= num_inference_steps <= self.num_timesteps:
            raise ValueError("num_inference_steps must be within the diffusion schedule")
        if method == "ddpm" and num_inference_steps != self.num_timesteps:
            raise ValueError("DDPM sampling must use every configured timestep")
        if guidance_scale < 0:
            raise ValueError("guidance_scale must be non-negative")

        model = self.model
        was_training = model.training
        model.eval()
        try:
            sample = torch.randn(
                labels.shape[0],
                1,
                28,
                28,
                device=labels.device,
                dtype=next(model.parameters()).dtype,
                generator=generator,
            )
            if method == "ddpm":
                timesteps = list(range(self.num_timesteps - 1, -1, -1))
            else:
                timesteps = (
                    torch.linspace(
                        self.num_timesteps - 1,
                        0,
                        num_inference_steps,
                    )
                    .round()
                    .long()
                    .unique_consecutive()
                    .tolist()
                )

            for index, timestep in enumerate(timesteps):
                current_timestep = torch.full(
                    (labels.shape[0],),
                    timestep,
                    device=labels.device,
                    dtype=torch.long,
                )
                if guidance_scale == 1:
                    predicted_noise = model(sample, current_timestep, labels)
                else:
                    model_input = torch.cat((sample, sample), dim=0)
                    timestep_input = torch.cat(
                        (current_timestep, current_timestep), dim=0
                    )
                    null_labels = torch.full_like(labels, self.num_classes)
                    label_input = torch.cat((null_labels, labels), dim=0)
                    unconditional, conditional = model(
                        model_input, timestep_input, label_input
                    ).chunk(2)
                    predicted_noise = unconditional + guidance_scale * (
                        conditional - unconditional
                    )
                alpha_bar = self.alpha_bars[timestep]
                predicted_clean = (
                    sample - (1 - alpha_bar).sqrt() * predicted_noise
                ) / alpha_bar.sqrt()
                predicted_clean = predicted_clean.clamp(-1, 1)
                if trajectory is not None:
                    trajectory.append({
                        "timestep": timestep,
                        "noisy": ((sample.clamp(-1, 1) + 1) / 2).cpu(),
                        "clean": ((predicted_clean + 1) / 2).cpu(),
                    })

                if method == "ddpm":
                    variance = self.posterior_variance[timestep]
                    sample = self.posterior_mean(
                        sample, predicted_clean, current_timestep
                    )
                    if timestep > 0:
                        noise = torch.randn(
                            sample.shape,
                            device=sample.device,
                            dtype=sample.dtype,
                            generator=generator,
                        )
                        sample = sample + variance.sqrt() * noise
                else:
                    predicted_noise = (
                        sample - alpha_bar.sqrt() * predicted_clean
                    ) / (1 - alpha_bar).sqrt()
                    previous_alpha_bar = (
                        self.alpha_bars[timesteps[index + 1]]
                        if index + 1 < len(timesteps)
                        else self.alpha_bars.new_tensor(1.0)
                    )
                    sample = previous_alpha_bar.sqrt() * predicted_clean
                    sample = sample + (1 - previous_alpha_bar).sqrt() * predicted_noise

            if not torch.isfinite(sample).all():
                raise RuntimeError("Sampler produced non-finite image values")
            return (sample.clamp(-1, 1) + 1) / 2
        finally:
            model.train(was_training)
