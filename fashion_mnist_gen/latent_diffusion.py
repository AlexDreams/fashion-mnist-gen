"""Level 3 extension: class-conditional diffusion in the latent space of the VAE.

The level-0 VAE samples its latent code from N(0, I), but the codes of real
images do not fill that prior uniformly: decoded prior samples are blurry and
off-class. Here a small diffusion model learns the real latent distribution
(standardized encoder means) per class, then the frozen VAE decoder turns the
generated latent into an image. The denoiser is an MLP on a 16-D vector, so it
trains and samples much faster than the pixel U-Net of level 2.

The noise schedule, classifier-free guidance and DDPM/DDIM loops are written by
hand, like the pixel diffusion, and reuse the same equations.
"""

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from fashion_mnist_gen.diffusion import sinusoidal_embedding
from fashion_mnist_gen.vae import ConditionalVAE


class LatentDenoiser(nn.Module):
    """Residual MLP predicting the noise added to a latent vector."""

    def __init__(
        self,
        latent_dim: int = 16,
        num_classes: int = 10,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        if min(latent_dim, num_classes, hidden_dim, num_blocks, embedding_dim) < 1:
            raise ValueError("all LatentDenoiser sizes must be positive")
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        # Index num_classes is the "no class" token used for guidance.
        self.class_embedding = nn.Embedding(num_classes + 1, embedding_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.input = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            nn.ModuleDict(
                {
                    "norm": nn.LayerNorm(hidden_dim),
                    "condition": nn.Linear(embedding_dim, hidden_dim * 2),
                    "linear1": nn.Linear(hidden_dim, hidden_dim),
                    "linear2": nn.Linear(hidden_dim, hidden_dim),
                }
            )
            for _ in range(num_blocks)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, latents: torch.Tensor, timesteps: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        if latents.ndim != 2 or latents.shape[1] != self.latent_dim:
            raise ValueError(f"latents must have shape (batch, {self.latent_dim})")
        embedding = self.time_projection(
            sinusoidal_embedding(timesteps, self.embedding_dim)
        ) + self.class_embedding(labels)
        hidden = self.input(latents)
        for block in self.blocks:
            scale, shift = block["condition"](F.silu(embedding)).chunk(2, dim=1)
            update = block["norm"](hidden) * (1 + scale) + shift
            update = block["linear2"](F.silu(block["linear1"](F.silu(update))))
            hidden = hidden + update
        return self.output(F.silu(self.output_norm(hidden)))


class LatentDiffusion(nn.Module):
    """Latent DDPM/DDIM sampler wrapped around a frozen conditional VAE decoder."""

    def __init__(
        self,
        vae: ConditionalVAE,
        latent_mean: torch.Tensor,
        latent_std: torch.Tensor,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        hidden_dim: int = 256,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        if num_timesteps < 2:
            raise ValueError("num_timesteps must be at least two")
        if not 0 < beta_start < beta_end < 1:
            raise ValueError("betas must satisfy 0 < beta_start < beta_end < 1")
        self.num_timesteps = num_timesteps
        self.num_classes = vae.num_classes
        self.latent_dim = vae.latent_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.vae = vae
        for parameter in self.vae.parameters():
            parameter.requires_grad_(False)
        self.model = LatentDenoiser(
            latent_dim=vae.latent_dim,
            num_classes=vae.num_classes,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
        )
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_previous = F.pad(alpha_bars[:-1], (1, 0), value=1.0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_previous", alpha_bars_previous)
        self.register_buffer(
            "posterior_variance", betas * (1 - alpha_bars_previous) / (1 - alpha_bars)
        )
        self.register_buffer("latent_mean", latent_mean.detach().clone().float())
        self.register_buffer("latent_std", latent_std.detach().clone().float())

    # ----- latent normalization -------------------------------------------------
    def normalize(self, latents: torch.Tensor) -> torch.Tensor:
        return (latents - self.latent_mean) / self.latent_std

    def denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        return latents * self.latent_std + self.latent_mean

    @torch.no_grad()
    def encode(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Standardized encoder means: the training data of the latent diffusion."""
        mean, _ = self.vae.encode(images, labels)
        return self.normalize(mean)

    # ----- training -------------------------------------------------------------
    def training_loss(
        self,
        latents: torch.Tensor,
        labels: torch.Tensor,
        class_dropout_probability: float = 0.1,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Noise-prediction MSE on standardized latents, with label dropout for CFG."""
        if latents.ndim != 2 or latents.shape[1] != self.latent_dim:
            raise ValueError(f"latents must have shape (batch, {self.latent_dim})")
        if labels.shape != (latents.shape[0],) or labels.dtype != torch.long:
            raise ValueError("labels must be a torch.long tensor with shape (batch,)")
        if not 0 <= class_dropout_probability <= 1:
            raise ValueError("class_dropout_probability must be between zero and one")
        batch_size = latents.shape[0]
        timesteps = torch.randint(
            self.num_timesteps, (batch_size,), device=latents.device, generator=generator
        )
        noise = torch.randn(
            latents.shape, device=latents.device, dtype=latents.dtype, generator=generator
        )
        alpha_bar = self.alpha_bars[timesteps].unsqueeze(1)
        noisy = alpha_bar.sqrt() * latents + (1 - alpha_bar).sqrt() * noise
        if class_dropout_probability:
            dropped = (
                torch.rand(batch_size, device=latents.device, generator=generator)
                < class_dropout_probability
            )
            labels = labels.masked_fill(dropped, self.num_classes)
        return F.mse_loss(self.model(noisy, timesteps, labels), noise)

    # ----- sampling -------------------------------------------------------------
    def _predict_noise(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        labels: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        if guidance_scale == 1:
            return self.model(sample, timestep, labels)
        null_labels = torch.full_like(labels, self.num_classes)
        unconditional, conditional = self.model(
            torch.cat((sample, sample)),
            torch.cat((timestep, timestep)),
            torch.cat((null_labels, labels)),
        ).chunk(2)
        return unconditional + guidance_scale * (conditional - unconditional)

    @torch.inference_mode()
    def sample_latents(
        self,
        labels: torch.Tensor,
        method: str = "ddim",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        generator: torch.Generator | None = None,
        trajectory: list[dict[str, Any]] | None = None,
    ) -> torch.Tensor:
        """Return standardized latents from a DDPM or DDIM reverse loop."""
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

        was_training = self.model.training
        self.model.eval()
        try:
            sample = torch.randn(
                labels.shape[0],
                self.latent_dim,
                device=labels.device,
                dtype=self.latent_mean.dtype,
                generator=generator,
            )
            if method == "ddpm":
                timesteps = list(range(self.num_timesteps - 1, -1, -1))
            else:
                timesteps = (
                    torch.linspace(self.num_timesteps - 1, 0, num_inference_steps)
                    .round()
                    .long()
                    .unique_consecutive()
                    .tolist()
                )
            for index, timestep in enumerate(timesteps):
                current = torch.full(
                    (labels.shape[0],), timestep, device=labels.device, dtype=torch.long
                )
                predicted_noise = self._predict_noise(sample, current, labels, guidance_scale)
                alpha_bar = self.alpha_bars[timestep]
                # Latents are standardized: |values| beyond ~5 std are artefacts.
                predicted_clean = (
                    (sample - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
                ).clamp(-5, 5)
                if trajectory is not None:
                    trajectory.append(
                        {"timestep": timestep, "noisy": sample.clone(),
                         "clean": predicted_clean.clone()}
                    )
                if method == "ddpm":
                    alpha = self.alphas[timestep]
                    beta = self.betas[timestep]
                    alpha_bar_previous = self.alpha_bars_previous[timestep]
                    sample = (
                        beta * alpha_bar_previous.sqrt() / (1 - alpha_bar) * predicted_clean
                        + alpha.sqrt() * (1 - alpha_bar_previous) / (1 - alpha_bar) * sample
                    )
                    if timestep > 0:
                        noise = torch.randn(
                            sample.shape,
                            device=sample.device,
                            dtype=sample.dtype,
                            generator=generator,
                        )
                        sample = sample + self.posterior_variance[timestep].sqrt() * noise
                else:
                    predicted_noise = (
                        sample - alpha_bar.sqrt() * predicted_clean
                    ) / (1 - alpha_bar).sqrt()
                    previous_alpha_bar = (
                        self.alpha_bars[timesteps[index + 1]]
                        if index + 1 < len(timesteps)
                        else self.alpha_bars.new_tensor(1.0)
                    )
                    sample = (
                        previous_alpha_bar.sqrt() * predicted_clean
                        + (1 - previous_alpha_bar).sqrt() * predicted_noise
                    )
            if not torch.isfinite(sample).all():
                raise RuntimeError("Latent sampler produced non-finite values")
            return sample
        finally:
            self.model.train(was_training)

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
        """Generate images in [0, 1]; same signature as ConditionalDDPM.sample.

        With ``trajectory``, each recorded latent (noisy z_t and predicted z_0)
        is decoded by the VAE so the steps can be shown as images.
        """
        latent_steps: list[dict[str, Any]] | None = [] if trajectory is not None else None
        latents = self.sample_latents(
            labels, method, num_inference_steps, guidance_scale, generator, latent_steps
        )
        self.vae.eval()
        if trajectory is not None:
            for step in latent_steps:
                trajectory.append({
                    "timestep": step["timestep"],
                    "noisy": self.vae.decode(self.denormalize(step["noisy"]), labels).cpu(),
                    "clean": self.vae.decode(self.denormalize(step["clean"]), labels).cpu(),
                })
        return self.vae.decode(self.denormalize(latents), labels)

    # ----- persistence ----------------------------------------------------------
    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "model_state_dict": {
                name: tensor.detach().cpu().clone()
                for name, tensor in self.model.state_dict().items()
            },
            "vae_state_dict": {
                name: tensor.detach().cpu().clone()
                for name, tensor in self.vae.state_dict().items()
            },
            "latent_dim": self.latent_dim,
            "latent_mean": self.latent_mean.cpu(),
            "latent_std": self.latent_std.cpu(),
            "num_timesteps": self.num_timesteps,
            "hidden_dim": self.hidden_dim,
            "num_blocks": self.num_blocks,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, Any]) -> "LatentDiffusion":
        vae = ConditionalVAE(
            latent_dim=checkpoint["latent_dim"],
            hidden_dim=checkpoint["vae_state_dict"]["encoder.0.weight"].shape[0],
        )
        vae.load_state_dict(checkpoint["vae_state_dict"])
        model = cls(
            vae,
            checkpoint["latent_mean"],
            checkpoint["latent_std"],
            num_timesteps=checkpoint["num_timesteps"],
            hidden_dim=checkpoint["hidden_dim"],
            num_blocks=checkpoint["num_blocks"],
        )
        model.model.load_state_dict(checkpoint["model_state_dict"])
        return model
