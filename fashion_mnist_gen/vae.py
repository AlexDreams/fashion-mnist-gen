"""Class-conditional variational autoencoder for Fashion-MNIST."""

import torch
from torch import nn
from torch.nn import functional as F


class ConditionalVAE(nn.Module):
    """MLP VAE conditioned on the requested Fashion-MNIST class."""

    def __init__(
        self,
        latent_dim: int = 16,
        num_classes: int = 10,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if latent_dim < 1 or num_classes < 1 or hidden_dim < 1:
            raise ValueError("latent_dim, num_classes, and hidden_dim must be positive")

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28 + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_variance = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 28 * 28),
            nn.Sigmoid(),
        )

    def encode(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
            raise ValueError("images must have shape (batch, 1, 28, 28)")
        if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
            raise ValueError("labels must have shape (batch,)")
        if labels.dtype != torch.long:
            raise ValueError("labels must have dtype torch.long")

        class_condition = F.one_hot(labels, num_classes=self.num_classes).to(
            dtype=images.dtype
        )
        encoded = self.encoder(
            torch.cat((images.flatten(start_dim=1), class_condition), dim=1)
        )
        return self.mean(encoded), self.log_variance(encoded)

    def decode(self, latent: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"latent must have shape (batch, {self.latent_dim})")
        if labels.ndim != 1 or labels.shape[0] != latent.shape[0]:
            raise ValueError("labels must have shape (batch,)")
        if labels.dtype != torch.long:
            raise ValueError("labels must have dtype torch.long")

        class_condition = F.one_hot(labels, num_classes=self.num_classes).to(
            dtype=latent.dtype
        )
        return self.decoder(torch.cat((latent, class_condition), dim=1)).reshape(
            -1, 1, 28, 28
        )

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_variance = self.encode(images, labels)
        noise = torch.randn_like(mean)
        latent = mean + noise * torch.exp(0.5 * log_variance)
        return self.decode(latent, labels), mean, log_variance

    @torch.no_grad()
    def sample(
        self,
        labels: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Generate one image per label from the learned standard-normal prior."""
        if labels.ndim != 1:
            raise ValueError("labels must have shape (batch,)")
        if labels.dtype != torch.long:
            raise ValueError("labels must have dtype torch.long")
        latent = torch.randn(
            labels.shape[0],
            self.latent_dim,
            device=labels.device,
            dtype=next(self.parameters()).dtype,
            generator=generator,
        )
        return self.decode(latent, labels)


def vae_loss(
    reconstructions: torch.Tensor,
    images: torch.Tensor,
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-image ELBO loss, binary cross entropy, and KL divergence."""
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if reconstructions.shape != images.shape:
        raise ValueError("reconstructions and images must have matching shapes")

    batch_size = images.shape[0]
    reconstruction_loss = F.binary_cross_entropy(
        reconstructions, images, reduction="sum"
    ) / batch_size
    kl_divergence = -0.5 * torch.sum(
        1 + log_variance - mean.square() - log_variance.exp()
    ) / batch_size
    return (
        reconstruction_loss + beta * kl_divergence,
        reconstruction_loss,
        kl_divergence,
    )
