"""Class-conditional GAN for Fashion-MNIST."""

import torch
from torch import nn


class ConditionalGenerator(nn.Module):
    """Generate 28 x 28 images conditioned on a Fashion-MNIST class."""

    def __init__(
        self,
        noise_dim: int = 64,
        num_classes: int = 10,
        class_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        if noise_dim < 1 or num_classes < 1 or class_embedding_dim < 1:
            raise ValueError("noise_dim, num_classes, and embedding size must be positive")

        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.class_embedding = nn.Embedding(num_classes, class_embedding_dim)
        self.project = nn.Sequential(
            nn.Linear(noise_dim + class_embedding_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if noise.ndim != 2 or noise.shape[1] != self.noise_dim:
            raise ValueError(f"noise must have shape (batch, {self.noise_dim})")
        if labels.ndim != 1 or labels.shape[0] != noise.shape[0]:
            raise ValueError("labels must have shape (batch,)")
        if labels.dtype != torch.long:
            raise ValueError("labels must have dtype torch.long")

        class_embedding = self.class_embedding(labels)
        projected = self.project(torch.cat((noise, class_embedding), dim=1))
        features = projected.reshape(noise.shape[0], 128, 7, 7)
        return self.upsample(features)

    @torch.no_grad()
    def sample(
        self,
        labels: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Generate one image per requested class label."""
        if labels.ndim != 1:
            raise ValueError("labels must have shape (batch,)")
        if labels.dtype != torch.long:
            raise ValueError("labels must have dtype torch.long")
        noise = torch.randn(
            labels.shape[0],
            self.noise_dim,
            device=labels.device,
            dtype=next(self.parameters()).dtype,
            generator=generator,
        )
        return self(noise, labels)


class ConditionalDiscriminator(nn.Module):
    """Distinguish real and generated images given their requested class."""

    def __init__(
        self,
        num_classes: int = 10,
        class_embedding_dim: int = 28 * 28,
    ) -> None:
        super().__init__()
        if num_classes < 1 or class_embedding_dim < 1:
            raise ValueError("num_classes and embedding size must be positive")

        self.num_classes = num_classes
        self.class_embedding = nn.Embedding(num_classes, class_embedding_dim)
        self.features = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.score = nn.Linear(128 * 7 * 7, 1)

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
            raise ValueError("images must have shape (batch, 1, 28, 28)")
        if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
            raise ValueError("labels must have shape (batch,)")
        if labels.dtype != torch.long:
            raise ValueError("labels must have dtype torch.long")

        class_map = self.class_embedding(labels).reshape(-1, 1, 28, 28)
        features = self.features(torch.cat((images, class_map), dim=1))
        return self.score(features.flatten(start_dim=1)).squeeze(dim=1)
