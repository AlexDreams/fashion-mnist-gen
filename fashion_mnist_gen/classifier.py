"""Classifier used to measure class consistency of generated images."""

import torch
from torch import nn
from torch.utils.data import DataLoader


class FashionClassifier(nn.Module):
    """Small CNN baseline for 28 x 28 Fashion-MNIST images."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(128, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return the penultimate classifier representation for image metrics."""
        features = self.features(images)
        return self.classifier[:3](features)


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """Return classification accuracy over a complete data loader."""
    model.eval()
    correct = 0
    total = 0
    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        predictions = model(images).argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += labels.numel()

    if total == 0:
        raise ValueError("Cannot evaluate accuracy on an empty data loader")
    return correct / total
