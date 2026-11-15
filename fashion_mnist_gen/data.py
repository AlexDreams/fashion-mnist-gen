"""Fashion-MNIST data loading helpers."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

VALIDATION_FRACTION = 0.1

CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


def make_dataloaders(
    data_dir: str | Path = "data",
    batch_size: int = 128,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Download Fashion-MNIST if needed and return train/test data loaders."""
    if batch_size < 1:
        raise ValueError("batch_size must be greater than zero")

    transform = ToTensor()
    train_dataset = FashionMNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = FashionMNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform,
    )

    generator = torch.Generator().manual_seed(seed)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def make_train_validation_loaders(
    data_dir: str | Path = "data",
    batch_size: int = 128,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Split the official training set for classifier training and validation."""
    if batch_size < 1:
        raise ValueError("batch_size must be greater than zero")

    dataset = FashionMNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=ToTensor(),
    )
    validation_size = int(len(dataset) * VALIDATION_FRACTION)
    train_size = len(dataset) - validation_size
    split_generator = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset = torch.utils.data.random_split(
        dataset,
        (train_size, validation_size),
        generator=split_generator,
    )

    loader_generator = torch.Generator().manual_seed(seed)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    return train_loader, validation_loader
