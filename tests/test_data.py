import pytest

from fashion_mnist_gen.data import (
    CLASS_NAMES,
    make_dataloaders,
    make_train_validation_loaders,
)


def test_fashion_mnist_class_names_cover_all_labels() -> None:
    assert len(CLASS_NAMES) == 10
    assert CLASS_NAMES[0] == "T-shirt/top"
    assert CLASS_NAMES[9] == "Ankle boot"


def test_make_dataloaders_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be greater than zero"):
        make_dataloaders(batch_size=0)


def test_make_train_validation_loaders_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be greater than zero"):
        make_train_validation_loaders(batch_size=0)


def test_train_validation_split_uses_only_official_training_images() -> None:
    train_loader, validation_loader = make_train_validation_loaders(batch_size=128)

    assert len(train_loader.dataset) == 54_000
    assert len(validation_loader.dataset) == 6_000
    assert len(train_loader.dataset) + len(validation_loader.dataset) == 60_000
