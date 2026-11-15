import pytest
import torch

from fashion_mnist_gen.metrics import class_fidelity


def test_class_fidelity_returns_overall_and_per_class_accuracy() -> None:
    predictions = torch.tensor([0, 1, 2, 2])
    labels = torch.tensor([0, 1, 2, 3])

    overall, per_class = class_fidelity(predictions, labels, num_classes=4)

    assert overall == pytest.approx(0.75)
    assert per_class == [1.0, 1.0, 1.0, 0.0]


def test_class_fidelity_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        class_fidelity(
            torch.tensor([], dtype=torch.long),
            torch.tensor([], dtype=torch.long),
            10,
        )


def test_class_fidelity_rejects_out_of_range_labels() -> None:
    with pytest.raises(ValueError, match="outside the configured classes"):
        class_fidelity(torch.tensor([0]), torch.tensor([10]), num_classes=10)
