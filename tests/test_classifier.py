import pytest
import torch

from fashion_mnist_gen.classifier import FashionClassifier, evaluate_accuracy


def test_classifier_returns_one_score_per_fashion_class() -> None:
    model = FashionClassifier()
    scores = model(torch.zeros(4, 1, 28, 28))

    assert scores.shape == (4, 10)


def test_classifier_features_return_one_embedding_per_image() -> None:
    model = FashionClassifier()
    features = model.extract_features(torch.zeros(4, 1, 28, 28))

    assert features.shape == (4, 128)


def test_evaluate_accuracy_rejects_empty_data_loader() -> None:
    model = FashionClassifier()
    empty_loader = torch.utils.data.DataLoader([])

    with pytest.raises(ValueError, match="empty data loader"):
        evaluate_accuracy(model, empty_loader, torch.device("cpu"))
