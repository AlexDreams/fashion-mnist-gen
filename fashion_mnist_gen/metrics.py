"""Metrics shared by generative model evaluations."""

import torch


def class_fidelity(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> tuple[float, list[float]]:
    """Return overall and per-class accuracy for class-conditioned samples."""
    if predictions.ndim != 1 or labels.ndim != 1:
        raise ValueError("predictions and labels must be one-dimensional")
    if predictions.shape != labels.shape:
        raise ValueError("predictions and labels must have matching shapes")
    if predictions.numel() == 0:
        raise ValueError("predictions and labels must not be empty")
    if predictions.dtype != torch.long or labels.dtype != torch.long:
        raise ValueError("predictions and labels must have dtype torch.long")
    if predictions.device != labels.device:
        raise ValueError("predictions and labels must be on the same device")
    if num_classes < 1:
        raise ValueError("num_classes must be greater than zero")
    if torch.any((labels < 0) | (labels >= num_classes)):
        raise ValueError("labels contain values outside the configured classes")

    matches = predictions == labels
    counts = torch.bincount(labels, minlength=num_classes)
    class_matches = torch.bincount(
        labels[matches],
        minlength=num_classes,
    )
    per_class = class_matches.float() / counts.clamp_min(1)
    return matches.float().mean().item(), per_class.cpu().tolist()


def fashion_kid(real_features: torch.Tensor, generated_features: torch.Tensor) -> float:
    """Unbiased polynomial-kernel MMD in the Fashion classifier feature space."""
    if real_features.ndim != 2 or generated_features.ndim != 2:
        raise ValueError("features must have shape (samples, dimensions)")
    if real_features.shape[1] != generated_features.shape[1]:
        raise ValueError("real and generated feature dimensions must match")
    if real_features.shape[0] < 2 or generated_features.shape[0] < 2:
        raise ValueError("KID requires at least two samples in each set")
    if not torch.isfinite(real_features).all() or not torch.isfinite(
        generated_features
    ).all():
        raise ValueError("features must contain only finite values")

    real = real_features.float()
    generated = generated_features.to(device=real.device, dtype=torch.float32)
    dimension = real.shape[1]
    real_kernel = (real @ real.T / dimension + 1).pow(3)
    generated_kernel = (generated @ generated.T / dimension + 1).pow(3)
    cross_kernel = (real @ generated.T / dimension + 1).pow(3)
    real_term = (real_kernel.sum() - real_kernel.diagonal().sum()) / (
        real.shape[0] * (real.shape[0] - 1)
    )
    generated_term = (
        generated_kernel.sum() - generated_kernel.diagonal().sum()
    ) / (generated.shape[0] * (generated.shape[0] - 1))
    return (real_term - 2 * cross_kernel.mean() + generated_term).item()


def feature_diversity(features: torch.Tensor) -> float:
    """Mean off-diagonal cosine distance in a feature set."""
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("features must contain at least two feature vectors")
    if not torch.isfinite(features).all():
        raise ValueError("features must contain only finite values")
    normalized = torch.nn.functional.normalize(features.float(), dim=1)
    similarities = normalized @ normalized.T
    sample_count = features.shape[0]
    return (
        1 - (similarities.sum() - similarities.diagonal().sum())
        / (sample_count * (sample_count - 1))
    ).item()


def relative_feature_coverage(
    real_features: torch.Tensor,
    generated_features: torch.Tensor,
) -> float:
    """Fraction of real features covered within their real nearest-neighbor radius."""
    if real_features.ndim != 2 or generated_features.ndim != 2:
        raise ValueError("features must have shape (samples, dimensions)")
    if real_features.shape[1] != generated_features.shape[1]:
        raise ValueError("real and generated feature dimensions must match")
    if real_features.shape[0] < 2 or generated_features.shape[0] < 1:
        raise ValueError("coverage requires two real samples and one generated sample")
    if not torch.isfinite(real_features).all() or not torch.isfinite(
        generated_features
    ).all():
        raise ValueError("features must contain only finite values")

    real = real_features.float()
    generated = generated_features.to(device=real.device, dtype=torch.float32)
    real_distances = torch.cdist(real, real)
    real_distances.fill_diagonal_(float("inf"))
    real_neighbor_radius = real_distances.min(dim=1).values
    generated_neighbor_distance = torch.cdist(real, generated).min(dim=1).values
    return (generated_neighbor_distance <= real_neighbor_radius).float().mean().item()


def pixel_diversity(images: torch.Tensor) -> float:
    """Mean pairwise Euclidean distance between images of one set (pixel space).

    This is the "distance moyenne entre images d'une même classe" of the project
    brief. Compare it with the same value on real images of the class: a ratio
    far below one reveals mode collapse (near-duplicate samples).
    """
    if images.ndim < 2 or images.shape[0] < 2:
        raise ValueError("images must contain at least two samples")
    flat = images.reshape(images.shape[0], -1).float()
    if not torch.isfinite(flat).all():
        raise ValueError("images must contain only finite values")
    distances = torch.cdist(flat, flat)
    count = flat.shape[0]
    return (distances.sum() / (count * (count - 1))).item()


def class_coverage(
    predictions: torch.Tensor,
    num_classes: int,
    min_share: float = 0.5,
) -> dict[str, float]:
    """How well the classifier-predicted classes of all samples cover the classes.

    ``coverage`` is the fraction of classes that receive at least ``min_share``
    times their uniform share (1 / num_classes) of the predictions.
    ``normalized_entropy`` is the entropy of the predicted class histogram divided
    by log(num_classes): 1 means perfectly balanced, 0 means a single class.
    """
    if predictions.ndim != 1 or predictions.numel() == 0:
        raise ValueError("predictions must be a non-empty one-dimensional tensor")
    if predictions.dtype != torch.long:
        raise ValueError("predictions must have dtype torch.long")
    if num_classes < 2:
        raise ValueError("num_classes must be at least two")
    if not 0 < min_share <= 1:
        raise ValueError("min_share must be in (0, 1]")
    if torch.any((predictions < 0) | (predictions >= num_classes)):
        raise ValueError("predictions contain values outside the configured classes")

    counts = torch.bincount(predictions.cpu(), minlength=num_classes).float()
    shares = counts / counts.sum()
    covered = (shares >= min_share / num_classes).float().mean().item()
    nonzero = shares[shares > 0]
    entropy = -(nonzero * nonzero.log()).sum().item()
    return {
        "coverage": covered,
        "normalized_entropy": entropy / torch.log(torch.tensor(float(num_classes))).item(),
    }
