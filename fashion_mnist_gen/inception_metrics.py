"""Standard Inception-based KID / FID through torchmetrics.

Fashion-MNIST images are grayscale 28 x 28 in [0, 1]. They are repeated on three
channels; torchmetrics' Inception feature extractor resizes them to 299 x 299.
Inception was trained on ImageNet, so on Fashion-MNIST these scores are mostly
useful to compare models with each other (the brief's FID/KID), while the local
"Fashion-KID" uses features that know about clothes. Both are reported.
"""

import torch


def _to_rgb(images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4 or images.shape[1] not in (1, 3):
        raise ValueError("images must have shape (batch, 1 or 3, height, width)")
    images = images.float().clamp(0, 1)
    return images.repeat(1, 3, 1, 1) if images.shape[1] == 1 else images


class InceptionKID:
    """Kernel Inception Distance with cached real-image statistics.

    Real images are added once; :meth:`score` can then be called for several
    generated sets (models, seeds) without recomputing real features.
    """

    def __init__(
        self,
        device: torch.device,
        subset_size: int = 1000,
        subsets: int = 100,
        batch_size: int = 128,
        compute_fid: bool = False,
    ) -> None:
        try:
            from torchmetrics.image.kid import KernelInceptionDistance
        except ImportError as error:  # pragma: no cover - dependency message
            raise ImportError(
                "Inception KID requires torchmetrics and torch-fidelity "
                "(pip install torch-fidelity)"
            ) from error
        self.device = device
        self.batch_size = batch_size
        self.requested_subset_size = subset_size
        self.subsets = subsets
        self._real_count = 0
        self.kid = KernelInceptionDistance(
            subsets=subsets,
            subset_size=subset_size,
            reset_real_features=False,
            normalize=True,
        ).to(device)
        self.fid = None
        if compute_fid:
            from torchmetrics.image.fid import FrechetInceptionDistance

            self.fid = FrechetInceptionDistance(
                reset_real_features=False, normalize=True
            ).to(device)

    def _update(self, images: torch.Tensor, real: bool) -> None:
        for start in range(0, images.shape[0], self.batch_size):
            batch = _to_rgb(images[start : start + self.batch_size]).to(self.device)
            self.kid.update(batch, real=real)
            if self.fid is not None:
                self.fid.update(batch, real=real)

    @torch.inference_mode()
    def add_real(self, images: torch.Tensor) -> None:
        self._update(images, real=True)
        self._real_count += images.shape[0]

    @torch.inference_mode()
    def score(self, generated: torch.Tensor) -> dict[str, float]:
        if self._real_count == 0:
            raise RuntimeError("add_real must be called before score")
        self.kid.reset()
        if self.fid is not None:
            self.fid.reset()
        self.kid.subset_size = min(
            self.requested_subset_size, self._real_count, generated.shape[0]
        )
        self._update(generated, real=False)
        kid_mean, kid_std = self.kid.compute()
        result = {
            "inception_kid": kid_mean.item(),
            "inception_kid_subset_std": kid_std.item(),
            "inception_kid_subset_size": self.kid.subset_size,
        }
        if self.fid is not None:
            result["inception_fid"] = self.fid.compute().item()
        return result

