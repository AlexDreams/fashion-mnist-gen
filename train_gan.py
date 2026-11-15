"""Train the class-conditional Fashion-MNIST GAN (level 1).

Besides the two losses, every epoch is monitored on the validation split (never
on the test set) with a fixed noise batch:

* D(real) and D(fake): mean discriminator probabilities, to see who wins;
* intra-class pixel diversity of generated images divided by the same value on
  real validation images: a ratio far below one flags mode collapse;
* if the evaluation classifier exists: class fidelity, class coverage and
  Fashion-KID against validation features.

Epoch checkpoints are kept so that a collapse can be inspected afterwards.
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import torch
from torch import nn
from torch.optim import Adam
from torchvision.utils import save_image

from fashion_mnist_gen.classifier import FashionClassifier
from fashion_mnist_gen.data import CLASS_NAMES, make_train_validation_loaders
from fashion_mnist_gen.gan import ConditionalDiscriminator, ConditionalGenerator
from fashion_mnist_gen.metrics import (
    class_coverage,
    class_fidelity,
    fashion_kid,
    pixel_diversity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--noise-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/gan"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/gan.pt"))
    parser.add_argument(
        "--classifier",
        type=Path,
        default=Path("checkpoints/fashion_classifier.pt"),
        help="Optional: enables fidelity and Fashion-KID monitoring when present.",
    )
    parser.add_argument(
        "--monitor-samples-per-class",
        type=int,
        default=100,
        help="Generated and real validation images per class for monitoring (0 = off).",
    )
    parser.add_argument(
        "--collapse-ratio",
        type=float,
        default=0.6,
        help="Flag a class when generated/real pixel diversity falls below this.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Keep a checkpoint every N epochs (the last epoch is always kept).",
    )
    parser.add_argument(
        "--selection",
        choices=("last", "validation_kid"),
        default="last",
        help=(
            "Generator saved in --checkpoint: last epoch, or the epoch with the best "
            "validation Fashion-KID (requires the classifier)."
        ),
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Limit batches per epoch for smoke tests (0 = full epoch).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def train_epoch(
    generator: ConditionalGenerator,
    discriminator: ConditionalDiscriminator,
    data_loader: torch.utils.data.DataLoader,
    generator_optimizer: Adam,
    discriminator_optimizer: Adam,
    loss_fn: nn.BCEWithLogitsLoss,
    device: torch.device,
    max_batches: int = 0,
) -> dict[str, float]:
    generator.train()
    discriminator.train()
    totals = {
        "generator_loss": 0.0,
        "discriminator_loss": 0.0,
        "d_real_probability": 0.0,
        "d_fake_probability": 0.0,
    }
    total_examples = 0

    for batch_index, (real_images, labels) in enumerate(data_loader):
        if max_batches and batch_index >= max_batches:
            break
        real_images = real_images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = labels.shape[0]
        real_targets = torch.ones(batch_size, device=device)
        fake_targets = torch.zeros(batch_size, device=device)

        discriminator_optimizer.zero_grad(set_to_none=True)
        real_logits = discriminator(real_images, labels)
        real_loss = loss_fn(real_logits, real_targets)

        noise = torch.randn(batch_size, generator.noise_dim, device=device)
        fake_images = generator(noise, labels)
        fake_logits = discriminator(fake_images.detach(), labels)
        fake_loss = loss_fn(fake_logits, fake_targets)
        discriminator_loss = 0.5 * (real_loss + fake_loss)
        discriminator_loss.backward()
        discriminator_optimizer.step()

        generator_optimizer.zero_grad(set_to_none=True)
        noise = torch.randn(batch_size, generator.noise_dim, device=device)
        generated_images = generator(noise, labels)
        fooled_logits = discriminator(generated_images, labels)
        generator_loss = loss_fn(fooled_logits, real_targets)
        generator_loss.backward()
        generator_optimizer.step()

        totals["generator_loss"] += generator_loss.item() * batch_size
        totals["discriminator_loss"] += discriminator_loss.item() * batch_size
        totals["d_real_probability"] += (
            torch.sigmoid(real_logits).sum().item()
        )
        totals["d_fake_probability"] += (
            torch.sigmoid(fake_logits).sum().item()
        )
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("Cannot train a GAN with an empty data loader")
    return {name: value / total_examples for name, value in totals.items()}


def collect_validation_reference(
    validation_loader: torch.utils.data.DataLoader,
    samples_per_class: int,
) -> list[torch.Tensor]:
    """First ``samples_per_class`` validation images of each class."""
    buckets: list[list[torch.Tensor]] = [[] for _ in CLASS_NAMES]
    counts = [0] * len(CLASS_NAMES)
    for images, labels in validation_loader:
        for class_id in range(len(CLASS_NAMES)):
            missing = samples_per_class - counts[class_id]
            if missing <= 0:
                continue
            class_images = images[labels == class_id][:missing]
            if len(class_images):
                buckets[class_id].append(class_images)
                counts[class_id] += len(class_images)
        if min(counts) >= samples_per_class:
            break
    if min(counts) < 2:
        raise ValueError("Validation split has too few images for monitoring")
    return [torch.cat(bucket) for bucket in buckets]


class CollapseMonitor:
    """Score a fixed noise batch on the validation split after every epoch."""

    def __init__(
        self,
        real_by_class: list[torch.Tensor],
        noise_dim: int,
        seed: int,
        device: torch.device,
        classifier: FashionClassifier | None,
        collapse_ratio: float,
    ) -> None:
        self.device = device
        self.classifier = classifier
        self.collapse_ratio = collapse_ratio
        self.samples_per_class = min(len(images) for images in real_by_class)
        self.real_by_class = [images[: self.samples_per_class] for images in real_by_class]
        self.real_pixel_diversity = [pixel_diversity(images) for images in self.real_by_class]
        self.labels = torch.arange(len(CLASS_NAMES), device=device).repeat_interleave(
            self.samples_per_class
        )
        generator = torch.Generator(device=device).manual_seed(seed + 7919)
        self.noise = torch.randn(
            self.labels.shape[0], noise_dim, device=device, generator=generator
        )
        self.real_features: list[torch.Tensor] | None = None
        if classifier is not None:
            with torch.inference_mode():
                self.real_features = [
                    classifier.extract_features(images.to(device)).cpu()
                    for images in self.real_by_class
                ]

    @torch.inference_mode()
    def __call__(self, generator: ConditionalGenerator) -> dict[str, Any]:
        generator.eval()
        images = generator(self.noise, self.labels).cpu()
        generator.train()
        per_class = images.split(self.samples_per_class)
        ratios = [
            pixel_diversity(class_images) / real
            for class_images, real in zip(per_class, self.real_pixel_diversity)
        ]
        collapsed = [
            name for name, ratio in zip(CLASS_NAMES, ratios) if ratio < self.collapse_ratio
        ]
        result: dict[str, Any] = {
            "pixel_diversity_ratio": sum(ratios) / len(ratios),
            "min_pixel_diversity_ratio": min(ratios),
            "pixel_diversity_ratio_per_class": dict(zip(CLASS_NAMES, ratios)),
            "collapsed_classes": collapsed,
        }
        if self.classifier is not None and self.real_features is not None:
            logits = []
            features = []
            for class_images in per_class:
                class_images = class_images.to(self.device)
                logits.append(self.classifier(class_images).cpu())
                features.append(self.classifier.extract_features(class_images).cpu())
            predictions = torch.cat(logits).argmax(dim=1)
            fidelity, _ = class_fidelity(
                predictions, self.labels.cpu(), num_classes=len(CLASS_NAMES)
            )
            kid = sum(
                fashion_kid(real, generated)
                for real, generated in zip(self.real_features, features)
            ) / len(CLASS_NAMES)
            coverage = class_coverage(predictions, len(CLASS_NAMES))
            result.update(
                {
                    "validation_class_fidelity": fidelity,
                    "validation_fashion_kid": kid,
                    "validation_class_coverage": coverage["coverage"],
                }
            )
        return result


def save_visualizations(
    generator: ConditionalGenerator,
    output_dir: Path,
    seed: int,
    device: torch.device,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    generator.eval()
    labels = torch.arange(len(CLASS_NAMES), device=device).repeat_interleave(8)
    random_generator = torch.Generator(device=device).manual_seed(seed)
    with torch.inference_mode():
        images = generator.sample(labels, generator=random_generator).cpu()
    save_image(
        images,
        output_dir / "samples_by_class.png",
        nrow=8,
        padding=2,
    )

    figure, axes = plt.subplots(10, 8, figsize=(10, 13))
    for class_index, row in enumerate(axes):
        for sample_index, axis in enumerate(row):
            axis.imshow(
                images[class_index * 8 + sample_index].squeeze(),
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axis.set_xticks([])
            axis.set_yticks([])
        row[0].set_ylabel(CLASS_NAMES[class_index], rotation=0, labelpad=55)
    figure.suptitle("Échantillons GAN générés par classe")
    figure.tight_layout()
    figure.savefig(output_dir / "samples_by_class_labeled.png", dpi=150)
    plt.close(figure)


def save_loss_plot(history: list[dict[str, Any]], path: Path) -> None:
    epochs = range(1, len(history) + 1)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(
        epochs,
        [entry["generator_loss"] for entry in history],
        label="Générateur",
    )
    axis.plot(
        epochs,
        [entry["discriminator_loss"] for entry in history],
        label="Discriminateur",
    )
    axis.set(xlabel="Époque", ylabel="Perte BCE", title="Entraînement du GAN")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_monitoring_plot(
    history: list[dict[str, Any]], collapse_ratio: float, path: Path
) -> None:
    if "pixel_diversity_ratio" not in history[0]:
        return
    epochs = list(range(1, len(history) + 1))
    has_classifier = "validation_class_fidelity" in history[0]
    columns = 3 if has_classifier else 2
    figure, axes = plt.subplots(1, columns, figsize=(5 * columns, 4))

    axes[0].plot(epochs, [h["d_real_probability"] for h in history], label="D(réel)")
    axes[0].plot(epochs, [h["d_fake_probability"] for h in history], label="D(généré)")
    axes[0].axhline(0.5, color="grey", linestyle=":", linewidth=1)
    axes[0].set(xlabel="Époque", ylabel="Probabilité moyenne", ylim=(0, 1),
                title="Sorties du discriminateur")
    axes[0].legend()

    axes[1].plot(epochs, [h["pixel_diversity_ratio"] for h in history], label="moyenne")
    axes[1].plot(epochs, [h["min_pixel_diversity_ratio"] for h in history],
                 label="pire classe")
    axes[1].axhline(collapse_ratio, color="tab:red", linestyle="--", linewidth=1,
                    label="seuil d'effondrement")
    axes[1].axhline(1.0, color="grey", linestyle=":", linewidth=1)
    axes[1].set(xlabel="Époque", ylabel="Diversité générée / réelle",
                title="Effondrement des modes")
    axes[1].legend()

    if has_classifier:
        axes[2].plot(epochs, [h["validation_fashion_kid"] for h in history],
                     color="tab:purple")
        axes[2].set(xlabel="Époque", ylabel="Fashion-KID (validation) ↓",
                    title="Réalisme et respect de la classe")
        twin = axes[2].twinx()
        twin.plot(epochs, [h["validation_class_fidelity"] for h in history],
                  color="tab:green")
        twin.set_ylabel("Précision classe (validation) ↑", color="tab:green")
        twin.set_ylim(0, 1)

    for axis in axes:
        axis.grid(alpha=0.25)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def state_copy(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be greater than zero")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be greater than zero")
    if args.noise_dim < 1:
        raise ValueError("--noise-dim must be greater than zero")
    if args.monitor_samples_per_class == 1 or args.monitor_samples_per_class < 0:
        raise ValueError("--monitor-samples-per-class must be 0 or at least 2")
    if args.save_every < 0:
        raise ValueError("--save-every must be non-negative")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader = make_train_validation_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    classifier = None
    if args.classifier.is_file():
        classifier = FashionClassifier().to(device)
        classifier.load_state_dict(
            torch.load(args.classifier, map_location="cpu", weights_only=True)[
                "model_state_dict"
            ]
        )
        classifier.eval()
    elif args.selection == "validation_kid":
        raise FileNotFoundError(
            f"--selection validation_kid needs the classifier: {args.classifier}"
        )

    generator = ConditionalGenerator(noise_dim=args.noise_dim).to(device)
    discriminator = ConditionalDiscriminator().to(device)
    monitor = None
    if args.monitor_samples_per_class:
        monitor = CollapseMonitor(
            collect_validation_reference(validation_loader, args.monitor_samples_per_class),
            args.noise_dim,
            args.seed,
            device,
            classifier,
            args.collapse_ratio,
        )
    elif args.selection == "validation_kid":
        raise ValueError("--selection validation_kid needs monitoring")

    generator_optimizer = Adam(
        generator.parameters(), lr=2e-4, betas=(0.5, 0.999)
    )
    discriminator_optimizer = Adam(
        discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999)
    )
    loss_fn = nn.BCEWithLogitsLoss()
    history: list[dict[str, Any]] = []
    epoch_dir = args.checkpoint.with_name(args.checkpoint.stem + "_epochs")
    best_kid = float("inf")
    best_epoch = args.epochs
    best_generator_state: dict[str, torch.Tensor] | None = None
    start_time = time.perf_counter()
    monitoring_seconds = 0.0

    print(f"Appareil : {device}")
    if device.type == "cuda":
        print(f"GPU : {torch.cuda.get_device_name(device)}")
    for epoch in range(1, args.epochs + 1):
        metrics: dict[str, Any] = {"epoch": epoch}
        metrics.update(
            train_epoch(
                generator,
                discriminator,
                train_loader,
                generator_optimizer,
                discriminator_optimizer,
                loss_fn,
                device,
                max_batches=args.max_batches,
            )
        )
        monitor_start = time.perf_counter()
        if monitor is not None:
            metrics.update(monitor(generator))
        monitoring_seconds += time.perf_counter() - monitor_start
        history.append(metrics)

        line = (
            f"Époque {epoch}/{args.epochs} "
            f"- perte G : {metrics['generator_loss']:.4f} "
            f"- perte D : {metrics['discriminator_loss']:.4f} "
            f"- D(réel) {metrics['d_real_probability']:.2f} "
            f"D(généré) {metrics['d_fake_probability']:.2f}"
        )
        if monitor is not None:
            line += f" - diversité/réel {metrics['pixel_diversity_ratio']:.2f}"
            line += f" (min {metrics['min_pixel_diversity_ratio']:.2f})"
            if "validation_fashion_kid" in metrics:
                line += (
                    f" - précision {metrics['validation_class_fidelity']:.1%}"
                    f" - Fashion-KID {metrics['validation_fashion_kid']:.2f}"
                )
            if metrics["collapsed_classes"]:
                line += f" - EFFONDREMENT : {', '.join(metrics['collapsed_classes'])}"
        print(line)

        if args.selection == "validation_kid" and metrics["validation_fashion_kid"] < best_kid:
            best_kid = metrics["validation_fashion_kid"]
            best_epoch = epoch
            best_generator_state = state_copy(generator)
        if args.save_every and (epoch % args.save_every == 0 or epoch == args.epochs):
            epoch_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "generator_state_dict": state_copy(generator),
                    "discriminator_state_dict": state_copy(discriminator),
                    "noise_dim": args.noise_dim,
                    "seed": args.seed,
                    "epochs": epoch,
                    "training_seconds": time.perf_counter() - start_time - monitoring_seconds,
                },
                epoch_dir / f"epoch_{epoch:03d}.pt",
            )

    # Training time excludes the monitoring so that it stays comparable with
    # the VAE and diffusion scripts.
    elapsed = time.perf_counter() - start_time - monitoring_seconds
    if best_generator_state is not None:
        generator.load_state_dict(best_generator_state)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "generator_state_dict": state_copy(generator),
            "discriminator_state_dict": state_copy(discriminator),
            "class_names": CLASS_NAMES,
            "noise_dim": args.noise_dim,
            "seed": args.seed,
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "selection": args.selection,
            "training_seconds": elapsed,
            "monitoring_seconds": monitoring_seconds,
        },
        args.checkpoint,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "epochs": args.epochs,
                "noise_dim": args.noise_dim,
                "selection": args.selection,
                "best_epoch": best_epoch,
                "collapse_ratio": args.collapse_ratio,
                "training_seconds": elapsed,
                "monitoring_seconds": monitoring_seconds,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_loss_plot(history, args.output_dir / "loss.png")
    save_monitoring_plot(history, args.collapse_ratio, args.output_dir / "monitoring.png")
    save_visualizations(generator, args.output_dir, args.seed, device)
    print(f"Modèle retenu : époque {best_epoch} ({args.selection})")
    print(f"Durée : {elapsed:.1f} s (dont suivi {monitoring_seconds:.1f} s)")
    print(f"Point de sauvegarde : {args.checkpoint}")
    print(f"Visualisations et mesures : {args.output_dir}")


if __name__ == "__main__":
    main()
