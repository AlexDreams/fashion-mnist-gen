"""Train the class-conditional Fashion-MNIST VAE baseline."""

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam

from fashion_mnist_gen.data import CLASS_NAMES, make_train_validation_loaders
from fashion_mnist_gen.vae import ConditionalVAE, vae_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/vae"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/vae.pt"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def run_epoch(
    model: ConditionalVAE,
    data_loader: torch.utils.data.DataLoader,
    optimizer: Adam | None,
    device: torch.device,
    beta: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0}
    total_examples = 0

    with torch.set_grad_enabled(training):
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            reconstructions, mean, log_variance = model(images, labels)
            loss, reconstruction, kl = vae_loss(
                reconstructions, images, mean, log_variance, beta=beta
            )
            if optimizer is not None:
                loss.backward()
                optimizer.step()

            count = images.shape[0]
            totals["loss"] += loss.item() * count
            totals["reconstruction"] += reconstruction.item() * count
            totals["kl"] += kl.item() * count
            total_examples += count

    if total_examples == 0:
        raise ValueError("Cannot run an epoch with an empty data loader")
    return {name: value / total_examples for name, value in totals.items()}


def save_visualizations(
    model: ConditionalVAE,
    validation_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: Path,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    images, image_labels = next(iter(validation_loader))
    images = images[:8].to(device)
    image_labels = image_labels[:8].to(device)
    with torch.no_grad():
        reconstructions, _, _ = model(images, image_labels)
        figure, axes = plt.subplots(8, 2, figsize=(4, 12))
        for index, (image, reconstruction, label) in enumerate(
            zip(images.cpu(), reconstructions.cpu(), image_labels.cpu())
        ):
            axes[index, 0].imshow(image.squeeze(), cmap="gray", vmin=0, vmax=1)
            axes[index, 1].imshow(
                reconstruction.squeeze(), cmap="gray", vmin=0, vmax=1
            )
            axes[index, 0].set_ylabel(CLASS_NAMES[int(label)], rotation=0, labelpad=40)
            for axis in axes[index]:
                axis.set_xticks([])
                axis.set_yticks([])
        axes[0, 0].set_title("Original")
        axes[0, 1].set_title("Reconstruction")
        figure.tight_layout()
        figure.savefig(output_dir / "reconstructions.png", dpi=150)
        plt.close(figure)

        requested_classes = torch.arange(
            len(CLASS_NAMES), device=device, dtype=torch.long
        ).repeat_interleave(8)
        generator = torch.Generator(device=device).manual_seed(seed)
        samples = model.sample(requested_classes, generator=generator).cpu()
        figure, axes = plt.subplots(10, 8, figsize=(10, 13))
        for class_index, row in enumerate(axes):
            for sample_index, axis in enumerate(row):
                axis.imshow(
                    samples[class_index * 8 + sample_index].squeeze(),
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                )
                axis.set_xticks([])
                axis.set_yticks([])
            row[0].set_ylabel(CLASS_NAMES[class_index], rotation=0, labelpad=55)
        figure.suptitle("Échantillons générés par classe")
        figure.tight_layout()
        figure.savefig(output_dir / "samples_by_class.png", dpi=150)
        plt.close(figure)

        interpolation_images: list[torch.Tensor] = []
        for batch_images, batch_labels in validation_loader:
            class_matches = (batch_labels == 9).nonzero().flatten()
            for match in class_matches:
                interpolation_images.append(batch_images[match])
            if len(interpolation_images) >= 64:
                break
        if len(interpolation_images) < 2:
            raise RuntimeError(
                "Could not find two ankle-boot images for interpolation"
            )
        interpolation_images = interpolation_images[:64]

        endpoints = torch.stack(interpolation_images).to(device)
        endpoint_labels = torch.full(
            (len(endpoints),), 9, dtype=torch.long, device=device
        )
        endpoint_means, _ = model.encode(endpoints, endpoint_labels)
        distances = torch.cdist(endpoint_means, endpoint_means)
        first_indices, second_indices = torch.triu_indices(
            len(endpoints), len(endpoints), offset=1, device=device
        )
        farthest_pair = distances[first_indices, second_indices].argmax()
        first_endpoint = endpoint_means[first_indices[farthest_pair]]
        second_endpoint = endpoint_means[second_indices[farthest_pair]]
        fractions = torch.linspace(0, 1, steps=8, device=device).unsqueeze(1)
        latent_path = (1 - fractions) * first_endpoint + fractions * second_endpoint
        interpolated = model.decode(
            latent_path, endpoint_labels[:1].expand(fractions.shape[0])
        ).cpu()
        figure, axes = plt.subplots(1, len(interpolated), figsize=(12, 2))
        for index, axis in enumerate(axes):
            axis.imshow(interpolated[index].squeeze(), cmap="gray", vmin=0, vmax=1)
            axis.set_xticks([])
            axis.set_yticks([])
        figure.suptitle("Interpolation latente — bottes")
        figure.tight_layout()
        figure.savefig(output_dir / "latent_interpolation.png", dpi=150)
        plt.close(figure)


def save_loss_plot(history: list[dict[str, float]], path: Path) -> None:
    epochs = range(1, len(history) + 1)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(epochs, [entry["train_loss"] for entry in history], label="Train")
    axis.plot(
        epochs,
        [entry["validation_loss"] for entry in history],
        label="Validation",
    )
    axis.set(xlabel="Époque", ylabel="Perte ELBO par image", title="Entraînement du VAE")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be greater than zero")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be greater than zero")
    if args.beta < 0:
        raise ValueError("--beta must be non-negative")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader = make_train_validation_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    model = ConditionalVAE(latent_dim=args.latent_dim).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    history: list[dict[str, float]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    start_time = time.perf_counter()

    print(f"Appareil : {device}")
    if device.type == "cuda":
        print(f"GPU : {torch.cuda.get_device_name(device)}")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, optimizer, device, beta=args.beta
        )
        validation_metrics = run_epoch(
            model, validation_loader, None, device, beta=args.beta
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_reconstruction": train_metrics["reconstruction"],
                "train_kl": train_metrics["kl"],
                "validation_loss": validation_metrics["loss"],
                "validation_reconstruction": validation_metrics["reconstruction"],
                "validation_kl": validation_metrics["kl"],
            }
        )
        print(
            f"Époque {epoch}/{args.epochs} "
            f"- perte train : {train_metrics['loss']:.2f} "
            f"(reconstruction {train_metrics['reconstruction']:.2f}, "
            f"KL {train_metrics['kl']:.2f}) "
            f"- validation : {validation_metrics['loss']:.2f}"
        )
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }

    elapsed = time.perf_counter() - start_time
    if best_state is None:
        raise RuntimeError("Training completed without saving a best model state")
    model.load_state_dict(best_state)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "class_names": CLASS_NAMES,
            "latent_dim": args.latent_dim,
            "beta": args.beta,
            "seed": args.seed,
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "training_seconds": elapsed,
        },
        args.checkpoint,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "epochs": args.epochs,
                "latent_dim": args.latent_dim,
                "beta": args.beta,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "training_seconds": elapsed,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_loss_plot(history, args.output_dir / "loss.png")
    save_visualizations(
        model, validation_loader, device, args.output_dir, seed=args.seed
    )
    print(f"Meilleure époque : {best_epoch} (validation {best_validation_loss:.2f})")
    print(f"Durée : {elapsed:.1f} s")
    print(f"Point de sauvegarde : {args.checkpoint}")
    print(f"Visualisations et mesures : {args.output_dir}")


if __name__ == "__main__":
    main()
