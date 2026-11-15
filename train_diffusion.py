"""Train a small class-conditional DDPM on Fashion-MNIST."""

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
from torch.optim import AdamW

from fashion_mnist_gen.data import CLASS_NAMES, make_train_validation_loaders
from fashion_mnist_gen.diffusion import ConditionalDDPM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--class-dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/diffusion"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/diffusion.pt"),
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


def run_epoch(
    diffusion: ConditionalDDPM,
    data_loader: torch.utils.data.DataLoader,
    optimizer: AdamW | None,
    device: torch.device,
    class_dropout: float,
    max_batches: int = 0,
    generator: torch.Generator | None = None,
) -> float:
    training = optimizer is not None
    diffusion.train(training)
    total_loss = 0.0
    total_examples = 0
    with torch.set_grad_enabled(training):
        for batch_index, (images, labels) in enumerate(data_loader):
            if max_batches and batch_index >= max_batches:
                break
            images = images.to(device, non_blocking=True).mul(2).sub(1)
            labels = labels.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss = diffusion.training_loss(
                images,
                labels,
                class_dropout_probability=class_dropout if training else 0.0,
                generator=generator,
            )
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * labels.shape[0]
            total_examples += labels.shape[0]

    if total_examples == 0:
        raise ValueError("Cannot run a diffusion epoch with no batches")
    return total_loss / total_examples


@torch.inference_mode()
def save_samples(
    diffusion: ConditionalDDPM,
    output_dir: Path,
    seed: int,
    device: torch.device,
) -> None:
    diffusion.eval()
    labels = torch.arange(len(CLASS_NAMES), device=device).repeat_interleave(8)
    generator = torch.Generator(device=device).manual_seed(seed)
    images = diffusion.sample(
        labels,
        method="ddim",
        num_inference_steps=min(50, diffusion.num_timesteps),
        guidance_scale=1.5,
        generator=generator,
    ).cpu()

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
    figure.suptitle("Échantillons DDIM générés par classe")
    figure.tight_layout()
    figure.savefig(output_dir / "samples_by_class.png", dpi=150)
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
    axis.set(
        xlabel="Époque",
        ylabel="MSE du bruit prédit",
        title="Entraînement du DDPM",
    )
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
    if args.timesteps < 2:
        raise ValueError("--timesteps must be at least two")
    if not 0 <= args.class_dropout <= 1:
        raise ValueError("--class-dropout must be between zero and one")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.max_batches < 0:
        raise ValueError("--max-batches must be non-negative")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader = make_train_validation_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    diffusion = ConditionalDDPM(
        num_timesteps=args.timesteps,
        base_channels=args.base_channels,
    ).to(device)
    optimizer = AdamW(diffusion.parameters(), lr=args.learning_rate)
    history: list[dict[str, float]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    start_time = time.perf_counter()

    print(f"Appareil : {device}")
    if device.type == "cuda":
        print(f"GPU : {torch.cuda.get_device_name(device)}")
    print(
        f"Paramètres : {sum(p.numel() for p in diffusion.model.parameters()):,} "
        f"- pas de bruit : {args.timesteps}"
    )
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            diffusion,
            train_loader,
            optimizer,
            device,
            class_dropout=args.class_dropout,
            max_batches=args.max_batches,
        )
        validation_generator = torch.Generator(device=device).manual_seed(args.seed)
        validation_loss = run_epoch(
            diffusion,
            validation_loader,
            None,
            device,
            class_dropout=args.class_dropout,
            max_batches=args.max_batches,
            generator=validation_generator,
        )
        history.append({"train_loss": train_loss, "validation_loss": validation_loss})
        print(
            f"Époque {epoch}/{args.epochs} "
            f"- MSE train : {train_loss:.5f} "
            f"- validation : {validation_loss:.5f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in diffusion.model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training completed without saving a best model state")
    elapsed = time.perf_counter() - start_time
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "num_timesteps": args.timesteps,
            "base_channels": args.base_channels,
            "class_names": CLASS_NAMES,
            "class_dropout": args.class_dropout,
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
                "num_timesteps": args.timesteps,
                "base_channels": args.base_channels,
                "class_dropout": args.class_dropout,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "training_seconds": elapsed,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    diffusion.model.load_state_dict(best_state)
    save_loss_plot(history, args.output_dir / "loss.png")
    save_samples(diffusion, args.output_dir, args.seed, device)
    print(f"Meilleure époque : {best_epoch} (validation {best_validation_loss:.5f})")
    print(f"Durée : {elapsed:.1f} s")
    print(f"Point de sauvegarde : {args.checkpoint}")
    print(f"Visualisations et mesures : {args.output_dir}")


if __name__ == "__main__":
    main()
