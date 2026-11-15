"""Train the level-3 latent diffusion in the latent space of the trained VAE.

The VAE (level 0) is frozen. Training and validation images are encoded once
(encoder means, standardized with training statistics), then a small MLP
denoiser is trained on these 16-D vectors with the usual noise-prediction loss
and class dropout for classifier-free guidance. The best epoch on the
validation MSE is kept. The checkpoint embeds the VAE weights, so it is
self-contained for evaluation and the interface.
"""

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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import save_image

from fashion_mnist_gen.data import CLASS_NAMES, make_train_validation_loaders
from fashion_mnist_gen.latent_diffusion import LatentDiffusion
from fashion_mnist_gen.vae import ConditionalVAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae", type=Path, default=Path("checkpoints/vae.pt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--class-dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/latent_diffusion"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/latent_diffusion.pt")
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def encode_split(
    vae: ConditionalVAE,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    vae.eval()
    latents, labels = [], []
    for images, batch_labels in loader:
        mean, _ = vae.encode(images.to(device), batch_labels.to(device))
        latents.append(mean.cpu())
        labels.append(batch_labels)
    return torch.cat(latents), torch.cat(labels)


def run_epoch(
    diffusion: LatentDiffusion,
    latents: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    optimizer: AdamW | None,
    class_dropout: float,
    device: torch.device,
    generator: torch.Generator,
) -> float:
    training = optimizer is not None
    diffusion.model.train(training)
    order = (
        torch.randperm(len(latents), generator=generator)
        if training
        else torch.arange(len(latents))
    )
    total, count = 0.0, 0
    for start in range(0, len(latents), batch_size):
        index = order[start : start + batch_size]
        batch = latents[index].to(device)
        batch_labels = labels[index].to(device)
        with torch.set_grad_enabled(training):
            loss = diffusion.training_loss(
                batch,
                batch_labels,
                class_dropout_probability=class_dropout,
                generator=None if training else generator,
            )
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.model.parameters(), 1.0)
            optimizer.step()
        total += loss.item() * len(index)
        count += len(index)
    return total / count


def save_outputs(
    diffusion: LatentDiffusion,
    history: list[dict[str, float]],
    output_dir: Path,
    seed: int,
    device: torch.device,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    epochs = range(1, len(history) + 1)
    axis.plot(epochs, [h["train_loss"] for h in history], label="Entraînement")
    axis.plot(epochs, [h["validation_loss"] for h in history], label="Validation")
    axis.set(xlabel="Époque", ylabel="MSE du bruit", title="Diffusion latente")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "loss.png", dpi=150)
    plt.close(figure)

    labels = torch.arange(len(CLASS_NAMES), device=device).repeat_interleave(8)
    generator = torch.Generator(device=device).manual_seed(seed)
    images = diffusion.sample(
        labels, method="ddim", num_inference_steps=50, guidance_scale=1.5,
        generator=generator,
    )
    save_image(images.cpu(), output_dir / "samples_by_class.png", nrow=8, padding=2)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive")
    if not args.vae.is_file():
        raise FileNotFoundError(f"Train the VAE first (missing {args.vae})")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_checkpoint = torch.load(args.vae, map_location="cpu", weights_only=True)
    vae = ConditionalVAE(latent_dim=vae_checkpoint["latent_dim"])
    vae.load_state_dict(vae_checkpoint["model_state_dict"])
    vae.to(device)

    train_loader, validation_loader = make_train_validation_loaders(
        data_dir=args.data_dir, batch_size=1024, seed=args.seed
    )
    start_time = time.perf_counter()
    train_latents, train_labels = encode_split(vae, train_loader, device)
    validation_latents, validation_labels = encode_split(vae, validation_loader, device)
    latent_mean = train_latents.mean(dim=0)
    latent_std = train_latents.std(dim=0).clamp_min(1e-3)

    diffusion = LatentDiffusion(
        vae,
        latent_mean,
        latent_std,
        num_timesteps=args.timesteps,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
    ).to(device)
    train_latents = diffusion.normalize(train_latents.to(device)).cpu()
    validation_latents = diffusion.normalize(validation_latents.to(device)).cpu()

    optimizer = AdamW(diffusion.model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    parameter_count = sum(p.numel() for p in diffusion.model.parameters())
    print(f"Appareil : {device} - paramètres du débruiteur : {parameter_count:,}")
    shuffle_generator = torch.Generator().manual_seed(args.seed)
    history: list[dict[str, float]] = []
    best_loss, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            diffusion, train_latents, train_labels, args.batch_size, optimizer,
            args.class_dropout, device, shuffle_generator,
        )
        scheduler.step()
        validation_generator = torch.Generator(device=device).manual_seed(args.seed)
        validation_loss = run_epoch(
            diffusion, validation_latents, validation_labels, args.batch_size, None,
            args.class_dropout, device, validation_generator,
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss}
        )
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in diffusion.model.state_dict().items()
            }
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Époque {epoch}/{args.epochs} - MSE train {train_loss:.4f} "
                f"- validation {validation_loss:.4f}"
            )

    elapsed = time.perf_counter() - start_time
    diffusion.model.load_state_dict(best_state)
    payload = diffusion.checkpoint_payload()
    payload.update(
        {
            "class_names": CLASS_NAMES,
            "seed": args.seed,
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "class_dropout": args.class_dropout,
            "training_seconds": elapsed,
            "vae_checkpoint": str(args.vae),
            "vae_training_seconds": vae_checkpoint.get("training_seconds"),
        }
    )
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "epochs": args.epochs,
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "training_seconds": elapsed,
                "vae_training_seconds": vae_checkpoint.get("training_seconds"),
                "denoiser_parameters": parameter_count,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_outputs(diffusion, history, args.output_dir, args.seed, device)
    print(f"Meilleure époque : {best_epoch} (validation {best_loss:.4f})")
    print(f"Durée : {elapsed:.1f} s (+ VAE {vae_checkpoint.get('training_seconds', 0):.0f} s)")
    print(f"Point de sauvegarde : {args.checkpoint}")


if __name__ == "__main__":
    main()
