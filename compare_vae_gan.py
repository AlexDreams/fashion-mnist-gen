"""Compare VAE and conditional GAN class fidelity on the same sample protocol."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from fashion_mnist_gen.classifier import FashionClassifier
from fashion_mnist_gen.data import CLASS_NAMES
from fashion_mnist_gen.gan import ConditionalGenerator
from fashion_mnist_gen.metrics import class_fidelity
from fashion_mnist_gen.vae import ConditionalVAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae", type=Path, default=Path("checkpoints/vae.pt"))
    parser.add_argument("--gan", type=Path, default=Path("checkpoints/gan.pt"))
    parser.add_argument(
        "--classifier",
        type=Path,
        default=Path("checkpoints/fashion_classifier.pt"),
    )
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/comparison/vae_vs_gan.json"),
    )
    return parser.parse_args()


def evaluate_generator(
    name: str,
    model: nn.Module,
    labels: torch.Tensor,
    classifier: FashionClassifier,
    device: torch.device,
    seed: int,
    training_seconds: float,
) -> dict[str, Any]:
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    random_generator = torch.Generator(device=device).manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        images = model.sample(labels, generator=random_generator)
        predictions = classifier(images).argmax(dim=1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    generation_seconds = time.perf_counter() - start
    overall, per_class = class_fidelity(
        predictions, labels, num_classes=len(CLASS_NAMES)
    )
    return {
        "name": name,
        "parameter_count": parameter_count,
        "training_seconds": training_seconds,
        "generation_seconds": generation_seconds,
        "seconds_per_image": generation_seconds / labels.numel(),
        "overall_accuracy": overall,
        "per_class_accuracy": dict(zip(CLASS_NAMES, per_class)),
    }


def main() -> None:
    args = parse_args()
    if args.samples_per_class < 1:
        raise ValueError("--samples-per-class must be greater than zero")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier_checkpoint = torch.load(
        args.classifier, map_location="cpu", weights_only=True
    )
    classifier = FashionClassifier().to(device)
    classifier.load_state_dict(classifier_checkpoint["model_state_dict"])
    classifier.eval()

    vae_checkpoint = torch.load(args.vae, map_location="cpu", weights_only=True)
    vae = ConditionalVAE(latent_dim=vae_checkpoint["latent_dim"]).to(device)
    vae.load_state_dict(vae_checkpoint["model_state_dict"])

    gan_checkpoint = torch.load(args.gan, map_location="cpu", weights_only=True)
    gan = ConditionalGenerator(noise_dim=gan_checkpoint["noise_dim"]).to(device)
    gan.load_state_dict(gan_checkpoint["generator_state_dict"])

    labels = torch.arange(len(CLASS_NAMES), device=device).repeat_interleave(
        args.samples_per_class
    )
    models = [
        evaluate_generator(
            "vae",
            vae,
            labels,
            classifier,
            device,
            args.seed,
            vae_checkpoint["training_seconds"],
        ),
        evaluate_generator(
            "gan",
            gan,
            labels,
            classifier,
            device,
            args.seed,
            gan_checkpoint["training_seconds"],
        ),
    ]
    results = {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "classifier": str(args.classifier),
        "seed": args.seed,
        "samples_per_class": args.samples_per_class,
        "class_names": list(CLASS_NAMES),
        "models": models,
        "gan_minus_vae_class_fidelity": {
            class_name: gan_score - vae_score
            for class_name, vae_score, gan_score in zip(
                CLASS_NAMES,
                models[0]["per_class_accuracy"].values(),
                models[1]["per_class_accuracy"].values(),
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for model_metrics in models:
        print(
            f"{model_metrics['name']} : fidélité "
            f"{model_metrics['overall_accuracy']:.2%}, "
            f"génération {model_metrics['seconds_per_image']:.5f} s/image, "
            f"entraînement {model_metrics['training_seconds']:.1f} s"
        )
    print(f"Résultats sauvegardés dans : {args.output}")


if __name__ == "__main__":
    main()
