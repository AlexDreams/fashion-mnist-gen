"""Measure class fidelity of VAE samples using the frozen real-image classifier."""

import argparse
import json
from pathlib import Path

import torch

from fashion_mnist_gen.classifier import FashionClassifier
from fashion_mnist_gen.data import CLASS_NAMES
from fashion_mnist_gen.metrics import class_fidelity
from fashion_mnist_gen.vae import ConditionalVAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=Path,
        default=[Path("checkpoints/vae.pt"), Path("checkpoints/vae_beta05.pt")],
    )
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--classifier",
        type=Path,
        default=Path("checkpoints/fashion_classifier.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/vae/class_fidelity.json"),
    )
    return parser.parse_args()


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

    labels = torch.arange(len(CLASS_NAMES), device=device).repeat_interleave(
        args.samples_per_class
    )
    models: dict[str, dict[str, object]] = {}
    results: dict[str, object] = {
        "seed": args.seed,
        "samples_per_class": args.samples_per_class,
        "classifier": str(args.classifier),
        "class_names": list(CLASS_NAMES),
        "models": models,
    }

    for checkpoint_path in args.checkpoints:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        model = ConditionalVAE(latent_dim=checkpoint["latent_dim"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        generator = torch.Generator(device=device).manual_seed(args.seed)
        with torch.inference_mode():
            images = model.sample(labels, generator=generator)
            predictions = classifier(images).argmax(dim=1)
        overall, per_class = class_fidelity(
            predictions, labels, num_classes=len(CLASS_NAMES)
        )
        models[checkpoint_path.stem] = {
            "checkpoint": str(checkpoint_path),
            "beta": checkpoint["beta"],
            "overall_accuracy": overall,
            "per_class_accuracy": dict(zip(CLASS_NAMES, per_class)),
        }
        print(f"{checkpoint_path.stem} : accord moyen {overall:.2%}")
        for class_name, accuracy in zip(CLASS_NAMES, per_class):
            print(f"  {class_name} : {accuracy:.2%}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Résultats sauvegardés dans : {args.output}")


if __name__ == "__main__":
    main()
