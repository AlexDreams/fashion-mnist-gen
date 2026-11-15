"""Measure conditional GAN class fidelity with the frozen real-image classifier."""

import argparse
import json
from pathlib import Path

import torch

from fashion_mnist_gen.classifier import FashionClassifier
from fashion_mnist_gen.data import CLASS_NAMES
from fashion_mnist_gen.gan import ConditionalGenerator
from fashion_mnist_gen.metrics import class_fidelity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/gan.pt"))
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
        default=Path("runs/gan/class_fidelity.json"),
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

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    generator = ConditionalGenerator(noise_dim=checkpoint["noise_dim"]).to(device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()
    labels = torch.arange(len(CLASS_NAMES), device=device).repeat_interleave(
        args.samples_per_class
    )
    random_generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.inference_mode():
        images = generator.sample(labels, generator=random_generator)
        predictions = classifier(images).argmax(dim=1)
    overall, per_class = class_fidelity(
        predictions, labels, num_classes=len(CLASS_NAMES)
    )
    results = {
        "checkpoint": str(args.checkpoint),
        "classifier": str(args.classifier),
        "seed": args.seed,
        "samples_per_class": args.samples_per_class,
        "class_names": list(CLASS_NAMES),
        "overall_accuracy": overall,
        "per_class_accuracy": dict(zip(CLASS_NAMES, per_class)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Accord moyen du classifieur : {overall:.2%}")
    for class_name, accuracy in zip(CLASS_NAMES, per_class):
        print(f"  {class_name} : {accuracy:.2%}")
    print(f"Résultats sauvegardés dans : {args.output}")


if __name__ == "__main__":
    main()
