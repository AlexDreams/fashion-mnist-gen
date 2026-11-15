"""Train the real-image classifier used by the generation metrics."""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from fashion_mnist_gen.classifier import FashionClassifier, evaluate_accuracy
from fashion_mnist_gen.data import CLASS_NAMES, make_train_validation_loaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/fashion_classifier.pt"),
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


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be greater than zero")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be greater than zero")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader = make_train_validation_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    model = FashionClassifier().to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    start_time = time.perf_counter()

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_examples += labels.size(0)

        accuracy = evaluate_accuracy(model, validation_loader, device)
        average_loss = total_loss / total_examples
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"- train loss: {average_loss:.4f} "
            f"- validation accuracy: {accuracy:.2%}"
        )

    elapsed = time.perf_counter() - start_time
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "class_names": CLASS_NAMES,
            "seed": args.seed,
            "epochs": args.epochs,
            "validation_accuracy": accuracy,
            "training_seconds": elapsed,
        },
        args.output,
    )
    print(f"Training time: {elapsed:.1f} s")
    print(f"Checkpoint saved to: {args.output}")


if __name__ == "__main__":
    main()
