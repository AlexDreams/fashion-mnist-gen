"""Evaluate generators on a fixed Fashion-MNIST test protocol.

Every level (VAE, GAN, diffusion settings) is scored on the same real test
images, with the same number of generated images per class and fixed seeds:

* realism: Inception KID (torchmetrics, the brief's KID) and Fashion-KID (same
  unbiased MMD in the features of the local Fashion classifier);
* class respect: accuracy of the frozen classifier on generated images;
* diversity: class coverage of the predicted labels, feature coverage of the
  real test images, intra-class pixel distance compared with real images
  (a ratio far below one flags mode collapse);
* cost: training time, seconds per image and parameter count.

A "real" reference row scores held-out real test images exactly like a
generator; it gives the ceiling that each metric can reach.
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from fashion_mnist_gen.classifier import FashionClassifier
from fashion_mnist_gen.data import CLASS_NAMES, make_dataloaders
from fashion_mnist_gen.diffusion import ConditionalDDPM
from fashion_mnist_gen.gan import ConditionalGenerator
from fashion_mnist_gen.latent_diffusion import LatentDiffusion
from fashion_mnist_gen.metrics import (
    class_coverage,
    class_fidelity,
    fashion_kid,
    feature_diversity,
    pixel_diversity,
    relative_feature_coverage,
)
from fashion_mnist_gen.vae import ConditionalVAE

PROTOCOL_VERSION = 2
LEVEL_LABELS = {"real": "Référence · vraies images", "vae": "0 · VAE", "gan": "1 · GAN"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("vae", "gan", "ddpm", "ldm"),
        default=("vae", "gan"),
    )
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=(2026, 2027, 2028))
    parser.add_argument(
        "--evaluation-partition",
        choices=("development", "final"),
        default="development",
    )
    parser.add_argument("--development-samples-per-class", type=int, default=100)
    parser.add_argument("--final-offset", type=int, default=0)
    parser.add_argument("--real-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--diffusion-sampler", choices=("ddpm", "ddim"), default="ddim")
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--diffusion-configs",
        nargs="+",
        default=None,
        metavar="SAMPLER:STEPS:GUIDANCE",
        help=(
            "Several diffusion settings scored in one run, e.g. "
            "ddpm:1000:1.5 ddim:50:1.5 ddim:10:1.5. Overrides "
            "--diffusion-sampler/--diffusion-steps/--guidance-scale."
        ),
    )
    parser.add_argument(
        "--inception-kid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Standard KID with torchmetrics' Inception features (slower on CPU).",
    )
    parser.add_argument(
        "--fid",
        action="store_true",
        help="Also report Inception FID (biased with 1 000 images; KID preferred).",
    )
    parser.add_argument("--kid-subsets", type=int, default=100)
    parser.add_argument(
        "--real-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Score held-out real test images as a ceiling row.",
    )
    parser.add_argument(
        "--collapse-ratio",
        type=float,
        default=0.6,
        help=(
            "A class is flagged as collapsed when its generated intra-class pixel "
            "distance is below this fraction of the real one."
        ),
    )
    parser.add_argument(
        "--class-coverage-share",
        type=float,
        default=0.5,
        help="A class is covered if it gets this fraction of its uniform share.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--classifier",
        type=Path,
        default=Path("checkpoints/fashion_classifier.pt"),
    )
    parser.add_argument("--vae", type=Path, default=Path("checkpoints/vae.pt"))
    parser.add_argument("--gan", type=Path, default=Path("checkpoints/gan.pt"))
    parser.add_argument(
        "--ddpm", type=Path, default=Path("checkpoints/diffusion.pt")
    )
    parser.add_argument(
        "--ldm",
        type=Path,
        default=Path("checkpoints/latent_diffusion.pt"),
        help="Level 3: latent diffusion checkpoint (train_latent_diffusion.py).",
    )
    parser.add_argument(
        "--human-results",
        type=Path,
        default=Path("runs/human_test/results.json"),
        help="Blind human test scores (human_test.py score); added when present.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/protocol"),
    )
    return parser.parse_args(argv)


def parse_diffusion_config(text: str) -> dict[str, Any]:
    """Parse ``sampler:steps:guidance`` such as ``ddim:50:1.5``."""
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError(f"Diffusion config must be SAMPLER:STEPS:GUIDANCE, got {text!r}")
    sampler, steps, guidance = parts[0].lower(), int(parts[1]), float(parts[2])
    if sampler not in {"ddpm", "ddim"}:
        raise ValueError(f"Unknown diffusion sampler {sampler!r}")
    if steps < 1:
        raise ValueError("Diffusion steps must be positive")
    if guidance < 0:
        raise ValueError("Guidance scale must be non-negative")
    return {"method": sampler, "steps": steps, "guidance_scale": guidance}


DIFFUSION_MODELS = ("ddpm", "ldm")


def diffusion_key(config: dict[str, Any], name: str = "ddpm") -> str:
    return f"{name}_{config['method']}{config['steps']}_g{config['guidance_scale']:g}"


def diffusion_label(config: dict[str, Any], name: str = "ddpm") -> str:
    prefix = "2 · Diffusion" if name == "ddpm" else "3 · Diffusion latente"
    return (
        f"{prefix} {config['method'].upper()} {config['steps']} pas, "
        f"guidance {config['guidance_scale']:g}"
    )


def load_generator(
    name: str,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if name == "vae":
        model = ConditionalVAE(latent_dim=checkpoint["latent_dim"])
        model.load_state_dict(checkpoint["model_state_dict"])
    elif name == "gan":
        model = ConditionalGenerator(noise_dim=checkpoint["noise_dim"])
        model.load_state_dict(checkpoint["generator_state_dict"])
    elif name == "ldm":
        model = LatentDiffusion.from_checkpoint(checkpoint)
    else:
        model = ConditionalDDPM(
            num_timesteps=checkpoint["num_timesteps"],
            base_channels=checkpoint["base_channels"],
        )
        model.model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint


def collect_real_images(
    test_loader: torch.utils.data.DataLoader,
) -> list[torch.Tensor]:
    """Return the official test images grouped by class, in dataset order."""
    images_by_class: list[list[torch.Tensor]] = [[] for _ in CLASS_NAMES]
    for images, labels in test_loader:
        for class_id in range(len(CLASS_NAMES)):
            mask = labels == class_id
            if mask.any():
                images_by_class[class_id].append(images[mask])
    if any(not class_images for class_images in images_by_class):
        raise ValueError("The Fashion-MNIST test set is missing a class")
    return [torch.cat(class_images) for class_images in images_by_class]


def select_real_feature_subset(
    features_by_class: list[torch.Tensor],
    samples_per_class: int,
    partition: str,
    development_samples_per_class: int,
    seed: int,
    final_offset: int = 0,
) -> list[torch.Tensor]:
    """Choose a fixed development or disjoint final subset from each test class.

    Works on any per-class tensors (images or features): the permutation only
    depends on ``seed`` and the class sizes.
    """
    if partition not in {"development", "final"}:
        raise ValueError("partition must be 'development' or 'final'")
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be positive")
    if development_samples_per_class < 1:
        raise ValueError("development_samples_per_class must be positive")
    if final_offset < 0:
        raise ValueError("final_offset must be non-negative")

    generator = torch.Generator().manual_seed(seed)
    selected: list[torch.Tensor] = []
    for features in features_by_class:
        permutation = torch.randperm(len(features), generator=generator)
        if partition == "development":
            start_index = 0
        else:
            start_index = development_samples_per_class + final_offset
        end_index = start_index + samples_per_class
        if end_index > len(features):
            raise ValueError(
                f"Requested {samples_per_class} samples per class for {partition}; "
                f"only {len(features) - start_index} are available"
            )
        selected.append(features[permutation[start_index:end_index]])
    return selected


def select_reference_subset(
    images_by_class: list[torch.Tensor],
    samples_per_class: int,
    seed: int,
    used_end_index: int,
) -> list[torch.Tensor]:
    """Take the last ``samples_per_class`` images of each class permutation.

    These real images act as a perfect "generator" for the ceiling row. They
    must not overlap the evaluated slice, which ends at ``used_end_index``.
    """
    generator = torch.Generator().manual_seed(seed)
    selected: list[torch.Tensor] = []
    for images in images_by_class:
        permutation = torch.randperm(len(images), generator=generator)
        start_index = len(images) - samples_per_class
        if start_index < used_end_index:
            raise ValueError(
                "Not enough test images for a disjoint real reference subset; "
                "use --no-real-reference or fewer samples per class"
            )
        selected.append(images[permutation[start_index:]])
    return selected


def sample_generated_images(
    name: str,
    model: torch.nn.Module,
    labels: torch.Tensor,
    batch_size: int,
    seed: int,
    device: torch.device,
    diffusion_sampler: str,
    diffusion_steps: int,
    guidance_scale: float,
) -> tuple[torch.Tensor, float]:
    warmup_labels = labels[: min(labels.shape[0], min(batch_size, 32))]
    warmup_generator = torch.Generator(device=device).manual_seed(seed + 1_000_003)
    if name in DIFFUSION_MODELS:
        model.sample(
            warmup_labels,
            method=diffusion_sampler,
            num_inference_steps=diffusion_steps,
            guidance_scale=guidance_scale,
            generator=warmup_generator,
        )
    else:
        model.sample(warmup_labels, generator=warmup_generator)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    random_generator = torch.Generator(device=device).manual_seed(seed)
    image_batches: list[torch.Tensor] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for start_index in range(0, labels.shape[0], batch_size):
        batch_labels = labels[start_index : start_index + batch_size]
        if name in DIFFUSION_MODELS:
            images = model.sample(
                batch_labels,
                method=diffusion_sampler,
                num_inference_steps=diffusion_steps,
                guidance_scale=guidance_scale,
                generator=random_generator,
            )
        else:
            images = model.sample(batch_labels, generator=random_generator)
        image_batches.append(images)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    generation_seconds = time.perf_counter() - start
    return torch.cat(image_batches).cpu(), generation_seconds


@torch.inference_mode()
def collect_generated_features(
    images: torch.Tensor,
    labels: torch.Tensor,
    classifier: FashionClassifier,
    batch_size: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    features_by_class: list[list[torch.Tensor]] = [
        [] for _ in range(len(CLASS_NAMES))
    ]
    predictions_by_class: list[list[torch.Tensor]] = [
        [] for _ in range(len(CLASS_NAMES))
    ]
    classifier.eval()
    for start_index in range(0, images.shape[0], batch_size):
        batch = images[start_index : start_index + batch_size].to(
            device, non_blocking=True
        )
        batch_labels = labels[start_index : start_index + batch_size]
        predictions = classifier(batch).argmax(dim=1)
        features = classifier.extract_features(batch)
        for class_id in range(len(CLASS_NAMES)):
            mask = batch_labels.to(device) == class_id
            if mask.any():
                features_by_class[class_id].append(features[mask].cpu())
                predictions_by_class[class_id].append(predictions[mask].cpu())

    return (
        [torch.cat(class_features) for class_features in features_by_class],
        [torch.cat(class_predictions) for class_predictions in predictions_by_class],
    )


def score_sample_run(
    real_features: list[torch.Tensor],
    generated_features: list[torch.Tensor],
    predictions: list[torch.Tensor],
    generated_images: list[torch.Tensor] | None = None,
    real_pixel_diversity: list[float] | None = None,
    collapse_ratio: float = 0.6,
    class_coverage_share: float = 0.5,
) -> dict[str, Any]:
    class_results: dict[str, dict[str, Any]] = {}
    kid_scores: list[float] = []
    diversity_scores: list[float] = []
    coverage_scores: list[float] = []
    fidelity_scores: list[float] = []
    pixel_ratios: list[float] = []
    collapsed_classes: list[str] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        expected = torch.full(
            (predictions[class_id].shape[0],),
            class_id,
            dtype=torch.long,
        )
        fidelity = class_fidelity(
            predictions[class_id], expected, num_classes=len(CLASS_NAMES)
        )[0]
        kid = fashion_kid(real_features[class_id], generated_features[class_id])
        generated_diversity = feature_diversity(generated_features[class_id])
        real_diversity = feature_diversity(real_features[class_id])
        coverage = relative_feature_coverage(
            real_features[class_id], generated_features[class_id]
        )
        class_results[class_name] = {
            "class_fidelity": fidelity,
            "fashion_kid": kid,
            "generated_feature_diversity": generated_diversity,
            "real_feature_diversity": real_diversity,
            "relative_feature_coverage": coverage,
        }
        if generated_images is not None and real_pixel_diversity is not None:
            generated_pixels = pixel_diversity(generated_images[class_id])
            ratio = generated_pixels / real_pixel_diversity[class_id]
            class_results[class_name].update(
                {
                    "generated_pixel_diversity": generated_pixels,
                    "real_pixel_diversity": real_pixel_diversity[class_id],
                    "pixel_diversity_ratio": ratio,
                    "mode_collapse_flag": ratio < collapse_ratio,
                }
            )
            pixel_ratios.append(ratio)
            if ratio < collapse_ratio:
                collapsed_classes.append(class_name)
        fidelity_scores.append(fidelity)
        kid_scores.append(kid)
        diversity_scores.append(generated_diversity)
        coverage_scores.append(coverage)

    all_predictions = torch.cat(predictions)
    predicted_coverage = class_coverage(
        all_predictions, len(CLASS_NAMES), min_share=class_coverage_share
    )
    result: dict[str, Any] = {
        "class_fidelity": sum(fidelity_scores) / len(fidelity_scores),
        "fashion_kid_mean": sum(kid_scores) / len(kid_scores),
        "fashion_kid_std_across_classes": torch.tensor(kid_scores).std(
            unbiased=False
        ).item(),
        "generated_feature_diversity": sum(diversity_scores) / len(diversity_scores),
        "relative_feature_coverage": sum(coverage_scores) / len(coverage_scores),
        "class_coverage": predicted_coverage["coverage"],
        "predicted_class_entropy": predicted_coverage["normalized_entropy"],
        "per_class": class_results,
    }
    if pixel_ratios:
        result["pixel_diversity_ratio"] = sum(pixel_ratios) / len(pixel_ratios)
        result["min_pixel_diversity_ratio"] = min(pixel_ratios)
        result["collapsed_class_count"] = len(collapsed_classes)
        result["collapsed_classes"] = collapsed_classes
    return result


def split_by_class(images: torch.Tensor, samples_per_class: int) -> list[torch.Tensor]:
    return [
        images[class_id * samples_per_class : (class_id + 1) * samples_per_class]
        for class_id in range(len(CLASS_NAMES))
    ]


def save_sample_grid(
    images: torch.Tensor,
    samples_per_class: int,
    path: Path,
) -> None:
    examples_per_class = min(8, samples_per_class)
    figure, axes = plt.subplots(
        len(CLASS_NAMES), examples_per_class, figsize=(examples_per_class * 1.4, 13)
    )
    for class_id, row in enumerate(axes):
        start = class_id * samples_per_class
        for image_index, axis in enumerate(row):
            axis.imshow(
                images[start + image_index].squeeze(),
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axis.set_xticks([])
            axis.set_yticks([])
        row[0].set_ylabel(CLASS_NAMES[class_id], rotation=0, labelpad=55)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


SCALAR_METRICS = (
    "class_fidelity",
    "fashion_kid_mean",
    "inception_kid",
    "inception_fid",
    "generated_feature_diversity",
    "relative_feature_coverage",
    "class_coverage",
    "predicted_class_entropy",
    "pixel_diversity_ratio",
    "min_pixel_diversity_ratio",
    "collapsed_class_count",
    "seconds_per_image",
)


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for metric_name in SCALAR_METRICS:
        values = [run[metric_name] for run in runs if metric_name in run]
        if not values:
            continue
        tensor = torch.tensor(values, dtype=torch.float64)
        summary[metric_name] = {
            "mean": tensor.mean().item(),
            "std": tensor.std(unbiased=False).item(),
        }
    return summary


def _format(value: float | None, pattern: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return pattern.format(value)


def training_cell(entry: dict[str, Any]) -> str:
    training = entry.get("training_seconds")
    if training is None:
        return "—"
    cell = f"{training:.0f} s ({entry.get('training_epochs')} ép.)"
    prerequisite = entry.get("prerequisite_training_seconds")
    if prerequisite:
        cell += f" + VAE {prerequisite:.0f} s"
    return cell


def results_markdown(results: dict[str, Any]) -> str:
    """Render the brief's level table (mean ± std over sampling seeds)."""
    has_fid = any(
        "inception_fid" in entry["summary"] for entry in results["models"].values()
    )
    has_human = any("human_test" in entry for entry in results["models"].values())
    header = [
        "Niveau",
        "KID Inception ×10³ ↓",
        *(["FID Inception ↓"] if has_fid else []),
        "Fashion-KID ↓",
        "Précision classe ↑",
        "Couverture classes ↑",
        "Couverture caract. ↑",
        "Diversité / réel",
        "Classes effondrées",
        *(["Test humain (pris pour vrai)"] if has_human else []),
        "Entraînement",
        "Temps / image",
        "Paramètres",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for entry in results["models"].values():
        summary = entry["summary"]

        def mean(name: str) -> float | None:
            return summary[name]["mean"] if name in summary else None

        def mean_std(name: str, scale: float, pattern: str) -> str:
            if name not in summary:
                return "—"
            return (
                pattern.format(summary[name]["mean"] * scale)
                + " ± "
                + pattern.format(summary[name]["std"] * scale)
            )

        row = [
            entry["label"],
            mean_std("inception_kid", 1000, "{:.2f}"),
            *([mean_std("inception_fid", 1, "{:.1f}")] if has_fid else []),
            mean_std("fashion_kid_mean", 1, "{:.2f}"),
            _format(mean("class_fidelity"), "{:.1%}"),
            _format(mean("class_coverage"), "{:.0%}"),
            _format(mean("relative_feature_coverage"), "{:.1%}"),
            _format(mean("pixel_diversity_ratio"), "{:.2f}"),
            _format(mean("collapsed_class_count"), "{:.0f}"),
            *(
                [
                    f"{entry['human_test']['judged_real_rate']:.0%} "
                    f"(n={entry['human_test']['answers']})"
                    if "human_test" in entry
                    else "—"
                ]
                if has_human
                else []
            ),
            training_cell(entry),
            "—" if entry["kind"] == "real" else _format(mean("seconds_per_image"), "{:.2e} s"),
            "—" if entry.get("parameter_count") is None else f"{entry['parameter_count']:,}".replace(",", " "),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append(
        f"Partition : {results['evaluation_partition']} · "
        f"{results['samples_per_class']} images par classe · graines "
        f"{', '.join(map(str, results['sampling_seeds']))} · appareil {results['device']}. "
        f"Précision du classifieur sur les vraies images évaluées : "
        f"{results['classifier_accuracy_on_evaluated_real_images']:.1%}."
    )
    return "\n".join(lines) + "\n"


def generator_parameter_count(name: str, model: torch.nn.Module) -> int:
    """Parameters used at generation time."""
    if name == "ddpm":
        return sum(p.numel() for p in model.model.parameters())
    if name == "ldm":
        return sum(p.numel() for p in model.model.parameters()) + sum(
            p.numel() for p in model.vae.decoder.parameters()
        )
    return sum(p.numel() for p in model.parameters())


def attach_human_results(results: dict[str, Any], path: Path) -> None:
    """Add the blind test score ("generated image judged real") when available."""
    if not path.is_file():
        return
    human = json.loads(path.read_text(encoding="utf-8"))
    results["human_test"] = {
        key: value for key, value in human.items() if key != "per_source"
    }
    for key, entry in results["models"].items():
        source = human.get("per_source", {}).get(key)
        if source:
            entry["human_test"] = source


def validate_args(args: argparse.Namespace) -> None:
    if args.samples_per_class < 2:
        raise ValueError("--samples-per-class must be at least two for KID")
    if args.development_samples_per_class < 1:
        raise ValueError("--development-samples-per-class must be positive")
    if args.final_offset < 0:
        raise ValueError("--final-offset must be non-negative")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be greater than zero")
    if not args.seeds:
        raise ValueError("At least one sampling seed is required")
    if args.guidance_scale < 0:
        raise ValueError("--guidance-scale must be non-negative")
    if not 0 < args.collapse_ratio <= 1:
        raise ValueError("--collapse-ratio must be in (0, 1]")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    validate_args(args)

    paths = {"vae": args.vae, "gan": args.gan, "ddpm": args.ddpm, "ldm": args.ldm}
    for name in args.models:
        if not paths[name].is_file():
            raise FileNotFoundError(f"Missing {name} checkpoint: {paths[name]}")
    if not args.classifier.is_file():
        raise FileNotFoundError(f"Missing classifier checkpoint: {args.classifier}")

    if args.diffusion_configs:
        diffusion_configs = [parse_diffusion_config(text) for text in args.diffusion_configs]
    else:
        diffusion_configs = [
            {
                "method": args.diffusion_sampler,
                "steps": args.diffusion_steps,
                "guidance_scale": args.guidance_scale,
            }
        ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier_checkpoint = torch.load(
        args.classifier, map_location="cpu", weights_only=True
    )
    classifier = FashionClassifier().to(device)
    classifier.load_state_dict(classifier_checkpoint["model_state_dict"])
    classifier.eval()
    _, test_loader = make_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=42,
    )
    real_images_all = collect_real_images(test_loader)
    evaluated_real_images = select_real_feature_subset(
        real_images_all,
        samples_per_class=args.samples_per_class,
        partition=args.evaluation_partition,
        development_samples_per_class=args.development_samples_per_class,
        seed=args.real_seed,
        final_offset=args.final_offset,
    )
    real_stack = torch.cat(evaluated_real_images)
    class_labels = torch.arange(len(CLASS_NAMES)).repeat_interleave(
        args.samples_per_class
    )
    selected_real_features, real_predictions = collect_generated_features(
        real_stack, class_labels, classifier, args.batch_size, device
    )
    classifier_accuracy = (
        (torch.cat(real_predictions) == class_labels).float().mean().item()
    )
    real_pixel_div = [pixel_diversity(images) for images in evaluated_real_images]

    inception = None
    if args.inception_kid:
        from fashion_mnist_gen.inception_metrics import InceptionKID

        print("Caractéristiques Inception des vraies images…")
        inception = InceptionKID(
            device,
            subset_size=min(1000, real_stack.shape[0]),
            subsets=args.kid_subsets,
            batch_size=args.batch_size,
            compute_fid=args.fid,
        )
        inception.add_real(real_stack)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluated_start = (
        0
        if args.evaluation_partition == "development"
        else args.development_samples_per_class + args.final_offset
    )
    results: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "Fashion-MNIST official test split",
        "evaluation_partition": args.evaluation_partition,
        "development_samples_per_class": args.development_samples_per_class,
        "final_offset": args.final_offset,
        "evaluated_real_indices_per_class": [
            evaluated_start,
            evaluated_start + args.samples_per_class - 1,
        ],
        "real_test_subset_seed": args.real_seed,
        "feature_extractor": "Frozen FashionClassifier penultimate 128-D features",
        "kid_definition": "Unbiased cubic polynomial-kernel MMD; not Inception KID",
        "inception_kid_definition": (
            "torchmetrics KernelInceptionDistance, Inception-v3 pool features, "
            "grayscale repeated on 3 channels, all classes pooled"
            if args.inception_kid
            else None
        ),
        "coverage_definition": (
            "Real test features with a generated neighbor within their nearest-real "
            "neighbor radius"
        ),
        "class_coverage_definition": (
            f"Fraction of classes receiving at least {args.class_coverage_share:g} x "
            "their uniform share of classifier predictions"
        ),
        "mode_collapse_definition": (
            "Intra-class mean pairwise pixel L2 distance of generated images divided "
            f"by the same value on real test images; flagged below {args.collapse_ratio:g}"
        ),
        "classifier_accuracy_on_evaluated_real_images": classifier_accuracy,
        "classifier_validation_accuracy": classifier_checkpoint.get("validation_accuracy"),
        "samples_per_class": args.samples_per_class,
        "sampling_seeds": list(args.seeds),
        "diffusion_configs": diffusion_configs,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "models": {},
    }

    def score(images: torch.Tensor) -> dict[str, Any]:
        generated_features, predictions = collect_generated_features(
            images, class_labels, classifier, args.batch_size, device
        )
        metrics = score_sample_run(
            selected_real_features,
            generated_features,
            predictions,
            generated_images=split_by_class(images, args.samples_per_class),
            real_pixel_diversity=real_pixel_div,
            collapse_ratio=args.collapse_ratio,
            class_coverage_share=args.class_coverage_share,
        )
        if inception is not None:
            metrics.update(inception.score(images))
        return metrics

    if args.real_reference:
        reference_images = torch.cat(
            select_reference_subset(
                real_images_all,
                args.samples_per_class,
                args.real_seed,
                used_end_index=max(
                    args.development_samples_per_class,
                    evaluated_start + args.samples_per_class,
                ),
            )
        )
        metrics = score(reference_images)
        metrics["seconds_per_image"] = 0.0
        results["models"]["real"] = {
            "label": LEVEL_LABELS["real"],
            "kind": "real",
            "source": "last samples_per_class images of each class permutation",
            "sample_runs": [metrics],
            "summary": summarize_runs([metrics]),
        }
        save_sample_grid(
            reference_images, args.samples_per_class, args.output_dir / "real_samples.png"
        )
        print(
            f"real : fidélité {metrics['class_fidelity']:.2%}, "
            f"KID-Fashion {metrics['fashion_kid_mean']:.3f}"
            + (
                f", KID Inception {metrics['inception_kid']:.5f}"
                if "inception_kid" in metrics
                else ""
            )
        )

    entries: list[tuple[str, str, dict[str, Any] | None]] = []
    for name in args.models:
        if name in DIFFUSION_MODELS:
            entries.extend(
                (diffusion_key(config, name), name, config) for config in diffusion_configs
            )
        else:
            entries.append((name, name, None))

    loaded: dict[str, tuple[torch.nn.Module, dict[str, Any]]] = {}
    for key, name, config in entries:
        if name not in loaded:
            loaded[name] = load_generator(name, paths[name], device)
        model, checkpoint = loaded[name]
        model_results: dict[str, Any] = {
            "label": diffusion_label(config, name) if config else LEVEL_LABELS[name],
            "kind": name,
            "checkpoint": str(paths[name]),
            "training_seed": checkpoint["seed"],
            "training_epochs": checkpoint["epochs"],
            "best_epoch": checkpoint.get("best_epoch"),
            "training_seconds": checkpoint["training_seconds"],
            # Level 3 also needs the VAE: its cost is reported separately.
            "prerequisite_training_seconds": checkpoint.get("vae_training_seconds"),
            "parameter_count": generator_parameter_count(name, model),
            "sampler": config,
            "sample_runs": [],
        }
        for seed_index, seed in enumerate(args.seeds):
            labels = class_labels.to(device)
            generated_images, generation_seconds = sample_generated_images(
                name,
                model,
                labels,
                args.batch_size,
                seed,
                device,
                config["method"] if config else "ddim",
                config["steps"] if config else 0,
                config["guidance_scale"] if config else 0.0,
            )
            metrics = score(generated_images)
            metrics["sampling_seed"] = seed
            metrics["generation_seconds"] = generation_seconds
            metrics["seconds_per_image"] = (
                generation_seconds / generated_images.shape[0]
            )
            model_results["sample_runs"].append(metrics)
            if seed_index == 0:
                save_sample_grid(
                    generated_images,
                    args.samples_per_class,
                    args.output_dir / f"{key}_samples.png",
                )
            print(
                f"{key} seed={seed}: fidélité {metrics['class_fidelity']:.2%}, "
                f"KID-Fashion {metrics['fashion_kid_mean']:.3f}, "
                + (
                    f"KID Inception {metrics['inception_kid']:.5f}, "
                    if "inception_kid" in metrics
                    else ""
                )
                + f"couverture classes {metrics['class_coverage']:.0%}, "
                f"diversité/réel {metrics['pixel_diversity_ratio']:.2f}, "
                f"temps {metrics['seconds_per_image']:.5f} s/image"
            )

        model_results["summary"] = summarize_runs(model_results["sample_runs"])
        results["models"][key] = model_results

    attach_human_results(results, args.human_results)
    output_path = args.output_dir / "results.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    table = results_markdown(results)
    (args.output_dir / "results.md").write_text(table, encoding="utf-8")
    print()
    print(table)
    print(f"Résultats sauvegardés dans : {output_path}")
    return results


if __name__ == "__main__":
    main()
