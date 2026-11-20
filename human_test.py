"""Blind "real or generated?" test (brief: 5 people x 20 images per level).

1. ``python human_test.py prepare`` builds a fixed pool: 20 images per
   available level (fixed seeds, random classes) and 20 real test images,
   saved under neutral random names with a hidden manifest.
2. Participants answer in the Streamlit interface (tab "Test humain"), which
   appends one line per answer to ``runs/human_test/answers.jsonl``.
3. ``python human_test.py score`` computes, per source, the share of images
   judged real. A generated image judged real is a success for the model; the
   real images give the control rate. ``eval.py`` adds the score to the table.
"""

import argparse
import json
import random
import secrets
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image

HUMAN_DIR = Path("runs/human_test")
POOL_DIR = HUMAN_DIR / "pool"
MANIFEST = HUMAN_DIR / "manifest.json"
ANSWERS = HUMAN_DIR / "answers.jsonl"
RESULTS = HUMAN_DIR / "results.json"


def save_png(image: torch.Tensor, path: Path) -> None:
    array = (image.squeeze().clamp(0, 1) * 255).round().byte().cpu().numpy()
    Image.fromarray(array).save(path)


def prepare(args: argparse.Namespace) -> None:
    from evaluate_protocol import (
        DIFFUSION_MODELS,
        HUMAN_TEST_DIFFUSION_CONFIGS,
        collect_real_images,
        diffusion_key,
        load_generator,
    )
    from fashion_mnist_gen.data import make_dataloaders

    if MANIFEST.exists() and not args.force:
        raise SystemExit(
            f"{MANIFEST} existe déjà : le pool doit rester identique pour tous les "
            "participants. Utilisez --force pour le recréer (et effacer les réponses)."
        )
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    for old in POOL_DIR.glob("*.png"):
        old.unlink()
    if ANSWERS.exists():
        ANSWERS.unlink()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)
    items: list[dict[str, Any]] = []
    diffusion_settings: dict[str, dict[str, Any]] = {}

    def add(source: str, images: torch.Tensor, labels: list[int]) -> None:
        for image, label in zip(images, labels):
            image_id = secrets.token_hex(6)
            save_png(image, POOL_DIR / f"{image_id}.png")
            items.append({"id": image_id, "source": source, "class_id": label})

    paths = {"vae": args.vae, "gan": args.gan, "ddpm": args.ddpm, "ldm": args.ldm}
    for name, path in paths.items():
        if not path.is_file():
            print(f"[human_test] {name} absent ({path}) : ignoré")
            continue
        model, _ = load_generator(name, path, device)
        labels = [rng.randrange(10) for _ in range(args.images_per_source)]
        label_tensor = torch.tensor(labels, device=device)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        if name in DIFFUSION_MODELS:
            config = HUMAN_TEST_DIFFUSION_CONFIGS[name]
            diffusion_settings[name] = config
            images = model.sample(
                label_tensor,
                method=config["method"],
                num_inference_steps=config["steps"],
                guidance_scale=config["guidance_scale"],
                generator=generator,
            )
            source = diffusion_key(config, name)
        else:
            images = model.sample(label_tensor, generator=generator)
            source = name
        add(source, images.cpu(), labels)

    _, test_loader = make_dataloaders(data_dir=args.data_dir, batch_size=512)
    real_by_class = collect_real_images(test_loader)
    labels = [rng.randrange(10) for _ in range(args.images_per_source)]
    real_images = [
        real_by_class[label][rng.randrange(len(real_by_class[label]))] for label in labels
    ]
    add("real", torch.stack(real_images), labels)

    rng.shuffle(items)
    MANIFEST.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "images_per_source": args.images_per_source,
                "diffusion_settings": diffusion_settings,
                "items": items,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    sources = sorted({item["source"] for item in items})
    print(f"Pool : {len(items)} images ({', '.join(sources)}) dans {POOL_DIR}")


def load_manifest() -> dict[str, Any]:
    if not MANIFEST.is_file():
        raise FileNotFoundError("Pool absent : lancez d'abord `python human_test.py prepare`")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def load_answers() -> list[dict[str, Any]]:
    if not ANSWERS.is_file():
        return []
    return [
        json.loads(line)
        for line in ANSWERS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def record_answer(participant: str, image_id: str, answer: str) -> None:
    if answer not in {"real", "generated"}:
        raise ValueError("answer must be 'real' or 'generated'")
    ANSWERS.parent.mkdir(parents=True, exist_ok=True)
    with ANSWERS.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"participant": participant, "image_id": image_id, "answer": answer})
            + "\n"
        )


def compute_scores(manifest: dict[str, Any], answers: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-source share of answers "real" (last answer per participant and image)."""
    source_of = {item["id"]: item["source"] for item in manifest["items"]}
    latest: dict[tuple[str, str], str] = {}
    for answer in answers:
        if answer["image_id"] in source_of:
            latest[(answer["participant"], answer["image_id"])] = answer["answer"]
    per_source: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"answers": 0, "judged_real": 0, "participants": set()}
    )
    correct = 0
    for (participant, image_id), answer in latest.items():
        source = source_of[image_id]
        stats = per_source[source]
        stats["answers"] += 1
        stats["judged_real"] += answer == "real"
        stats["participants"].add(participant)
        correct += (answer == "real") == (source == "real")
    result_sources = {
        source: {
            "answers": stats["answers"],
            "participants": len(stats["participants"]),
            "judged_real_rate": stats["judged_real"] / stats["answers"],
        }
        for source, stats in sorted(per_source.items())
    }
    return {
        "participants": len({participant for participant, _ in latest}),
        "answers": len(latest),
        "human_accuracy": correct / len(latest) if latest else None,
        "definition": (
            "judged_real_rate = share of answers 'vraie' for the images of a source; "
            "for generated sources higher is better, real images give the control rate"
        ),
        "per_source": result_sources,
    }


def score(args: argparse.Namespace) -> None:
    excluded = set(args.exclude)
    answers = [answer for answer in load_answers() if answer["participant"] not in excluded]
    results = compute_scores(load_manifest(), answers)
    results["excluded_participants"] = sorted(excluded)
    RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"{results['participants']} participants, {results['answers']} réponses")
    for source, stats in results["per_source"].items():
        print(
            f"  {source:<22} pris pour vrai : {stats['judged_real_rate']:.0%} "
            f"({stats['answers']} réponses, {stats['participants']} personnes)"
        )
    print(f"Résultats : {RESULTS}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="build the fixed image pool")
    prepare_parser.add_argument("--images-per-source", type=int, default=20)
    prepare_parser.add_argument("--seed", type=int, default=2026)
    prepare_parser.add_argument("--force", action="store_true")
    prepare_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    prepare_parser.add_argument("--vae", type=Path, default=Path("checkpoints/vae.pt"))
    prepare_parser.add_argument("--gan", type=Path, default=Path("checkpoints/gan.pt"))
    prepare_parser.add_argument("--ddpm", type=Path, default=Path("checkpoints/diffusion.pt"))
    prepare_parser.add_argument(
        "--ldm", type=Path, default=Path("checkpoints/latent_diffusion.pt")
    )
    prepare_parser.set_defaults(func=prepare)
    score_parser = commands.add_parser("score", help="compute the scores")
    score_parser.add_argument(
        "--exclude", nargs="*", default=[],
        help="participants left out of the scores (e.g. non-blind organisers)",
    )
    score_parser.set_defaults(func=score)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
