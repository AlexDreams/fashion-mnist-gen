"""Figures for the report and the video: ``python make_figures.py``.

Writes into ``runs/figures/`` (only for the checkpoints that exist):

* ``comparaison_niveaux.png``: one image per class for every level and for real
  test images, same seed (the brief's "same request" gallery);
* ``guidance_<model>.png`` and ``pas_<model>.png``: effect of the guidance
  scale and of the number of DDIM steps, same seed on every row;
* ``echecs_<model>.png``: the generated images the frozen classifier finds the
  least like their requested class (material for the 5 failure analyses);
* ``courbe_guidance.png``: class accuracy and KID against guidance, from
  ``runs/guidance_sweep/results.json`` if that sweep was run;
* ``cout_qualite.png``: time per image against Inception KID, from the latest
  ``eval.py`` results.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from evaluate_protocol import (
    DIFFUSION_MODELS,
    HUMAN_TEST_DIFFUSION_CONFIGS,
    collect_real_images,
    load_generator,
)
from fashion_mnist_gen import visualize
from fashion_mnist_gen.classifier import FashionClassifier
from fashion_mnist_gen.data import CLASS_NAMES, make_dataloaders

LEVEL_NAMES = {
    "real": "Vraies images",
    "vae": "0 · VAE",
    "gan": "1 · GAN",
    "ddpm": "2 · Diffusion",
    "ldm": "3 · Diffusion latente",
}
GUIDANCE_VALUES = (0.0, 1.0, 1.5, 3.0, 5.0)
STEP_VALUES = (5, 10, 20, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--guidance", type=float, default=None,
                        help="Same guidance for every model (default: the selected one "
                             "per model, 2.0 pixels and 3.0 latent).")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--with-ddpm",
        action="store_true",
        help="Add a 1 000-step DDPM row to the steps figure (slow without GPU).",
    )
    parser.add_argument("--failure-pool", type=int, default=100,
                        help="Generated images per class searched for failures.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--classifier", type=Path,
                        default=Path("checkpoints/fashion_classifier.pt"))
    parser.add_argument("--vae", type=Path, default=Path("checkpoints/vae.pt"))
    parser.add_argument("--gan", type=Path, default=Path("checkpoints/gan.pt"))
    parser.add_argument("--ddpm", type=Path, default=Path("checkpoints/diffusion.pt"))
    parser.add_argument("--ldm", type=Path,
                        default=Path("checkpoints/latent_diffusion.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/figures"))
    return parser.parse_args()


def guidance_for(name: str, args: argparse.Namespace) -> float:
    """Guidance used for a model: --guidance if given, else the frozen final value."""
    if args.guidance is not None:
        return args.guidance
    return HUMAN_TEST_DIFFUSION_CONFIGS.get(name, {"guidance_scale": 1.0})["guidance_scale"]


def sample(name, model, labels, seed, device, method="ddim", steps=50, guidance=1.5):
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.inference_mode():
        if name in DIFFUSION_MODELS:
            return model.sample(labels, method=method, num_inference_steps=steps,
                                guidance_scale=guidance, generator=generator).cpu()
        return model.sample(labels, generator=generator).cpu()


def image_rows(rows: list[tuple[str, torch.Tensor]], path: Path, title: str,
               column_titles: list[str] | None = None) -> None:
    columns = rows[0][1].shape[0]
    figure, axes = plt.subplots(len(rows), columns,
                                figsize=(columns * 1.05, len(rows) * 1.15 + 0.6),
                                squeeze=False)
    for row_index, (label, images) in enumerate(rows):
        for column_index in range(columns):
            axis = axes[row_index][column_index]
            axis.imshow(images[column_index].squeeze(), cmap="gray", vmin=0, vmax=1)
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0 and column_titles:
                axis.set_title(column_titles[column_index], fontsize=7)
        axes[row_index][0].set_ylabel(label, rotation=0, ha="right", va="center",
                                      fontsize=8)
    figure.suptitle(title, fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  {path}")


def failures_figure(name, model, classifier, args, device, path: Path) -> None:
    labels = torch.arange(len(CLASS_NAMES), device=device).repeat_interleave(
        args.failure_pool
    )
    images = []
    for start in range(0, len(labels), 500):
        images.append(sample(name, model, labels[start:start + 500],
                             args.seed + start, device, "ddim", args.steps,
                             guidance_for(name, args)))
    images = torch.cat(images)
    with torch.inference_mode():
        probabilities = torch.cat([
            classifier(images[start:start + 500].to(device)).softmax(dim=1).cpu()
            for start in range(0, len(images), 500)
        ])
    target = probabilities[torch.arange(len(labels)), labels.cpu()]
    worst = target.argsort()[:10]
    figure, axes = plt.subplots(1, len(worst), figsize=(len(worst) * 1.3, 2.1))
    for axis, index in zip(axes, worst.tolist()):
        requested = CLASS_NAMES[labels[index]]
        predicted = CLASS_NAMES[probabilities[index].argmax()]
        axis.imshow(images[index].squeeze(), cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"demandé : {requested}\nvu : {predicted}\np={target[index]:.2f}",
                       fontsize=6)
        axis.set_xticks([])
        axis.set_yticks([])
    rate = (probabilities.argmax(dim=1) != labels.cpu()).float().mean().item()
    figure.suptitle(
        f"{LEVEL_NAMES[name]} : 10 pires images sur {len(labels)} "
        f"({rate:.1%} mal classées)", fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  {path}")


def guidance_curve(path: Path) -> None:
    source = Path("runs/guidance_sweep/results.json")
    if not source.is_file():
        return
    results = json.loads(source.read_text(encoding="utf-8"))
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))
    series: dict[str, list[tuple[float, dict]]] = {}
    for entry in results["models"].values():
        if entry.get("sampler"):
            series.setdefault(entry["kind"], []).append(
                (entry["sampler"]["guidance_scale"], entry["summary"]))
    for kind, points in series.items():
        points.sort(key=lambda point: point[0])
        scales = [point[0] for point in points]
        left.plot(scales, [p[1]["class_fidelity"]["mean"] for p in points], marker="o",
                  label=LEVEL_NAMES[kind])
        metric = "inception_kid" if "inception_kid" in points[0][1] else "fashion_kid_mean"
        right.plot(scales, [p[1][metric]["mean"] for p in points], marker="o",
                   label=LEVEL_NAMES[kind])
        right.set_ylabel("KID Inception ↓" if metric == "inception_kid" else "Fashion-KID ↓")
    left.set(xlabel="Échelle de guidance", ylabel="Précision de classe ↑")
    right.set(xlabel="Échelle de guidance")
    for axis in (left, right):
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Effet de la guidance (partition de développement)")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"  {path}")


def cost_quality(path: Path) -> None:
    for folder in (Path("runs/eval_final"), Path("runs/eval_dev")):
        source = folder / "results.json"
        if source.is_file():
            break
    else:
        return
    results = json.loads(source.read_text(encoding="utf-8"))
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for entry in results["models"].values():
        summary = entry["summary"]
        if entry["kind"] == "real" or "inception_kid" not in summary:
            continue
        x = summary["seconds_per_image"]["mean"]
        y = summary["inception_kid"]["mean"] * 1000
        axis.scatter(x, y, s=50)
        axis.annotate(entry["label"], (x, y), fontsize=7, xytext=(5, 3),
                      textcoords="offset points")
    axis.set_xscale("log")
    axis.set(xlabel=f"Secondes par image ({results['device']}, échelle log)",
             ylabel="KID Inception ×10³ ↓",
             title=f"Coût contre qualité ({results['evaluation_partition']})")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"  {path}")


def model_views(models, real, args, device) -> None:
    """Figures asked after the progress review: diffusion steps, VAE latent space, GAN."""
    out = args.output_dir
    noise_steps = [0, 50, 100, 200, 300, 400, 500, 650, 800, 999]
    examples = torch.stack([images[0] for images in real[:10:2]])
    noisy = visualize.forward_noising(examples, noise_steps,
                                      generator=torch.Generator().manual_seed(args.seed))
    image_rows([(CLASS_NAMES[2 * index], torch.stack([step[index] for step in noisy]))
                for index in range(len(examples))],
               out / "bruitage_direct.png",
               "Bruitage direct q(x_t | x_0) : la même image de plus en plus bruitée",
               [f"t = {t}" for t in noise_steps])

    for name, model in models.items():
        if name not in DIFFUSION_MODELS:
            continue
        trajectory = []
        labels = torch.arange(0, 10, 2, device=device)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        with torch.inference_mode():
            model.sample(labels, method="ddim", num_inference_steps=args.steps,
                         guidance_scale=guidance_for(name, args), generator=generator,
                         trajectory=trajectory)
        selected = visualize.pick_steps(trajectory, 10)
        rows = []
        for index, class_id in enumerate(labels.tolist()):
            rows.append((f"{CLASS_NAMES[class_id]} x_t",
                         torch.stack([step["noisy"][index] for step in selected])))
            rows.append(("x̂_0 estimée", torch.stack([step["clean"][index] for step in selected])))
        image_rows(rows, out / f"diffusion_etapes_{name}.png",
                   f"{LEVEL_NAMES[name]} : débruitage DDIM {args.steps} pas (image courante "
                   "et image finale estimée)", [f"t = {step['timestep']}" for step in selected])

    if "vae" in models:
        vae = models["vae"]
        images = torch.cat([class_images[:300] for class_images in real])
        labels = torch.arange(10).repeat_interleave(
            torch.tensor([min(300, len(class_images)) for class_images in real]))
        codes = visualize.latent_codes(vae, images, labels)
        points = visualize.project_2d(codes).numpy()
        figure, axes = plt.subplots(2, 5, figsize=(13, 5.6), sharex=True, sharey=True)
        for index, axis in enumerate(axes.flat):
            axis.scatter(points[:, 0], points[:, 1], s=3, color="#c9c8c2", linewidths=0)
            mask = (labels == index).numpy()
            axis.scatter(points[mask, 0], points[mask, 1], s=6, color="#2a78d6", linewidths=0)
            axis.set_title(CLASS_NAMES[index], fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])
        figure.suptitle("VAE : codes latents (ACP 2D) de 3 000 images de test, une classe "
                        "en bleu par panneau", fontsize=11)
        figure.tight_layout()
        figure.savefig(out / "vae_espace_latent.png", dpi=150, bbox_inches="tight")
        plt.close(figure)
        print(f"  {out / 'vae_espace_latent.png'}")
        maps = [visualize.latent_map(vae, codes[labels == c], c, size=9) for c in (0, 3, 7)]
        image_rows([(f"{CLASS_NAMES[c]} ({row - 4:+d})", grid[row * 9:(row + 1) * 9])
                    for c, grid in zip((0, 3, 7), maps) for row in (0, 4, 8)],
                   out / "vae_carte_latente.png",
                   "VAE : parcours des 2 directions principales de l'espace latent",
                   [f"{value:+.1f} σ" for value in torch.linspace(-2.5, 2.5, 9).tolist()])

    if "gan" in models:
        gan = models["gan"]
        rows = [(CLASS_NAMES[c], visualize.interpolate(gan, "gan", c, args.seed + c,
                                                       args.seed + 100 + c))
                for c in range(10)]
        image_rows(rows, out / "gan_interpolations.png",
                   "GAN : interpolation sphérique entre deux bruits, par classe",
                   [f"{f:.1f}" for f in torch.linspace(0, 1, 10).tolist()])
        grid = visualize.same_noise_all_classes(gan, "gan", args.seed, rows=6)
        image_rows([(f"bruit {row + 1}", grid[row * 10:(row + 1) * 10]) for row in range(6)],
                   out / "gan_meme_bruit.png",
                   "GAN : même bruit par ligne, une classe par colonne", list(CLASS_NAMES))


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"vae": args.vae, "gan": args.gan, "ddpm": args.ddpm, "ldm": args.ldm}
    models = {name: load_generator(name, path, device)[0]
              for name, path in paths.items() if path.is_file()}
    labels = torch.arange(len(CLASS_NAMES), device=device)
    print("Figures :")

    _, test_loader = make_dataloaders(data_dir=args.data_dir, batch_size=512)
    real = collect_real_images(test_loader)
    generator = torch.Generator().manual_seed(args.seed)
    rows = [(LEVEL_NAMES["real"], torch.stack([
        images[torch.randint(len(images), (1,), generator=generator)].squeeze(0)
        for images in real]))]
    for name, model in models.items():
        rows.append((LEVEL_NAMES[name], sample(name, model, labels, args.seed, device,
                                               "ddim", args.steps,
                                               guidance_for(name, args))))
    guidances = ", ".join(f"{name} {guidance_for(name, args):g}"
                          for name in models if name in DIFFUSION_MODELS)
    image_rows(rows, args.output_dir / "comparaison_niveaux.png",
               f"Même demande pour chaque niveau (graine {args.seed}, diffusion DDIM "
               f"{args.steps} pas, guidance {guidances or '-'})", list(CLASS_NAMES))

    for name, model in models.items():
        if name not in DIFFUSION_MODELS:
            continue
        image_rows(
            [(f"guidance {value:g}", sample(name, model, labels, args.seed, device,
                                            "ddim", args.steps, value))
             for value in GUIDANCE_VALUES],
            args.output_dir / f"guidance_{name}.png",
            f"{LEVEL_NAMES[name]} : effet de la guidance (DDIM {args.steps} pas)",
            list(CLASS_NAMES))
        step_rows = [(f"DDIM {value}", sample(name, model, labels, args.seed, device,
                                              "ddim", value, guidance_for(name, args)))
                     for value in STEP_VALUES]
        if args.with_ddpm:
            step_rows.append(("DDPM 1000", sample(name, model, labels, args.seed, device,
                                                  "ddpm", model.num_timesteps,
                                                  guidance_for(name, args))))
        image_rows(step_rows, args.output_dir / f"pas_{name}.png",
                   f"{LEVEL_NAMES[name]} : effet du nombre de pas "
                   f"(guidance {guidance_for(name, args):g})", list(CLASS_NAMES))

    if args.classifier.is_file():
        classifier = FashionClassifier().to(device)
        classifier.load_state_dict(torch.load(args.classifier, map_location="cpu",
                                              weights_only=True)["model_state_dict"])
        classifier.eval()
        for name, model in models.items():
            failures_figure(name, model, classifier, args, device,
                            args.output_dir / f"echecs_{name}.png")

    model_views(models, real, args, device)
    guidance_curve(args.output_dir / "courbe_guidance.png")
    cost_quality(args.output_dir / "cout_qualite.png")


if __name__ == "__main__":
    main()
