"""Reproduce the project's results table with one command: ``python eval.py``.

The frozen protocol (partition, real-image slice, seeds, diffusion settings) is
written below so that the report table can be regenerated from the provided
checkpoints. Any argument of ``evaluate_protocol.py`` can be appended to
override it, e.g. ``python eval.py --no-inception-kid`` or
``python eval.py --dev`` while tuning (development slice, never the final one).
The final protocol uses DDPM guidance 2.0 and latent-diffusion guidance 3.0,
with the same selected scale across that model's samplers.

Missing generator checkpoints are skipped with a warning, so the table can be
built level by level during the project.
"""

import argparse
import sys
from pathlib import Path

import evaluate_protocol

# Frozen evaluation protocol: change it only together with the report.
FINAL_PROTOCOL = [
    "--evaluation-partition", "final",
    "--development-samples-per-class", "100",
    "--final-offset", "100",
    "--real-seed", "2026",
    "--samples-per-class", "100",
    "--seeds", "2029", "2030", "2031",
]
DEVELOPMENT_PROTOCOL = [
    "--evaluation-partition", "development",
    "--real-seed", "2026",
    "--samples-per-class", "100",
    "--seeds", "2026", "2027", "2028",
]
# Development baseline: one row per sampler, at the common guidance 1.5.
DIFFUSION_CONFIGS = ["ddpm:1000:1.5", "ddim:50:1.5", "ddim:10:1.5"]
CHECKPOINTS = {
    "vae": Path("checkpoints/vae.pt"),
    "gan": Path("checkpoints/gan.pt"),
    "ddpm": Path("checkpoints/diffusion.pt"),
    "ldm": Path("checkpoints/latent_diffusion.pt"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--dev", action="store_true", help="Use the development slice.")
    parser.add_argument("-h", "--help", action="store_true")
    own, passthrough = parser.parse_known_args()
    if own.help:
        print(__doc__)
        evaluate_protocol.parse_args(["--help"])
        return

    def override(flag: str) -> str | None:
        for index, value in enumerate(passthrough):
            if value == flag and index + 1 < len(passthrough):
                return passthrough[index + 1]
            if value.startswith(flag + "="):
                return value.split("=", 1)[1]
        return None

    models = []
    for name, default_path in CHECKPOINTS.items():
        path = Path(override(f"--{name}") or default_path)
        if path.is_file():
            models.append(name)
        else:
            print(f"[eval] checkpoint absent pour {name} ({path}) : niveau ignoré.")
    if not models:
        sys.exit(
            "[eval] Aucun checkpoint de générateur trouvé dans checkpoints/. "
            "Téléchargez-les (voir README) ou relancez les scripts train_*.py."
        )

    output_dir = "runs/eval_dev" if own.dev else "runs/eval_final"
    has_diffusion_override = any(
        argument == flag or argument.startswith(flag + "=")
        for argument in passthrough
        for flag in (
            "--diffusion-configs",
            "--diffusion-sampler",
            "--diffusion-steps",
            "--guidance-scale",
        )
    )
    if has_diffusion_override:
        diffusion_args = []
    elif own.dev:
        diffusion_args = ["--diffusion-configs", *DIFFUSION_CONFIGS]
    else:
        diffusion_args = []
        for model_name, configs in evaluate_protocol.FINAL_DIFFUSION_CONFIGS_BY_MODEL.items():
            diffusion_args.extend(
                [
                    f"--{model_name}-configs",
                    *[
                        f"{config['method']}:{config['steps']}:{config['guidance_scale']:g}"
                        for config in configs
                    ],
                ]
            )

    argv = [
        *(DEVELOPMENT_PROTOCOL if own.dev else FINAL_PROTOCOL),
        "--models", *models,
        *diffusion_args,
        "--vae", str(CHECKPOINTS["vae"]),
        "--gan", str(CHECKPOINTS["gan"]),
        "--ddpm", str(CHECKPOINTS["ddpm"]),
        "--ldm", str(CHECKPOINTS["ldm"]),
        "--output-dir", output_dir,
        *passthrough,
    ]
    evaluate_protocol.main(argv)


if __name__ == "__main__":
    main()
