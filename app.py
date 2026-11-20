"""Streamlit interface: ``streamlit run app.py``.

Tabs: generate with one level, gallery of every class, a look inside each
model (diffusion step by step, VAE latent space, GAN interpolations), comparison
of the levels, effect of guidance and steps, protocol results, blind human test.
"""

import hashlib
import io
import random
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from torchvision.utils import make_grid  # noqa: E402

from evaluate_protocol import (  # noqa: E402
    DIFFUSION_MODELS,
    HUMAN_TEST_DIFFUSION_CONFIGS,
    load_generator,
)
from fashion_mnist_gen import visualize  # noqa: E402
from fashion_mnist_gen.data import CLASS_NAMES  # noqa: E402

CHECKPOINTS = {
    "vae": Path("checkpoints/vae.pt"),
    "gan": Path("checkpoints/gan.pt"),
    "ddpm": Path("checkpoints/diffusion.pt"),
    "ldm": Path("checkpoints/latent_diffusion.pt"),
}
LEVEL_NAMES = {
    "vae": "Niveau 0 · VAE",
    "gan": "Niveau 1 · GAN",
    "ddpm": "Niveau 2 · Diffusion (pixels)",
    "ldm": "Niveau 3 · Diffusion latente",
}
LEVEL_PITCH = {
    "vae": "Compresse l'image en 16 nombres puis la reconstruit. Rapide mais flou.",
    "gan": "Un faussaire contre un détective. Net et instantané.",
    "ddpm": "Part du bruit pur et le retire pas à pas avec un U-Net.",
    "ldm": "La diffusion, mais dans les 16 nombres du VAE.",
}
CLASS_LABELS_FR = (
    "T-shirt / haut", "Pantalon", "Pull", "Robe", "Manteau",
    "Sandale", "Chemise", "Basket", "Sac", "Bottine",
)
ACCENT = "#2a78d6"
MUTED = "#c9c8c2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="Catalogue de mode imaginaire", page_icon="👗", layout="wide")
st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
.hero {padding: 1.4rem 1.6rem; border-radius: 14px; margin-bottom: 1rem;
       background: linear-gradient(120deg, #eaf2fc 0%, #f6f3ee 100%);
       border: 1px solid #e3e1da;}
.hero h1 {margin: 0; font-size: 2.1rem; letter-spacing: -0.02em; color: #0b0b0b;}
.hero p {margin: .35rem 0 0; color: #52514e;}
.level-card {border: 1px solid #e3e1da; border-radius: 12px; padding: .8rem 1rem;
             background: #ffffff; min-height: 6.2rem;}
.level-card b {color: #0b0b0b;} .level-card span {color: #52514e; font-size: .9rem;}
.level-card.off {opacity: .45;}
.stTabs [data-baseweb="tab-list"] {gap: .3rem; flex-wrap: wrap;}
div[data-testid="stMetric"] {background: #ffffff; border: 1px solid #e3e1da;
                             border-radius: 10px; padding: .5rem .8rem;}
img {image-rendering: pixelated;}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- utils
@st.cache_resource(show_spinner="Chargement du modèle…")
def get_model(name: str, path: str, modified: float) -> tuple[torch.nn.Module, dict[str, Any]]:
    del modified  # part of the cache key: reload when the checkpoint changes
    return load_generator(name, Path(path), DEVICE)


@st.cache_resource(show_spinner="Chargement des images de test…")
def test_images(per_class: int = 300) -> tuple[torch.Tensor, torch.Tensor]:
    """First ``per_class`` official test images of each class (never seen in training)."""
    from torchvision.datasets import FashionMNIST

    dataset = FashionMNIST("data", train=False, download=True)
    images = dataset.data.float().div(255).unsqueeze(1)
    labels = dataset.targets
    keep = torch.cat([torch.nonzero(labels == c).squeeze(1)[:per_class] for c in range(10)])
    return images[keep], labels[keep]


@st.cache_resource(show_spinner="Projection de l'espace latent…")
def latent_projection(modified: float, method_2d: str) -> tuple[torch.Tensor, torch.Tensor]:
    del modified  # cache key: recompute when the VAE checkpoint changes
    vae, _ = model_for("vae")
    images, labels = test_images()
    codes = visualize.latent_codes(vae, images, labels)
    return codes, visualize.project_2d(codes, method_2d)


def available_levels() -> list[str]:
    return [name for name, path in CHECKPOINTS.items() if path.is_file()]


def model_for(name: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    path = CHECKPOINTS[name]
    return get_model(name, str(path), path.stat().st_mtime)


def parameter_count(name: str, model: torch.nn.Module) -> int:
    from evaluate_protocol import generator_parameter_count

    return generator_parameter_count(name, model)


def generate(name: str, labels: list[int] | torch.Tensor, seed: int, method: str = "ddim",
             steps: int = 50, guidance: float | None = None,
             trajectory: list | None = None) -> tuple[torch.Tensor, float]:
    """``guidance=None`` uses the value selected for that model (final protocol)."""
    model, _ = model_for(name)
    if guidance is None and name in DIFFUSION_MODELS:
        guidance = HUMAN_TEST_DIFFUSION_CONFIGS[name]["guidance_scale"]
    labels = torch.as_tensor(labels, dtype=torch.long, device=DEVICE)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    start = time.perf_counter()
    with torch.inference_mode():
        if name in DIFFUSION_MODELS:
            images = model.sample(labels, method=method, num_inference_steps=steps,
                                  guidance_scale=guidance, generator=generator,
                                  trajectory=trajectory)
        else:
            images = model.sample(labels, generator=generator)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return images.cpu(), time.perf_counter() - start


def to_picture(images: torch.Tensor, columns: int = 8, scale: int = 4) -> np.ndarray:
    images = F.interpolate(images.float(), scale_factor=scale, mode="nearest")
    grid = make_grid(images, nrow=columns, padding=2, pad_value=1.0)
    return (grid[0].clamp(0, 1).numpy() * 255).astype(np.uint8)


def png_bytes(picture: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(picture).save(buffer, format="PNG")
    return buffer.getvalue()


def cost_metrics(name: str, seconds: float, count: int) -> None:
    model, checkpoint = model_for(name)
    training = checkpoint.get("training_seconds")
    columns = st.columns(4)
    columns[0].metric("Temps total", f"{seconds:.2f} s")
    columns[1].metric(f"Par image ({DEVICE.type.upper()})", f"{seconds / count * 1000:.1f} ms")
    columns[2].metric("Paramètres", f"{parameter_count(name, model):,}".replace(",", " "))
    if training:
        text = f"{training / 60:.0f} min"
        if checkpoint.get("vae_training_seconds"):
            text += f" + VAE {checkpoint['vae_training_seconds'] / 60:.0f}"
        columns[3].metric(f"Entraînement ({checkpoint.get('epochs')} ép.)", text)


def class_name(index: int) -> str:
    return f"{CLASS_LABELS_FR[index]} ({CLASS_NAMES[index]})"


def default_index(options: list[str], preferred: str) -> int:
    return options.index(preferred) if preferred in options else 0


# --------------------------------------------------------------------------- sidebar
levels = available_levels()
st.sidebar.header("Votre demande")
class_id = st.sidebar.selectbox("Vêtement", range(len(CLASS_NAMES)), format_func=class_name)
seed = int(st.sidebar.number_input("Graine", min_value=0, max_value=2**31 - 1, value=42))
count = st.sidebar.slider("Nombre d'images", 4, 64, 16, step=4)
st.sidebar.divider()
st.sidebar.subheader("Diffusion (niveaux 2 et 3)")
method = st.sidebar.radio("Échantillonneur", ("ddim", "ddpm"), horizontal=True,
                          format_func=str.upper)
if method == "ddpm":
    steps = 1000
    st.sidebar.caption("DDPM utilise les 1 000 pas du bruitage (lent sur CPU).")
else:
    steps = st.sidebar.select_slider("Nombre de pas DDIM", (5, 10, 20, 50, 100, 250), 50)
selected_guidance = st.sidebar.toggle(
    "Guidance retenue par modèle", value=True,
    help="Valeurs choisies par le balayage de développement : "
         + ", ".join(f"{LEVEL_NAMES[name]} {config['guidance_scale']:g}"
                     for name, config in HUMAN_TEST_DIFFUSION_CONFIGS.items()),
)
if selected_guidance:
    guidance = None
    st.sidebar.caption(" · ".join(f"{name.upper()} : {config['guidance_scale']:g}"
                                  for name, config in HUMAN_TEST_DIFFUSION_CONFIGS.items()))
else:
    guidance = st.sidebar.slider("Échelle de guidance", 0.0, 6.0, 1.5, step=0.5)
st.sidebar.caption("Guidance 1 = conditionnel simple, 0 = sans classe, > 1 = classe renforcée.")

st.markdown(
    """<div class="hero"><h1>Catalogue de mode imaginaire</h1>
<p>Fashion-MNIST 28 × 28 · quatre générateurs entraînés par nos soins, du plus simple
au plus avancé, comparés avec le même protocole.</p></div>""",
    unsafe_allow_html=True,
)
cards = st.columns(4)
for column, name in zip(cards, CHECKPOINTS):
    state = "" if name in levels else " off"
    note = "" if name in levels else " <i>(absent)</i>"
    column.markdown(
        f'<div class="level-card{state}"><b>{LEVEL_NAMES[name]}</b>{note}<br>'
        f"<span>{LEVEL_PITCH[name]}</span></div>",
        unsafe_allow_html=True,
    )
st.write("")
if not levels:
    st.error("Aucun checkpoint dans checkpoints/. Entraînez ou téléchargez les modèles (README).")
    st.stop()

(tab_generate, tab_gallery, tab_inside, tab_compare, tab_sweep, tab_results,
 tab_human) = st.tabs(["Générer", "Galerie", "Comprendre les modèles", "Comparer",
                       "Guidance et pas", "Résultats", "Test humain"])

# --------------------------------------------------------------------------- generate
with tab_generate:
    level = st.radio("Niveau", levels, format_func=LEVEL_NAMES.get, horizontal=True,
                     index=default_index(levels, "gan"))
    if st.button("Générer", type="primary"):
        images, seconds = generate(level, [class_id] * count, seed, method, steps, guidance)
        picture = to_picture(images)
        st.image(picture, caption=f"{LEVEL_NAMES[level]} · {CLASS_LABELS_FR[class_id]} · "
                                  f"graine {seed}")
        cost_metrics(level, seconds, count)
        st.download_button("Télécharger la planche (PNG)", png_bytes(picture),
                           file_name=f"{level}_{CLASS_NAMES[class_id]}_{seed}.png")

# --------------------------------------------------------------------------- gallery
with tab_gallery:
    st.write("Une ligne par vêtement : tout le catalogue d'un niveau en un clic.")
    left, right = st.columns([2, 1])
    gallery_level = left.radio("Niveau à afficher", levels, format_func=LEVEL_NAMES.get,
                               horizontal=True, index=default_index(levels, "gan"))
    per_class = right.slider("Images par vêtement", 4, 16, 10)
    if st.button("Remplir le catalogue", type="primary"):
        labels = torch.arange(10).repeat_interleave(per_class)
        images, seconds = generate(gallery_level, labels, seed, method, steps, guidance)
        for index in range(10):
            label_column, image_column = st.columns([1, 6])
            label_column.markdown(f"**{CLASS_LABELS_FR[index]}**")
            image_column.image(to_picture(images[index * per_class:(index + 1) * per_class],
                                          columns=per_class, scale=3))
        cost_metrics(gallery_level, seconds, len(labels))
        st.download_button("Télécharger le catalogue (PNG)",
                           png_bytes(to_picture(images, columns=per_class)),
                           file_name=f"catalogue_{gallery_level}_{seed}.png")

# --------------------------------------------------------------------------- inside
with tab_inside:
    view = st.radio(
        "Que voulez-vous voir ?",
        ("diffusion", "vae", "gan"),
        horizontal=True,
        format_func={"diffusion": "Diffusion pas à pas", "vae": "Espace latent du VAE",
                     "gan": "Interpolations du GAN"}.get,
    )

    if view == "diffusion":
        st.markdown(
            "**1. Bruitage (sens direct, sans réseau).** On ajoute du bruit gaussien à une "
            "vraie image selon le calendrier linéaire de 1 000 pas : "
            "x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε."
        )
        real_images, real_labels = test_images()
        real = real_images[real_labels == class_id][seed % 300:seed % 300 + 1]
        noise_steps = [0, 50, 100, 200, 300, 400, 500, 650, 800, 999]
        noisy = visualize.forward_noising(real, noise_steps,
                                          generator=torch.Generator().manual_seed(seed))
        st.image(to_picture(torch.cat(noisy), columns=len(noise_steps), scale=4),
                 caption="t = " + " · ".join(str(t) for t in noise_steps))

        st.markdown(
            "**2. Débruitage (sens inverse, appris par le modèle).** À chaque pas, le réseau "
            "prédit le bruit ; on en déduit l'image finale estimée (x̂_0) et on avance vers "
            "une image un peu moins bruitée (x_t)."
        )
        diffusion_levels = [name for name in levels if name in DIFFUSION_MODELS]
        if not diffusion_levels:
            st.info("Aucun modèle de diffusion disponible pour l'instant.")
        else:
            left, middle, right = st.columns(3)
            diffusion_level = left.radio("Modèle de diffusion", diffusion_levels,
                                         format_func=LEVEL_NAMES.get)
            shown = middle.slider("Étapes affichées", 6, 16, 10)
            examples = right.slider("Exemples", 1, 4, 2)
            if diffusion_level == "ldm":
                st.caption("Niveau 3 : le débruitage a lieu sur 16 nombres ; chaque étape "
                           "est décodée par le VAE pour être vue.")
            if st.button("Lancer le débruitage", type="primary"):
                trajectory: list = []
                generate(diffusion_level, [class_id] * examples, seed, method, steps,
                         guidance, trajectory=trajectory)
                st.session_state["trajectory"] = (diffusion_level, trajectory)
            saved = st.session_state.get("trajectory")
            if saved and saved[0] == diffusion_level:
                trajectory = saved[1]
                selected = visualize.pick_steps(trajectory, shown)
                caption = "t = " + " · ".join(str(step["timestep"]) for step in selected)
                for example in range(trajectory[0]["noisy"].shape[0]):
                    st.markdown(f"Exemple {example + 1} : image courante x_t")
                    st.image(to_picture(torch.stack([s["noisy"][example] for s in selected]),
                                        columns=len(selected)), caption=caption)
                    st.markdown(f"Exemple {example + 1} : image finale estimée x̂_0")
                    st.image(to_picture(torch.stack([s["clean"][example] for s in selected]),
                                        columns=len(selected)), caption=caption)
                st.markdown("**Rejouer pas à pas**")
                index = st.slider("Pas", 1, len(trajectory), len(trajectory))
                step = trajectory[index - 1]
                left, right = st.columns(2)
                left.image(to_picture(step["noisy"], columns=4, scale=6),
                           caption=f"x_t, t = {step['timestep']}")
                right.image(to_picture(step["clean"], columns=4, scale=6),
                            caption=f"x̂_0 estimée à t = {step['timestep']}")

    elif view == "vae":
        if "vae" not in levels:
            st.info("Checkpoint du VAE absent.")
        else:
            vae, _ = model_for("vae")
            st.markdown(
                "Chaque vraie image est encodée en **16 nombres**, projetés ici en 2D. "
                "Le VAE est *conditionnel* : la classe lui est donnée à part, donc l'espace "
                "latent code surtout le **style** (largeur, contraste, épaisseur) et les "
                "classes s'y superposent. C'est attendu."
            )
            method_2d = st.radio("Projection", ("pca", "tsne"), horizontal=True,
                                 format_func={"pca": "ACP (rapide)",
                                              "tsne": "t-SNE (≈ 20 s)"}.get)
            _, real_labels = test_images()
            codes, points_tensor = latent_projection(CHECKPOINTS["vae"].stat().st_mtime,
                                                     method_2d)
            points = points_tensor.numpy()
            labels_np = real_labels.numpy()
            figure, axes = plt.subplots(2, 5, figsize=(13, 5.6), sharex=True, sharey=True)
            for index, axis in enumerate(axes.flat):
                axis.scatter(points[:, 0], points[:, 1], s=3, color=MUTED, linewidths=0)
                mask = labels_np == index
                axis.scatter(points[mask, 0], points[mask, 1], s=6, color=ACCENT,
                             linewidths=0)
                axis.set_title(CLASS_LABELS_FR[index], fontsize=10,
                               fontweight="bold" if index == class_id else "normal")
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_color("#e3e1da")
            figure.suptitle("Codes latents de 3 000 vraies images de test (en bleu : la "
                            "classe du panneau ; en gris : toutes les autres)", fontsize=11)
            figure.tight_layout()
            st.pyplot(figure)
            plt.close(figure)

            left, right = st.columns(2)
            with left:
                st.markdown(f"**Carte de l'espace latent : {CLASS_LABELS_FR[class_id]}**")
                st.caption("On parcourt les deux directions principales (−2,5 à +2,5 "
                           "écarts-types autour du code moyen) et on décode chaque point.")
                grid = visualize.latent_map(vae, codes[real_labels == class_id], class_id)
                st.image(to_picture(grid, columns=9, scale=2))
            with right:
                st.markdown("**Même code latent, toutes les classes**")
                st.caption("Chaque ligne garde le même style ; seule la classe change.")
                st.image(to_picture(visualize.same_noise_all_classes(vae, "vae", seed),
                                    columns=10, scale=2))
                st.markdown("**Interpolation entre deux codes**")
                st.image(to_picture(visualize.interpolate(vae, "vae", class_id, seed,
                                                          seed + 1), columns=10, scale=2),
                         caption=f"graine {seed} → graine {seed + 1}")

    else:
        if "gan" not in levels:
            st.info("Checkpoint du GAN absent.")
        else:
            gan, _ = model_for("gan")
            st.markdown(
                "Le générateur transforme un vecteur de **64 nombres aléatoires** en image. "
                "En glissant d'un vecteur à l'autre, l'image change progressivement : le GAN "
                "n'a pas appris par cœur quelques images, il a appris un espace continu."
            )
            rows = st.slider("Nombre d'interpolations", 1, 8, 4)
            interpolations = torch.cat([
                visualize.interpolate(gan, "gan", class_id, seed + 2 * row,
                                      seed + 2 * row + 1, steps=10)
                for row in range(rows)
            ])
            st.image(to_picture(interpolations, columns=10, scale=3),
                     caption=f"{CLASS_LABELS_FR[class_id]} : chaque ligne passe d'un bruit "
                             "à un autre en 10 étapes")
            st.markdown("**Même bruit, toutes les classes**")
            st.image(to_picture(visualize.same_noise_all_classes(gan, "gan", seed, rows=6),
                                columns=10, scale=3),
                     caption=" · ".join(CLASS_LABELS_FR))
            st.caption("Pour générer beaucoup d'images d'un coup : onglet « Galerie ».")

# --------------------------------------------------------------------------- compare
with tab_compare:
    st.write("Même vêtement, même graine et même nombre d'images pour chaque niveau. "
             "Les réglages de diffusion de la barre latérale s'appliquent aux niveaux 2 et 3.")
    if st.button("Comparer", type="primary"):
        for name in levels:
            images, seconds = generate(name, [class_id] * count, seed, method, steps,
                                       guidance)
            st.subheader(LEVEL_NAMES[name])
            st.image(to_picture(images))
            cost_metrics(name, seconds, count)

# --------------------------------------------------------------------------- sweeps
with tab_sweep:
    sweep_levels = [name for name in levels if name in DIFFUSION_MODELS]
    if not sweep_levels:
        st.info("Aucun modèle de diffusion disponible pour l'instant.")
    else:
        sweep_level = st.radio("Modèle étudié", sweep_levels, format_func=LEVEL_NAMES.get,
                               horizontal=True)
        sweep_kind = st.radio(
            "Faire varier", ("guidance", "pas"), horizontal=True,
            format_func=lambda kind: "l'échelle de guidance" if kind == "guidance"
            else "le nombre de pas DDIM",
        )
        per_row = st.slider("Images par ligne", 4, 12, 8)
        if st.button("Lancer", type="primary"):
            values = ((0.0, 1.0, 1.5, 3.0, 5.0) if sweep_kind == "guidance"
                      else (5, 10, 20, 50, 100))
            for value in values:
                kwargs = (
                    {"method": "ddim", "steps": steps if method == "ddim" else 50,
                     "guidance": value}
                    if sweep_kind == "guidance"
                    else {"method": "ddim", "steps": value, "guidance": guidance}
                )
                images, seconds = generate(sweep_level, [class_id] * per_row, seed, **kwargs)
                label = (f"guidance {value:g}" if sweep_kind == "guidance"
                         else f"DDIM {value} pas")
                st.image(to_picture(images, columns=per_row),
                         caption=f"{label} · {seconds / per_row * 1000:.1f} ms/image")
            st.caption("Même graine sur chaque ligne : seule la valeur testée change.")

# --------------------------------------------------------------------------- results
with tab_results:
    found = False
    for folder, title in (
        (Path("runs/eval_final"), "Évaluation finale (`python eval.py`)"),
        (Path("runs/eval_dev"), "Développement (`python eval.py --dev`)"),
        (Path("runs/guidance_sweep"), "Balayage de la guidance (développement)"),
    ):
        table = folder / "results.md"
        if table.is_file():
            found = True
            st.subheader(title)
            st.markdown(table.read_text(encoding="utf-8"))
            grids = sorted(folder.glob("*_samples.png"))
            if grids:
                with st.expander("Échantillons aléatoires (première graine)"):
                    columns = st.columns(min(4, len(grids)))
                    for index, grid in enumerate(grids):
                        columns[index % len(columns)].image(
                            str(grid), caption=grid.stem.replace("_samples", ""))
    figures = sorted(Path("runs/figures").glob("*.png"))
    if figures:
        found = True
        st.subheader("Figures du rapport (`python make_figures.py`)")
        choice = st.selectbox("Figure", figures, format_func=lambda path: path.stem)
        st.image(str(choice))
    for path, caption in (
        (Path("runs/gan/monitoring.png"), "Suivi du GAN : discriminateur, effondrement, qualité"),
        (Path("runs/gan/loss.png"), "Pertes du GAN"),
    ):
        if path.is_file():
            found = True
            with st.expander(caption):
                st.image(str(path))
    if not found:
        st.info("Pas encore de résultats : lancez `python eval.py`.")

# --------------------------------------------------------------------------- human test
with tab_human:
    import human_test

    if not human_test.MANIFEST.is_file():
        st.info("Pool d'images absent : lancez `python human_test.py prepare`.")
    else:
        manifest = human_test.load_manifest()
        items = manifest["items"]
        st.write(
            "Pour chaque image, dites si c'est une **vraie** photo du catalogue "
            "Fashion-MNIST ou une image **générée**. Répondez vite, à l'instinct. "
            f"{len(items)} images."
        )
        participant = st.text_input("Votre prénom ou un pseudonyme").strip()
        if participant:
            state_key = "human_" + hashlib.sha1(participant.encode()).hexdigest()[:10]
            if state_key not in st.session_state:
                order = list(range(len(items)))
                random.Random(participant).shuffle(order)
                already = {
                    answer["image_id"]
                    for answer in human_test.load_answers()
                    if answer["participant"] == participant
                }
                st.session_state[state_key] = [
                    index for index in order if items[index]["id"] not in already
                ]
            remaining = st.session_state[state_key]
            if remaining:
                st.caption(
                    f"{len(remaining)} images à juger. Cochez une réponse sous chaque "
                    "image, puis envoyez tout en bas de la page."
                )
                choices = {"Vraie": "real", "Générée": "generated"}
                columns_per_row = 5
                with st.form(f"form_{state_key}"):
                    picked = {}
                    for start in range(0, len(remaining), columns_per_row):
                        row = st.columns(columns_per_row)
                        for offset, (column, index) in enumerate(
                            zip(row, remaining[start:start + columns_per_row])
                        ):
                            item = items[index]
                            image = np.array(
                                Image.open(human_test.POOL_DIR / f"{item['id']}.png")
                            )
                            with column:
                                st.image(np.kron(image, np.ones((5, 5), dtype=np.uint8)),
                                         caption=f"n° {start + offset + 1}")
                                picked[item["id"]] = st.radio(
                                    f"Image {item['id']}", list(choices), index=None,
                                    horizontal=True, label_visibility="collapsed",
                                    key=f"pick_{state_key}_{item['id']}",
                                )
                    submitted = st.form_submit_button("Envoyer mes réponses", type="primary")
                if submitted:
                    missing = [image_id for image_id, value in picked.items() if value is None]
                    if missing:
                        st.error(f"Il manque {len(missing)} réponse(s) : rien n'a été enregistré.")
                    else:
                        for image_id, value in picked.items():
                            human_test.record_answer(participant, image_id, choices[value])
                        st.session_state[state_key] = []
                        st.rerun()
            else:
                st.success("Merci, c'est terminé !")
        with st.expander("Résultats provisoires (organisateurs)"):
            scores = human_test.compute_scores(manifest, human_test.load_answers())
            st.write(f"{scores['participants']} participants, {scores['answers']} réponses.")
            if scores["per_source"]:
                st.table([
                    {
                        "source": source,
                        "pris pour vrai": f"{stats['judged_real_rate']:.0%}",
                        "réponses": str(stats["answers"]),
                        "personnes": str(stats["participants"]),
                    }
                    for source, stats in scores["per_source"].items()
                ])
            st.caption("`python human_test.py score` enregistre ces résultats pour eval.py.")
