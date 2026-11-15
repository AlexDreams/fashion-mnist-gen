# Plan du rapport (4 pages max, PDF) et de la vidéo (3 à 5 min)

Chaque point renvoie au barème du sujet. Chiffres à prendre dans
`runs/eval_final/results.md` (après `python eval.py`), figures dans `runs/figures/`.

## Rapport

### 1. Parcours et cas d'usage (≈ ¼ page)
- Parcours Image, Fashion-MNIST 28 × 28, 10 classes : « catalogue de mode imaginaire ».
- Demande type : « 16 robes, graine 42 ». Contraintes : sans GPU possible, modèles entraînés par nous.

### 2. Architecture (≈ ½ page)
- Schéma : données → 4 générateurs → protocole commun (classifieur figé, Inception) → tableau + interface.
- Niveaux 0 à 3 en une ligne chacun (taille, conditionnement, échantillonneur).
- Composant sans framework : bruitage et échantillonnage DDPM/DDIM (niveaux 2 et 3), boucles d'entraînement.
- Figures « dans le modèle » : `diffusion_etapes_ddpm.png` (x_t et x̂_0), `vae_espace_latent.png`, `gan_interpolations.png`.

### 3. Protocole et choix justifiés par des mesures (≈ 1 page)
- Découpage : 54 000 / 6 000 (graine 42) ; test officiel → développement (100 par classe) / final (tranche 200–299, graine 2026), jamais vu à l'entraînement.
- 100 images par classe × 3 graines ; budget égal (30 époques), même machine.
- Métriques et leurs limites : KID Inception (standard, mais Inception ne connaît pas les vêtements) et Fashion-KID ; précision de classe (plafond = vraies images, ≈ 89 %) ; couverture des classes ; diversité / réel (< 0,6 = effondrement) ; test humain ; coûts.
- Choix appuyés par une mesure : guidance retenue (figure `courbe_guidance.png`), nombre de pas DDIM (`pas_ddpm.png`), sélection du GAN (suivi par époque, `runs/gan/monitoring.png`).

### 4. Tableau des niveaux 0 à 3 (≈ ½ page)
- Copier `results.md` (une ligne par réglage de diffusion : DDPM 1000, DDIM 50, DDIM 10).
- Figure `cout_qualite.png` : ce que chaque niveau apporte et ce qu'il coûte.

### 5. Analyse de 5 échecs (≈ ¾ page)
Pistes déjà observées (à confirmer sur les modèles finaux, figures `echecs_*.png`) :
1. VAE : chemises et sandales mal reconnues (37 % et 40 %), images moyennes et floues.
2. GAN : précision de classe supérieure aux vraies images (91,9 % contre 89,4 %) ; vêtements « trop typiques », la métrique récompense la caricature.
3. Diffusion latente : une bottine générée pour « sandale » (confusion des chaussures).
4. Diffusion avec guidance 0 ou trop forte : perte de la classe ou images saturées.
5. DDIM 10 pas : détails perdus par rapport à 50 pas.
Pour chacun : image, mesure qui le montre, cause probable, piste de correction.

### 6. Limites et coût (≈ ¼ page)
- Un seul entraînement par modèle (la variance des graines d'échantillonnage n'est pas celle des entraînements).
- 28 × 28, 10 classes ; métriques propres au projet (Fashion-KID, couverture) ; test humain sur 5 personnes.
- Coûts : temps d'entraînement, temps par image (GPU et CPU), paramètres.

## Vidéo (3 à 5 min), tout dans `streamlit run app.py`
1. (30 s) Le problème et les 4 niveaux.
2. (60 s) Onglets « Galerie » et « Comparer » : tout le catalogue, même demande, temps affichés.
   (45 s) Onglet « Comprendre les modèles » : diffusion pas à pas, espace latent du VAE, interpolations du GAN.
3. (60 s) Onglet « Guidance et nombre de pas » : guidance 0 → 5, DDIM 5 → 100 pas, puis DDPM 1000.
4. (45 s) Onglet « Résultats » : tableau, suivi du GAN (pas d'effondrement), coût contre qualité.
5. (45 s) Un échec analysé (figure `echecs_*.png`) et le test humain.
