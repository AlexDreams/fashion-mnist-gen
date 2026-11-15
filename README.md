# Génération d’images de mode avec Fashion-MNIST

Projet du parcours « Image » du cours d’IA générative 2026. L’objectif est
d’entraîner et de comparer des générateurs conditionnels sur Fashion-MNIST
(images 28 × 28, dix catégories), puis de présenter les résultats dans une
petite application interactive.

| Niveau | Modèle | Script |
|---|---|---|
| 0 | VAE conditionnel (référence) | `train_vae.py` |
| 1 | GAN conditionnel, suivi de l’effondrement des modes | `train_gan.py` |
| 2 | DDPM conditionnel (U-Net), guidance sans classifieur, DDPM puis DDIM | `train_diffusion.py` |
| 3 | Extension : diffusion latente dans l’espace du VAE | `train_latent_diffusion.py` |

Composant écrit sans framework : le bruitage et l’échantillonnage DDPM/DDIM
(niveaux 2 et 3) ainsi que les boucles d’entraînement.

## Lancer la démo

```powershell
.\.venv\Scripts\streamlit.exe run app.py     # interface (http://localhost:8501)
.\.venv\Scripts\python.exe eval.py           # tableau de résultats (protocole figé)
.\.venv\Scripts\python.exe eval.py --dev     # même tableau sur la partition de développement
.\.venv\Scripts\python.exe make_figures.py   # figures du rapport dans runs/figures/
.\.venv\Scripts\python.exe -m pytest -q      # tests
```

L’interface propose : génération par niveau (classe, graine, échantillonneur,
nombre de pas, guidance, temps et coût affichés), galerie de tout le catalogue
(une ligne par vêtement, téléchargeable), onglet « Comprendre les modèles »
(diffusion pas à pas : bruitage puis débruitage avec x_t et x̂_0 ; espace latent
du VAE en ACP ou t-SNE, carte latente et interpolations ; interpolations du GAN
et « même bruit, toutes les classes »), comparaison de tous les niveaux sur la
même demande, effet de la guidance et du nombre de pas, résultats du protocole
et test humain à l’aveugle.

`make_figures.py` produit les mêmes visualisations pour le rapport :
`bruitage_direct.png`, `diffusion_etapes_<modèle>.png`, `vae_espace_latent.png`,
`vae_carte_latente.png`, `gan_interpolations.png`, `gan_meme_bruit.png`.

Pipeline complet sur la machine GPU (entraînement à budget égal de 30 époques,
évaluation de développement, balayage de la guidance, figures, pool du test
humain, archive des checkpoints) :

```powershell
powershell -ExecutionPolicy Bypass -File scripts\pipeline_gpu.ps1
```

## Niveau 3 : diffusion latente

Symptômes mesurés qui motivent l’extension : le VAE est rapide mais produit des
images floues, peu variées et parfois hors classe (le prior N(0, I) ne
correspond pas à la répartition réelle des codes latents) ; la diffusion sur les
pixels est lente (50 à 1 000 passes d’un U-Net par image). La diffusion latente
apprend la vraie distribution des codes latents du VAE, par classe, avec un
petit MLP sur des vecteurs de dimension 16, puis le décodeur du VAE (figé)
produit l’image. Mesures attendues face au niveau 2 : temps d’entraînement et
par image, qualité (KID, précision de classe, diversité, test humain).

```powershell
.\.venv\Scripts\python.exe train_latent_diffusion.py --epochs 100
```

Le checkpoint contient les poids du VAE utilisé : il est autonome. Son coût
d’entraînement est rapporté avec celui du VAE (« + VAE … s » dans le tableau).

## Test humain à l’aveugle

```powershell
.\.venv\Scripts\python.exe human_test.py prepare   # pool fixe : 20 images par niveau + 20 vraies
.\.venv\Scripts\streamlit.exe run app.py           # onglet « Test humain », 5 personnes
.\.venv\Scripts\python.exe human_test.py score     # runs/human_test/results.json
```

Chaque participant voit le même pool dans un ordre propre à son pseudonyme. Le
score d’un niveau est la part de ses images jugées « vraies » (un succès pour
le générateur) ; les vraies images donnent le taux de contrôle. `eval.py`
ajoute automatiquement une colonne « Test humain » quand le fichier de
résultats existe.

## Démarrage rapide

Environnement initial de développement : Windows, Python 3.12, PyTorch avec
CUDA et GPU NVIDIA. Depuis PowerShell, à la racine du dépôt :

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe --index-strategy unsafe-best-match -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Sans GPU NVIDIA, installer la version CPU de PyTorch :

```powershell
uv pip install --python .venv\Scripts\python.exe --index-strategy unsafe-best-match -r requirements-cpu.txt
```

Le premier lancement télécharge Fashion-MNIST dans `data/`. Le premier calcul
du KID Inception télécharge aussi les poids d’Inception-v3 (environ 90 Mo). Pour vérifier
PyTorch et CUDA :

```powershell
.\.venv\Scripts\python.exe -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Les scripts choisissent CUDA lorsqu’il est disponible et peuvent fonctionner
sur CPU, avec un temps d’exécution supérieur.

## Organisation du code

| Fichier | Rôle |
|---|---|
| `fashion_mnist_gen/data.py` | Téléchargement, prétraitement et partitions Fashion-MNIST |
| `fashion_mnist_gen/classifier.py` | Classifieur de référence et caractéristiques 128-D |
| `fashion_mnist_gen/vae.py` | VAE conditionnel |
| `fashion_mnist_gen/gan.py` | GAN conditionnel |
| `fashion_mnist_gen/diffusion.py` | U-Net, DDPM et échantillonneur DDIM |
| `fashion_mnist_gen/metrics.py` | Fidélité, Fashion-KID, diversité (caractéristiques et pixels), couverture des caractéristiques et des classes |
| `fashion_mnist_gen/inception_metrics.py` | KID (et FID optionnel) Inception via torchmetrics |
| `train_classifier.py` | Entraînement du classifieur d’évaluation |
| `train_vae.py`, `train_gan.py`, `train_diffusion.py` | Entraînement des générateurs |
| `evaluate_protocol.py` | Évaluation commune et reproductible des modèles |
| `eval.py` | Commande unique qui reproduit le tableau de résultats (protocole figé) |
| `fashion_mnist_gen/latent_diffusion.py`, `train_latent_diffusion.py` | Niveau 3 : diffusion latente |
| `app.py` | Interface Streamlit |
| `human_test.py` | Test humain à l’aveugle : pool, réponses, scores |
| `make_figures.py` | Figures du rapport et de la vidéo |
| `scripts/pipeline_gpu.ps1` | Pipeline complet sur la machine GPU |
| `compare_vae_gan.py` | Comparaison initiale VAE/GAN |
| `tests/` | Tests des modèles, métriques et partitions du protocole |

## Entraînement

Les poids du classifieur et des générateurs sont nécessaires à l’évaluation.
Ils ne sont pas inclus dans Git; les commandes ci-dessous les recréent.
Lien de téléchargement des checkpoints de référence : **à compléter**
(archive à décompresser dans `checkpoints/`).

Le VAE et la diffusion enregistrent le meilleur modèle selon la perte de
validation. Le GAN n’a pas de perte de validation significative : par défaut
il enregistre la dernière époque (`--selection last`), ou l’époque de meilleur
Fashion-KID sur la validation avec `--selection validation_kid`.

```powershell
.\.venv\Scripts\python.exe train_classifier.py --epochs 5
.\.venv\Scripts\python.exe train_vae.py --epochs 20 --seed 42
.\.venv\Scripts\python.exe train_gan.py --epochs 20 --seed 42
.\.venv\Scripts\python.exe train_diffusion.py --epochs 10 --seed 42
```

Suivi du GAN (niveau 1). À chaque époque, sur la partition de validation et
avec un bruit fixé, `train_gan.py` enregistre dans `runs/gan/metrics.json` et
`runs/gan/monitoring.png` : D(réel) et D(généré), le rapport entre la diversité
intra-classe des images générées et celle des vraies images (distance L2
moyenne entre paires, par classe) et, si le classifieur existe, la précision de
classe et le Fashion-KID. Une classe dont le rapport passe sous 0,6
(`--collapse-ratio`) est signalée comme effondrée. Un point de sauvegarde est
conservé toutes les 5 époques dans `checkpoints/gan_epochs/` (`--save-every`).

Les sorties vont dans `checkpoints/` et `runs/`. Une expérience séparée peut
utiliser des chemins différents afin de ne pas écraser la référence.
Exemple de test rapide de la diffusion :

```powershell
.\.venv\Scripts\python.exe train_diffusion.py --epochs 1 --max-batches 2 --timesteps 20 --base-channels 16 --batch-size 8 --checkpoint checkpoints\diffusion_smoke.pt --output-dir runs\diffusion_smoke
```

## Protocole d’évaluation

- Entraînement : 54 000 images Fashion-MNIST; validation : 6 000 images,
  séparation déterministe avec la graine 42. Le classifieur est entraîné
  uniquement sur la partition d’entraînement.
- Test officiel : séparation déterministe par classe avec la graine 2026.
  Les 100 premières images de chaque classe constituent le jeu de développement;
  les images restantes sont réservées à l’évaluation finale.
- Les paramètres se choisissent sur le développement. Le jeu final ne doit être
  consulté qu’une fois les paramètres gelés. `--final-offset` permet de prendre
  une tranche finale ultérieure disjointe, par exemple `100` pour ne pas
  réutiliser les 100 premières images finales.
- Par défaut, l’évaluation génère 100 images par classe pour trois graines. Ces
  répétitions mesurent la variabilité de l’échantillonnage, pas celle des
  modèles entraînés.
- La fidélité de classe est la précision du classifieur figé sur les images
  générées. Fashion-KID est un MMD non biaisé à noyau polynomial cubique dans
  les caractéristiques 128-D du classifieur local; ce n’est **pas** le KID
  standard calculé avec Inception.
- KID Inception (protocole v2) : `KernelInceptionDistance` de torchmetrics,
  images en niveaux de gris répétées sur trois canaux et redimensionnées en
  299 × 299, toutes classes confondues (1 000 images réelles contre 1 000
  générées, 100 sous-ensembles). C’est la mesure standard demandée par le
  sujet; Inception ne connaît pas Fashion-MNIST, d’où le maintien du
  Fashion-KID à côté. `--fid` ajoute le FID, biaisé avec si peu d’images.
- Couverture des classes : part des classes qui reçoivent au moins la moitié
  de leur part uniforme des prédictions du classifieur, plus l’entropie
  normalisée de l’histogramme des classes prédites.
- Diversité et effondrement des modes : distance L2 moyenne entre images
  générées d’une même classe divisée par la même distance sur les vraies
  images de test de la classe. Un rapport inférieur à 0,6 signale une classe
  effondrée (`--collapse-ratio`).
- Ligne de référence « vraies images » : 100 autres images de test par classe
  (les dernières de chaque permutation, disjointes des tranches évaluées)
  passent par exactement le même calcul. Elle donne le plafond atteignable
  par chaque métrique, par exemple la précision du classifieur sur de vraies
  images.
- La diversité cosinus et la couverture de caractéristiques sont des métriques
  définies pour ce projet, et non des mesures standardisées. Les métriques de
  caractéristiques sont des diagnostics; elles ne remplacent pas le jugement
  humain.
- Le temps d’échantillonnage exclut l’inférence du classifieur et son calcul de
  métriques. Les temps dépendent du matériel et de la taille des lots.

### Reproduire le tableau : `python eval.py`

`eval.py` fige le protocole final (tranche réelle 200–299 de chaque classe,
graines 2029–2031, 100 images par classe) et produit une ligne par niveau, plus
une ligne par réglage de diffusion (DDPM 1 000, DDIM 50, DDIM 10, guidance
1,5). Les checkpoints absents sont ignorés avec un avertissement. Résultats :
`runs/eval_final/results.md` et `results.json`. Pendant les réglages, utiliser
`python eval.py --dev` (partition de développement). Tout argument de
`evaluate_protocol.py` peut être ajouté, par exemple `--no-inception-kid` ou
`--ddpm checkpoints\diffusion_30epochs.pt`.

### Premiers résultats du protocole v2 (CPU, partition de développement)

Modèles réentraînés sur CPU (graine 42, 20 époques chacun, même machine),
classifieur de 5 époques (90,3 % en validation). Commande : `python eval.py --dev`.
Ces chiffres ne se comparent pas directement au tableau v1 ci-dessous
(autre machine, autre classifieur, autre tranche réelle).

| Niveau | KID Inception ×10³ ↓ | Fashion-KID ↓ | Précision classe ↑ | Couverture classes ↑ | Couverture caract. ↑ | Diversité / réel | Classes effondrées | Entraînement | Temps / image (CPU) | Paramètres |
|---|---|---|---|---|---|---|---|---|---|---|
| Référence · vraies images | ≈ 0 | 0,29 | 89,4 % | 100 % | 49,8 % | 1,02 | 0 | — | — | — |
| 0 · VAE | 51,5 ± 1,1 | 176,0 ± 4,9 | 69,6 % | 90 % | 27,7 % | 0,83 | 0 | 376 s | 1,7e-05 s | 551 728 |
| 1 · GAN | 34,6 ± 0,2 | 9,4 ± 1,4 | 91,9 % | 100 % | 34,7 % | 0,99 | 0 | 2 090 s | 1,6e-04 s | 653 025 |
| 3 · Diffusion latente, DDPM 1 000 | 29,2 ± 0,5 | 101,8 ± 3,7 | 87,3 % | 100 % | 36,5 % | 0,81 | 0 | 643 s + VAE 376 s | 5,8e-02 s | 1 110 176 |
| 3 · Diffusion latente, DDIM 50 | 29,8 ± 1,0 | 100,7 ± 2,1 | 88,1 % | 100 % | 35,9 % | 0,80 | 0 | 643 s + VAE 376 s | 2,8e-03 s | 1 110 176 |
| 3 · Diffusion latente, DDIM 10 | 29,5 ± 0,8 | 102,3 ± 2,2 | 89,1 % | 100 % | 34,5 % | 0,73 | 0 | 643 s + VAE 376 s | 7,3e-04 s | 1 110 176 |

Lecture : le GAN bat nettement le VAE sur toutes les mesures de qualité, sans
effondrement des modes (rapport de diversité ≥ 0,93 dans chaque classe pendant
tout l’entraînement, voir `runs/gan/monitoring.png`). Sa précision de classe
dépasse celle du classifieur sur de vraies images (91,9 % contre 89,4 %) : il
produit des vêtements plus « typiques » que les vrais, notamment pour Shirt
(80 % contre 56 %). Le VAE ne couvre que 9 classes sur 10 : ses chemises
(37 %) et sandales (40 %) sont mal reconnues, et ses images sont moins variées
que les vraies (0,83). Le GAN coûte 5,6 fois plus de temps d’entraînement à
nombre d’époques égal.

Diffusion latente (niveau 3, guidance 1,5, mesurée ici contre les niveaux 0 et 1
faute de niveau 2 sur cette machine) : elle corrige le défaut principal du VAE
avec le même décodeur (précision 69,6 % → 88,1 %, toutes les classes couvertes)
et obtient le meilleur KID Inception, mais reste floue (Fashion-KID 101 contre
9 pour le GAN, diversité 0,80). Les deux KID se contredisent : Inception
pénalise le grain des images du GAN, le Fashion-KID pénalise le flou du
décodeur. DDIM 10 pas garde la qualité de DDPM 1 000 pas pour 80 fois moins de
temps. Échec typique : 13 % des images mal classées, surtout des sandales
générées comme des bottines (`runs/figures/echecs_ldm.png`) ; une guidance de 3
corrige visiblement ce cas (`runs/figures/guidance_ldm.png`), à confirmer par le
balayage de la guidance.

### Comparaison finale de référence (protocole v1)

Comparaison avec 100 images générées par classe et trois graines
d’échantillonnage. Les trois modèles ont été comparés sur la même tranche du
jeu de test, indices 200 à 299 dans chaque classe de la permutation déterminée
par la graine 2026 (`--final-offset 100`). Cette tranche est disjointe du
développement et de l’évaluation finale initiale. Réglages de diffusion :
DDIM, 50 pas, guidance 1,5.

| Modèle | Fidélité de classe ↑ | Fashion-KID ↓ | Couverture ↑ | Temps par image |
|---|---:|---:|---:|---:|
| VAE | 70,23 % | 128,65 | 29,00 % | 0,0000036 s |
| GAN conditionnel | **91,63 %** | **7,53** | **37,40 %** | 0,0000067 s |
| DDIM conditionnel, 30 époques | 88,83 % | 32,17 | 31,37 % | 0,00968 s |

L’entraînement prolongé améliore nettement la diffusion par rapport au premier
DDPM de 10 époques : fidélité de 76,40 % à 88,83 %, Fashion-KID de 148,13 à
32,17 et couverture de 19,17 % à 31,37 %. Le GAN reste meilleur en Fashion-KID
et en couverture; son échantillonnage est aussi beaucoup plus rapide. Ces
résultats restent limités à un seul entraînement par modèle.

Exemples et résultats de référence inclus dans le dépôt :
[VAE](examples/reference/vae_samples.png),
[GAN](examples/reference/gan_samples.png),
[DDIM](examples/reference/ddim_samples.png),
[courbe d’entraînement diffusion](examples/reference/diffusion_loss.png) et
[résultats détaillés](examples/reference/results_final.json). Ces grilles
facilitent la revue sans devoir télécharger les checkpoints. Les résultats
finaux initiaux à 10 époques et les résultats de développement à 30 époques
sont également conservés dans `examples/reference/`.

### Résultats de réglage de la diffusion

Résultats exploratoires sur la partition de développement (100 images par
classe, trois graines), avant l’expérience prolongée :

| Échantillonneur / entraînement | Guidance | Fidélité | Fashion-KID | Couverture | Secondes/image |
|---|---:|---:|---:|---:|---:|
| DDPM, 1 000 pas, 10 époques | 1,0 | 65,23 % | 198,73 | 16,83 % | 0,0873 |
| DDIM, 10 pas, 10 époques | 1,0 | 61,37 % | 252,60 | 15,57 % | 0,00079 |
| DDIM, 50 pas, 10 époques | 1,0 | 59,47 % | 230,88 | 14,73 % | 0,00318 |
| DDIM, 50 pas, 10 époques | 1,5 | 76,07 % | 146,21 | 20,17 % | 0,00629 |
| DDIM, 50 pas, 30 époques | 1,5 | **88,40 %** | **31,04** | **28,63 %** | 0,01001 |

Le DDPM 30 époques a été réentraîné depuis l’initialisation, avec la graine 42
et un checkpoint distinct. Sa meilleure MSE de validation est passée de
0,04259 (10 époques) à 0,03883 (30 époques); son entraînement a duré 689,8 s
sur la RTX 3060 Laptop. Le gain a d’abord été constaté sur le développement,
puis le tableau final a été recalculé pour les trois modèles sur une tranche
disjointe commune. Les chiffres des deux évaluations finales (10 et 30 époques)
ne doivent pas être comparés directement, car ils utilisent des tranches réelles
différentes.

Pour reproduire la comparaison de développement :

```powershell
.\.venv\Scripts\python.exe evaluate_protocol.py --models ddpm --ddpm checkpoints\diffusion_30epochs.pt --samples-per-class 100 --seeds 2026 2027 2028 --evaluation-partition development --diffusion-sampler ddim --diffusion-steps 50 --guidance-scale 1.5 --output-dir runs\protocol_diffusion_30epochs
```

Pour l’évaluation finale déjà réalisée :

```powershell
.\.venv\Scripts\python.exe evaluate_protocol.py --models vae gan ddpm --ddpm checkpoints\diffusion_30epochs.pt --samples-per-class 100 --seeds 2029 2030 2031 --evaluation-partition final --development-samples-per-class 100 --final-offset 100 --real-seed 2026 --diffusion-sampler ddim --diffusion-steps 50 --guidance-scale 1.5 --output-dir runs\protocol_final_30epochs
```

Les fichiers de résultats et les graphiques sont écrits dans le répertoire
indiqué avec `--output-dir`. Ils sont ignorés par Git avec les données et les
poids afin de garder le dépôt léger. Pour reproduire seulement l’évaluation,
il faut donc fournir les checkpoints séparément; sinon, les commandes
d’entraînement ci-dessus les régénèrent.

## Proposition de reprise avec le collaborateur

1. Installer l’environnement et lancer les tests.
2. Examiner les exemples et les résultats; discuter les compromis entre qualité
   et coût de calcul, ainsi que les limites des métriques actuelles.
3. Choisir ensemble une extension du niveau 3 qui répond à une limite observée.
4. Construire l’interface interactive et organiser le test humain à l’aveugle.
5. Refaire une évaluation finale après avoir gelé les modèles et réglages.
6. Finaliser le rapport et la vidéo.

Avant de publier une nouvelle comparaison finale, figer le modèle et les
paramètres. Si le jeu final a déjà été utilisé, choisir un `--final-offset`
qui réserve de nouvelles images, puis consigner ce choix et les graines dans
le rapport.

Pour une première revue, les questions prioritaires sont : le protocole et ses
métriques sont-ils assez clairement justifiés, l’extension choisie répond-elle
aux limites observées, et quelles améliorations seraient les plus utiles avant
de construire l’interface ?
