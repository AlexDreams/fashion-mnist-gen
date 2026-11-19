# Catalogue de mode imaginaire : du VAE à la diffusion latente sur Fashion-MNIST

Alexandre Garreau, Nicolas Donier · ECE ING5, Mineur IA générative · novembre 2026
Dépôt : https://github.com/AlexDreams/fashion-mnist-gen · reproduction : `python eval.py`

> Brouillon. Les valeurs `[à compléter]` viennent de `runs/eval_final/results.md`
> (après `python eval.py`) et de `runs/human_test/results.json`.

## 1. Parcours et cas d'usage

Nous avons suivi le parcours Image, sur Fashion-MNIST : 70 000 images 28 × 28 en
niveaux de gris, réparties en 10 classes de vêtements. Le cas d'usage est un
**catalogue de mode imaginaire**. Une demande type est « 16 robes, graine 42 » :
on choisit la classe et la graine, et le résultat doit être reproductible.
Quatre générateurs, du plus simple au plus avancé, répondent à la même demande.
Ils sont comparés avec un protocole unique, à budget d'entraînement égal (30
époques) et sur la même machine (RTX 3060). Tous les modèles ont été entraînés
par nous ; aucun modèle pré-entraîné ne génère d'images.

## 2. Architecture

| Niveau | Modèle | Taille | Génération |
|---|---|---|---|
| 0 | VAE conditionnel (MLP, latent de dimension 16) | 551 728 param. | 1 passe du décodeur |
| 1 | GAN conditionnel (type DCGAN, bruit de dimension 64) | 653 025 param. | 1 passe du générateur |
| 2 | DDPM conditionnel (U-Net), guidance sans classifieur | [à compléter] param. | 1 000 pas (DDPM) ou 50/10 (DDIM) |
| 3 | Diffusion latente : MLP résiduel dans l'espace du VAE + décodeur figé | 1 110 176 param. | 50 pas sur 16 nombres + 1 décodage |

La classe est donnée à chaque modèle : en one-hot pour le VAE, par un
plongement (*embedding*) pour le GAN et la diffusion. Pour la guidance, la
diffusion apprend aussi une classe « vide » : l'étiquette est retirée dans 10 %
des cas à l'entraînement.

**Composant écrit sans framework.** Le calendrier de bruit linéaire (β de 10⁻⁴
à 0,02, T = 1 000), le bruitage q(x_t | x_0), la moyenne a posteriori exacte, et
les boucles d'échantillonnage DDPM et DDIM avec guidance sont écrits en PyTorch
pur. Ils servent aux niveaux 2 et 3. Les boucles d'entraînement sont aussi
écrites à la main.

**Pourquoi le niveau 3.** Nos mesures montraient deux défauts. Le VAE est rapide
mais flou, et il se trompe souvent de classe : le prior N(0, I) ne correspond
pas à la vraie répartition des codes latents. La diffusion sur pixels est de
bonne qualité mais coûteuse : 50 à 1 000 passes d'un U-Net par image. La
diffusion latente apprend la vraie distribution des codes du VAE avec un petit
MLP, puis réutilise le décodeur. Elle garde la rapidité du VAE et apporte la
fidélité de la diffusion.

**Interface.** L'application Streamlit permet de choisir le niveau, la classe,
la graine, l'échantillonneur, le nombre de pas et la guidance ; elle affiche le
temps de génération et le coût de chaque modèle. Elle propose aussi une galerie
de tout le catalogue et une comparaison des niveaux. L'onglet « Comprendre les
modèles » montre la diffusion pas à pas (x_t et x̂_0), l'espace latent du VAE
et les interpolations du GAN.

## 3. Protocole et choix justifiés par des mesures

**Données.** L'ensemble d'entraînement officiel est découpé en 54 000 images
d'entraînement et 6 000 de validation (graine 42). L'ensemble de test officiel,
jamais vu à l'entraînement, est séparé en deux tranches par classe :
- une tranche de **développement** (100 images par classe) pour tous les
  réglages ;
- une tranche **finale** (images 200 à 299 de chaque classe), utilisée **une
  seule fois** par `python eval.py`.

**Mesures.** Chaque mesure porte sur 100 images par classe, avec 3 graines
d'échantillonnage (2029 à 2031 pour l'évaluation finale) ; on rapporte la
moyenne ± l'écart-type.
- **KID Inception** (torchmetrics) : la mesure standard. Elle a une limite :
  Inception a appris sur des photos couleur, pas sur des vêtements 28 × 28.
- **Fashion-KID** : le même calcul, sur les caractéristiques de notre
  classifieur de vêtements.
- **Précision de classe** : mesurée par un classifieur entraîné une seule fois
  puis figé. Son plafond est sa précision sur les vraies images, environ 89 %.
- **Couverture** des classes et des caractéristiques.
- **Diversité** : distance moyenne entre images d'une même classe, divisée par
  celle des vraies images. Une classe sous 0,6 compte comme effondrée.
- **Test humain à l'aveugle** : 5 personnes, 20 images par source.
- **Coûts** : temps d'entraînement, temps par image et nombre de paramètres.

**Choix appuyés par une mesure.**
- *Guidance.* Nous avons balayé la guidance de 0 à 5 en DDIM 50 pas, sur la
  tranche de développement (`courbe_guidance.png`). Nous retenons **2,0 pour
  la diffusion sur pixels** (précision de classe 92,3 %) et **3,0 pour la
  diffusion latente** (88,9 %). Au-delà, la précision ne gagne presque plus
  rien, et la diversité et le KID se dégradent.
- *Nombre de pas.* DDIM 10 pas garde une qualité proche de DDPM 1 000 pas pour
  environ 100 fois moins de temps (`pas_ddpm.png`). Le tableau donne les trois
  réglages.
- *Sélection du GAN.* Nous suivons à chaque époque D(réel), D(faux), la
  diversité par classe, la précision et le Fashion-KID sur la validation
  (`runs/gan/monitoring.png`). Aucun effondrement n'a été observé : le ratio de
  diversité reste au-dessus de 0,9. Nous gardons donc la dernière époque.

## 4. Résultats (tranche finale, RTX 3060)

[à compléter : copier `runs/eval_final/results.md`. Lignes : vraies images, VAE,
GAN, DDPM 1000 / DDIM 50 / DDIM 10 (guidance 2,0), diffusion latente DDPM 1000
/ DDIM 50 / DDIM 10 (guidance 3,0) ; colonne test humain.]

Temps d'entraînement (30 époques) :

| Modèle | Temps d'entraînement |
|---|---|
| VAE | 259 s |
| GAN | 356 s |
| Diffusion sur pixels | 1 063 s |
| Débruiteur latent | 132 s + VAE 259 s |
| Classifieur d'évaluation | 43 s |

**Lecture.**
- Chaque niveau corrige un défaut du précédent. Le GAN rend des images nettes
  là où le VAE est flou. La diffusion couvre mieux les classes et se pilote par
  la guidance. La diffusion latente obtient une fidélité proche de la
  diffusion sur pixels pour une fraction du coût.
- Les deux KID ne classent pas toujours les modèles dans le même ordre : le KID
  Inception pénalise le grain des images du GAN, le Fashion-KID pénalise le
  flou du décodeur. Nous gardons donc les deux.
- [à compléter : le meilleur niveau sur chaque mesure, et ce que le test humain
  confirme ou contredit.]

## 5. Analyse de cinq échecs

Pour chaque échec : ce qu'on voit, la mesure qui le montre, la cause probable et
une piste de correction.

1. **VAE : chemises et sandales hors classe** (`echecs_vae.png`).
   - Ce qu'on voit : des images moyennes et floues.
   - Mesure : précision de classe basse sur Chemise et Sandale [à compléter].
   - Cause : le prior N(0, I) tombe dans des zones peu peuplées de l'espace
     latent, et la reconstruction en MSE moyenne les détails. L'espace latent
     (`vae_espace_latent.png`) montre que les classes s'y superposent : le VAE
     conditionnel code le style, pas le vêtement.
   - Correction : c'est précisément ce que corrige le niveau 3.
2. **GAN : vêtements « trop typiques »**.
   - Mesure : sa précision de classe peut dépasser celle des vraies images.
   - Cause : le générateur exagère les traits distinctifs de chaque classe. La
     métrique récompense la caricature, alors que le grain visible le trahit
     au test humain [à compléter].
   - Correction : régularisation spectrale, ou entraînement plus long avec
     sélection sur le KID de validation.
3. **Diffusion latente : sandales générées comme des bottines**
   (`echecs_ldm.png`).
   - Cause : les deux chaussures sont proches dans l'espace du VAE, et une
     guidance faible ne suffit pas à les séparer.
   - Correction : la guidance 3,0 corrige visiblement ce cas
     (`guidance_ldm.png`).
4. **Guidance mal réglée.**
   - À 0, la classe est souvent perdue.
   - À 5, les images saturent et se ressemblent, et la diversité chute
     (`guidance_ddpm.png`).
   - Le choix 2,0 / 3,0 est un compromis mesuré.
5. **DDIM 10 pas : détails perdus.**
   - Mesure : la diversité baisse par rapport à 50 pas.
   - Cause : chaque pas doit retirer une grande part du bruit d'un coup.
     L'image finale estimée x̂_0 se fige très tôt (`diffusion_etapes_ddpm.png`)
     et les détails fins n'ont pas le temps d'apparaître.
   - Correction : rester à 20–50 pas, ou passer à un échantillonneur d'ordre
     supérieur.

## 6. Limites et coût

- Chaque modèle n'a été entraîné qu'une fois. Les écarts-types reflètent les
  graines d'échantillonnage, pas la variance d'un entraînement à l'autre.
- Le format 28 × 28 en niveaux de gris est simple. Le Fashion-KID et la
  couverture sont des mesures propres au projet.
- Le test humain ne porte que sur 5 personnes, avec un intervalle large.
- Coût : la diffusion sur pixels est la plus chère à entraîner (1 063 s contre
  356 s pour le GAN) et à utiliser (50 à 1 000 passes d'un U-Net par image). La
  diffusion latente divise ces deux coûts.
- Piste : appliquer le même pipeline à un jeu de vêtements en couleur. Le
  niveau 3 s'y adapte en changeant seulement l'autoencodeur.
