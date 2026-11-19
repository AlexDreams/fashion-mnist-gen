# Catalogue de mode imaginaire : du VAE à la diffusion latente sur Fashion-MNIST

Alexandre Garreau, Nicolas Donier · ECE ING5, Mineur IA générative · novembre 2026
Dépôt : https://github.com/AlexDreams/fashion-mnist-gen · reproduction : `python eval.py`

> Brouillon. Chiffres actuels : partition de **développement** (RTX 3060, guidance
> 1,5 pour tous). À remplacer en section 4 par `runs/eval_final/results.md`
> (guidance retenue 2,0 / 3,0) et les scores du test humain : `[à compléter]`.

## 1. Parcours et cas d'usage

Nous avons suivi le parcours Image, sur Fashion-MNIST : 70 000 images 28 × 28 en
niveaux de gris, réparties en 10 classes de vêtements. Le cas d'usage est un
**catalogue de mode imaginaire**. Une demande type est « 16 robes, graine 42 » :
on choisit la classe et la graine, et le résultat doit être reproductible.
Quatre générateurs, du plus simple au plus avancé, répondent à la même demande.
Ils sont comparés avec un protocole unique, à budget égal (30 époques) et sur la
même machine (RTX 3060 Laptop). Tous les modèles ont été entraînés par nous ;
aucun modèle pré-entraîné ne génère d'images.

## 2. Architecture

| Niveau | Modèle | Paramètres | Génération d'une image |
|---|---|---|---|
| 0 | VAE conditionnel (MLP, latent de dimension 16) | 551 728 | 1 passe du décodeur |
| 1 | GAN conditionnel (type DCGAN, bruit de dimension 64) | 653 025 | 1 passe du générateur |
| 2 | DDPM conditionnel (U-Net), guidance sans classifieur | 351 329 | 1 000 pas (DDPM) ou 50/10 (DDIM) |
| 3 | Diffusion latente : MLP résiduel sur les 16 nombres du VAE + décodeur figé | 1 110 176 | 10 à 1 000 pas sur 16 nombres + 1 décodage |

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
mais flou, et il se trompe de classe : le prior N(0, I) ne correspond pas à la
vraie répartition des codes latents. La diffusion sur pixels est la meilleure,
mais elle coûte 50 à 1 000 passes d'un U-Net par image. La diffusion latente
apprend la vraie distribution des codes du VAE avec un petit MLP, puis réutilise
le décodeur.

**Interface (Streamlit).** On y choisit le niveau, la classe, la graine,
l'échantillonneur, le nombre de pas et la guidance ; le temps et le coût de
chaque génération sont affichés. L'application propose aussi une galerie de
tout le catalogue et une comparaison des niveaux. L'onglet « Comprendre les
modèles » montre la diffusion pas à pas (x_t et x̂_0,
`diffusion_etapes_ddpm.png`), l'espace latent du VAE et les interpolations du
GAN.

## 3. Protocole et choix justifiés par des mesures

**Données.** L'ensemble d'entraînement officiel est découpé en 54 000 images
d'entraînement et 6 000 de validation (graine 42). L'ensemble de test officiel,
jamais vu à l'entraînement, est séparé en deux tranches par classe :
- une tranche de **développement** (100 images par classe) pour tous les
  réglages ;
- une tranche **finale** (images 200 à 299 de chaque classe), utilisée **une
  seule fois** par `python eval.py`.

**Mesures.** Chaque mesure porte sur 100 images par classe, avec 3 graines
d'échantillonnage ; on rapporte la moyenne ± l'écart-type.
- **KID Inception** (torchmetrics) : la mesure standard. Elle a une limite :
  Inception a appris sur des photos couleur, pas sur des vêtements 28 × 28.
- **Fashion-KID** : le même calcul, sur les caractéristiques de notre
  classifieur de vêtements.
- **Précision de classe** : mesurée par un classifieur entraîné une seule fois
  puis figé (43 s d'entraînement). Son plafond est sa précision sur les vraies
  images : 89,2 %.
- **Couverture** des classes et des caractéristiques.
- **Diversité** : distance moyenne entre images d'une même classe, divisée par
  celle des vraies images. Une classe sous 0,6 compte comme effondrée.
- **Test humain à l'aveugle** : 5 personnes, 20 images par source.
- **Coûts** : temps d'entraînement, temps par image et nombre de paramètres.

**Choix 1 : la guidance** (balayage DDIM 50 pas, développement,
`courbe_guidance.png`).

| Guidance | 0 | 1 | 1,5 | **2** | 3 | 5 |
|---|---|---|---|---|---|---|
| Pixels : précision | 10,0 % | 76,7 % | 88,4 % | **92,3 %** | 95,9 % | 98,0 % |
| Pixels : Fashion-KID | 365 | 107 | 31,0 | **29,5** | 95,7 | 199 |
| Pixels : couverture caract. | 5,0 % | 24,0 % | 28,6 % | **28,0 %** | 23,7 % | 16,4 % |
| Latente : KID Inception ×10³ | 42,4 | 30,0 | 27,8 | 26,4 | **25,0** | 25,5 |
| Latente : Fashion-KID | 107 | 78,4 | 69,1 | 62,8 | **56,1** | 56,5 |

Au-delà de 2, la précision de la diffusion sur pixels monte encore, mais la
couverture des caractéristiques s'effondre (16 % à 5) : le modèle ne produit
plus que des vêtements « types ». Nous retenons **2,0 pour les pixels**, qui
donne le meilleur KID et le meilleur Fashion-KID. Nous retenons **3,0 pour la
diffusion latente**, qui donne les meilleurs KID.

**Choix 2 : le nombre de pas.** Sur pixels, réduire les pas dégrade nettement
la qualité : le KID passe de 10,8 (DDPM 1 000) à 14,0 (DDIM 50), puis à 20,8
(DDIM 10). En latent, il ne bouge pas : 27,7, 27,8 puis 27,4. Une trajectoire
de 16 nombres est assez simple pour 10 pas.

**Choix 3 : la sélection du GAN.** Nous suivons à chaque époque D(réel),
D(faux), la diversité par classe, la précision et le Fashion-KID sur la
validation (`runs/gan/monitoring.png`). Aucun effondrement n'est apparu : la
diversité reste à 1,01 fois celle du réel en fin d'entraînement. Nous gardons
donc la dernière époque.

## 4. Résultats

Partition de développement, guidance 1,5 pour tous (le tableau final utilisera
2,0 / 3,0) :

| Niveau | KID Inc. ×10³ ↓ | Fashion-KID ↓ | Précision ↑ | Couv. classes | Diversité | Entraînement | Temps / image |
|---|---|---|---|---|---|---|---|
| Vraies images | ≈ 0 | 0,6 | 89,2 % | 100 % | 1,02 | — | — |
| 0 · VAE | 44,0 ± 0,7 | 111 ± 3 | 76,0 % | 87 % | 0,79 | 259 s | 0,009 ms |
| 1 · GAN | 28,2 ± 0,9 | **5,1 ± 1,5** | **91,8 %** | 100 % | 1,01 | 356 s | 0,023 ms |
| 2 · DDPM 1 000 | **10,8 ± 0,4** | 18,5 ± 1,6 | 90,5 % | 100 % | 0,91 | 1 063 s | 788 ms |
| 2 · DDIM 50 | 14,0 ± 0,7 | 31,0 ± 1,7 | 88,4 % | 100 % | 0,97 | 1 063 s | 35,6 ms |
| 2 · DDIM 10 | 20,8 ± 1,1 | 51,4 ± 2,8 | 88,1 % | 93 % | 0,87 | 1 063 s | 7,5 ms |
| 3 · Latente DDPM 1 000 | 27,7 ± 0,5 | 68,3 ± 1,4 | 86,2 % | 90 % | 0,82 | 132 s + VAE 259 s | 51,5 ms |
| 3 · Latente DDIM 50 | 27,8 ± 0,2 | 69,1 ± 2,1 | 87,2 % | 97 % | 0,80 | 132 s + VAE 259 s | 2,6 ms |
| 3 · Latente DDIM 10 | 27,4 ± 0,3 | 69,5 ± 2,2 | 88,1 % | 100 % | 0,74 | 132 s + VAE 259 s | 0,5 ms |

[à compléter : tableau final de `python eval.py` et colonne « test humain (pris
pour vrai) ».]

**Lecture.**
- **Ce qu'apporte chaque niveau.**
  - Le GAN divise par 20 le Fashion-KID du VAE et couvre toutes les classes.
  - La diffusion sur pixels obtient le meilleur KID Inception (10,8 contre
    28,2 pour le GAN), mais coûte 3 fois plus d'entraînement et plus de
    30 000 fois plus de temps par image en DDPM 1 000.
  - La diffusion latente corrige le VAE avec le même décodeur : KID de 44,0 à
    27,8, précision de 76 % à 87 %. Elle est 14 fois plus rapide que la
    diffusion sur pixels à 50 pas et coûte 2,7 fois moins d'entraînement
    (391 s contre 1 063 s). Mais elle hérite du flou du décodeur : elle ne
    rattrape pas les pixels.
- **Les deux KID se contredisent sur le GAN**, meilleur en Fashion-KID et
  moyen en KID Inception. Inception voit son grain, notre classifieur ne voit
  que des vêtements bien formés. Une seule mesure aurait donc suffi à fausser
  le classement : nous gardons les deux.

## 5. Analyse de cinq échecs

Pour chaque échec : ce qu'on voit, la mesure qui le montre, la cause probable et
une piste de correction.

1. **VAE : sandales et chemises hors classe** (`echecs_vae.png`).
   - Mesure : précision de 39 % sur les sandales et 40 % sur les chemises ;
     une classe sur huit n'est pas couverte (87 %).
   - Cause : le prior N(0, I) tombe dans des zones peu peuplées de l'espace
     latent, et la perte MSE moyenne les détails fins comme les lanières.
     L'espace latent (`vae_espace_latent.png`) montre que les classes s'y
     superposent : le VAE conditionnel code le style, pas le vêtement.
   - Correction : le niveau 3, qui monte les sandales à 61 %.
2. **Diffusion sans guidance : la classe est ignorée.**
   - Mesure : à guidance 0, précision de 10 % (le hasard) ; seules 70 % des
     classes sont produites.
   - Cause : sans guidance, le modèle génère des vêtements plausibles mais
     sans lien avec la classe demandée.
   - Correction : c'est la raison d'être de la guidance sans classifieur.
3. **Guidance trop forte : des caricatures** (`guidance_ddpm.png`).
   - Mesure : à guidance 5, la précision atteint 98 %, au-dessus des vraies
     images (89 %), mais la couverture des caractéristiques tombe à 16 % et le
     Fashion-KID est multiplié par 7.
   - Cause : la précision récompense les vêtements les plus typiques. C'est la
     raison pour laquelle aucune mesure seule ne choisit la guidance.
4. **Diffusion latente : la guidance déplace l'erreur au lieu de la corriger.**
   - Mesure : de guidance 1,5 à 5, les sandales passent de 61 % à 79 %, mais
     les chemises tombent de 49 % à 33 %.
   - Cause : dans l'espace du VAE, chemises, T-shirts et pulls sont voisins ;
     pousser vers la classe exagère les traits les plus distinctifs, et une
     chemise devient un pull.
   - Correction : un autoencodeur entraîné avec une perte de classification
     pour séparer les classes.
5. **DDIM 10 pas sur pixels : qualité perdue.**
   - Mesure : le KID double (10,8 → 20,8), les chemises tombent à 42 % et une
     classe passe sous le seuil de couverture (93 %).
   - Cause : chaque pas doit retirer une grande part du bruit d'un coup, et
     l'image estimée x̂_0 se fige trop tôt (`diffusion_etapes_ddpm.png`).
   - Correction : rester à 50 pas, ou utiliser un échantillonneur d'ordre
     supérieur.

## 6. Limites et coût

- **Plafond de la mesure de précision.** Notre classifieur ne reconnaît que
  57 % des vraies chemises. Les scores « chemise » de tous les modèles sont
  donc plafonnés par la mesure, pas seulement par les générateurs.
- **Un seul entraînement par modèle.** Les écarts-types reflètent les graines
  d'échantillonnage, pas la variance d'un entraînement à l'autre.
- **Données et mesures simples.** Le format 28 × 28 en niveaux de gris est
  simple. Le Fashion-KID et la couverture sont des mesures propres au projet.
- **Test humain** sur 5 personnes seulement, avec un intervalle large.
- **Coût.** Le meilleur compromis dépend de l'usage :
  - pour générer en masse, le GAN : 0,02 ms par image, très bonne précision ;
  - pour la meilleure qualité mesurée, la diffusion sur pixels à 50 pas ;
  - la diffusion latente est le compromis entre les deux.
- **Piste.** Appliquer le même pipeline à des vêtements en couleur : le niveau
  3 s'y adapte en changeant seulement l'autoencodeur.
