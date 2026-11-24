# Implémentations propres du projet

## 4. Implémentations propres du projet

### 4.1 `bilinear`

**Fichier** : `backends/bilinear/channelwise.py`

Interpolation bilinéaire canal par canal : noyau 5 points pour le vert,
9 points pour rouge/bleu, normalisés par le nombre d'échantillons
réellement sommés au bord de l'image.

### 4.2 `green_edge_based`

**Fichier** : `backends/green_edge_based/directional.py`

Le vert (canal le mieux échantillonné) est interpolé en choisissant,
pixel à pixel, entre un noyau horizontal/vertical/isotrope selon la
direction de plus faible contraste (gradient du vert brut). Rouge et
bleu sont interpolés sur leur différence au vert.

### 4.3 `malvar_bilateral`

**Fichier** : `backends/malvar_bilateral/bilateral.py`

Vert estimé par le noyau de Malvar-He-Cutler (identique à `malvar`),
puis rouge/bleu interpolés par un filtre bilatéral guidé par ce vert
(pondération spatiale + proximité radiométrique).

**Limites** : boucle Python par pixel manquant — coûteux sur de grandes
images. Motif RGGB uniquement.

### 4.4 `edge_aware` / `edge_aware_simplified`

**Fichier** : `backends/edge_aware/full.py`, `backends/edge_aware/simplified.py`

`edge_aware` : pour chaque pixel manquant, ne retient que les voisins
alignés avec la direction perpendiculaire au gradient local du vert
(le long d'un éventuel contour), pondérés par distance spatiale et
proximité radiométrique.

`edge_aware_simplified` : version vectorisable, fenêtre fixe 3×3,
pondération uniquement radiométrique (pas de terme spatial ni de
sélection directionnelle) — beaucoup plus rapide, qualité proche sur
images peu texturées.

**Limites** : `edge_aware` boucle en Python pur par pixel manquant (très
lent sur de grandes images) ; les deux sont RGGB uniquement.

### 4.5 `ahd`

**Fichier** : `backends/ahd/homogeneity.py`

Inspiré du principe général de Hirakawa & Parks (2005) — pas un portage
de l'algorithme original (qui compare deux directions candidates dans
un espace de couleur perceptuel). Le vert est interpolé
directionnellement, pondéré par le gradient local ; rouge/bleu sur leur
chrominance au vert.

**Limites** : approximation, pas l'algorithme AHD publié. Motif RGGB
uniquement.

### 4.6 `mrf`

**Fichier** : `backends/mrf/diffusion.py`

Inspiré des approches par champ de Markov — approxime l'énergie
(lissage de la chrominance sous contrainte edge-aware) par diffusion
itérative par points fixes, pas une véritable inférence MRF (belief
propagation, non praticable en Python pur).

**Limites** : approximation, 20 itérations par défaut (compromis
vitesse/qualité). Motif RGGB uniquement.
