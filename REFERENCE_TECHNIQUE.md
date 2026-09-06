# Référence technique — eoqual-demosaicing

Détail de chaque méthode : principe, implémentation, limites connues.
Pour le catalogue résumé et l'installation, voir `README.md` ; pour
l'audit des licences, `THIRD_PARTY_LICENSES.md`.

## 1. Conventions

- Toutes les fonctions publiques ont la signature
  `xxx_demosaic(image: np.ndarray, ...) -> np.ndarray`.
- `image` est une image Bayer 2D (mosaïque brute, un seul canal). La
  sortie est une image RGB `(H, W, 3)`.
- Le motif de Bayer supporté (`pattern`) varie selon les méthodes — voir
  ci-dessous et le tableau de `README.md`. Les méthodes qui ne
  paramètrent pas `pattern` sont câblées en dur pour `RGGB` uniquement.

## 2. Interpolation générique

### 2.1 `interp_bilinear` / `interp_bicubic` / `interp_spline`

**Fichier** : `backends/interpolation/generic.py` (`interp_demosaic`)

Chaque canal R, G, B est extrait de la mosaïque puis interpolé
indépendamment par un filtre générique : noyau moyenneur
(`'bilinear'`), noyau gaussien séparable (`'bicubic'`), ou interpolation
par spline cubique (`'spline'`) — avec normalisation par le nombre
d'échantillons valides pour ignorer les pixels manquants dans la
convolution.

**Limites** : ignore la structure du motif de Bayer au-delà de
l'extraction par canal — pas de guidage inter-canal (contrairement à
`malvar` ou `ha`), donc plus sensible au moiré sur les hautes
fréquences.

## 3. OpenCV et colour-demosaicing

### 3.1 `opencv_bilinear` / `opencv_vng` / `opencv_ea`

**Fichier** : `backends/opencv/wrapper.py` (`opencv_demosaic`)

Enrobage direct de `cv2.cvtColor` avec les codes
`COLOR_BAYER_*2RGB[_VNG|_EA]` d'OpenCV : interpolation bilinéaire,
Variable Number of Gradients (VNG) et Edge-Aware (EA).

**Limites** : VNG requiert une image `uint8` ; les autres méthodes
`uint8`/`uint16` — conversion automatique avec avertissement sinon (voir
le code, non testé sur tous les cas de dépassement).

### 3.2 `colour_bilinear` / `colour_malvar2004` / `colour_menon2007`

**Fichier** : `backends/colour_science/wrapper.py` (`colour_science_demosaic`)

Enrobage du paquet [colour-demosaicing](https://github.com/colour-science/colour-demosaicing) :
bilinéaire, Malvar-He-Cutler (2004), et DDFAPD de Menon, Andriani &
Calvagno (2007) — filtrage directionnel avec décision a posteriori,
généralement la méthode la plus fidèle du socle sur des images
naturelles peu bruitées.

## 4. Implémentations propres du projet

### 4.1 `bilinear`

**Fichier** : `backends/bilinear/amram.py`

Interpolation bilinéaire canal par canal : noyau 5 points pour le vert,
9 points pour rouge/bleu, normalisés par le nombre d'échantillons
réellement sommés au bord de l'image.

### 4.2 `green_edge_based`

**Fichier** : `backends/green_edge_based/amram.py`

Le vert (canal le mieux échantillonné) est interpolé en choisissant,
pixel à pixel, entre un noyau horizontal/vertical/isotrope selon la
direction de plus faible contraste (gradient du vert brut). Rouge et
bleu sont interpolés sur leur différence au vert.

### 4.3 `malvar_bilateral`

**Fichier** : `backends/malvar_bilateral/amram.py`

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

**Fichier** : `backends/ahd/amram.py`

Inspiré du principe général de Hirakawa & Parks (2005) — pas un portage
de l'algorithme original (qui compare deux directions candidates dans
un espace de couleur perceptuel). Le vert est interpolé
directionnellement, pondéré par le gradient local ; rouge/bleu sur leur
chrominance au vert.

**Limites** : approximation, pas l'algorithme AHD publié. Motif RGGB
uniquement.

### 4.6 `mrf`

**Fichier** : `backends/mrf/amram.py`

Inspiré des approches par champ de Markov — approxime l'énergie
(lissage de la chrominance sous contrainte edge-aware) par diffusion
itérative par points fixes, pas une véritable inférence MRF (belief
propagation, non praticable en Python pur).

**Limites** : approximation, 20 itérations par défaut (compromis
vitesse/qualité). Motif RGGB uniquement.

## 5. `malvar` — Malvar, He & Cutler

**Fichier** : `backends/malvar/he_cutler.py`
**Référence** : Malvar, H. S., He, L.-W., Cutler, R. (2004). "High-quality
linear interpolation for demosaicing of Bayer-patterned color images."
IEEE ICASSP.

Interpolation par noyaux 5×5 corrigés du gradient (figure 2 de
l'article) : un noyau pour le vert aux positions R/B, trois noyaux pour
rouge/bleu selon leur position relative (même ligne, même colonne,
diagonale). Adapté de `prysm.bayer` (licence MIT, voir
`THIRD_PARTY_LICENSES.md`).

**Limites** : motifs `RGGB`/`BGGR` uniquement.

## 6. Famille IPOL — `ha`, `ari`, `ri`, `gbtf`, `mlri`, `wmlri`

**Fichiers** : `backends/ha/ha.py`, `backends/ari/ari.py`, `backends/ri/ri.py`
**Référence principale** : Jin, Q., Guo, Y., Facciolo, G., Morel, J.-M.
(2021). "A Mathematical Analysis and Implementation of Residual
Interpolation Demosaicking Algorithms." Image Processing On Line.

### 6.1 `ha` — Hamilton & Adams (1997)

Interpolation directionnelle du vert (choix entre gradient horizontal
et vertical selon le contraste local), puis rouge/bleu par un schéma
similaire guidé par le vert. Référence historique : brevet US 5,629,734.

### 6.2 `ari` — Adaptive Residual Interpolation

Monno et al. (2016). Le vert est interpolé par un filtre guidé (guided
filter) adaptatif, puis rouge/bleu par résidu (différence au vert
lissée) en deux passes (diagonale, puis horizontale/verticale).

### 6.3 `ri` / `gbtf` / `mlri` / `wmlri` — Residual Interpolation et variantes

Un seul point d'entrée (`ri_demosaic`, paramètre `algorithm`) couvre
quatre articles :

- `gbtf` : Pekkucuksen & Altunbasak (2010), "Gradient Based Threshold
  Free Color Filter Array Interpolation," IEEE ICIP.
- `ri` : Monno et al. (2016), résiduel guidé simple.
- `mlri` / `wmlri` : Kiku et al. (2014/2016), résiduel minimisant le
  Laplacien (et sa variante pondérée).

**Limites communes à la famille IPOL** : `sigma` (régularisation du
filtre guidé) est ignoré par `gbtf` mais doit rester passé (valeur par
défaut `2.0`) ; pas de gestion native du bruit au-delà du filtre guidé
lui-même — pas de débruitage séparé.

## 7. `dlmmse` — LMMSE directionnelle (Zhang & Wu)

**Fichier** : `backends/dlmmse/zhang_wu.py`
**Référence** : Zhang, L., Wu, X. (2005). "Color demosaicking via
directional linear minimum mean square-error estimation." IEEE
Transactions on Image Processing, 14(12), 2167-2178.

Le vert est interpolé horizontalement et verticalement par un filtre
directionnel court (5 points), puis les deux estimations sont
fusionnées par pondération LMMSE : chaque direction est pondérée par la
variance de l'AUTRE direction (plus une direction est incertaine/
bruitée, moins elle pèse dans la fusion), estimée localement sur une
fenêtre de 9 pixels. Rouge et bleu sont ensuite obtenus par propagation
de leur différence au vert : diagonale entre positions rouge et bleu
(toujours diagonalement adjacentes dans un motif de Bayer), puis
axiale vers les positions vertes.

Réimplémentation vectorisée NumPy depuis les équations du code de
référence BSD de Pascal Getreuer (IPOL) — voir `backends/dlmmse/NOTICE.md`
pour le détail des deux points où cette réimplémentation diffère
volontairement du code source d'origine (gestion des bords, constante
de stabilisation numérique — équivalence vérifiée par comparaison
numérique, pas seulement par argument).

**Limites** : la variante « papier » de l'estimation de la moyenne
locale est implémentée (pas la variante alternative du code MATLAB de
référence de Zhang, très proche numériquement).

## 8. `cdm` — LDI-NAT (Zhang, Wu, Buades & Li)

**Fichier** : `backends/cdm/nat_cdm.py`
**Référence** : Zhang, L., Wu, X., Buades, A., Li, X. (2011). "Color
demosaicking by local directional interpolation and nonlocal adaptive
thresholding." Journal of Electronic Imaging, 20(2), 023016.

Pipeline en quatre étapes (Fig. 1 de l'article) :

1. **LDI** (Local Directional Interpolation) du vert : à chaque position
   rouge/bleue, la différence vert-rouge (ou vert-bleu) est estimée
   dans 4 directions sur une fenêtre compacte, fusionnées par pondération
   inversement proportionnelle au gradient directionnel.
2. **NAT** (Nonlocal Adaptive Thresholding) du vert : chaque position
   interpolée est raffinée par recherche des patchs les plus similaires
   (fenêtre 31×31, patchs 5×5, 100 candidats), projection sur leur propre
   base ACP (SVD locale) et seuillage doux — une alternative structurelle
   à la simple moyenne non locale (NLM), qui préserve mieux les contours.
3. **LDI diagonal** de rouge/bleu à l'aide du vert reconstruit (rouge et
   bleu sont toujours diagonalement adjacents dans un motif de Bayer),
   puis complétion axiale aux positions vertes.
4. **NAT** de rouge et bleu.

Réimplémentation depuis les équations et paramètres publiés (§2, §3.2)
— aucun code tiers repris. La complétion axiale (étape 3) distingue les
deux sous-types de pixel vert du motif de Bayer (axe « couleur cible
native » : lignes ou colonnes selon le sous-type) en appliquant la même
formule à l'image transposée pour l'un des deux — validé par comparaison
numérique sur les 4 motifs de Bayer (RMSE cohérente).

**Limites** : recherche de patchs + SVD par pixel manquant — nettement
plus coûteux que les autres méthodes du catalogue (non chronométré pour
être adapté aux grandes images sans optimisation). Le seuil NAT
(`t = 0.03 × g_Y`) utilise une formule de « gradient moyen des patchs »
non explicitée en détail dans l'article ; interprétée ici comme la
moyenne des différences successives absolues des patchs retenus.

## 9. `lslcd` — Démultiplexage luma-chroma (Dubois)

**Fichier** : `backends/lslcd/dubois.py`
**Référence** : Dubois, E. (2005). "Frequency-domain methods for
demosaicking of Bayer-sampled color images." IEEE Signal Processing
Letters, 12(12), 847-850 ; architecture non bruitée reprise de Jeon &
Dubois (2013), §III (voir §11 pour l'extension bruitée, non
implémentée).

Le signal Bayer est modélisé en fréquence comme une luminance en bande
de base plus deux chrominances modulées (une à la fréquence diagonale de
Nyquist, une aux deux fréquences axiales en opposition de phase). Le
démultiplexage démodule puis filtre passe-bas chaque porteuse ; les deux
estimées de chrominance axiale sont combinées par pondération inverse à
leur énergie locale ; la luminance s'obtient en retranchant les chromas
remodulées au signal brut. R, G, B sont enfin reconstruits par inversion
de la matrice luma-chrominance (`G = L+C1`, `R = (L-C1)+2C2`,
`B = (L-C1)-2C2`).

Réimplémentation depuis les équations publiées — aucun code tiers
repris. **Différences assumées avec l'article** : les filtres passe-bas
sont conçus ici par moindres carrés séparables (`scipy.signal.firls`,
11 points), pas jointement optimisés sur un jeu d'entraînement comme
dans l'article ; la combinaison adaptative des deux chromas axiales est
une heuristique par énergie locale, non spécifiée en détail dans
l'article. Les trois porteuses de modulation sont dérivées directement
des masques du motif de Bayer (pas d'une parité fixe supposée), pour
rester correctes quel que soit le motif — validé par comparaison
numérique sur les 4 motifs (RMSE identique à 0.0003 près).

**Limites** : qualité en retrait par rapport aux filtres publiés
(entraînés) de l'article — voir §11 pour l'extension bruitée non
implémentée.

## 10. Méthodes non implémentées

Voir `THIRD_PARTY_LICENSES.md` §3 : `LSLCD-NE` (extension bruitée de
`lslcd`, §9 — banc de filtres entraîné pour 11 niveaux de bruit non
reproduit) ; méthodes bloquées par une licence copyleft ou des droits
restreints (`RCD` : GPL-3.0 ; `SSD` : LGPLv3 + mention de brevets par
ses propres auteurs) ; ou simple citation bibliographique sans code
disponible (`AFD`, `AMaZE`, `CS`, `EAHD`, `HPHD`, `JCNN`, `LSSC`, `PPG`,
`RCNN`, `RTF`, `VCD`) ; ainsi que le démosaïçage multispectral/MSFA,
hors périmètre (motif de Bayer standard uniquement dans ce dépôt).
