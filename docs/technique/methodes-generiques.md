# Conventions et méthodes génériques

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
