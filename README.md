<p align="center">
  <img src="resources/eoqual_rounded.png" alt="EOQual" width="280">
</p>

# eoqual-demosaicing

Collection de méthodes de démosaïçage d'images issues d'un capteur à
matrice de filtres colorés (motif de Bayer) — voir
**[REFERENCE_TECHNIQUE.md](REFERENCE_TECHNIQUE.md)** pour le détail de
chaque méthode (principe, implémentation, limites) et
**[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)** pour l'audit
complet des licences du code adapté.

## Sommaire

1. [Installation](#installation)
2. [Démarrage rapide](#démarrage-rapide)
3. [Catalogue des méthodes](#catalogue-des-méthodes)
4. [Licence](#licence)
5. [Contribuer](#contribuer)
6. [Credits](#credits)

---

## Installation

```bash
pip install git+https://github.com/EOQual/eoqual-demosaicing.git
```

Installe tout ce qui est nécessaire pour les 26 méthodes du catalogue —
numpy, scipy, opencv-python, colour-demosaicing, rich (voir
`pyproject.toml`). Aucun extra optionnel : toutes les méthodes sont sous
licence permissive et sans dépendance lourde (voir
`THIRD_PARTY_LICENSES.md`).

Pour lister, à tout moment, les méthodes disponibles et leur licence
amont :

```python
from eoqual.demosaicing.runner import list_methods
list_methods()  # tableau Rich : méthode, motifs Bayer, licence, référence
```

---

## Démarrage rapide

```python
import numpy as np
import eoqual.demosaicing as d

# Image Bayer (mosaïque brute), motif RGGB
bayer = ...  # np.ndarray, 2D

rgb = d.malvar_demosaic(bayer, pattern="RGGB")
rgb = d.ari_demosaic(bayer, pattern="RGGB")
rgb = d.ha_demosaic(bayer, pattern="RGGB")

# Ou par le dispatcher générique
from eoqual.demosaicing.runner import demosaic
rgb = demosaic(bayer, method="malvar")
rgb = demosaic(bayer, method="mlri")  # ri_demosaic(..., algorithm="MLRI")
```

Voir `examples/all_methods_overview.py` pour une comparaison des 26
méthodes sur une image réelle, et `examples/one_method_cli.py` pour un
usage en ligne de commande.

---

## Catalogue des méthodes

Toutes les méthodes attendent une image Bayer 2D (mosaïque brute) en
entrée et renvoient une image RGB `(H, W, 3)`. Les méthodes marquées
« RGGB » uniquement n'exposent pas de paramètre `pattern`.

| Méthode | Motifs Bayer | Principe |
|---|---|---|
| `interp_bilinear` / `interp_bicubic` / `interp_spline` | RGGB/BGGR/GRBG/GBRG | Interpolation générique canal par canal (noyau moyenneur / gaussien / spline) |
| `opencv_bilinear` / `opencv_vng` / `opencv_ea` | RGGB/BGGR/GRBG/GBRG | Méthodes natives OpenCV (bilinéaire, VNG, Edge-Aware) |
| `colour_bilinear` / `colour_malvar2004` / `colour_menon2007` | RGGB/BGGR/GRBG/GBRG | Méthodes du paquet `colour-demosaicing` |
| `bilinear` | RGGB/BGGR/GRBG/GBRG | Interpolation bilinéaire canal par canal (implémentation propre) |
| `green_edge_based` | RGGB/BGGR/GRBG/GBRG | Interpolation basée sur les contours du canal vert |
| `malvar` | RGGB/BGGR | Malvar, He & Cutler (2004) — noyaux corrigés du gradient |
| `malvar_bilateral` | RGGB | Malvar (vert) + filtre bilatéral guidé (rouge/bleu) |
| `edge_aware` / `edge_aware_simplified` | RGGB | Interpolation edge-aware (complète / vectorisée simplifiée) |
| `ahd` | RGGB | Interpolation edge-aware guidée par l'homogénéité locale, inspirée d'AHD |
| `mrf` | RGGB | Diffusion edge-aware de la chrominance, inspirée des champs de Markov |
| `ha` | RGGB/BGGR/GRBG/GBRG | Hamilton & Adams (1997) — interpolation directionnelle |
| `ari` | RGGB/BGGR/GRBG/GBRG | Monno et al. (2016) — Adaptive Residual Interpolation |
| `ri` / `gbtf` / `mlri` / `wmlri` | RGGB/BGGR/GRBG/GBRG | Residual Interpolation et variantes (Monno, Pekkucuksen & Altunbasak, Kiku et al.) |
| `dlmmse` | RGGB/BGGR/GRBG/GBRG | Zhang & Wu (2005) — LMMSE directionnelle |
| `cdm` | RGGB/BGGR/GRBG/GBRG | Zhang, Wu, Buades & Li (2011) — interpolation directionnelle + seuillage non local |
| `lslcd` | RGGB/BGGR/GRBG/GBRG | Dubois — démultiplexage luma-chroma (fréquentiel) |

Détail de chaque méthode (principe, implémentation, limites) :
**[REFERENCE_TECHNIQUE.md](REFERENCE_TECHNIQUE.md)**.

**D'autres méthodes ne sont pas encore portées** (roadmap — code de
référence trouvé mais pas encore intégré, ou simple citation
bibliographique sans code disponible) : voir
**[THIRD_PARTY_LICENSES.md §3](THIRD_PARTY_LICENSES.md#3-méthodes-non-implémentées-roadmap)**.

---

## Licence

**Le code propre à eoqual-demosaicing est sous licence MIT** (`license.txt`).

Toutes les méthodes cataloguées sont MIT, y compris le code adapté de
`prysm` (`malvar`), de l'article IPOL de Jin, Guo, Facciolo & Morel
(`ha`, `ari`, `ri` et variantes — licence BSD 2-Clause modifiée de leur
code source), de l'algorithme de Getreuer/Zhang-Wu (`dlmmse` —
réimplémenté depuis une source BSD), et des réimplémentations propres
depuis les articles publiés pour `cdm` et `lslcd`. Inventaire complet,
fichier par fichier, avec vérification de chaque source :
**[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)**.

---

## Contribuer

Bug, méthode manquante, algorithme à ajouter, erreur de licence
constatée : les contributions sont bienvenues — voir
**[CONTRIBUTING.md](CONTRIBUTING.md)** pour comment signaler un problème,
proposer une méthode, ou soumettre une implémentation.

---

## Credits

- prysm [https://github.com/brandondube/prysm] (Brandon Dube — `malvar`)
- Résidual demosaicking (IPOL) [https://www.ipol.im/pub/art/2021/358/] (Jin, Guo, Facciolo, Morel — `ha`, `ari`, `ri` et variantes)
- Zhang-Wu Directional LMMSE Image Demosaicking (IPOL) [https://www.ipol.im/pub/art/2011/g_zwld/] (Pascal Getreuer — `dlmmse`)
- Zhang, Wu, Buades & Li (2011), J. Electronic Imaging 20(2), 023016 (LDI-NAT — `cdm`)
- Dubois (2005), IEEE SPL 12(12), 847-850 ; Jeon & Dubois (2013), IEEE TIP 22(1), 146-156 (luma-chroma demultiplexing — `lslcd`)
- colour-demosaicing [https://github.com/colour-science/colour-demosaicing]
- NumPy [https://numpy.org/]
- SciPy [https://scipy.org/]
- OpenCV [https://opencv.org/]

Jeu de données de test (`tests/images/`) : Kodak Lossless True Color
Image Suite (R. Franzen) [http://r0k.us/graphics/kodak/].
