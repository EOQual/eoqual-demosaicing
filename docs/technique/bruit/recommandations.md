# Démosaïçage et bruit : recommandations et mode opératoire

## 11.9 Recommandations

Dans tous les cas : **démosaïquer, puis stabiliser avec ``(a, b)``
mesurés sur la mosaïque ou le capteur, puis débruiter avec
σ = √κ_Y + 0.2** (κ_Y du [§11.3](modele.md#113-ce-que-le-démosaïçage-fait-du-bruit)), puis inverser la stabilisation.

| Contrainte | Chaîne (C = √κ_Y + 0.2) | Gain (100 / 400 / 1600 e⁻) | ΔE | Temps démosaïçage 4 Mpx |
|---|---|---|---|---|
| remplacement direct de HA | **`gbtf` → GAT → NLB, C = 1.75** | +1.2 / +1.1 / +1.6 dB | −14 à −19 % | 0.4 s |
| qualité maximale sans réseau, bruit moyen à faible | `ari` → GAT → NLB, C = 1.0 | +0.5 / +1.2 / +2.1 dB | −5 à −18 % | 66 s |
| vitesse maximale | `malvar` → GAT → NLB, C = 1.6 | +0.8 / +1.2 / +1.7 dB | −12 à −14 % | 0.14 s |
| HA imposé | HA → GAT → NLB, C = 1.36 (+ médian chroma 3×3) | +0.6 / +0.1 / −0.1 dB | −3 à −7 % | 0.5 s |
| PyTorch disponible | `ari` → DRUNet 1.5σ₀ | +1.6 / +1.6 / +2.2 dB ([§11.6](etudes.md#116-chaînes-démosaïçage--débruitage)) | −20 à −24 % | 66 s |

Gains par rapport à HA → GAT → NLB avec σ = 1 (le σ du capteur sans
majoration), même bruit et mêmes zones ([§11.7](etudes.md#117-comparaison-des-démosaïqueurs--qualité-et-temps-de-calcul)) ; dernière ligne : par
rapport à HA → NLB σ₀ ([§11.6](etudes.md#116-chaînes-démosaïçage--débruitage)).

**Répartition entre bibliothèques.** Le démosaïçage (eoqual-demosaicing)
et le débruitage (eoqual-denoising) restent séparés : une image n'est
pas toujours démosaïquée, et un débruiteur sert à bien d'autres images.
Ce qui relie les deux est une **information** — κ_Y du démosaïqueur
utilisé, et ``(a, b)`` du capteur — plutôt qu'un traitement. Deux options,
non tranchées :

- **enchaîner** les deux appels en documentant la recette (ce chapitre,
  et eoqual-denoising [§6](../methodes-publiees.md#6-famille-ipol--ha-ari-ri-gbtf-mlri-wmlri)) : aucune dépendance entre bibliothèques ;
- **exposer κ_Y** dans le registre d'eoqual-demosaicing (champ de
  `METHODS_CONFIGS`, mesuré une fois pour toutes), et laisser à
  eoqual-denoising la stabilisation et le facteur C (paramètre
  ``noise_gain`` par exemple) : chaque bibliothèque garde son rôle, la
  chaîne s'écrit en trois lignes.

## 11.10 Limites

- L'orthophoto est un produit déjà traité (démosaïqué, rehaussé,
  compressé) : « sans bruit » est une hypothèse, et sa linéarisation par
  le gamma sRGB une approximation. Les planchers absolus des
  démosaïqueurs dépendent de ce traitement préalable ; le classement est
  plus robuste que les valeurs.
- Un seul capteur simulé (RGGB, pixels indépendants, sans diaphonie ni
  non-uniformité de réponse), quatre zones d'une même scène.
- κ est mesuré sur un champ uniforme : un minorant pour les méthodes
  adaptatives sur scène texturée (d'où le terme +0.2, et l'écart de la
  famille RI).
- ΔE2000 calculé en supposant un sRGB D65 ; CPSNR et ΔE mesurent des
  erreurs moyennes, pas l'aspect des artefacts locaux, d'où les vignettes.
- Paramètres du Plug-and-Play réglés sur une zone (tuiles, 400 e⁻).


## 11.11 Mode opératoire sur données réelles : Hamilton-Adams et GBTF

Pour vérifier sur ses propres images ce que prévoit l'étude, et choisir
entre HA et GBTF. Exemple exécutable : `examples/real_data_protocol.py` ;
tracés : notebook `examples/demosaicing_noise.ipynb` (Plotly).

**Valeurs à tester** (σ = C dans le domaine stabilisé) :

| Démosaïqueur | κ_Y | C de la règle | Valeurs balayées par défaut |
|---|---|---|---|
| `ha` | 1.35 | **1.36** | 1.0 (σ du capteur), 1.16 (√κ_Y), 1.36, 1.5 |
| `gbtf` | 2.41 | **1.75** | 1.25, 1.55 (√κ_Y), 1.75, 2.0 |

**Étapes.**

0. **Unités.** Tout se fait en luminance linéaire normalisée, avant
   balance des blancs et gamma : ``L = (DN − offset) / S``,
   ``S = DN_max − offset``. Un modèle de bruit donné en DN,
   ``σ²(DN) = a_DN + b_DN · (DN − offset)`` (``a_DN`` et ``b_DN = 1/K``
   de la PTC d'eoqual-measure), se convertit en ``a = a_DN / S²`` et
   ``b = b_DN / S``. ``(a, b)`` doivent avoir été
   estimés **sur la mosaïque brute** (ou par la PTC) — sur une image
   démosaïquée, ils seraient faussés d'un facteur κ — et valoir pour les
   quatre sous-plans (avant balance des blancs).
1. **Démosaïquer** (`ha`, `gbtf`) avec le motif du capteur.
2. **Stabiliser** : ``generalized_anscombe(rgb, gain=1/b, read_noise_dn=√a)``.
3. **Débruiter** : `nl_bayes_denoise(g, sigma=C, data_range=(g.min(), g.max()))`,
   mode couleur `"yuv"` (défaut).
4. **Inverser** : ``inverse_generalized_anscombe(d, 1/b, √a)``.
5. Balance des blancs, matrice couleur, gamma : ensuite seulement.

```bash
python examples/real_data_protocol.py --raw mosaique.npy --offset 64 --dn-max 4095 \
    --a 406.2 --b 10.08 --pattern RGGB --crop 1000 1000 1024 --out protocole/
python examples/real_data_protocol.py --simulate capitole_2048_cog.tif --out protocole/   # sans données
```

Sorties : `protocol_results.json` (critères par démosaïqueur et par C),
`protocol_vignettes.png` (une ligne par démosaïqueur, une colonne par C ;
zone texturée puis zone uniforme, aperçu avec balance des blancs « gris
moyen » et gamma sRGB). Sur 1024×1024 : environ 7 s par valeur de C
(NL-Bayes ; 1.7 s sur 512×512), quelques dixièmes de seconde pour le
démosaïçage.

**Critères sans vérité terrain.**

- **bruit de chrominance résiduel** — les défauts visés : écart-type de U
  et V (base orthonormée du [§11.3](modele.md#113-ce-que-le-démosaïçage-fait-du-bruit)) dans des zones uniformes, après
  retrait du plan moyen de chaque zone (un dégradé d'éclairement n'est pas
  du bruit ; `eoqual.core.detrend.fit_plane_residual`) ;
- **bruit de luminance résiduel** : même mesure sur Y ;
- **détails conservés** : énergie du gradient de Y dans des zones
  texturées, rapportée à celle de l'image démosaïquée non débruitée (1 =
  rien perdu ; elle baisse aussi parce que le bruit disparaît) ;
- **vignettes** à l'échelle 1:1, mêmes zones pour toutes les valeurs.

Les zones sont choisies automatiquement (blocs 32×32 de gradient minimal
et maximal, sans chevauchement, hors saturation ; un avertissement
signale une image sans vraie zone uniforme) ou imposées par
`--roi-flat` / `--roi-texture` (coordonnées dans l'image recadrée).

**Choisir C.** Prendre la plus petite valeur au-delà de laquelle σ_U et
σ_V ne baissent presque plus, en vérifiant que les détails ne chutent pas
nettement par rapport à la valeur précédente. Comparer HA et GBTF chacun
à **son** C (à C égal, GBTF paraît plus bruité : il concentre davantage de
bruit en luminance, κ_Y = 2.41 contre 1.35).

**Résultat attendu** (mode `--simulate`, zone « tuiles », 400 e⁻ à pleine
échelle, 12 bits) :

| Chaîne | σ_U | σ_V | σ_Y | détails | CPSNR | ΔE2000 |
|---|---|---|---|---|---|---|
| HA, sans débruitage | 0.0162 | 0.0184 | 0.0508 | 1.00 | 30.86 | 4.21 |
| HA, C = 1.0 | 0.0034 | 0.0046 | 0.0342 | 0.92 | 33.25 | 2.57 |
| HA, C = 1.36 (règle) | 0.0030 | 0.0041 | 0.0332 | 0.89 | 33.32 | 2.47 |
| GBTF, C = 1.25 | 0.0026 | 0.0037 | 0.0362 | 0.89 | 34.11 | 2.18 |
| GBTF, C = 1.75 (règle) | 0.0023 | 0.0031 | 0.0326 | 0.85 | 34.06 | 2.23 |

HA : la majoration réduit la chrominance résiduelle de 11 à 12 % et
l'erreur de couleur de 4 %. GBTF, à son C, a 23 % de bruit de chrominance en
moins que HA à son C, et 0.7 à 0.9 dB de plus. Sur cette zone, l'optimum
de GBTF est un peu sous la règle (C ≈ 1.55 : 34.20 dB) : la règle est un
point de départ, le balayage le confirme ou l'ajuste. Si les mesures
réelles s'écartent nettement de ce comportement, vérifier d'abord les
unités de ``(a, b)``.

**Références** : Q. Jin, G. Facciolo, J.-M. Morel (2020), CVPR Workshops,
arXiv 2004.11577 ; M. Mäkitalo, A. Foi (2013), « Optimal inversion of the
generalized Anscombe transformation for Poisson-Gaussian noise », IEEE
TIP 22(1) ; D. L. Donoho, I. M. Johnstone (1994), Biometrika 81(3)
(estimation de σ) ; J. R. Janesick (2007), *Photon Transfer*, SPIE Press
(modèle de bruit et PTC) ; I. Pekkucuksen, Y. Altunbasak (2010),
« Gradient based threshold free color filter array interpolation »,
ICIP (GBTF) ; W. T. Freeman (1988), brevet US 4,774,565 (médian sur les
différences de couleur) ; F. Heide et al. (2014), « FlexISP », ACM TOG
33(6) ; M. Gharbi, G. Chaurasia, S. Paris, F. Durand (2016), « Deep joint
demosaicking and denoising », ACM TOG 35(6) ; M. Lebrun, A. Buades,
J.-M. Morel (2013), SIAM J. Imaging Sci. 6(3) (NL-Bayes) ; K. Zhang et
al. (2021), IEEE TPAMI 44(10) (DRUNet).
