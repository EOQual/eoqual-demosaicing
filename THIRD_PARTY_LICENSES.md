# Licences tierces — inventaire et statut

> **Avertissement.** Cet inventaire est tenu au meilleur effort, à partir
> d'une vérification directe des dépôts/paquets sources (fichier LICENSE,
> champ `license` de l'API GitHub, classifieur PyPI). Si une erreur y est
> identifiée (source mal vérifiée, statut incorrect, nouvelle source
> vendored non recensée), elle sera corrigée dès que signalée — et si
> nécessaire, l'implémentation concernée sera retirée plutôt que sa
> licence « arrangée » a posteriori. Pour signaler un problème, voir
> `CONTRIBUTING.md`.

Ce document recense, pour chaque méthode de `src/eoqual/demosaicing/`,
sa source, la licence **réellement vérifiée** de cette source, et le
statut de compatibilité avec la licence MIT annoncée pour
`eoqual-demosaicing` (`license.txt`).

**État actuel** : toutes les méthodes cataloguées sont MIT ou reposent
sur des dépendances/sources MIT, BSD ou Apache-2.0 permissives. Aucune
méthode n'est écartée pour raison de licence à ce jour. Voir §3 pour les
méthodes non implémentées (roadmap) — absentes faute de portage, pas
pour raison de licence, sauf mention contraire explicite dans leur
entrée.

## 1. Légende

- 🔴 **Bloquant** — licence copyleft ou absente, et atteignable sans
  action explicite de l'utilisateur. *Aucune entrée dans cet état
  actuellement — voir §3 pour deux méthodes en roadmap dont la source
  disponible est copyleft (RCD, GPL-3.0) ou restreinte (SSD, LGPLv3 +
  brevets), non intégrées pour cette raison.*
- 🟢 **Compatible** — licence permissive (MIT/BSD/Apache), compatible
  avec MIT moyennant la conservation de la notice d'origine.

## 2. Inventaire

### 2.1 Sources permissives confirmées (🟢)

| Méthode | Fichier | Source | Licence vérifiée |
|---|---|---|---|
| `interp_bilinear`, `interp_bicubic`, `interp_spline` | `backends/interpolation/generic.py` | Implémentation originale (auteur de ce dépôt) | MIT |
| `opencv_bilinear`, `opencv_vng`, `opencv_ea` | `backends/opencv/wrapper.py` | Enrobage de `cv2.cvtColor` (OpenCV) | MIT (wrapper) — OpenCV **Apache-2.0** |
| `colour_bilinear`, `colour_malvar2004`, `colour_menon2007` | `backends/colour_science/wrapper.py` | Enrobage de https://github.com/colour-science/colour-demosaicing | MIT (wrapper) — vérifié (`LICENSE` du dépôt) **BSD-3-Clause** |
| `bilinear` | `backends/bilinear/amram.py` | Implémentation originale (auteur de ce dépôt) | MIT |
| `green_edge_based` | `backends/green_edge_based/amram.py` | Implémentation originale (auteur de ce dépôt) | MIT |
| `malvar_bilateral` | `backends/malvar_bilateral/amram.py` | Implémentation originale (auteur de ce dépôt) | MIT |
| `edge_aware`, `edge_aware_simplified` | `backends/edge_aware/` | Implémentation originale (auteur de ce dépôt) | MIT |
| `ahd` | `backends/ahd/amram.py` | Implémentation originale, inspirée du principe de Hirakawa & Parks (2005) — aucun code tiers repris | MIT |
| `mrf` | `backends/mrf/amram.py` | Implémentation originale, inspirée des approches par champ de Markov — aucun code tiers repris | MIT |
| `malvar` | `backends/malvar/he_cutler.py` | Adapté de https://github.com/brandondube/prysm (`prysm/bayer.py`, Brandon Dube) | **MIT** — vérifié (`LICENSE.md` du dépôt) |
| `ha`, `ari`, `ri`, `gbtf`, `mlri`, `wmlri` | `backends/ha/`, `backends/ari/`, `backends/ri/` | Adapté du code de référence IPOL (Jin, Guo, Facciolo, Morel, 2021), lui-même basé sur les gists de Facciolo/Guo/Jin pour `ha` — voir `NOTICE.md` dans chaque dossier | **BSD 2-Clause modifiée** — vérifié (`LICENSE.txt` du paquet source `residual_demosaicking`, Copyright (c) 2021 Qiyu Jin, Yu Guo, Gabriele Facciolo) |
| `dlmmse` | `backends/dlmmse/zhang_wu.py` | Réimplémentation NumPy des équations, depuis la description algorithmique du code de référence IPOL (Pascal Getreuer) — voir `NOTICE.md` | **BSD simplifiée** — vérifié (`LICENSE` du paquet source `dmzhangwu`, Copyright (c) 2010-2011 Pascal Getreuer) |
| `cdm` | `backends/cdm/nat_cdm.py` | Réimplémentation depuis les équations publiées (Zhang, Wu, Buades & Li, 2011) — aucun code tiers repris (le code MATLAB de référence des auteurs n'est distribué sans licence identifiée) | MIT (auteur de ce dépôt) |
| `lslcd` | `backends/lslcd/dubois.py` | Réimplémentation depuis les équations publiées (Dubois 2005 ; Jeon & Dubois 2013, §III) — filtres passe-bas propres conçus par moindres carrés, aucun code tiers repris | MIT (auteur de ce dépôt) |

Dépendances du socle (numpy, scipy, opencv-python, colour-demosaicing,
rich) : toutes BSD/MIT/Apache-2.0.

### 2.2 Jeux de données de test

`tests/images/lighthouse.tif` / `lighthouse_bayer.tif` : suite Kodak,
mise à disposition par Eastman Kodak Company pour un usage sans
restriction (aucun texte de licence formel — voir
`tests/images/NOTICE.md`).

## 3. Méthodes non implémentées (roadmap)

Ces méthodes ne sont **pas encore portées** dans `eoqual-demosaicing` —
absence de code, pas nécessairement de licence (voir raison indiquée
par entrée).

### 3.1 Code source disponible, non intégré

| Méthode | Référence | Raison |
|---|---|---|
| **LSLCD-NE** (LSLCD avec estimation de bruit adaptative) | Jeon, G., Dubois, E. (2013). "Demosaicking of noisy Bayer-sampled color images with least-squares luma-chroma demultiplexing and noise level estimation." IEEE TIP, 22(1), 146-156. | Extension du `lslcd` du socle (§2.1) à des images bruitées : nécessite un banc de filtres pré-entraînés pour 11 niveaux de bruit (procédure d'entraînement par moindres carrés sur un jeu de données synthétique) et un estimateur de bruit dédié (analyse d'homogénéité par blocs) — hors périmètre de ce portage, qui couvre uniquement le cas non bruité. Pas de blocage de licence : article en libre accès (page web des auteurs), simplement pas encore porté. |
| **RCD** (Ratio Corrected Demosaicing) | Sanz Rodríguez, L. — pas de publication académique à comité de lecture identifiée (algorithme documenté dans le README du dépôt `LuisSR/RCD-Demosaicing`, également repris par RawTherapee). | Code C de référence, sous licence **GPL-3.0** (copyleft fort, vérifié — `LICENSE` du dépôt) : incompatible avec une inclusion directe dans le socle MIT sans isoler `eoqual-demosaicing` sous GPL ou réécrire clean-room depuis la description algorithmique du README (discrimination directionnelle locale + filtre passe-bas + interpolation par ratio). |
| **SSD** (Self-similarity driven demosaicking) | Buades, A., Coll, B., Morel, J.-M., Sbert, C. (2009). "Self-similarity driven color demosaicking." IEEE Transactions on Image Processing, 18(6), 1192-1202. | Code C++ de référence (IPOL) : licence **LGPLv3** pour la majorité des fichiers, mais le cœur de l'algorithme (`demosaickingIpol.cpp`, `libdemosaicking.*`) est explicitement documenté par ses auteurs comme pouvant relever du brevet US 5,629,734 (Hamilton-Adams), du brevet US 4,642,678 (Cok) et du brevet EP 1,749,278 (Buades, Coll, Morel), « fournis à des fins scientifiques et éducatives uniquement ». Non intégré en l'état — nécessiterait une clarification de licence/brevet ou une réécriture depuis l'article IPOL seul. |

### 3.2 Aucun code source disponible (citation seule)

| Méthode | Référence |
|---|---|
| **AFD** (Alternating projection Frequency-Domain) | Dubois, E. (2005). "Frequency-domain methods for demosaicking of Bayer-sampled color images." IEEE Signal Processing Letters, 12(12), 847-850. Voir `lslcd` (§2.1), issu de la même équipe/architecture fréquentielle. |
| **AMaZE** (Aliasing Minimization and Zipper Elimination) | Algorithme distribué avec RawTherapee — pas de publication académique séparée identifiée. |
| **CS** (Contour stencils) | Getreuer, P. (2011). "Color demosaicing with contour stencils." 17th IEEE International Conference on Digital Signal Processing (DSP). |
| **EAHD** (Enhanced AHD, dite « Horvath's AHD ») | Variante non publiée formellement de l'AHD (Hirakawa & Parks, 2005) — voir `ahd` (§2.1) pour notre implémentation, elle-même une approximation inspirée du principe AHD, pas un portage de cette variante. |
| **HPHD** (Heterogeneity-Projection Hard-Decision) | Pas de référence bibliographique précise identifiée à ce jour. |
| **JCNN** (Joint CNN demosaicking/denoising) | Gharbi, M., Chaurasia, G., Paris, S., Durand, F. (2016). "Deep joint demosaicking and denoising." ACM Transactions on Graphics, 35(6), 191. |
| **LSSC** (Learned Simultaneous Sparse Coding) | Mairal, J., Bach, F., Ponce, J., Sapiro, G., Zisserman, A. (2009). "Non-local sparse models for image restoration." IEEE ICCV. |
| **PPG** (Patterned Pixel Grouping) | Lin, C.-K. (2004). "Pixel Grouping for Color Filter Array Demosaicing." Portland State University. |
| **RCNN** (Residual CNN demosaicking) | Tan, R., Zhang, K., Zuo, W., Zhang, L. (2017). "Color image demosaicking via deep residual learning." IEEE ICME. |
| **RTF** (Regression Tree Fields) | Khashabi, D., Nowozin, S., Jancsary, J., Fitzgibbon, A. W. (2014). "Joint Demosaicing and Denoising via Learned Nonparametric Random Fields." IEEE TIP, 23(12), 4968-4981. |
| **VCD** (Variance of Color Differences) | Chung, K.-H., Chan, Y.-H. "Color Demosaicing Using Variance of Color Differences." |

### 3.3 Hors périmètre (autre famille de capteur)

Ces méthodes concernent des capteurs à matrice de filtres
**multispectrale** (MSFA/snapshot spectral, plusieurs bandes au-delà de
R/G/B), pas le motif de Bayer standard traité par ce dépôt — extension
possible dans un projet dédié plutôt que dans `eoqual-demosaicing` :

- Tsagkatakis, G., Bloemen, M., Geelen, B., Jayapala, M., Tsakalides, P.
  (2019). "Graph and Rank Regularized Matrix Recovery for Snapshot
  Spectral Image Demosaicing." IEEE Transactions on Computational
  Imaging. (Dépend de dénébruiteurs GAP-TV/WNNM/V-BM4D aux licences
  hétérogènes, dont une en "nonprofit purposes only, tous droits
  réservés" pour V-BM4D.)
- Démosaïquage multispectral CAM25 : pas de publication ni de licence
  identifiée à ce jour.

## 4. Pistes de remédiation (facultatif)

- `LSLCD-NE` : entraîner le banc de filtres adapté au bruit (11 niveaux)
  sur un jeu de données synthétique et implémenter l'estimateur de bruit
  par homogénéité de blocs décrits dans Jeon & Dubois (2013) §IV-V.
- `RCD` : contacter l'auteur pour une double licence, ou réécrire
  clean-room depuis la description algorithmique du README (GPL-3.0
  sinon incompatible avec le socle MIT).
- `SSD` : clarifier le statut brevet auprès des auteurs, ou réécrire
  clean-room depuis l'article IPOL seul.
