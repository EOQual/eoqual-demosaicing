# NOTICE — code adapté (Jin, Guo, Facciolo & Morel, IPOL)

Les fichiers de ce dossier (``ri.py`` et ``lib/``) sont adaptés du code de
référence distribué avec l'article IPOL :

- Jin, Q., Guo, Y., Facciolo, G., Morel, J.-M. (2021). "A Mathematical
  Analysis and Implementation of Residual Interpolation Demosaicking
  Algorithms." Image Processing On Line.
  https://www.ipol.im/pub/art/2021/358/
- Licence : **BSD 2-Clause modifiée**, Copyright (c) 2021, Qiyu Jin, Yu
  Guo, Gabriele Facciolo — voir `LICENSE` dans ce dossier (texte exact du
  paquet source `residual_demosaicking`).
- Algorithmes originaux couverts par ``ri_demosaic`` :
  - `"GBTF"` : Pekkucuksen & Altunbasak (2010), "Gradient Based
    Threshold Free Color Filter Array Interpolation", IEEE ICIP.
  - `"RI"` : Monno, Y., Kiku, D., Tanaka, M., Okutomi, M. (2016).
    "Adaptive Residual Interpolation for Color and Multispectral Image
    Demosaicking." Sensors, 17(12), 2787.
  - `"MLRI"` / `"WMLRI"` : Kiku, D., Monno, Y., Tanaka, M., Okutomi, M.
    (2014/2016). "Minimized-Laplacian Residual Interpolation for Color
    Image Demosaicking" et sa variante pondérée.

Modifications apportées : fonctions renommées et regroupées sous
`lib/` (ex. `RIgreen_interpolation.py`), imports adaptés à la structure
du paquet `eoqual-demosaicing` ; logique de calcul inchangée.

Voir `THIRD_PARTY_LICENSES.md` (racine du projet) pour l'entrée
correspondante dans l'audit des licences.
