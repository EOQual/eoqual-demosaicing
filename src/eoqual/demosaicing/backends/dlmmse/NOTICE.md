# NOTICE — algorithme réimplémenté depuis une source BSD (Getreuer/Zhang-Wu)

`zhang_wu.py` est une **réimplémentation vectorisée NumPy** (pas un
portage ligne à ligne) de l'algorithme décrit et publié en C dans :

- Getreuer, P. (2011). "Zhang-Wu Directional LMMSE Image Demosaicking."
  Image Processing On Line. https://www.ipol.im/pub/art/2011/g_zwld/
  Code source : `dmzhangwu.c`/`dmzhangwu.h`.
- Licence du code source : **BSD simplifiée**, Copyright (c) 2010-2011
  Pascal Getreuer — voir `LICENSE` dans ce dossier (texte exact du
  paquet source).
- Algorithme original : Zhang, L., Wu, X. (2005). "Color demosaicking
  via directional linear minimum mean square-error estimation." IEEE
  Transactions on Image Processing, 14(12), 2167-2178.

## Fidélité au code source

Les équations (filtre d'interpolation directionnelle 5 points, filtre
de lissage 9 points, fusion LMMSE par variance locale, propagation
diagonale/axiale des différences de couleur) sont reprises à
l'identique. Deux choix d'implémentation diffèrent volontairement de
`dmzhangwu.c` :

- **Gestion des bords** : `dmzhangwu.c` traite spécialement chaque cas
  de bord/coin dans `DiagonalAverage`/`AxialAverage`. Cette
  réimplémentation utilise un remplissage symétrique « whole-sample »
  (`np.pad(mode="reflect")`) puis moyenne toujours 4 voisins — les deux
  approches sont mathématiquement équivalentes (la réflexion
  whole-sample duplique exactement les voisins valides aux bords, ce
  qui reproduit la même moyenne réduite que le code C), vérifié par
  comparaison numérique.
- **Constante de stabilisation** (`DivEpsilon`) : `0.1/(255²)` dans le
  code source (image en `[0, 255]`), reproportionnée à `0.1` pour la
  convention `[0, 1]` de ce projet (même rôle : éviter une division par
  une variance quasi nulle dans les zones plates).
- Seule la variante « papier » de l'estimation de la moyenne locale est
  implémentée (`UseZhangCodeEst=0` dans le code source) — pas la
  variante alternative du code MATLAB de référence de Zhang.

Voir `THIRD_PARTY_LICENSES.md` (racine du projet) pour l'entrée
correspondante dans l'audit des licences.
