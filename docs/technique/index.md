# Documents techniques

Détail de chaque méthode : principe, implémentation, limites connues. Les
chapitres gardent une numérotation unique d'un document à l'autre : un
renvoi « §n » désigne toujours le même chapitre. Ces documents décrivent
l'état présent du code ; ce qui change d'une version à l'autre est dans les
[notes de version](../changelog.md). Pour le catalogue résumé et
l'installation, voir le `README.md` ; pour l'audit des licences,
`THIRD_PARTY_LICENSES.md`.

| Document | Chapitres | Contenu |
|---|---|---|
| [Conventions et méthodes génériques](methodes-generiques.md) | 1-3 | Conventions, interpolation générique, OpenCV et colour-demosaicing |
| [Implémentations propres du projet](methodes-projet.md) | 4 | Bilinéaire, gradient de vert, Malvar bilatéral, contours, AHD, MRF |
| [Méthodes publiées réimplémentées](methodes-publiees.md) | 5-10 | Malvar, famille IPOL (HA, ARI, RI, GBTF, MLRI), DLMMSE, LDI-NAT, LSLCD ; méthodes non implémentées |
| [Démosaïçage et bruit](bruit/index.md) | 11 | Bruit du capteur, choix de σ, choix du démosaïqueur : modèle, études, recommandations |

```{toctree}
:hidden:
:maxdepth: 2

methodes-generiques
methodes-projet
methodes-publiees
bruit/index
```
