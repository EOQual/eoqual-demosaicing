# Change Log

## Version History

0.1.0 (2026-09-06)

:   -   Première version publique.
    -   **26 méthodes** cataloguées : interpolation générique
        (`interp_bilinear`, `interp_bicubic`, `interp_spline`), OpenCV
        (`opencv_bilinear`, `opencv_vng`, `opencv_ea`), colour-demosaicing
        (`colour_bilinear`, `colour_malvar2004`, `colour_menon2007`),
        implémentations propres du projet (`bilinear`,
        `green_edge_based`, `malvar_bilateral`, `edge_aware`,
        `edge_aware_simplified`, `ahd`, `mrf`), `malvar` (adapté de
        prysm), la famille IPOL (`ha`, `ari`, `ri`, `gbtf`, `mlri`,
        `wmlri`), `dlmmse` (Zhang & Wu, réimplémenté depuis une source
        BSD de Pascal Getreuer), `cdm` (Zhang, Wu, Buades & Li —
        interpolation directionnelle + seuillage non local) et `lslcd`
        (Dubois — démultiplexage luma-chroma fréquentiel).
    -   Registre central unique (`config.py::METHODS_CONFIGS`) et
        dispatcher générique `eoqual.demosaicing.runner.demosaic(image,
        method=...)`, en plus de l'import direct de chaque fonction.
    -   Audit de licence complet des sources amont
        (`THIRD_PARTY_LICENSES.md`) : vérification directe (fichier
        LICENSE, champ `license` de l'API GitHub) pour chaque dépendance
        tierce ou code adapté. Aucune méthode cataloguée n'est écartée
        pour raison de licence à ce jour ; les méthodes non implémentées
        sont documentées en roadmap (`THIRD_PARTY_LICENSES.md` §3).
    -   Documentation : `README.md` (installation, catalogue, licence),
        `REFERENCE_TECHNIQUE.md` (principe/implémentation/limites par
        méthode), `THIRD_PARTY_LICENSES.md` (audit des licences
        tierces), `CONTRIBUTING.md` (signaler un bug, proposer/
        implémenter une méthode).
    -   Tests : smoke tests (`tests/test_backends.py`, 26/26 méthodes)
        sur une image Bayer réelle (suite Kodak), exemples
        (`examples/`).

## Évolutions futures

Voir `THIRD_PARTY_LICENSES.md` §4 (pistes de remédiation sur
`LSLCD-NE`, `RCD`, `SSD`) et `CONTRIBUTING.md` (proposer/implémenter
une méthode).
