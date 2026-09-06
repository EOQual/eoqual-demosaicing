#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dubois — Least-Squares Luma-Chroma Demultiplexing (LSLCD), démosaïçage
fréquentiel.

Références :
    Dubois, E. (2005). "Frequency-domain methods for demosaicking of
    Bayer-sampled color images." IEEE Signal Processing Letters, 12(12),
    847-850.
    Jeon, G., Dubois, E. (2013). "Demosaicking of noisy Bayer-sampled
    color images with least-squares luma-chroma demultiplexing and
    noise level estimation." IEEE Transactions on Image Processing,
    22(1), 146-156. (§III présente le système LSLCD non bruité repris
    ici ; l'article introduit par ailleurs une extension adaptative au
    bruit, non implémentée — voir plus bas.)

Principe (Jeon & Dubois, §III, Fig. 1 — variante non bruitée) : le
signal CFA de Bayer se décompose en fréquence comme un signal de
luminance en bande de base plus deux signaux de chrominance modulés
(C1 à la fréquence diagonale de Nyquist, C2 à deux fréquences en
opposition de phase le long des axes) :

.. math::

    f_{CFA} = f_L + f_{C1} \\cdot p_{diag} + f_{C2} \\cdot (p_{lignes} - p_{colonnes})

où :math:`p_{diag}`, :math:`p_{lignes}`, :math:`p_{colonnes}` valent
+1/-1 en damier selon, respectivement, la position verte/non-verte, la
ligne contenant le rouge ou le bleu, et la colonne contenant le rouge ou
le bleu — ces trois porteuses sont dérivées ici directement des masques
du motif de Bayer (``core.bayer``) plutôt que d'une parité fixe, pour
rester correctes quel que soit le motif (RGGB/BGGR/GRBG/GBRG).

Le démultiplexage démodule puis filtre passe-bas chaque porteuse pour
extraire une estimée de C1 et deux estimées de C2 (une par porteuse),
combinées de façon adaptative ; la luminance est obtenue par
soustraction des chromas remodulées au signal brut. R, G, B sont enfin
reconstruits par inversion de la matrice luma-chrominance (G = L+C1,
R = (L-C1)+2C2, B = (L-C1)-2C2).

Réimplémentation depuis les équations publiées — aucun code tiers
repris. **Limites assumées par rapport à l'article** : les filtres
passe-bas sont conçus ici par moindres carrés séparables
(`scipy.signal.firls`, 11 points, dans l'esprit « least-squares » de la
méthode) plutôt que jointement optimisés sur un jeu d'entraînement comme
dans l'article d'origine ; la combinaison adaptative de C2 utilise une
pondération par énergie locale, heuristique raisonnable mais non
spécifiée explicitement dans l'article. Il s'agit donc de « LSLCD »
(cas non bruité) — l'extension adaptative au bruit (« LSLCD-NE », banc
de filtres pré-entraînés pour 11 niveaux de bruit + estimateur de bruit
dédié) n'est **pas** implémentée ici (voir ``THIRD_PARTY_LICENSES.md``).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.signal import firls

from ...core.bayer import get_mosaic_masks_RGB

__all__ = ["lslcd_demosaic"]

_NUMTAPS = 11
_CUTOFF = 0.25       # bande passante normalisée (Nyquist = 1)
_TRANSITION = 0.15


def _lowpass_2d(numtaps: int = _NUMTAPS, cutoff: float = _CUTOFF) -> np.ndarray:
    """Noyau passe-bas 2D séparable conçu par moindres carrés (`firls`)."""
    edge = min(cutoff + _TRANSITION, 0.999)
    proto = firls(numtaps, [0, cutoff, edge, 1.0], [1, 1, 0, 0])
    kernel_2d = np.outer(proto, proto)
    return kernel_2d / kernel_2d.sum()


def lslcd_demosaic(image: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """
    Dématriçe une image Bayer par démultiplexage luma-chroma (LSLCD).

    Parameters
    ----------
    image : np.ndarray
        Image Bayer 2D (mosaïque brute).
    pattern : str
        Motif de Bayer (``'RGGB'``, ``'BGGR'``, ``'GRBG'``, ``'GBRG'``).
        Par défaut ``'RGGB'``.

    Returns
    -------
    np.ndarray
        Image RGB démosaïquée, forme ``(H, W, 3)``.
    """
    mosaic = np.asarray(image, dtype=np.float64)
    red_mask, green_mask, blue_mask = get_mosaic_masks_RGB(pattern, mosaic.shape)

    # Porteuses en damier dérivées du motif : +1/-1 selon vert/non-vert
    # (fréquence diagonale), et selon que la ligne/colonne porte le
    # rouge ou le bleu (fréquences axiales) — correct pour les 4 motifs
    # sans logique de repli supplémentaire.
    diag_carrier = np.where(green_mask.astype(bool), 1.0, -1.0)

    red_rows = red_mask.astype(bool).any(axis=1)
    row_carrier = np.where(red_rows[:, None], 1.0, -1.0) * np.ones_like(mosaic)
    red_cols = red_mask.astype(bool).any(axis=0)
    col_carrier = np.where(red_cols[None, :], 1.0, -1.0) * np.ones_like(mosaic)

    kernel = _lowpass_2d()

    # Chrominance C1 (porteuse diagonale) : démodulation puis passe-bas
    c1 = ndimage.convolve(mosaic * diag_carrier, kernel, mode="mirror")

    # Chrominance C2 : deux estimées indépendantes (une par porteuse),
    # combinées par pondération inversement proportionnelle à leur
    # énergie locale (favorise l'estimée la moins bruitée/aliasée).
    c2a = ndimage.convolve(mosaic * row_carrier, kernel, mode="mirror")
    c2b = ndimage.convolve(mosaic * col_carrier, kernel, mode="mirror")

    energy_kernel = np.ones((5, 5)) / 25.0
    energy_a = ndimage.convolve(c2a**2, energy_kernel, mode="mirror") + 1e-8
    energy_b = ndimage.convolve(c2b**2, energy_kernel, mode="mirror") + 1e-8
    weight_a = 1.0 / energy_a
    weight_b = 1.0 / energy_b
    c2 = (weight_a * c2a + weight_b * c2b) / (weight_a + weight_b)

    # Luminance : signal brut moins les chromas remodulées
    luma = mosaic - c1 * diag_carrier - c2 * row_carrier - c2 * col_carrier

    # Inversion de la matrice luma-chrominance :
    # G = L + C1 ; R = (L - C1) + 2*C2 ; B = (L - C1) - 2*C2
    green = luma + c1
    red = (luma - c1) + 2 * c2
    blue = (luma - c1) - 2 * c2

    return np.stack([red, green, blue], axis=-1)
