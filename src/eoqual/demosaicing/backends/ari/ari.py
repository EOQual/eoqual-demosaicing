#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jin, Guo, Facciolo & Morel — Adaptive Residual Interpolation (ARI).

Adapté du code de référence IPOL fourni avec :

    Jin, Q., Guo, Y., Facciolo, G., Morel, J.-M. (2021). "A Mathematical
    Analysis and Implementation of Residual Interpolation Demosaicking
    Algorithms." Image Processing On Line.

Basé sur l'algorithme original :

    Monno, Y., Kiku, D., Tanaka, M., Okutomi, M. (2016). "Adaptive
    Residual Interpolation for Color and Multispectral Image
    Demosaicking." Sensors, 17(12), 2787.

Licence du code amont : BSD 2-Clause modifiée (Copyright (c) 2021, Qiyu
Jin, Yu Guo, Gabriele Facciolo — voir ``LICENSE`` dans ce dossier).
Fonctions renommées et regroupées sous ``lib/`` ; logique de calcul
inchangée. Voir ``THIRD_PARTY_LICENSES.md`` (racine du projet).
"""
from __future__ import annotations

import numpy as np

from ...core.bayer import get_mosaic_masks_RGB
from .lib.ARIgreen_interpolation import ARIgreen_interpolation
from .lib.ARIred_blue_interpolation_first import ARIred_blue_interpolation_first
from .lib.ARIred_blue_interpolation_second import ARIred_blue_interpolation_second

__all__ = ["ari_demosaic"]


def ari_demosaic(image: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """
    Dématriçe une image Bayer par Adaptive Residual Interpolation (ARI).

    Parameters
    ----------
    image : np.ndarray
        Image Bayer 2D (mosaïque brute, un seul canal).
    pattern : str
        Motif de Bayer (``"RGGB"``, ``"BGGR"``, ``"GRBG"`` ou ``"GBRG"``).
        Par défaut ``"RGGB"``.

    Returns
    -------
    np.ndarray
        Image RGB démosaïquée, forme ``(H, W, 3)``.
    """
    # Epsilon du filtre guidé
    eps = 1e-10

    # Masque RGB de la mosaïque
    mask = np.stack(get_mosaic_masks_RGB(pattern, image.shape), -1)

    # Génération de la mosaïque par canal
    mosaic = (image[..., None] * mask).astype(np.float32)

    # Interpolation du canal vert
    green = ARIgreen_interpolation(mosaic, mask, pattern, eps)

    # Interpolation rouge/bleu — première étape (diagonale)
    red, blue = ARIred_blue_interpolation_first(green, mosaic, mask, eps)

    # Interpolation rouge/bleu — seconde étape (horizontale/verticale)
    red, blue = ARIred_blue_interpolation_second(green, red, blue, mask, eps)

    rgb = np.zeros(mosaic.shape, dtype=np.float32)
    rgb[:, :, 0] = red
    rgb[:, :, 1] = green
    rgb[:, :, 2] = blue

    return rgb
