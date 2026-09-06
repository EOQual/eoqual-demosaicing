#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jin, Guo, Facciolo & Morel — Residual Interpolation (RI, GBTF, MLRI, WMLRI).

Adapté du code de référence IPOL fourni avec :

    Jin, Q., Guo, Y., Facciolo, G., Morel, J.-M. (2021). "A Mathematical
    Analysis and Implementation of Residual Interpolation Demosaicking
    Algorithms." Image Processing On Line.

Un seul point d'entrée (``ri_demosaic``) couvre quatre variantes
sélectionnées par ``algorithm``, chacune correspondant à un article :

    - ``"GBTF"`` : Pekkucuksen & Altunbasak (2010), "Gradient Based
      Threshold Free Color Filter Array Interpolation."
    - ``"RI"``   : Monno et al. (2016), "Adaptive Residual Interpolation
      for Color and Multispectral Image Demosaicking."
    - ``"MLRI"`` / ``"WMLRI"`` : Kiku et al. (2014/2016), "Minimized-
      Laplacian Residual Interpolation for Color Image Demosaicking" et
      sa variante pondérée.

Licence du code amont : BSD 2-Clause modifiée (Copyright (c) 2021, Qiyu
Jin, Yu Guo, Gabriele Facciolo — voir ``LICENSE`` dans ce dossier).
Fonctions renommées et regroupées sous ``lib/`` ; logique de calcul
inchangée. Voir ``THIRD_PARTY_LICENSES.md`` (racine du projet).
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from ...core.bayer import get_mosaic_masks_RGB
from .lib.RIblue_interpolation import blue_interpolation
from .lib.RIgreen_interpolation import green_interpolation
from .lib.RIred_interpolation import red_interpolation

__all__ = ["ri_demosaic"]


def ri_demosaic(
    image: np.ndarray,
    pattern: str = "RGGB",
    sigma: Optional[float] = 2.0,
    algorithm: Literal["GBTF", "RI", "MLRI", "WMLRI"] = "RI",
) -> np.ndarray:
    """
    Dématriçe une image Bayer par Residual Interpolation.

    Parameters
    ----------
    image : np.ndarray
        Image Bayer 2D (mosaïque brute, un seul canal).
    pattern : str
        Motif de Bayer (``"RGGB"``, ``"BGGR"``, ``"GRBG"`` ou ``"GBRG"``).
        Par défaut ``"RGGB"``.
    sigma : float, optional
        Paramètre de régularisation du filtre guidé. Ignoré par
        ``"GBTF"``. Par défaut ``2.0``.
    algorithm : Literal["GBTF", "RI", "MLRI", "WMLRI"]
        Variante à utiliser. Par défaut ``"RI"``.

    Returns
    -------
    np.ndarray
        Image RGB démosaïquée, forme ``(H, W, 3)``.
    """
    # Masque RGB de la mosaïque
    mask = np.stack(get_mosaic_masks_RGB(pattern, image.shape), -1)

    # Génération de la mosaïque par canal
    mosaic = (image[..., None] * mask).astype(np.float32)

    # Interpolation du canal vert
    green, residual = green_interpolation(mosaic, mask, pattern, sigma, algorithm)

    # Paramètres du filtre guidé (upsampling)
    window_h, window_v, eps = 5, 5, 0

    # Interpolation rouge et bleue
    red = red_interpolation(green, mosaic, mask, pattern, window_h, window_v, eps, residual, algorithm)
    blue = blue_interpolation(green, mosaic, mask, pattern, window_h, window_v, eps, residual, algorithm)

    rgb = np.zeros((*mosaic.shape[:2], 3), dtype=np.float32)
    rgb[:, :, 0] = red
    rgb[:, :, 1] = green
    rgb[:, :, 2] = blue

    return rgb
