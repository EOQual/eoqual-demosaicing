#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interpolation basée sur les contours du canal vert (implémentation
propre du projet, aucun code tiers repris).

Le canal vert (le mieux échantillonné du motif de Bayer) est interpolé
en choisissant, pixel à pixel, entre un noyau horizontal, vertical ou
isotrope selon la direction où le contraste local est le plus faible
(estimée par un gradient horizontal/vertical du vert brut). Les canaux
rouge et bleu sont ensuite interpolés sur leur différence au vert
(hypothèse de lissage chromatique), qui est plus lisse que le signal
brut.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import ndimage

from ...core.bayer import combine_channels, get_mosaic_masks_RGB

__all__ = ["green_edge_based_demosaic"]


def green_edge_based_demosaic(
    image: np.ndarray,
    pattern: Literal['RGGB', 'BGGR', 'GBRG', 'GRBG'] = "RGGB"
) -> np.ndarray:
    """
    Dématriçe une image Bayer par interpolation basée sur les contours du vert.

    Parameters
    ----------
    image : np.ndarray
        Image Bayer 2D (mosaïque brute).
    pattern : Literal['RGGB', 'BGGR', 'GBRG', 'GRBG']
        Motif de Bayer. Par défaut ``"RGGB"``.

    Returns
    -------
    np.ndarray
        Image RGB démosaïquée, forme ``(H, W, 3)``.
    """
    typ = image.dtype
    mask_red, mask_green, mask_blue = get_mosaic_masks_RGB(pattern, image.shape)

    # Extraction des valeurs de base de la couleur verte
    greenbase = image * mask_green
    greenh = np.abs(ndimage.convolve(greenbase, [[1, 0, -1]], mode='mirror'))
    greenv = np.abs(ndimage.convolve(greenbase, [[1], [0], [-1]], mode='mirror'))

    # Détection des zones de transition pour la couleur verte
    gbool  = greenh > greenv
    gbool2 = greenh < greenv
    gbool3 = greenh == greenv
    
    # Convolution pour la couleur verte
    green  = ndimage.convolve(greenbase, [[0, 0, 0], [0.5, 1.0, 0.5], [0, 0, 0]], mode='mirror').astype(np.float32) * gbool2
    green += ndimage.convolve(greenbase, [[0, 0.5, 0], [0, 1.0, 0], [0, 0.5, 0]], mode='mirror').astype(np.float32) * gbool
    green += ndimage.convolve(greenbase, [[0, 0.25, 0], [0.25, 1.0, 0.25], [0, 0.25, 0]], mode='mirror').astype(np.float32) * gbool3

    # Convolutions pour les couleurs rouge et bleue
    red  = (image - green) * mask_red
    blue = (image - green) * mask_blue
    
    kernel_rb = np.array([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]])

    red  = ndimage.convolve(red, kernel_rb, mode='mirror').astype(np.float32) + green
    blue = ndimage.convolve(blue, kernel_rb, mode='mirror').astype(np.float32) + green

    # Normalisation pour les couleurs rouge et bleue
    red_count  = ndimage.convolve(mask_red, kernel_rb, mode='mirror').astype(np.float32)
    blue_count = ndimage.convolve(mask_blue, kernel_rb, mode='mirror').astype(np.float32)

    red  = np.where(red_count > 0, red / red_count, 0)
    blue = np.where(blue_count > 0, blue / blue_count, 0)

    # Clamping pour s'assurer que les valeurs sont dans la plage du type uint8 ou uint16 (après normalisation)
    if (typ == np.uint8) or (typ == np.uint16):
        red   = red + 0.5
        green = green + 0.5
        blue  = blue + 0.5
    
        red   = np.clip(red, np.iinfo(typ).min, np.iinfo(typ).max).astype(typ)
        green = np.clip(green, np.iinfo(typ).min, np.iinfo(typ).max).astype(typ)
        blue  = np.clip(blue, np.iinfo(typ).min, np.iinfo(typ).max).astype(typ)

    return combine_channels(red, green, blue)

