#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interpolation bilinéaire canal par canal (implémentation propre du
projet, aucun code tiers repris).

Chaque canal (R, G, B) est extrait de la mosaïque Bayer puis convolué
séparément par un noyau bilinéaire adapté à sa densité d'échantillonnage
(5 points pour le vert, 9 pour le rouge/bleu), avec normalisation par le
nombre d'échantillons réellement disponibles au bord de l'image.
"""
from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
from scipy import ndimage

from ...core.bayer import combine_channels, get_mosaic_masks_RGB

__all__ = ["bilinear_demosaic"]


def _bilinear_channels(
    image: np.ndarray,
    pattern: Literal['RGGB', 'BGGR', 'GBRG', 'GRBG'] = 'RGGB'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpole séparément les canaux R, G, B par convolution bilinéaire.

    Parameters
    ----------
    image : np.ndarray
        Image Bayer 2D (mosaïque brute).
    pattern : Literal['RGGB', 'BGGR', 'GBRG', 'GRBG']
        Motif de Bayer. Par défaut ``'RGGB'``.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Les canaux rouge, vert et bleu interpolés.
    """
    typ = image.dtype
    red_mask, green_mask, blue_mask = get_mosaic_masks_RGB(pattern, image.shape)

    red = image * red_mask
    green = image * green_mask
    blue = image * blue_mask

    kernel_rb = np.array([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]])
    kernel_g = np.array([[0, 0.25, 0], [0.25, 1.0, 0.25], [0, 0.25, 0]])

    red2 = ndimage.convolve(red, kernel_rb, mode='mirror').astype(np.float32)
    green2 = ndimage.convolve(green, kernel_g, mode='mirror').astype(np.float32)
    blue2 = ndimage.convolve(blue, kernel_rb, mode='mirror').astype(np.float32)

    # Normalisation par le nombre d'échantillons réellement sommés
    red_count = ndimage.convolve(red_mask, kernel_rb, mode='mirror').astype(np.float32)
    green_count = ndimage.convolve(green_mask, kernel_g, mode='mirror').astype(np.float32)
    blue_count = ndimage.convolve(blue_mask, kernel_rb, mode='mirror').astype(np.float32)

    if typ in (np.uint8, np.uint16):
        red2 = np.where(red_count > 0, red2 / red_count + 0.5, 0).astype(typ)
        green2 = np.where(green_count > 0, green2 / green_count + 0.5, 0).astype(typ)
        blue2 = np.where(blue_count > 0, blue2 / blue_count + 0.5, 0).astype(typ)
    else:
        red2 = np.where(red_count > 0, red2 / red_count, 0).astype(typ)
        green2 = np.where(green_count > 0, green2 / green_count, 0).astype(typ)
        blue2 = np.where(blue_count > 0, blue2 / blue_count, 0).astype(typ)

    return red2, green2, blue2


def bilinear_demosaic(
    image: np.ndarray,
    pattern: Literal['RGGB', 'BGGR', 'GBRG', 'GRBG'] = 'RGGB'
) -> np.ndarray:
    """
    Dématriçe une image Bayer par interpolation bilinéaire canal par canal.

    Parameters
    ----------
    image : np.ndarray
        Image Bayer 2D (mosaïque brute).
    pattern : Literal['RGGB', 'BGGR', 'GBRG', 'GRBG']
        Motif de Bayer. Par défaut ``'RGGB'``.

    Returns
    -------
    np.ndarray
        Image RGB démosaïquée, forme ``(H, W, 3)``.
    """
    red, green, blue = _bilinear_channels(image, pattern)
    return combine_channels(red, green, blue)
