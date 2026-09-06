#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Malvar, He & Cutler — interpolation par noyaux corrigés du gradient.

Référence :
    Malvar, H. S., He, L.-W., Cutler, R. (2004). "High-quality linear
    interpolation for demosaicing of Bayer-patterned color images."
    IEEE ICASSP.

Adapté de ``prysm.bayer`` (Brandon Dube) :
https://github.com/brandondube/prysm/blob/master/prysm/bayer.py — licence
**MIT**, voir ``THIRD_PARTY_LICENSES.md``. Noyaux identiques (figure 2 de
l'article) ; adaptation aux conventions du projet.

Limite : motifs Bayer ``RGGB`` et ``BGGR`` uniquement.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import ndimage

__all__ = ["malvar_demosaic"]


top_left     = (slice(0, None, 2), slice(0, None, 2))
top_right    = (slice(1, None, 2), slice(0, None, 2))
bottom_left  = (slice(0, None, 2), slice(1, None, 2))
bottom_right = (slice(1, None, 2), slice(1, None, 2))

# Kernels from Malvar et al, fig 2.
# names derived from the paper,
# in demosaic_malvar the naming
# may be more clear
# "G at R locations" or G at B locations
kernel_G_at_R_or_B = [
    [ 0, 0, -1, 0,  0], # NOQA
    [ 0, 0,  2, 0,  0], # NOQA
    [-1, 2,  4, 2, -1], # NOQA
    [ 0, 0,  2, 0,  0], # NOQA
    [ 0, 0, -1, 0,  0], # NOQA
]

# R at green in R row, B column
kernel_R_at_G_in_RB = [
    [ 0,  0, .5, 0,  0], # NOQA
    [ 0, -1, 0, -1,  0], # NOQA
    [-1,  4, 5,  4, -1], # NOQA
    [ 0, -1, 0, -1,  0], # NOQA
    [ 0,  0, .5, 0,  0], # NOQA
]

kernel_R_at_G_in_BR = [
    [0,  0, -1,  0, 0 ], # NOQA
    [0, -1,  4, -1, 0 ], # NOQA
    [.5, 0,  5,  0, .5], # NOQA
    [0, -1,  4, -1, 0 ], # NOQA
    [0,  0, -1,  0, 0 ], # NOQA
]

kernel_R_at_B_in_BB = [
    [0,    0, -3/2, 0,  0],   # NOQA
    [0,    2,  0,   2,  0],   # NOQA
    [-3/2, 0,  6,   0, -3/2], # NOQA
    [0,    2,  0,   2,  0],   # NOQA
    [0,    0, -3/2, 0,  0],   # NOQA
]

kernel_B_at_G_BR    = kernel_R_at_G_in_RB
kernel_B_at_G_RB    = kernel_R_at_G_in_BR
kernel_B_at_R_in_RR = kernel_R_at_B_in_BB


def malvar_demosaic(image: np.ndarray, pattern: Literal['RGGB', 'BGGR'] = 'RGGB') -> np.ndarray:
    """
    Dématriçe une image Bayer par la méthode de Malvar, He & Cutler.

    Parameters
    ----------
    image : np.ndarray
        Image Bayer 2D (mosaïque brute).
    pattern : Literal['RGGB', 'BGGR']
        Motif de Bayer. Par défaut ``'RGGB'``.

    Returns
    -------
    np.ndarray
        Image RGB démosaïquée, forme ``(H, W, 3)``.
    """
    typ = image.dtype
    # Convertir l'image en float pour éviter les problèmes de débordement et de précision
    img_float = image.astype(np.float32)

    # create all of our convolution kernels (FIR filters)
    # division by 8 is to make the kernel sum to 1
    # (preserve energy)
    kgreen           = np.asarray(kernel_G_at_R_or_B) / 8
    kgreensameColumn = np.asarray(kernel_R_at_G_in_RB) / 8
    kgreensameRow    = np.asarray(kernel_R_at_G_in_BR) / 8
    kdiagonalRB      = np.asarray(kernel_R_at_B_in_BB) / 8
 
    # there is only one filter for G
    Gest = ndimage.convolve(img_float, kgreen)
 
    # there are only three unique convolutions remaining
    c1 = ndimage.convolve(img_float, kgreensameColumn)
    c2 = ndimage.convolve(img_float, kgreensameRow)
    c3 = ndimage.convolve(img_float, kdiagonalRB)
 
    red   = np.empty_like(img_float)
    green = Gest
    blue  = np.empty_like(img_float)
 
    green[top_right]   = img_float[top_right]
    green[bottom_left] = img_float[bottom_left]
 
    if pattern == 'RGGB':
        red[top_left]     = img_float[top_left]
        red[top_right]    = c2[top_right]
        red[bottom_left]  = c1[bottom_left]
        red[bottom_right] = c3[bottom_right]
 
        blue[top_left]     = c3[top_left]
        blue[top_right]    = c1[top_right]
        blue[bottom_left]  = c2[bottom_left]
        blue[bottom_right] = img_float[bottom_right]
    elif pattern == 'BGGR':
        blue[top_left]     = img_float[top_left]
        blue[top_right]    = c2[top_right]
        blue[bottom_left]  = c1[bottom_left]
        blue[bottom_right] = c3[bottom_right]
 
        red[top_left]     = c3[top_left]
        red[top_right]    = c1[top_right]
        red[bottom_left]  = c2[bottom_left]
        red[bottom_right] = img_float[bottom_right]
    else:
        raise NotImplementedError('only rggb, bggr bayer patterns currently implemented')

    # Clamping pour s'assurer que les valeurs sont dans la plage du type uint8 ou uint16 (après normalisation)
    if (typ == np.uint8) or (typ == np.uint16):
        red   = red + 0.5
        green = green + 0.5
        blue  = blue + 0.5
    
        red   = np.clip(red, np.iinfo(typ).min, np.iinfo(typ).max).astype(typ)
        green = np.clip(green, np.iinfo(typ).min, np.iinfo(typ).max).astype(typ)
        blue  = np.clip(blue, np.iinfo(typ).min, np.iinfo(typ).max).astype(typ)

    return np.stack((red, green, blue), axis=2)

