#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AHD — interpolation edge-aware guidée par l'homogénéité locale
(implémentation propre du projet, aucun code tiers repris).

Inspirée du principe général de :

    Hirakawa, K., Parks, T. W. (2005). "Adaptive homogeneity-directed
    demosaicing algorithm." IEEE Transactions on Image Processing,
    14(3), 360-369.

Il ne s'agit pas d'un portage de l'algorithme original (qui compare deux
directions candidates dans un espace de couleur perceptuel et sélectionne
la plus homogène) : cette implémentation approxime l'idée par une
interpolation directionnelle du vert pondérée par le gradient local
(``ahd_interpolate_green``), puis interpole rouge/bleu sur leur
chrominance au vert (``ahd_interpolate_red_blue``).

Limite : motif Bayer RGGB uniquement (``compute_masks``).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import convolve

__all__ = ["ahd_demosaic"]


def compute_masks(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construit les masques R, G, B d'un motif de Bayer RGGB.

    Parameters
    ----------
    shape : Tuple[int, int]
        Forme de l'image (hauteur, largeur).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Masques R, G, B.
    """
    r_mask = np.zeros(shape, dtype=np.float32)
    g_mask = np.zeros(shape, dtype=np.float32)
    b_mask = np.zeros(shape, dtype=np.float32)

    r_mask[0::2, 0::2] = 1
    g_mask[0::2, 1::2] = 1
    g_mask[1::2, 0::2] = 1
    b_mask[1::2, 1::2] = 1
    return r_mask, g_mask, b_mask


def ahd_interpolate_green(image: np.ndarray, g_mask: np.ndarray) -> np.ndarray:
    """
    Interpolation edge-aware du canal vert.

    Parameters
    ----------
    image : np.ndarray
        Mosaïque Bayer.
    g_mask : np.ndarray
        Masque des échantillons verts.

    Returns
    -------
    np.ndarray
        Canal vert interpolé.
    """
    green_kernel = np.array([
        [0, 0.25, 0],
        [0.25, 0, 0.25],
        [0, 0.25, 0]
    ], dtype=np.float32)
    green = image * g_mask
    bilinear_g = convolve(green, green_kernel, mode='mirror')
    green += bilinear_g * (1 - g_mask)
    green = np.clip(green, 0, 1)

    gx_kernel = np.array([
        [0, 0, 0],
        [-1, 0, 1],
        [0, 0, 0]
    ], dtype=np.float32)
    gy_kernel = gx_kernel.T

    grad_x = np.abs(convolve(green, gx_kernel, mode='mirror'))
    grad_y = np.abs(convolve(green, gy_kernel, mode='mirror'))

    denom = grad_x + grad_y
    denom = np.where(denom < 1e-3, 1e-3, denom)  # Stabilise les divisions
    weight_x = grad_y / denom
    weight_y = grad_x / denom

    green_h = convolve(green, np.array([
        [0, 0.5, 0],
        [0, 0, 0],
        [0, 0.5, 0]
    ]), mode='mirror')
    green_v = convolve(green, np.array([
        [0, 0, 0],
        [0.5, 0, 0.5],
        [0, 0, 0]
    ]), mode='mirror')

    green_interp = weight_x * green_h + weight_y * green_v

    # Repli bilinéaire dans les régions très plates
    flat_region = (grad_x + grad_y < 0.02)
    green_interp = np.where(flat_region, bilinear_g, green_interp)

    green = green * g_mask + green_interp * (1 - g_mask)
    return np.clip(green, 0, 1)


def ahd_interpolate_red_blue(
    image: np.ndarray,
    green: np.ndarray,
    r_mask: np.ndarray,
    b_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolation edge-aware des canaux rouge et bleu, guidée par le vert.

    Parameters
    ----------
    image : np.ndarray
        Mosaïque Bayer.
    green : np.ndarray
        Canal vert interpolé.
    r_mask : np.ndarray
        Masque des échantillons rouges.
    b_mask : np.ndarray
        Masque des échantillons bleus.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Canaux rouge et bleu interpolés.
    """
    # Chrominance = R-G et B-G
    red = image * r_mask
    blue = image * b_mask

    chrom_r = (red - green) * r_mask
    chrom_b = (blue - green) * b_mask

    # Interpolation bilinéaire de la chrominance
    bilinear_kernel = np.array([
        [0.25, 0.5, 0.25],
        [0.5, 0, 0.5],
        [0.25, 0.5, 0.25]
    ], dtype=np.float32)

    interp_chrom_r = convolve(chrom_r, bilinear_kernel, mode='mirror')
    interp_chrom_b = convolve(chrom_b, bilinear_kernel, mode='mirror')

    # Bornage de la chrominance pour éviter les artefacts de couleur
    interp_chrom_r = np.clip(interp_chrom_r, -0.5, 0.5)
    interp_chrom_b = np.clip(interp_chrom_b, -0.5, 0.5)

    red_full = green + interp_chrom_r
    blue_full = green + interp_chrom_b

    red_full = red * r_mask + red_full * (1 - r_mask)
    blue_full = blue * b_mask + blue_full * (1 - b_mask)

    return np.clip(red_full, 0, 1), np.clip(blue_full, 0, 1)


def ahd_demosaic(image: np.ndarray) -> np.ndarray:
    """
    Dématriçe une mosaïque Bayer RGGB par interpolation edge-aware AHD-like.

    Parameters
    ----------
    image : np.ndarray
        Mosaïque Bayer 2D, normalisée dans ``[0, 1]``.

    Returns
    -------
    np.ndarray
        Image RGB démosaïquée, forme ``(H, W, 3)``.
    """
    r_mask, g_mask, b_mask = compute_masks(image.shape)
    green = ahd_interpolate_green(image, g_mask)
    red, blue = ahd_interpolate_red_blue(image, green, r_mask, b_mask)

    rgb = np.stack([red, green, blue], axis=-1)
    return np.clip(rgb, 0, 1)
