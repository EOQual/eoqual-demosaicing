#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interpolation générique canal par canal (implémentation propre du
projet, aucun code tiers repris).

Chaque canal R, G, B est extrait de la mosaïque puis interpolé
indépendamment par un filtre générique — noyau moyenneur (``'bilinear'``),
noyau gaussien séparable (``'bicubic'``), ou interpolation par spline
cubique (``'spline'``) — avec normalisation par le nombre d'échantillons
valides pour ignorer les pixels manquants dans la convolution.
"""
from __future__ import annotations

import cv2
import numpy as np
import scipy.ndimage as ndimage

from ...core.bayer import extract_channel

__all__ = ["interp_demosaic"]


def interpolate_channel(
    channel: np.ndarray,
    method: str = 'bilinear',
    radius: int = 2
) -> np.ndarray:
    """
    Interpolates a channel while ignoring missing pixels in the convolution.
    
    Parameters
    ----------
    channel : np.ndarray
        2D matrix representing a partially filled color channel
    method : str, optional
        Interpolation method ('bilinear', 'bicubic', 'spline'). Default is 'bilinear'
    radius : int, optional
        Convolution kernel radius (default is 2)
    
    Returns
    -------
    np.ndarray
        Interpolated channel with estimated missing values.
    """
    if method == 'bilinear':
        kernel = np.ones((2*radius+1, 2*radius+1)) / ((2*radius+1)**2)
    elif method == 'bicubic':
        kernel = cv2.getGaussianKernel(2*radius+1, radius) @ cv2.getGaussianKernel(2*radius+1, radius).T
    elif method == 'spline':
        mask = (channel > 0).astype(np.float32)
        interpolated = ndimage.spline_filter(channel * mask, order=3)
        norm_factor = ndimage.spline_filter(mask, order=3)
        norm_factor[norm_factor == 0] = 1
        return interpolated / norm_factor # For spline, it's a direct interpolation
    else:
        raise ValueError("Méthode non supportée")

    # Mask of valid pixels
    mask = (channel > 0).astype(np.float32)

    # Convolution of the image and the mask
    convolved   = ndimage.convolve(channel, kernel, mode='mirror')
    norm_factor = ndimage.convolve(mask, kernel, mode='mirror')

    # Avoid division by zero
    norm_factor[norm_factor == 0] = 1

    # Normalization to avoid being influenced by black pixels
    return convolved / norm_factor


def interp_demosaic(
    image: np.ndarray,
    pattern: str = 'RGGB',
    method: str = 'bilinear',
    radius: int = 2
) -> np.ndarray:
    """
    Dématriçe une image Bayer par interpolation générique canal par canal.

    Parameters
    ----------
    image : np.ndarray
        Image Bayer 2D (mosaïque brute).
    pattern : str
        Motif de Bayer (``'RGGB'``, ``'BGGR'``, ``'GRBG'``, ``'GBRG'``).
        Par défaut ``'RGGB'``.
    method : str
        Méthode d'interpolation (``'bilinear'``, ``'bicubic'``,
        ``'spline'``). Par défaut ``'bilinear'``.
    radius : int
        Rayon du noyau de convolution. Par défaut ``2``.

    Returns
    -------
    np.ndarray
        Image RGB démosaïquée, forme ``(H, W, 3)``.
    """
    y_size, x_size = image.shape
    debayered_image = np.zeros((y_size, x_size, 3), dtype=image.dtype)
    for idx, band in enumerate(['R', 'G', 'B']):
        sub_plane = extract_channel(image, band, pattern)

        # Normalisation pour éviter les problèmes numériques
        max_sub_plane = np.max(sub_plane)
        if max_sub_plane > 0:
            sub_plane /= max_sub_plane

        debayered_image[..., idx] = interpolate_channel(sub_plane, radius=radius, method=method) * max_sub_plane

    return debayered_image