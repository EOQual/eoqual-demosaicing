#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enrobage des méthodes de démosaïçage natives d'OpenCV
(``cv2.cvtColor``) : interpolation bilinéaire, Variable Number of
Gradients (VNG) et Edge-Aware (EA). Aucun code tiers repris — appelle
uniquement l'API publique d'OpenCV (dépendance ``opencv-python``, du
socle du projet).

Voir https://docs.opencv.org/4.11.0/de/d25/imgproc_color_conversions.html
"""
from __future__ import annotations

import warnings

import numpy as np

__all__ = ["opencv_demosaic"]


def opencv_demosaic(
        image: np.ndarray,
        pattern: str = 'RGGB',
        method: str = 'bilinear'
    ) -> np.ndarray:
    """
    Dématriçe une image Bayer avec les méthodes natives d'OpenCV.

    Parameters
    ----------
    image : np.ndarray
        Image Bayer 2D (mosaïque brute).
    pattern : str
        Motif de Bayer (``'RGGB'``, ``'BGGR'``, ``'GRBG'``, ``'GBRG'``).
        Par défaut ``'RGGB'``.
    method : str
        Méthode de démosaïçage (``'bilinear'``, ``'vng'``, ``'ea'``).
        Par défaut ``'bilinear'``.

    Returns
    -------
    np.ndarray
        Image RGB démosaïquée.
    """
    import cv2

    if method == 'vng':
        if image.dtype not in [np.uint8]:
            warnings.warn(
                f"opencv_demosaic(method='vng') requires CV_8U ; got {image.dtype}, "
                "converting to uint8 for demosaicing.",
                stacklevel=2,
            )
            image_data_max = np.max(image)
            bayer_image_uint8 = np.uint8(image.astype(np.float32) / image_data_max * 255 + 0.5)
    else:
        if image.dtype not in [np.uint8, np.uint16]:
            warnings.warn(
                f"opencv_demosaic requires CV_8U or CV_16U ; got {image.dtype}, "
                "converting to uint16 for demosaicing.",
                stacklevel=2,
            )
            bayer_image_uint16 = np.uint16(image * 65535 + 0.5)

    if pattern == 'RGGB':
        if method == 'bilinear':
            code = cv2.COLOR_BAYER_RG2RGB
        elif method == 'vng':
            code = cv2.COLOR_BAYER_RG2RGB_VNG
        elif method == 'ea':
            code = cv2.COLOR_BAYER_RG2RGB_EA
    elif pattern == 'BGGR':
        if method == 'bilinear':
            code = cv2.COLOR_BAYER_BG2RGB
        elif method == 'vng':
            code = cv2.COLOR_BAYER_BG2RGB_VNG
        elif method == 'ea':
            code = cv2.COLOR_BAYER_BG2RGB_EA
    elif pattern == 'GRBG':
        if method == 'bilinear':
            code = cv2.COLOR_BAYER_GR2RGB
        elif method == 'vng':
            code = cv2.COLOR_BAYER_GR2RGB_VNG
        elif method == 'ea':
            code = cv2.COLOR_BAYER_GR2RGB_EA
    elif pattern == 'GBRG':
        if method == 'bilinear':
            code = cv2.COLOR_BAYER_GB2RGB
        elif method == 'vng':
            code = cv2.COLOR_BAYER_GB2RGB_VNG
        elif method == 'ea':
            code = cv2.COLOR_BAYER_GB2RGB_EA
    else:
        raise ValueError(f"Unknown Bayer pattern: {pattern}")

    if method == 'vng':
        if image.dtype not in [np.uint8]:
            debayered_image = cv2.cvtColor(bayer_image_uint8, code)  # type: ignore
            debayered_image = cv2.cvtColor(debayered_image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
            debayered_image = debayered_image.astype(np.float32) / 255 * image_data_max
    else:
        if image.dtype not in [np.uint8, np.uint16]:
            debayered_image = cv2.cvtColor(bayer_image_uint16, code)  # type: ignore
            debayered_image = cv2.cvtColor(debayered_image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
            debayered_image = debayered_image.astype(np.float32) / 65535
        else:
            debayered_image = cv2.cvtColor(image, code)
            debayered_image = cv2.cvtColor(debayered_image, cv2.COLOR_BGR2RGB)  # BGR -> RGB

    return debayered_image
