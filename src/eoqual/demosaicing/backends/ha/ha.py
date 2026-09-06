#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hamilton & Adams — interpolation directionnelle (Hamilton-Adams, HA).

Référence :
    Hamilton, J. Jr., Adams, J. Jr. (1997). "Adaptive color plan
    interpolation in single sensor color electronic camera." US Patent
    5,629,734.

Adapté du code de Gabriele Facciolo, Yu Guo et Qiyu Jin
(https://gist.github.com/gfacciol/48d946731e8cdac0693b3b4b3ea6d6c1,
même équipe et même licence BSD 2-Clause modifiée que les backends
``ari`` et ``ri`` — voir ``THIRD_PARTY_LICENSES.md``), modifié par
Olivier Amram (CNES).
"""
from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from ...core.bayer import get_mosaic_masks_RGB, get_mosaic_masks_GGRB

__all__ = ["ha_demosaic"]


def conv_optimized(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve the 2D image with the 2D kernel and return a 2D image.
    Pads the image to preserve the shape of the input image by replicating boundaries.
    Optimized version using cv2.filter2D and kernel inversion.

    Parameters
    ----------
    image : np.ndarray
        The input 2D image.
    kernel : np.ndarray
        The 2D convolution kernel.

    Returns
    -------
    np.ndarray
        The convolved 2D image.

    Examples
    --------
    >>> image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    >>> kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=int)
    >>> result = conv_optimized(image, kernel)
    >>> print(result)
    [[ -4.  -4.  -4.]
     [  0.   0.   0.]
     [  4.   4.   4.]]
    """
    # Invert the kernel
    ker_inverted = kernel[::-1, ::-1]

    # Apply convolution using cv2.filter2D
    return cv2.filter2D(image,  -1, kernel=ker_inverted, borderType=cv2.BORDER_REPLICATE)


def ha_green_interpolation(mosaic: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Hamilton-Adams green channel interpolation.
    This function implements Algorithm 1 for green channel interpolation.

    Parameters
    ----------
    mosaic : np.ndarray
        The Bayer mosaic image.
    mask : np.ndarray
        The corresponding mask for the mosaic.

    Returns
    -------
    np.ndarray
        The interpolated green channel.
    """
    Kh = np.array([[1/2, 0, 1/2]], dtype=np.float32)
    Kv = Kh.T
    Deltah = np.array([[1, 0, -2, 0, 1]], dtype=int)
    Deltav = Deltah.T

    Diffh = np.array([[1, 0, -1]], dtype=int)
    Diffv = Diffh.T

    # Get raw CFA data
    rawq = np.sum(mosaic, axis=2)

    rawh = conv_optimized(rawq, Kh) - conv_optimized(rawq, Deltah / 4)
    rawv = conv_optimized(rawq, Kv) - conv_optimized(rawq, Deltav / 4)

    CLh = np.abs(conv_optimized(rawq, Diffh)) + np.abs(conv_optimized(rawq, Deltah))
    CLv = np.abs(conv_optimized(rawq, Diffv)) + np.abs(conv_optimized(rawq, Deltav))

    # Implement the logic to assign:
    #    rawh when CLv > CLh
    #    rawv when CLv < CLh
    #    (rawh+rawv)/2 otherwise
    CLlocation = np.sign(CLh - CLv)
    green = (1 + CLlocation) * rawv / 2 + (1 - CLlocation) * rawh / 2

    # Inverse mask
    imask = (mask == 0)
    # Apply mask to the interpolated green channel
    green = green * imask[:, :, 1] + rawq * mask[:, :, 1]

    return green


def ha_blue_interpolation(green: np.ndarray, mosaicB: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Hamilton-Adams blue channel processing.
    This function implements Algorithm 2 for blue channel interpolation.

    Parameters
    ----------
    green : np.ndarray
        The interpolated green channel.
    mosaicB : np.ndarray
        The blue channel of Bayer mosaic image).
    mask : np.ndarray
        The mask corresponding to the mosaic.

    Returns
    -------
    np.ndarray
        The interpolated blue channel.
    """
    maskGr, maskGb, maskR, _ = mask

    Kh = np.array([[1, 0, 1]], dtype=int)
    Kv = Kh.T
    Kp = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.int8)
    Kn = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=np.int8)

    Deltap = np.array([[1, 0, 0], [0, -2, 0], [0, 0, 1]], dtype=np.int8)
    Deltan = np.array([[0, 0, 1], [0, -2, 0], [1, 0, 0]], dtype=np.int8)

    Deltah = np.array([[1, -2, 1]], dtype=int)
    Deltav = Deltah.T

    Diffp = np.array([[-1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.int8)
    Diffn = np.array([[0, 0, -1], [0, 0, 0], [1, 0, 0]], dtype=np.int8)

    Bh = maskGb * (0.5 * conv_optimized(mosaicB, Kh) - 0.25 * conv_optimized(green, Deltah))
    Bv = maskGr * (0.5 * conv_optimized(mosaicB, Kv) - 0.25 * conv_optimized(green, Deltav))
    Bp = maskR  * (0.5 * conv_optimized(mosaicB, Kp) - 0.25 * conv_optimized(green, Deltap))
    Bn = maskR  * (0.5 * conv_optimized(mosaicB, Kn) - 0.25 * conv_optimized(green, Deltan))

    CLp = maskR * (np.abs(conv_optimized(mosaicB, Diffp)) + np.abs(conv_optimized(green, Deltap)))
    CLn = maskR * (np.abs(conv_optimized(mosaicB, Diffn)) + np.abs(conv_optimized(green, Deltan)))

    CLlocation = np.sign(CLp - CLn)
    blue = (1 + CLlocation) * Bn / 2 + (1 - CLlocation) * Bp / 2
    blue = blue + Bh + Bv + mosaicB

    return blue


def ha_red_interpolation(green: np.ndarray, mosaicR: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Hamilton-Adams red channel processing.
    This function implements Algorithm 2 for red channel interpolation.

    Parameters
    ----------
    green : np.ndarray
        The interpolated green channel.
    mosaicR : np.ndarray
        The red channel of Bayer mosaic image.
    mask : np.ndarray
        The mask corresponding to the mosaic.

    Returns
    -------
    np.ndarray
        The interpolated red channel.
    """
    maskGr, maskGb, _, maskB = mask

    Kh = np.array([[1, 0, 1]], dtype=int)
    Kv = Kh.T
    Kp = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=int)
    Kn = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=int)

    Deltap = np.array([[1, 0, 0], [0, -2, 0], [0, 0, 1]], dtype=int)
    Deltan = np.array([[0, 0, 1], [0, -2, 0], [1, 0, 0]], dtype=int)

    Deltah = np.array([[1, -2, 1]], dtype=int)
    Deltav = Deltah.T

    Diffp = np.array([[-1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=int)
    Diffn = np.array([[0, 0, -1], [0, 0, 0], [1, 0, 0]], dtype=int)

    Rh = maskGr * (0.5 * conv_optimized(mosaicR, Kh) - 0.25 * conv_optimized(green, Deltah))
    Rv = maskGb * (0.5 * conv_optimized(mosaicR, Kv) - 0.25 * conv_optimized(green, Deltav))
    Rp = maskB  * (0.5 * conv_optimized(mosaicR, Kp) - 0.25 * conv_optimized(green, Deltap))
    Rn = maskB  * (0.5 * conv_optimized(mosaicR, Kn) - 0.25 * conv_optimized(green, Deltan))

    CLp = maskB * (np.abs(conv_optimized(mosaicR, Diffp)) + np.abs(conv_optimized(green, Deltap)))
    CLn = maskB * (np.abs(conv_optimized(mosaicR, Diffn)) + np.abs(conv_optimized(green, Deltan)))

    CLlocation = np.sign(CLp - CLn)
    red = (1 + CLlocation) * Rn / 2 + (1 - CLlocation) * Rp / 2
    red = red + Rh + Rv + mosaicR

    return red


def ha_demosaic(image: np.ndarray, pattern: Literal['RGGB', 'BGGR', 'GRBG', 'GBRG'] = 'RGGB') -> np.ndarray:
    """
    Dématriçe une image Bayer par la méthode de Hamilton-Adams.

    Parameters
    ----------
    image : np.ndarray
        Image Bayer 2D (mosaïque brute, un seul canal).
    pattern : Literal['RGGB', 'BGGR', 'GRBG', 'GBRG']
        Motif de Bayer. Par défaut ``'RGGB'``.

    Returns
    -------
    np.ndarray
        Image RGB démosaïquée, forme ``(H, W, 3)``.
    """
    # Masque RGB de la mosaïque
    mask_rgb = np.stack(get_mosaic_masks_RGB(pattern, image.shape), -1)

    # Génération de la mosaïque par canal
    mosaic = (image[..., None] * mask_rgb).astype(np.float32)

    # Interpolation du vert (Algorithme 1)
    green = ha_green_interpolation(mosaic, mask_rgb)
    del mask_rgb

    # Masque Gr/Gb/R/B de la mosaïque
    mask = get_mosaic_masks_GGRB(pattern, image.shape)

    # Interpolation rouge et bleue (Algorithme 2)
    red = ha_red_interpolation(green, mosaic[:, :, 0], mask)
    blue = ha_blue_interpolation(green, mosaic[:, :, 2], mask)

    rgb = np.zeros((*mosaic.shape[:2], 3), dtype=np.float32)
    rgb[:, :, 0] = red
    rgb[:, :, 1] = green
    rgb[:, :, 2] = blue

    return rgb