#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fonctions communes de manipulation des motifs de Bayer (masques par
canal, extraction/combinaison de plans, simulation de mosaïque),
partagées par tous les backends.

Toutes dérivent de la table unique de motifs d'``eoqual.core.bayer``
(``RGGB``, ``BGGR``, ``GRBG``, ``GBRG``, lus ligne par ligne depuis le
pixel en haut à gauche) : les quatre motifs sont cohérents d'une
fonction à l'autre par construction.
"""
from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

from eoqual.core.bayer import bayer_masks, bayer_offsets

__all__ = [
    'combine_channels',
    'extract_channel',
    'extract_subsampled_plane',
    'get_mosaic_masks_GGRB',
    'get_mosaic_masks_RGB',
    'extract_bayer_separated_planes',
    'rgb_to_bayer',
]


def combine_channels(ch1: np.ndarray, ch2: np.ndarray, ch3: np.ndarray) -> np.ndarray:
    """
    Combines three single-channel images into an RGB image.

    Parameters
    ----------
    ch1 : np.ndarray
        The first channel (e.g., red) as a NumPy array.
    ch2 : np.ndarray
        The second channel (e.g., green) as a NumPy array.
    ch3 : np.ndarray
        The third channel (e.g., blue) as a NumPy array.

    Returns
    -------
    np.ndarray
        The combined RGB image as a NumPy array.
    """
    return np.dstack((ch1, ch2, ch3))


def extract_channel(
    bayer_image: np.ndarray,
    channel: str,
    bayer_pattern: str = 'RGGB'
) -> np.ndarray:
    """
    Extracts a sub-plane from a Bayer image based on the given pattern.

    Parameters
    ----------
    bayer_image : np.ndarray
        Bayer image.
    channel : str
        Name of the channel to extract ('R', 'G', 'B').
    bayer_pattern : str, optional
        Bayer pattern ('RGGB', 'BGGR', 'GRBG', 'GBRG'). Default is 'RGGB'.

    Returns
    -------
    np.ndarray
        Sub-plane.
    """
    masks = bayer_masks(bayer_image.shape[:2], bayer_pattern)
    sub_plane = np.zeros_like(bayer_image, dtype=np.float32)
    if channel in ('R', 'G', 'B'):
        mask = masks[channel]
        sub_plane[mask] = bayer_image[mask].astype(np.float32)
    return sub_plane


def extract_subsampled_plane(
    bayer_image: np.ndarray,
    bayer_pattern: str = 'RGGB'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the R, G, and B channels from a Bayer image based on the given pattern.
    
    Parameters
    ---------- 
    bayer_image : np.ndarray
        Bayer image.
    bayer_pattern : str, optional
        Bayer pattern ('RGGB', 'BGGR', 'GRBG', 'GBRG'). Default is 'RGGB'.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        The extracted R, G, and B channels as NumPy arrays.
    """
    R = extract_channel(bayer_image, 'R', bayer_pattern)
    G = extract_channel(bayer_image, 'G', bayer_pattern)
    B = extract_channel(bayer_image, 'B', bayer_pattern)

    return R, G, B


def get_mosaic_masks_GGRB(
    pattern: Literal['GRBG', 'RGGB', 'GBRG', 'BGGR'],
    shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate the mosaic masks assuming a given pattern.

    Parameters
    ----------
    pattern : Literal['BGGR', 'GBRG', 'GRBG', 'RGGB']
        The Bayer pattern to recognize.
    shape : Tuple[int, int]
        The shape of the image (height, width).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the masks for Gr, Gb, R, and B channels, respectively.
    """
    masks = bayer_masks(shape, pattern)
    r_row = bayer_offsets(pattern)['R'][0]
    g1_row = bayer_offsets(pattern)['G1'][0]
    # Gr : vert sur la ligne du rouge ; Gb : vert sur la ligne du bleu.
    green_on_red, green_on_blue = ('G1', 'G2') if g1_row == r_row else ('G2', 'G1')
    maskGr = masks[green_on_red].astype(np.float64)
    maskGb = masks[green_on_blue].astype(np.float64)
    maskR = masks['R'].astype(np.float64)
    maskB = masks['B'].astype(np.float64)
    return (maskGr, maskGb, maskR, maskB)


def get_mosaic_masks_RGB(
    pattern: Literal['GRBG', 'RGGB', 'GBRG', 'BGGR'],
    shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates masks for red, green, and blue channels based on a given Bayer pattern and image shape.

    Parameters
    ----------
    pattern : Literal['BGGR', 'GBRG', 'GRBG', 'RGGB']
        The Bayer pattern to recognize.
    shape : Tuple[int, int]
        The shape of the image (height, width).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the red, green, and blue channel masks as NumPy arrays.
    """
    masks = bayer_masks(shape, pattern)
    red_mask = masks['R'].astype(np.int64)
    blue_mask = masks['B'].astype(np.int64)
    green_mask = masks['G'].astype(np.uint8)
    return red_mask, green_mask, blue_mask


def extract_bayer_separated_planes(
    bayer: np.ndarray,
    pattern: Literal['RGGB', 'BGGR', 'GRBG', 'GBRG'] = 'RGGB'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts separated Bayer planes from a Bayer image based on the specified pattern.
   
    For the given Bayer pattern, this function creates:
    - A mosaic image with three channels, where each channel contains only its sampled Bayer positions (zeros elsewhere).
    - A mask indicating the sampling locations of each color in the Bayer pattern.


    Parameters
    ----------
    bayer : np.ndarray
        Bayer image (H, W).
    pattern : Literal['RGGB', 'BGGR', 'GRBG', 'GBRG'], optional
        Bayer pattern to use for extraction. Default is 'RGGB'.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the bayer plane image and the mask used to generate it.
    """
    masks = bayer_masks(bayer.shape[:2], pattern)
    mask: np.ndarray = np.stack([masks['R'], masks['G'], masks['B']], axis=-1).astype(np.float64)
    bayer_plane: np.ndarray = bayer[..., None] * mask
    return bayer_plane, mask


def rgb_to_bayer(
    rgb: np.ndarray,
    pattern: Literal['RGGB', 'BGGR', 'GRBG', 'GBRG'] = 'RGGB'
) -> np.ndarray:
    """
    Simulate a Bayer mosaic from an RGB image for any Bayer pattern.
    
    Parameters
    ----------
    rgb : ndarray
        Input RGB image (H, W, 3)
    pattern : str
        Bayer pattern ('RGGB', 'BGGR', 'GRBG', 'GBRG')
        
    Returns
    -------
    bayer : ndarray
        Single-channel Bayer image (H, W)
    """
    masks = bayer_masks(rgb.shape[:2], pattern)
    bayer = np.zeros(rgb.shape[:2], dtype=rgb.dtype)
    for idx, channel in enumerate(('R', 'G', 'B')):
        bayer[masks[channel]] = rgb[..., idx][masks[channel]]
    return bayer
