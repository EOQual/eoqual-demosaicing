#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fonctions communes de manipulation des motifs de Bayer (masques par
canal, extraction/combinaison de plans, simulation de mosaïque) —
implémentation propre du projet, partagée par tous les backends.
"""
from __future__ import annotations

from typing import List, Literal, Tuple

import numpy as np

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
    sub_plane = np.zeros_like(bayer_image, dtype=np.float32)

    if bayer_pattern == 'RGGB':
        if channel == 'R':
            sub_plane[0::2, 0::2] = bayer_image[0::2, 0::2].astype(np.float32)
        elif channel == 'G':
            sub_plane[1::2, 0::2] = bayer_image[1::2, 0::2].astype(np.float32)
            sub_plane[0::2, 1::2] = bayer_image[0::2, 1::2].astype(np.float32)
        elif channel == 'B':
            sub_plane[1::2, 1::2] = bayer_image[1::2, 1::2].astype(np.float32)
    elif bayer_pattern == 'BGGR':
        if channel == 'B':
            sub_plane[0::2, 0::2] = bayer_image[0::2, 0::2].astype(np.float32)
        elif channel == 'G':
            sub_plane[1::2, 0::2] = bayer_image[1::2, 0::2].astype(np.float32)
            sub_plane[0::2, 1::2] = bayer_image[0::2, 1::2].astype(np.float32)
        elif channel == 'R':
            sub_plane[1::2, 1::2] = bayer_image[1::2, 1::2].astype(np.float32)
    elif bayer_pattern == 'GRBG':
        if channel == 'G':
            sub_plane[0::2, 0::2] = bayer_image[0::2, 0::2].astype(np.float32)
            sub_plane[1::2, 1::2] = bayer_image[1::2, 1::2].astype(np.float32)
        elif channel == 'R':
            sub_plane[0::2, 1::2] = bayer_image[0::2, 1::2].astype(np.float32)
        elif channel == 'B':
            sub_plane[1::2, 0::2] = bayer_image[1::2, 0::2].astype(np.float32)
    elif bayer_pattern == 'GBRG':
        if channel == 'G':
            sub_plane[0::2, 1::2] = bayer_image[0::2, 1::2].astype(np.float32)
            sub_plane[1::2, 0::2] = bayer_image[1::2, 0::2].astype(np.float32)
        elif channel == 'B':
            sub_plane[0::2, 0::2] = bayer_image[0::2, 0::2].astype(np.float32)
        elif channel == 'R':
            sub_plane[1::2, 1::2] = bayer_image[1::2, 1::2].astype(np.float32)
    else:
        raise ValueError(f"Unknown Bayer pattern: {bayer_pattern}")

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
    size_rawq = shape
    maskGr: np.ndarray = np.zeros((size_rawq[0], size_rawq[1]))
    maskGb: np.ndarray = np.zeros((size_rawq[0], size_rawq[1]))
    maskR: np.ndarray  = np.zeros((size_rawq[0], size_rawq[1]))
    maskB: np.ndarray  = np.zeros((size_rawq[0], size_rawq[1]))

    if pattern == 'GRBG':
        maskGr[0::2, 0::2] = 1
        maskGb[1::2, 1::2] = 1
        maskR[0::2, 1::2]  = 1
        maskB[1::2, 0::2]  = 1
    elif pattern == 'RGGB':
        maskGr[0::2, 1::2] = 1
        maskGb[1::2, 0::2] = 1
        maskB[1::2, 1::2]  = 1
        maskR[0::2, 0::2]  = 1
    elif pattern == 'GBRG':
        maskGb[0::2, 0::2] = 1
        maskGr[1::2, 1::2] = 1
        maskR[1::2, 0::2]  = 1
        maskB[0::2, 1::2]  = 1
    elif pattern == 'BGGR':
        maskGb[0::2, 1::2] = 1
        maskGr[1::2, 0::2] = 1
        maskB[0::2, 0::2]  = 1
        maskR[1::2, 1::2]  = 1

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
    s = shape
    i1, i2, i3, i4 = (0, 0, 1, 1)
    if pattern == 'BGGR':
        i1, i2, i3, i4 = (1, 1, 0, 0)
    elif pattern == 'GBRG':
        i1, i2, i3, i4 = (1, 0, 0, 1)
    elif pattern == 'GRBG':
        i1, i2, i3, i4 = (0, 1, 1, 0)
    elif pattern == 'RGGB':
        i1, i2, i3, i4 = (0, 0, 1, 1)
    v1  = np.array([int(x % 2 == i1) for x in range(s[0])]).reshape(-1, 1)
    v1t = np.array([int(x % 2 == i2) for x in range(s[1])]).reshape(1, -1)
    v2  = np.array([int(x % 2 == i3) for x in range(s[0])]).reshape(-1, 1)
    v2t = np.array([int(x % 2 == i4) for x in range(s[1])]).reshape(1, -1)
    red_mask   = v1 * v1t
    blue_mask  = v2 * v2t
    green_mask = (np.ones(s) - red_mask - blue_mask).astype(np.uint8)
    
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
    # Initialize the mask and mosaic image 
    size_rgb: Tuple[int, int, int] = bayer.shape
    mask: np.ndarray = np.zeros((size_rgb[0], size_rgb[1], 3))

    # Create a mapping for the Bayer pattern
    # 'R' -> 0, 'G' -> 1, 'B' -> 2
    num: np.ndarray = np.zeros(len(pattern))
    pattern_list: List[str] = list(pattern.lower())
    p: int = pattern_list.index('r')
    num[p] = 0
    p: List[int] = [idx for idx, i in enumerate(pattern_list) if i == 'g'] # type: ignore
    num[p] = 1
    p: int = pattern_list.index('b')
    num[p] = 2

    # Generate mask
    mask[0::2, 0::2, int(num[0])] = 1
    mask[0::2, 1::2, int(num[1])] = 1
    mask[1::2, 0::2, int(num[2])] = 1
    mask[1::2, 1::2, int(num[3])] = 1

    # Generate bayer plane
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
    H, W, _ = rgb.shape
    bayer = np.zeros((H, W), dtype=rgb.dtype)
    
    # Map letters to RGB channel indices
    channel_map = {'r': 0, 'g': 1, 'b': 2}
    
    # Parse pattern into channel indices
    pattern_lower = pattern.lower()
    if len(pattern_lower) != 4 or any(c not in 'rgb' for c in pattern_lower):
        raise ValueError(f"Invalid Bayer pattern: {pattern}")
    idx = [channel_map[c] for c in pattern_lower]
    
    # Assign channels according to the 2x2 pattern
    bayer[0::2, 0::2] = rgb[0::2, 0::2, idx[0]]
    bayer[0::2, 1::2] = rgb[0::2, 1::2, idx[1]]
    bayer[1::2, 0::2] = rgb[1::2, 0::2, idx[2]]
    bayer[1::2, 1::2] = rgb[1::2, 1::2, idx[3]]
    
    return bayer

