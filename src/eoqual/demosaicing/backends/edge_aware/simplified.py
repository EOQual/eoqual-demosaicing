#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interpolation edge-aware simplifiée (implémentation propre du projet,
aucun code tiers repris).

Version vectorisable de ``full.py`` : la fenêtre de voisinage est fixée
à 3×3 (au lieu d'une recherche de voisins alignés au contour), pondérée
uniquement par la proximité radiométrique au vert (guide) — pas de terme
spatial ni de sélection directionnelle. Beaucoup plus rapide, à qualité
proche pour des images peu texturées.
"""
from __future__ import annotations

import numpy as np

from ...core.bayer import extract_subsampled_plane

__all__ = ["edge_aware_simplified_demosaic"]


def edge_aware_interpolate(
    channel: np.ndarray,
    G: np.ndarray,
    mask: np.ndarray,
    sigma: float = 0.05
) -> np.ndarray:
    """
    Interpolation simple edge-aware d'un canal sparse guidée par G.

    Parameters
    ----------
    channel : np.ndarray
        Le canal à interpoler (R, G ou B).
    G : np.ndarray
        Le canal vert (G) utilisé comme guide pour l'interpolation.
    mask : np.ndarray
        Masque binaire indiquant les pixels valides dans le canal.
    sigma : float
        Paramètre de lissage pour la pondération des voisins. Par défaut ``0.05``.

    Returns
    -------
    np.ndarray
        Le canal interpolé, de la même forme que ``channel``.
    """
    H, W = channel.shape
    padded_channel = np.pad(channel, 1, mode='reflect')
    padded_mask = np.pad(mask.astype(bool), 1, mode='reflect')
    padded_G = np.pad(G, 1, mode='reflect')
    channel_full = np.zeros_like(channel, dtype=np.float64)

    for y in range(H):
        for x in range(W):
            mask_window = padded_mask[y:y+3, x:x+3].flatten()
            green_window = padded_G[y:y+3, x:x+3].flatten()
            channel_window = padded_channel[y:y+3, x:x+3].flatten()

            if mask_window[4]:
                channel_full[y, x] = channel_window[4]
            else:
                neighbors = channel_window[np.arange(9) != 4]
                neighbors_mask = mask_window[np.arange(9) != 4]
                green_center = green_window[4]
                green_neighbors = green_window[np.arange(9) != 4]

                weights = np.exp(-np.abs(green_neighbors - green_center) / sigma)
                weights *= neighbors_mask

                if weights.sum() > 0:
                    channel_full[y, x] = np.sum(weights * neighbors) / np.sum(weights)
                else:
                    channel_full[y, x] = 0.0

    return channel_full


def edge_aware_simplified_demosaic(
    image: np.ndarray,
    sigma: float = 0.05
) -> np.ndarray:
    """
    Dématriçe une mosaïque Bayer RGGB par interpolation edge-aware simplifiée.

    Parameters
    ----------
    image : np.ndarray
        Mosaïque Bayer RGGB (H, W).
    sigma : float
        Paramètre de lissage pour l'interpolation. Par défaut ``0.05``.

    Returns
    -------
    np.ndarray
        Image RGB démosaïquée, forme ``(H, W, 3)``, valeurs dans ``[0, 1]``.
    """
    R, G, B = extract_subsampled_plane(image, bayer_pattern='RGGB')
    mask_R = R > 0
    mask_G = G > 0
    mask_B = B > 0

    G_full = edge_aware_interpolate(G, G, mask_G, sigma)
    R_full = edge_aware_interpolate(R, G_full, mask_R, sigma)
    B_full = edge_aware_interpolate(B, G_full, mask_B, sigma)

    rgb = np.stack([R_full, G_full, B_full], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    return rgb
