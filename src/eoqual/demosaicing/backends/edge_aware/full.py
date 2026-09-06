#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interpolation edge-aware (implémentation propre du projet, aucun code
tiers repris).

Pour chaque pixel manquant, les voisins connus sont pondérés par leur
distance spatiale et leur proximité radiométrique au vert (guide), en
ne retenant que les voisins alignés avec la direction perpendiculaire
au gradient local du vert (i.e. le long d'un éventuel contour, jamais à
travers) — cf. ``compute_gradients``/``edge_aware_interpolate_channel``.

Coûteux (boucle Python par pixel manquant) : voir ``edge_aware_simplified.py``
pour une version vectorisée bien plus rapide, à qualité proche.

Limite : motif Bayer RGGB uniquement (``extract_subsampled_plane`` ne
prend pas de paramètre de motif).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import sobel

from ...core.bayer import extract_subsampled_plane

__all__ = ["edge_aware_demosaic"]


def compute_gradients(G: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the gradients of the green channel G using Sobel operator.

    Parameters
    ----------
    G : np.ndarray
        The green channel of the Bayer mosaic.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The gradients in the x and y directions.
    """
    # Calcul des gradients avec Sobel
    # Utilisation de mode 'reflect' pour éviter les artefacts aux bords
    Gx = sobel(G, axis=1, mode='reflect')
    Gy = sobel(G, axis=0, mode='reflect')
    return Gx, Gy


def edge_aware_interpolate_channel(
    channel: np.ndarray,
    G: np.ndarray,
    mask: np.ndarray, 
    window_size: int=5, 
    sigma_color: float=0.1,
    sigma_space: float=2.0,
) -> np.ndarray:
    """
    Edge-aware interpolation using gradient direction of G.
    This function fills in missing pixels in a channel based on the gradients of the green channel G.

    Parameters
    ----------
    channel : np.ndarray
        The channel to interpolate (R, G, or B).
    G : np.ndarray
        The green channel used as a guide for interpolation.
    mask : np.ndarray
        A binary mask indicating valid pixels in the channel.
    window_size : int, optional
        Size of the square window used for interpolation (default is 5).
    sigma_color : float, optional
        Standard deviation for color distance weighting (default is 0.1).
    sigma_space : float, optional
        Standard deviation for spatial distance weighting (default is 2.0).

    Returns
    -------
    np.ndarray
        The interpolated channel, with the same shape as `channel`.
    """
    H, W = channel.shape
    half = window_size // 2
    channel_out = channel.copy().astype(np.float64)
    Gx, Gy = compute_gradients(G)

    padded_channel = np.pad(channel, half, mode='reflect')
    padded_mask = np.pad(mask, half, mode='reflect')
    padded_G = np.pad(G, half, mode='reflect')
    padded_Gx = np.pad(Gx, half, mode='reflect')
    padded_Gy = np.pad(Gy, half, mode='reflect')

    for y in range(H):
        for x in range(W):
            if mask[y, x]:
                # Pixel connu, on garde
                continue

            # Coordonnées dans le patch
            yp, xp = y + half, x + half

            # Direction gradient au centre
            gx = padded_Gx[yp, xp]
            gy = padded_Gy[yp, xp]
            norm = np.hypot(gx, gy)
            if norm < 1e-5:
                # gradient nul, interpoler classiquement avec voisins
                neighbors_coords = [(dy, dx) for dy in range(-half, half+1) for dx in range(-half, half+1) if dy != 0 or dx != 0]
            else:
                # Direction perpendiculaire au gradient (bord)
                dx_perp = -gy / norm
                dy_perp = gx / norm

                neighbors_coords = []
                for dy in range(-half, half+1):
                    for dx in range(-half, half+1):
                        if dy == 0 and dx == 0:
                            continue
                        # vecteur pixel voisin
                        vx, vy = dx, dy
                        # angle cosinus entre vecteur (vx, vy) et direction perpendiculaire
                        cos_angle = (vx * dx_perp + vy * dy_perp) / (np.hypot(vx, vy) * 1.0)
                        # Garde voisins avec angle proche de 0 (dans la direction perpendiculaire au bord)
                        if abs(cos_angle) > 0.7:
                            neighbors_coords.append((dy, dx))

            weights = []
            values = []
            for dy, dx in neighbors_coords:
                ny, nx = yp + dy, xp + dx
                if padded_mask[ny, nx]:
                    spatial_dist = np.hypot(dy, dx)
                    color_dist = abs(padded_G[ny, nx] - padded_G[yp, xp])
                    w_spatial = np.exp(- (spatial_dist**2) / (2 * sigma_space**2))
                    w_color = np.exp(- (color_dist**2) / (2 * sigma_color**2))
                    w = w_spatial * w_color
                    weights.append(w)
                    values.append(padded_channel[ny, nx])

            if weights:
                weights = np.array(weights)
                values = np.array(values)
                channel_out[y, x] = np.sum(weights * values) / np.sum(weights)
            else:
                # fallback : moyenne simple des pixels connus voisins 3x3
                window_mask = padded_mask[yp-1:yp+2, xp-1:xp+2]
                window_vals = padded_channel[yp-1:yp+2, xp-1:xp+2]
                if np.any(window_mask):
                    channel_out[y, x] = np.mean(window_vals[window_mask])
                else:
                    channel_out[y, x] = 0

    return channel_out


def edge_aware_demosaic(
    image: np.ndarray,
    sigma_color: float = 0.1,
    sigma_space: float = 2.0
) -> np.ndarray:
    """
    Dématriçe une mosaïque Bayer RGGB par interpolation edge-aware.

    Parameters
    ----------
    image : np.ndarray
        Mosaïque Bayer RGGB (H, W).
    sigma_color : float
        Paramètre de lissage pour la pondération des couleurs. Par défaut ``0.1``.
    sigma_space : float
        Paramètre de lissage pour la pondération spatiale. Par défaut ``2.0``.

    Returns
    -------
    np.ndarray
        Image RGB démosaïquée, forme ``(H, W, 3)``, valeurs dans ``[0, 1]``.
    """
    R, G, B = extract_subsampled_plane(image)
    mask_R = (R != 0)
    mask_G = (G != 0)
    mask_B = (B != 0)

    G_full = edge_aware_interpolate_channel(G, G, mask_G, sigma_color=sigma_color, sigma_space=sigma_space)
    R_full = edge_aware_interpolate_channel(R, G_full, mask_R, sigma_color=sigma_color, sigma_space=sigma_space)
    B_full = edge_aware_interpolate_channel(B, G_full, mask_B, sigma_color=sigma_color, sigma_space=sigma_space)

    rgb = np.stack([R_full, G_full, B_full], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    return rgb

