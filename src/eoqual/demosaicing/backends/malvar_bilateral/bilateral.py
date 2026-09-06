#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Malvar (vert) + filtre bilatéral guidé (rouge/bleu) — implémentation
propre du projet, aucun code tiers repris.

Le vert est estimé par le noyau de Malvar, He & Cutler (2004) (même
noyau que ``backends/malvar/he_cutler.py``), puis rouge et bleu sont
interpolés par un filtre bilatéral (pondération spatiale + proximité
radiométrique au vert) guidé par ce vert — variante plus coûteuse
(boucle Python par pixel manquant) mais plus robuste aux contours que la
simple convolution bilinéaire.

Limite : motif Bayer RGGB uniquement.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d

__all__ = ["malvar_bilateral_demosaic"]


def demosaic_green_malvar(
    image: np.ndarray
) -> np.ndarray:
    """
    Interpolation du canal vert via Malvar-He-Cutler.
    
    Parameters
    ----------
    image : np.ndarray
        Mosaïque Bayer RGGB (2D array).
        
    Returns
    -------
    np.ndarray
        Canal vert interpolé (2D array).
    """
    h_G = np.array([[0, 0, -1, 0, 0],
                    [0, 0, 2, 0, 0],
                    [-1, 2, 4, 2, -1],
                    [0, 0, 2, 0, 0],
                    [0, 0, -1, 0, 0]]) / 8.0

    G_mask = np.zeros_like(image, dtype=bool)
    G_mask[0::2, 1::2] = True
    G_mask[1::2, 0::2] = True

    G = convolve2d(image, h_G, mode='same', boundary='symm')
    G = np.where(G_mask, image, G)
    return G


def bilateral_filter_guide(
    img: np.ndarray,
    guide: np.ndarray,
    mask: np.ndarray,
    window_size: int=5,
    sigma_color: float=0.1,
    sigma_space: float=2.0,
    epsilon: float = 1e-6
) -> np.ndarray:
    """
    Filtre bilatéral guidé sur img avec guide.
    Interpole les pixels où mask == False.

    Parameters
    ----------
    img : np.ndarray
        Image à interpoler (2D array).
    guide : np.ndarray
        Image guide pour le filtrage (2D array).
    mask : np.ndarray
        Masque binaire indiquant les pixels connus (True = connu, False = inconnu).
    window_size : int, optional
        Taille de la fenêtre carrée pour le filtrage (doit être impair, par défaut 5).
    sigma_color : float, optional
        Écart-type pour la pondération des différences de couleur (par défaut 0.1).
    sigma_space : float, optional
        Écart-type pour la pondération des distances spatiales (par défaut 2.0).
    epsilon : float, optional
        Small value to avoid division by zero, default is 1e-6.

    Returns
    -------
    np.ndarray
        Image interpolée (2D array).
    """
    H, W = img.shape
    half = window_size // 2

    padded_img = np.pad(img, half, mode='reflect')
    padded_guide = np.pad(guide, half, mode='reflect')
    padded_mask = np.pad(mask, half, mode='reflect')

    img_out = img.copy()

    for y in range(H):
        for x in range(W):
            if mask[y, x]:
                continue  # pixel connu

            yp, xp = y + half, x + half

            weights = []
            values = []

            center_guide = padded_guide[yp, xp]

            for dy in range(-half, half + 1):
                for dx in range(-half, half + 1):
                    ny, nx = yp + dy, xp + dx

                    if not padded_mask[ny, nx]:
                        continue  # voisin inconnu, on ignore

                    spatial_dist = np.hypot(dy, dx)
                    color_dist = abs(padded_guide[ny, nx] - center_guide)

                    w_spatial = np.exp(-(spatial_dist**2) / (2 * sigma_space**2))
                    w_color = np.exp(-(color_dist**2) / (2 * sigma_color**2))
                    w = w_spatial * w_color

                    weights.append(w)
                    values.append(padded_img[ny, nx])

            if weights:
                weights = np.array(weights)
                values = np.array(values)
                img_out[y, x] = np.sum(weights * values) / (np.sum(weights) + epsilon)
            else:
                # fallback simple : moyenne locale
                neighbors = padded_img[yp - 1:yp + 2, xp - 1:xp + 2]
                img_out[y, x] = np.mean(neighbors)

    return img_out


def malvar_bilateral_demosaic(
    image: np.ndarray
) -> np.ndarray:
    """
    Démosaicing edge-aware : 
    - Estime vert avec Malvar-He-Cutler
    - Interpole R et B par filtre bilatéral guidé par vert

    Parameters
    ----------
    image : np.ndarray
        Mosaïque Bayer RGGB (2D array).

    Returns
    -------
    np.ndarray
        Image RGB démosaïquée (3D array).
    2D array with shape (H, W, 3).
    """
    # Masques
    R_mask = np.zeros_like(image, dtype=bool)
    G_mask = np.zeros_like(image, dtype=bool)
    B_mask = np.zeros_like(image, dtype=bool)

    R_mask[0::2, 0::2] = True
    G_mask[0::2, 1::2] = True
    G_mask[1::2, 0::2] = True
    B_mask[1::2, 1::2] = True

    # Interpolation canal vert
    G = demosaic_green_malvar(image)

    # Canaux R et B initiaux
    R = np.zeros_like(image, dtype=float)
    B = np.zeros_like(image, dtype=float)
    R[R_mask] = image[R_mask]
    B[B_mask] = image[B_mask]

    # Interpolation R et B avec filtre bilatéral guidé par G
    R_interp = bilateral_filter_guide(R, G, R_mask, window_size=7, sigma_color=0.05, sigma_space=3.0)
    B_interp = bilateral_filter_guide(B, G, B_mask, window_size=7, sigma_color=0.05, sigma_space=3.0)

    rgb = np.stack([R_interp, G, B_interp], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    return rgb

