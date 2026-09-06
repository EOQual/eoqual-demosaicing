#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zhang & Wu — Directional LMMSE Image Demosaicking (DLMMSE).

Référence :
    Zhang, L., Wu, X. (2005). "Color demosaicking via directional linear
    minimum mean square-error estimation." IEEE Transactions on Image
    Processing, 14(12), 2167-2178.

Réimplémentation vectorisée NumPy des équations publiées, à partir de la
description algorithmique du code de référence BSD de Pascal Getreuer
(IPOL, https://www.ipol.im/pub/art/2011/g_zwld/, `dmzhangwu.c`) — voir
``NOTICE.md`` dans ce dossier pour le détail
des deux points où cette réimplémentation diffère volontairement du code
source (gestion des bords, constante de stabilisation), et
``THIRD_PARTY_LICENSES.md`` (racine du projet).

Principe : le vert est interpolé horizontalement et verticalement par un
filtre directionnel court, puis les deux estimations sont fusionnées par
une pondération LMMSE (Linear Minimum Mean Square Error) fondée sur leur
variance locale respective — l'estimation la moins bruitée reçoit le
plus de poids. Rouge et bleu sont ensuite obtenus par propagation de
leur différence au vert (diagonale entre positions R et B, puis axiale
vers les positions vertes).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ...core.bayer import get_mosaic_masks_RGB

__all__ = ["dlmmse_demosaic"]

# Filtre d'interpolation directionnelle (5 points, symétrique)
_INTERP_KERNEL = np.array([-0.25, 0.5, 0.5, 0.5, -0.25], dtype=np.float64)
# Filtre de lissage du signal de différence (9 points, quasi gaussien)
_SMOOTH_KERNEL = np.array(
    [0.03125, 0.0703125, 0.1171875, 0.1796875, 0.203125,
     0.1796875, 0.1171875, 0.0703125, 0.03125],
    dtype=np.float64,
)
# Demi-largeur de la fenêtre d'estimation LMMSE (fenêtre totale = 2*M+1 = 9)
_M = 4
_N = 2 * _M + 1
# Constante de stabilisation (évite une division par une variance ~0 dans
# les zones plates) — voir NOTICE.md pour la reproportion depuis le
# DivEpsilon = 0.1/255² du code source (image en [0, 255]).
_DIV_EPSILON = 0.1


def _diagonal_average(a: np.ndarray) -> np.ndarray:
    """Moyenne des 4 voisins diagonaux (bord : symétrie whole-sample)."""
    padded = np.pad(a, 1, mode="reflect")
    return (padded[:-2, :-2] + padded[:-2, 2:] + padded[2:, :-2] + padded[2:, 2:]) / 4.0


def _axial_average(a: np.ndarray) -> np.ndarray:
    """Moyenne des 4 voisins axiaux (bord : symétrie whole-sample)."""
    padded = np.pad(a, 1, mode="reflect")
    return (padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]) / 4.0


def _window_stats(smooth: np.ndarray, diff: np.ndarray, axis: int):
    """
    Statistiques locales (fenêtre 2M+1, zéro-paddées) pour la fusion LMMSE.

    Returns
    -------
    mean, variance, residual_variance : np.ndarray
        Moyenne locale du signal lissé, sa variance (estimateur non
        biaisé), et la variance résiduelle (bruit) entre signal lissé et
        signal brut de différence.
    """
    mom1 = ndimage.uniform_filter1d(smooth, size=_N, axis=axis, mode="constant", cval=0.0) * _N
    sumsq = ndimage.uniform_filter1d(smooth**2, size=_N, axis=axis, mode="constant", cval=0.0) * _N
    residual = ndimage.uniform_filter1d((smooth - diff) ** 2, size=_N, axis=axis, mode="constant", cval=0.0)

    mean = mom1 / _N
    variance = sumsq / (2 * _M) - mom1**2 / (2 * _M * _N)
    residual_variance = residual + _DIV_EPSILON
    return mean, variance, residual_variance


def dlmmse_demosaic(image: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """
    Dématriçe une image Bayer par LMMSE directionnelle (Zhang & Wu).

    Parameters
    ----------
    image : np.ndarray
        Image Bayer 2D (mosaïque brute), valeurs dans ``[0, 1]``.
    pattern : str
        Motif de Bayer (``'RGGB'``, ``'BGGR'``, ``'GRBG'``, ``'GBRG'``).
        Par défaut ``'RGGB'``.

    Returns
    -------
    np.ndarray
        Image RGB démosaïquée, forme ``(H, W, 3)``.
    """
    image = np.asarray(image, dtype=np.float64)
    red_mask, green_mask, blue_mask = get_mosaic_masks_RGB(pattern, image.shape)
    green_mask = green_mask.astype(bool)
    red_mask = red_mask.astype(bool)
    blue_mask = blue_mask.astype(bool)

    # 1. Interpolation directionnelle (bord : symétrie whole-sample)
    filtered_h = ndimage.convolve1d(image, _INTERP_KERNEL, axis=1, mode="mirror")
    filtered_v = ndimage.convolve1d(image, _INTERP_KERNEL, axis=0, mode="mirror")

    # 2. Signal de différence locale (le signe distingue vert / non-vert)
    diff_h = np.where(green_mask, image - filtered_h, filtered_h - image)
    diff_v = np.where(green_mask, image - filtered_v, filtered_v - image)

    # 3. Lissage du signal de différence
    smooth_h = ndimage.convolve1d(diff_h, _SMOOTH_KERNEL, axis=1, mode="mirror")
    smooth_v = ndimage.convolve1d(diff_v, _SMOOTH_KERNEL, axis=0, mode="mirror")

    # 4. Fusion LMMSE des estimations horizontale et verticale
    mean_h, var_h, res_h = _window_stats(smooth_h, diff_h, axis=1)
    mean_v, var_v, res_v = _window_stats(smooth_v, diff_v, axis=0)

    est_h = mean_h + (var_h / (var_h + res_h)) * (diff_h - mean_h)
    conf_h = var_h - (var_h / (var_h + res_h)) * var_h + _DIV_EPSILON
    est_v = mean_v + (var_v / (var_v + res_v)) * (diff_v - mean_v)
    conf_v = var_v - (var_v / (var_v + res_v)) * var_v + _DIV_EPSILON

    green_fused = image + (conf_v * est_h + conf_h * est_v) / (conf_h + conf_v)
    green = np.where(green_mask, image, green_fused)

    # 5. Différences de couleur au vert, connues à leurs positions natives
    diff_gr = np.where(red_mask, green - image, 0.0)
    diff_gb = np.where(blue_mask, green - image, 0.0)

    # 6. Propagation diagonale (R <-> B, toujours diagonalement adjacents
    #    dans un motif de Bayer, quel que soit le motif précis)
    diff_gr = np.where(blue_mask, _diagonal_average(diff_gr), diff_gr)
    diff_gb = np.where(red_mask, _diagonal_average(diff_gb), diff_gb)

    # 7. Propagation axiale vers les positions vertes
    diff_gr = np.where(green_mask, _axial_average(diff_gr), diff_gr)
    diff_gb = np.where(green_mask, _axial_average(diff_gb), diff_gb)

    red = green - diff_gr
    blue = green - diff_gb

    return np.stack([red, green, blue], axis=-1)
