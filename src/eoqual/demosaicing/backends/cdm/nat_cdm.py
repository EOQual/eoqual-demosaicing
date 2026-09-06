#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zhang, Wu, Buades & Li — Color Demosaicking by Local Directional
Interpolation and Nonlocal Adaptive Thresholding (LDI-NAT / "CDM").

Référence :
    Zhang, L., Wu, X., Buades, A., Li, X. (2011). "Color demosaicking by
    local directional interpolation and nonlocal adaptive thresholding."
    Journal of Electronic Imaging, 20(2), 023016.
    https://www4.comp.polyu.edu.hk/~cslzhang/paper/NAT_CDM_JEI.pdf

Réimplémentation depuis les équations et paramètres publiés dans
l'article (§2, §3.2) — aucun code tiers repris (le code MATLAB de
référence des auteurs n'est distribué sans licence identifiée, voir
``THIRD_PARTY_LICENSES.md``).

Principe (article, Fig. 1) :

1. **LDI** (Local Directional Interpolation) du canal vert : à chaque
   position rouge/bleue, la différence de couleur vert-rouge (ou
   vert-bleu) est estimée dans 4 directions (nord/sud/ouest/est) sur une
   fenêtre locale compacte, puis fusionnée par une pondération inversement
   proportionnelle au gradient directionnel (Eq. 2-1 à 2-6).
2. **NAT** (Nonlocal Adaptive Thresholding) du canal vert : chaque
   position interpolée est raffinée en recherchant les patchs les plus
   similaires dans une large fenêtre, puis en seuillant doucement leur
   projection sur la base de leurs propres vecteurs propres (PCA locale)
   — une alternative structurelle au filtrage par moyenne non locale
   (Eq. 2-7 à 2-13).
3. **LDI** de rouge/bleu à l'aide du vert reconstruit : interpolation
   diagonale aux positions de l'autre couleur (Eq. 2-14/2-15/2-16), puis
   complétion axiale aux positions vertes (Eq. 2-17/2-18) — cette
   dernière étape distingue les deux sous-types de pixel vert du motif
   de Bayer (``Gr``/``Gb``, cf. ``core.bayer.get_mosaic_masks_GGRB``),
   dont l'axe « couleur cible native » est respectivement les lignes ou
   les colonnes ; la même formule est appliquée à l'image transposée
   pour le second sous-type plutôt que d'écrire deux variantes.
4. **NAT** de rouge et bleu (§2.5).

Limite majeure : la recherche de patchs similaires et la SVD par pixel
(complexité O(M³) par pixel, M=25) rendent cette méthode nettement plus
coûteuse que les autres du catalogue — voir ``README.md``. Non adaptée
aux grandes images sans optimisation supplémentaire.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage

from ...core.bayer import get_mosaic_masks_GGRB, get_mosaic_masks_RGB

__all__ = ["cdm_demosaic"]

# Paramètres de l'article (§3.2) pour LDI-NAT
_PATCH_RADIUS = 2       # patchs 5x5
_SEARCH_RADIUS = 15     # fenêtre de recherche 31x31
_N_SIMILAR = 100        # nombre de patchs similaires retenus
_THRESHOLD_C = 0.03     # t = c * g_Y
_EPS = 1e-6


def _shift(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """``_shift(arr, dy, dx)[y, x] == arr[y + dy, x + dx]`` (bord : symétrie whole-sample)."""
    pad_y, pad_x = abs(dy), abs(dx)
    padded = np.pad(arr, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    h, w = arr.shape
    y0, x0 = pad_y + dy, pad_x + dx
    return padded[y0:y0 + h, x0:x0 + w]


# ---------------------------------------------------------------------------
# Étape 1-2 : vert par LDI puis NAT
# ---------------------------------------------------------------------------

def _ldi_green(mosaic: np.ndarray, nongreen_mask: np.ndarray) -> np.ndarray:
    """LDI du canal vert (Eq. 2-1 à 2-6), généralisé à tout motif de Bayer."""
    d_n = _shift(mosaic, -1, 0) - (mosaic + _shift(mosaic, -2, 0)) / 2
    d_s = _shift(mosaic, 1, 0) - (mosaic + _shift(mosaic, 2, 0)) / 2
    d_w = _shift(mosaic, 0, -1) - (mosaic + _shift(mosaic, 0, -2)) / 2
    d_e = _shift(mosaic, 0, 1) - (mosaic + _shift(mosaic, 0, 2)) / 2

    g_vert = np.abs(_shift(mosaic, -1, 0) - _shift(mosaic, 1, 0))
    g_horiz = np.abs(_shift(mosaic, 0, -1) - _shift(mosaic, 0, 1))

    grad_n = g_vert + np.abs(mosaic - _shift(mosaic, -2, 0)) \
        + 0.5 * np.abs(_shift(mosaic, 0, -1) - _shift(mosaic, -2, -1)) \
        + 0.5 * np.abs(_shift(mosaic, 0, 1) - _shift(mosaic, -2, 1)) + _EPS
    grad_s = g_vert + np.abs(mosaic - _shift(mosaic, 2, 0)) \
        + 0.5 * np.abs(_shift(mosaic, 0, -1) - _shift(mosaic, 2, -1)) \
        + 0.5 * np.abs(_shift(mosaic, 0, 1) - _shift(mosaic, 2, 1)) + _EPS
    grad_w = g_horiz + np.abs(mosaic - _shift(mosaic, 0, -2)) \
        + 0.5 * np.abs(_shift(mosaic, -1, 0) - _shift(mosaic, -1, -2)) \
        + 0.5 * np.abs(_shift(mosaic, 1, 0) - _shift(mosaic, 1, -2)) + _EPS
    grad_e = g_horiz + np.abs(mosaic - _shift(mosaic, 0, 2)) \
        + 0.5 * np.abs(_shift(mosaic, -1, 0) - _shift(mosaic, -1, 2)) \
        + 0.5 * np.abs(_shift(mosaic, 1, 0) - _shift(mosaic, 1, 2)) + _EPS

    weights = [1.0 / grad_n, 1.0 / grad_s, 1.0 / grad_w, 1.0 / grad_e]
    total = sum(weights)
    fused_diff = sum(w * d for w, d in zip(weights, (d_n, d_s, d_w, d_e))) / total

    return np.where(nongreen_mask, mosaic + fused_diff, mosaic)


def _nat_enhance(channel: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    """
    Renforcement non local par seuillage adaptatif (NAT, Eq. 2-7 à 2-13).

    Seuls les pixels de ``target_mask`` (positions initialement
    manquantes dans ce canal) sont raffinés — les échantillons connus du
    capteur ne sont pas modifiés.
    """
    patch_size = 2 * _PATCH_RADIUS + 1
    pad = _PATCH_RADIUS + _SEARCH_RADIUS
    h, w = channel.shape

    offsets = [(dy, dx) for dy in range(-_SEARCH_RADIUS, _SEARCH_RADIUS + 1)
               for dx in range(-_SEARCH_RADIUS, _SEARCH_RADIUS + 1)]
    box = np.ones((patch_size, patch_size)) / (patch_size * patch_size)
    dist_stack = np.empty((len(offsets), h, w), dtype=np.float64)
    for k, (dy, dx) in enumerate(offsets):
        dist_stack[k] = ndimage.convolve(np.abs(channel - _shift(channel, dy, dx)), box, mode="mirror")

    padded = np.pad(channel, pad, mode="reflect")
    patches_all = sliding_window_view(padded, (patch_size, patch_size))
    patches_all = patches_all.reshape(*patches_all.shape[:2], patch_size * patch_size)

    dy_arr = np.array([o[0] for o in offsets])
    dx_arr = np.array([o[1] for o in offsets])

    result = channel.copy()
    ys, xs = np.nonzero(target_mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        dists = dist_stack[:, y, x]
        top = np.argpartition(dists, _N_SIMILAR - 1)[:_N_SIMILAR]

        rows = y + dy_arr[top] + pad
        cols = x + dx_arr[top] + pad
        # patches_all[i, j] est le patch centré sur l'image paddée en
        # (i + patch_radius, j + patch_radius) ; on veut celui centré en
        # (rows, cols), d'où l'indexation décalée de -patch_radius.
        patch_matrix = patches_all[rows - _PATCH_RADIUS, cols - _PATCH_RADIUS].T  # (M, N)

        row_means = patch_matrix.mean(axis=1, keepdims=True)
        centered = patch_matrix - row_means

        # PCA locale : Y = Phi Gamma, seuillage doux de Gamma (Eq. 2-12/2-13)
        phi, _, _ = np.linalg.svd(centered, full_matrices=False)
        gamma = phi.T @ centered

        g_y = np.mean(np.abs(np.diff(centered, axis=0))) + np.mean(np.abs(np.diff(centered, axis=1)))
        threshold = _THRESHOLD_C * g_y
        lam = np.sign(gamma) * np.maximum(np.abs(gamma) - threshold, 0.0)

        reconstructed = phi @ lam + row_means
        center_idx = (patch_size * patch_size) // 2
        result[y, x] = reconstructed[center_idx, 0]

    return result


# ---------------------------------------------------------------------------
# Étape 3 : rouge/bleu par LDI diagonal puis complétion axiale
# ---------------------------------------------------------------------------

def _ldi_cross(green: np.ndarray, mosaic: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    LDI diagonal d'un canal couleur aux positions de l'autre couleur
    (Eq. 2-14/2-15/2-16) — R et B sont toujours diagonalement adjacents
    dans un motif de Bayer, quel que soit le motif précis.
    """
    d_nw = _shift(mosaic, -1, -1) - _shift(green, -1, -1)
    d_ne = _shift(mosaic, -1, 1) - _shift(green, -1, 1)
    d_se = _shift(mosaic, 1, 1) - _shift(green, 1, 1)
    d_sw = _shift(mosaic, 1, -1) - _shift(green, 1, -1)

    diag1 = np.abs(_shift(mosaic, -1, -1) - _shift(mosaic, 1, 1))   # axe nw-se
    diag2 = np.abs(_shift(mosaic, -1, 1) - _shift(mosaic, 1, -1))   # axe ne-sw

    grad_nw = diag1 + np.abs(_shift(mosaic, -2, -2) - mosaic) + np.abs(_shift(green, -1, -1) - green) + _EPS
    grad_ne = diag2 + np.abs(_shift(mosaic, -2, 2) - mosaic) + np.abs(_shift(green, -1, 1) - green) + _EPS
    grad_se = diag1 + np.abs(_shift(mosaic, 2, 2) - mosaic) + np.abs(_shift(green, 1, 1) - green) + _EPS
    grad_sw = diag2 + np.abs(_shift(mosaic, 2, -2) - mosaic) + np.abs(_shift(green, 1, -1) - green) + _EPS

    weights = [1.0 / grad_nw, 1.0 / grad_ne, 1.0 / grad_se, 1.0 / grad_sw]
    total = sum(weights)
    fused_diff = sum(w * d for w, d in zip(weights, (d_nw, d_ne, d_se, d_sw))) / total

    return np.where(mask, green + fused_diff, 0.0)


def _axial_fill_rows(green: np.ndarray, target_partial: np.ndarray, other_mosaic: np.ndarray) -> np.ndarray:
    """
    Complétion axiale à l'axe « lignes » natif de la couleur cible
    (Eq. 2-17/2-18, dérivées pour la position G1 de la figure 3 de
    l'article — A = axe des lignes, B = axe des colonnes).
    """
    d_n = _shift(target_partial, -1, 0) - _shift(green, -1, 0)
    d_s = _shift(target_partial, 1, 0) - _shift(green, 1, 0)
    d_w = _shift(target_partial, 0, -1) - _shift(green, 0, -1)
    d_e = _shift(target_partial, 0, 1) - _shift(green, 0, 1)

    g_vert = np.abs(_shift(green, -2, 0) - green)
    g_horiz = np.abs(green - _shift(green, 0, -2))
    t_vert = np.abs(_shift(target_partial, -1, 0) - _shift(target_partial, 1, 0))
    o_horiz = np.abs(_shift(other_mosaic, 0, 1) - _shift(other_mosaic, 0, -1))

    grad_n = np.abs(_shift(green, -2, 0) - green) + t_vert \
        + 0.5 * np.abs(_shift(other_mosaic, -2, -1) - _shift(other_mosaic, 0, -1)) \
        + 0.5 * np.abs(_shift(other_mosaic, -2, 1) - _shift(other_mosaic, 0, 1)) + _EPS
    grad_s = np.abs(_shift(green, 2, 0) - green) + t_vert \
        + 0.5 * np.abs(_shift(other_mosaic, 2, -1) - _shift(other_mosaic, 0, -1)) \
        + 0.5 * np.abs(_shift(other_mosaic, 2, 1) - _shift(other_mosaic, 0, 1)) + _EPS
    grad_w = g_horiz + o_horiz \
        + 0.5 * np.abs(_shift(target_partial, -1, -2) - _shift(target_partial, -1, 0)) \
        + 0.5 * np.abs(_shift(target_partial, 1, -2) - _shift(target_partial, 1, 0)) + _EPS
    grad_e = np.abs(green - _shift(green, 0, 2)) + o_horiz \
        + 0.5 * np.abs(_shift(target_partial, -1, 0) - _shift(target_partial, -1, 2)) \
        + 0.5 * np.abs(_shift(target_partial, 1, 0) - _shift(target_partial, 1, 2)) + _EPS

    weights = [1.0 / grad_n, 1.0 / grad_s, 1.0 / grad_w, 1.0 / grad_e]
    total = sum(weights)
    fused_diff = sum(w * d for w, d in zip(weights, (d_n, d_s, d_w, d_e))) / total

    return green + fused_diff


def _axial_fill(green: np.ndarray, target_partial: np.ndarray, other_mosaic: np.ndarray,
                rows_native_mask: np.ndarray, cols_native_mask: np.ndarray) -> np.ndarray:
    """
    Complétion axiale de ``target_partial`` aux positions vertes,
    couvrant les deux sous-types de pixel vert (``Gr``/``Gb``) : la
    formule (dérivée pour l'axe des lignes) est appliquée telle quelle
    là où l'axe natif de la couleur cible est celui des lignes, et sur
    l'image transposée (puis re-transposée) là où c'est celui des
    colonnes — les deux motifs sont symétriques par transposition.
    """
    filled_rows_axis = _axial_fill_rows(green, target_partial, other_mosaic)
    filled_cols_axis = _axial_fill_rows(green.T, target_partial.T, other_mosaic.T).T

    result = target_partial.copy()
    result = np.where(rows_native_mask, filled_rows_axis, result)
    result = np.where(cols_native_mask, filled_cols_axis, result)
    return result


def cdm_demosaic(image: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """
    Dématriçe une image Bayer par LDI-NAT (Zhang, Wu, Buades & Li, 2011).

    Parameters
    ----------
    image : np.ndarray
        Image Bayer 2D (mosaïque brute).
    pattern : str
        Motif de Bayer (``'RGGB'``, ``'BGGR'``, ``'GRBG'``, ``'GBRG'``).
        Par défaut ``'RGGB'``.

    Returns
    -------
    np.ndarray
        Image RGB démosaïquée, forme ``(H, W, 3)``.

    Notes
    -----
    Coûteux : recherche de patchs similaires (fenêtre 31x31, patchs 5x5,
    100 candidats retenus) et SVD par pixel manquant. Voir
    ``REFERENCE_TECHNIQUE.md``.
    """
    mosaic = np.asarray(image, dtype=np.float64)
    red_mask, green_mask, blue_mask = get_mosaic_masks_RGB(pattern, mosaic.shape)
    green_mask = green_mask.astype(bool)
    red_mask = red_mask.astype(bool)
    blue_mask = blue_mask.astype(bool)
    nongreen_mask = ~green_mask

    mask_gr, mask_gb, _, _ = get_mosaic_masks_GGRB(pattern, mosaic.shape)
    mask_gr = mask_gr.astype(bool)
    mask_gb = mask_gb.astype(bool)

    # 1-2. Vert : LDI puis NAT (aux positions initialement manquantes)
    green = _ldi_green(mosaic, nongreen_mask)
    green = _nat_enhance(green, nongreen_mask)

    # 3. Rouge : LDI diagonal aux positions bleues, puis complétion axiale
    #    aux positions vertes (axe natif du rouge : colonnes en Gr, lignes en Gb).
    red_at_blue = _ldi_cross(green, mosaic, blue_mask)
    red_partial = np.where(red_mask, mosaic, red_at_blue)
    red = _axial_fill(green, red_partial, mosaic, rows_native_mask=mask_gb, cols_native_mask=mask_gr)
    red = np.where(green_mask, red, red_partial)

    # Bleu : LDI diagonal aux positions rouges, puis complétion axiale
    # (axe natif du bleu : lignes en Gr, colonnes en Gb).
    blue_at_red = _ldi_cross(green, mosaic, red_mask)
    blue_partial = np.where(blue_mask, mosaic, blue_at_red)
    blue = _axial_fill(green, blue_partial, mosaic, rows_native_mask=mask_gr, cols_native_mask=mask_gb)
    blue = np.where(green_mask, blue, blue_partial)

    # 4. Renforcement non local de rouge et bleu (§2.5)
    red = _nat_enhance(red, ~red_mask)
    blue = _nat_enhance(blue, ~blue_mask)

    return np.stack([red, green, blue], axis=-1)
