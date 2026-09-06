#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
MRF — diffusion edge-aware de la chrominance, inspirée des approches par
champ de Markov (implémentation propre du projet, aucun code tiers
repris).

Un démosaïçage MRF (Markov Random Field) au sens plein :
    - modélise la relation locale entre canaux (p.ex. lissage de la
      différence de couleur) ;
    - formule un coût / une énergie sur l'image ;
    - minimise (ou approxime) cette énergie en imposant la cohérence
      locale des différences chromatiques.

La vraie inférence MRF par belief propagation n'est pas praticable en
Python pur (trop lente) — cette implémentation en approxime l'esprit par
une formulation quadratique locale (les différences R-G et B-G varient
lentement), résolue par diffusion itérative edge-aware (poids réduit sur
les forts gradients) :

.. math::

    E(R, B) = \sum_{i,j} \bigl[(R_{ij} - G_{ij} - c_{R,ij})^2 + (B_{ij} - G_{ij} - c_{B,ij})^2\bigr] \\
            + \lambda \sum_{(i,j),(k,l) \in \text{neighbors}} w_{ij,kl} \bigl[(c_{R,ij} - c_{R,kl})^2 + (c_{B,ij} - c_{B,kl})^2\bigr]

où :math:`c_R = R - G` et :math:`c_B = B - G` sont les cartes de
chrominance, :math:`w_{ij,kl}` des poids edge-aware pénalisant le
lissage à travers les forts gradients du vert, et :math:`\lambda`
contrôle la force du terme de lissage. En pratique : le vert est
interpolé en premier (bilinéaire), puis les cartes de chrominance sont
lissées par points fixes successifs en imposant les échantillons connus
au capteur.

Limite : motif Bayer RGGB uniquement (``compute_masks``).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import convolve

__all__ = ["mrf_demosaic"]


def compute_masks(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construit les masques R, G, B d'un motif de Bayer RGGB.

    Parameters
    ----------
    shape : Tuple[int, int]
        Forme de l'image (hauteur, largeur).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Masques R, G, B.
    """
    r_mask = np.zeros(shape, dtype=np.float32)
    g_mask = np.zeros(shape, dtype=np.float32)
    b_mask = np.zeros(shape, dtype=np.float32)

    r_mask[0::2, 0::2] = 1
    g_mask[0::2, 1::2] = 1
    g_mask[1::2, 0::2] = 1
    b_mask[1::2, 1::2] = 1
    return r_mask, g_mask, b_mask


def interpolate_green_bilinear(image: np.ndarray, g_mask: np.ndarray) -> np.ndarray:
    """Interpolation bilinéaire du canal vert."""
    green_kernel = np.array([
        [0, 0.25, 0],
        [0.25, 0, 0.25],
        [0, 0.25, 0]
    ], dtype=np.float32)

    green = image * g_mask
    bilinear = convolve(green, green_kernel, mode='mirror')
    green += bilinear * (1 - g_mask)
    return np.clip(green, 0, 1)


def mrf_diffusion(
    chroma: np.ndarray,
    mask: np.ndarray,
    edge_weight: np.ndarray,
    num_iter: int = 20,
    lam: float = 0.2
) -> np.ndarray:
    """
    Diffusion edge-aware itérative d'une carte de chrominance.

    Parameters
    ----------
    chroma : np.ndarray
        Carte de chrominance initiale (p.ex. R - G).
    mask : np.ndarray
        Masque des échantillons connus (1 si mesuré, 0 sinon).
    edge_weight : np.ndarray
        Carte de poids edge-aware.
    num_iter : int
        Nombre d'itérations de diffusion. Par défaut ``20``.
    lam : float
        Poids de régularisation. Par défaut ``0.2``.

    Returns
    -------
    np.ndarray
        Carte de chrominance diffusée.
    """
    kernel = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ], dtype=np.float32)

    result = chroma.copy()
    for _ in range(num_iter):
        neighbor_sum = convolve(result * edge_weight, kernel, mode='mirror')
        weight_sum = convolve(edge_weight, kernel, mode='mirror')
        updated = (result * mask + lam * neighbor_sum / (weight_sum + 1e-4) * (1 - mask)) / (mask + lam * (1 - mask))
        result = np.clip(updated, -0.5, 0.5)  # Limite l'excursion de chrominance
    return result


def mrf_demosaic(image: np.ndarray) -> np.ndarray:
    """
    Dématriçe une mosaïque Bayer RGGB par diffusion edge-aware de la chrominance.

    Parameters
    ----------
    image : np.ndarray
        Mosaïque Bayer 2D, normalisée dans ``[0, 1]``.

    Returns
    -------
    np.ndarray
        Image RGB démosaïquée, forme ``(H, W, 3)``.
    """
    r_mask, g_mask, b_mask = compute_masks(image.shape)

    # 1. Estimation du vert
    green = interpolate_green_bilinear(image, g_mask)

    # 2. Poids edge-aware à partir des gradients du vert
    gx_kernel = np.array([
        [0, 0, 0],
        [-1, 0, 1],
        [0, 0, 0]
    ], dtype=np.float32)
    gy_kernel = gx_kernel.T
    grad_x = np.abs(convolve(green, gx_kernel, mode='mirror'))
    grad_y = np.abs(convolve(green, gy_kernel, mode='mirror'))
    grad_mag = grad_x + grad_y
    edge_weight = np.exp(-5 * grad_mag)

    # 3. Cartes de chrominance initiales
    red_init = image * r_mask
    blue_init = image * b_mask
    chrom_r = (red_init - green) * r_mask
    chrom_b = (blue_init - green) * b_mask

    # 4. Diffusion des cartes de chrominance
    chrom_r_smooth = mrf_diffusion(chrom_r, r_mask, edge_weight)
    chrom_b_smooth = mrf_diffusion(chrom_b, b_mask, edge_weight)

    # 5. Recombinaison
    red_full = green + chrom_r_smooth
    blue_full = green + chrom_b_smooth

    # Restauration des échantillons d'origine
    red_full = red_full * (1 - r_mask) + red_init
    blue_full = blue_full * (1 - b_mask) + blue_init

    rgb = np.stack([np.clip(red_full, 0, 1),
                     np.clip(green, 0, 1),
                     np.clip(blue_full, 0, 1)], axis=-1)
    return rgb
