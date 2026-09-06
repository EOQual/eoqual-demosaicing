#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enrobage des méthodes de démosaïçage du paquet `colour-demosaicing
<https://github.com/colour-science/colour-demosaicing>`_ (licence
**BSD 3-Clause**, dépendance ``colour_demosaicing`` du socle du projet).
Aucun code tiers repris — appelle uniquement l'API publique du paquet.

Références :
    - ``bilinear`` : interpolation bilinéaire canal par canal.
    - ``malvar2004`` : Malvar, H. S., He, L.-W., Cutler, R. (2004).
      "High-quality linear interpolation for demosaicing of
      Bayer-patterned color images." IEEE ICASSP.
    - ``menon2007`` : Menon, D., Andriani, S., Calvagno, G. (2007).
      "Demosaicing With Directional Filtering and a posteriori
      Decision." IEEE Transactions on Image Processing, 16(1), 132-141
      (DDFAPD).
"""
from __future__ import annotations

from typing import Literal

import numpy as np

__all__ = ["colour_science_demosaic"]


def colour_science_demosaic(
    image: np.ndarray,
    pattern: str = 'RGGB',
    method: Literal['bilinear', 'malvar2004', 'menon2007'] = 'malvar2004',
) -> np.ndarray:
    """
    Dématriçe une image Bayer via `colour-demosaicing`.

    Parameters
    ----------
    image : np.ndarray
        Image Bayer 2D (mosaïque brute).
    pattern : str
        Motif de Bayer (``'RGGB'``, ``'BGGR'``, ``'GRBG'``, ``'GBRG'``).
        Par défaut ``'RGGB'``.
    method : Literal['bilinear', 'malvar2004', 'menon2007']
        Méthode à utiliser. Par défaut ``'malvar2004'``.

    Returns
    -------
    np.ndarray
        Image RGB démosaïquée, forme ``(H, W, 3)``.
    """
    from colour_demosaicing import (
        demosaicing_CFA_Bayer_bilinear,
        demosaicing_CFA_Bayer_Malvar2004,
        demosaicing_CFA_Bayer_Menon2007,
    )

    _FUNCS = {
        'bilinear': demosaicing_CFA_Bayer_bilinear,
        'malvar2004': demosaicing_CFA_Bayer_Malvar2004,
        'menon2007': demosaicing_CFA_Bayer_Menon2007,
    }
    if method not in _FUNCS:
        raise ValueError(f"Méthode '{method}' inconnue. Choix possibles : {list(_FUNCS)}")

    return _FUNCS[method](image, pattern=pattern.lower())
