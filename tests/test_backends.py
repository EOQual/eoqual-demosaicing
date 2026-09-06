#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smoke tests — pour chaque méthode, vérifie que l'appel sur une image
Bayer réelle ne lève pas d'exception et renvoie une image RGB de forme
cohérente. Ce ne sont PAS des tests de qualité numérique (pas
d'assertion sur la fidélité colorimétrique) — voir
REFERENCE_TECHNIQUE.md.
"""
import os

import cv2
import numpy as np
import pytest

from eoqual.demosaicing.config import METHODS_CONFIGS
from eoqual.demosaicing.runner import demosaic

_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")


@pytest.fixture
def bayer_image() -> np.ndarray:
    """
    Recadrage 64x64 (motif RGGB) de ``lighthouse_bayer.tif`` (suite Kodak,
    voir ``tests/images/NOTICE.md``), normalisé en float32 dans [0, 1].
    Recadrage volontairement petit : plusieurs backends (``edge_aware``,
    ``malvar_bilateral``) bouclent pixel à pixel en Python pur.
    """
    path = os.path.join(_IMAGES_DIR, "lighthouse_bayer.tif")
    image = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    assert image is not None, f"Image de test introuvable : {path}"
    return image[:64, :64].astype(np.float32) / 255.0


@pytest.mark.parametrize("method", list(METHODS_CONFIGS.keys()))
def test_method_smoke(bayer_image, method):
    """Chaque méthode du registre s'exécute et renvoie une image RGB valide."""
    cfg = METHODS_CONFIGS[method]
    kwargs = {"pattern": "RGGB"} if len(cfg["patterns"]) > 1 else {}

    result = demosaic(bayer_image, method=method, **kwargs)

    assert result.shape == (*bayer_image.shape, 3)
    assert np.all(np.isfinite(result))


def test_unknown_method_raises(bayer_image):
    with pytest.raises(ValueError):
        demosaic(bayer_image, method="not_a_real_method")


def test_list_methods_matches_registry():
    """Toutes les méthodes de METHODS_CONFIGS sont bien dispatchables."""
    from eoqual.demosaicing.runner import _DISPATCH

    assert set(METHODS_CONFIGS.keys()) == set(_DISPATCH.keys())
