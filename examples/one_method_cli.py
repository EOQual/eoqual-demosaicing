#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exemple d'utilisation (CLI) — applique une méthode de démosaïçage sur
une image Bayer. Script de démonstration, PAS un test automatisé (voir
tests/test_backends.py pour les smoke tests).

Usage
-----
    PYTHONPATH=src python examples/one_method_cli.py \\
        --image tests/images/lighthouse_bayer.tif --method malvar --pattern RGGB
"""
__author__  = "Olivier Amram"
__version__ = "20260905"
__status__  = "Development"

# ── Imports standard ──────────────────────────────────────────────────────────
import sys
import os
import argparse

# ── Imports tiers ─────────────────────────────────────────────────────────────
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ── Imports eoqual.demosaicing ────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eoqual.demosaicing.runner import demosaic, list_methods
from eoqual.demosaicing.config import METHODS_CONFIGS


# ── Programme principal ───────────────────────────────────────────────────────
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Démosaïçage d'une image Bayer par une méthode nommée")
    parser.add_argument(
        "--method",
        choices=list(METHODS_CONFIGS.keys()),
        required=True,
        help="Méthode à appliquer — voir list_methods()",
    )
    parser.add_argument("--image", required=True, help="Image Bayer à dématriçer (chemin)")
    parser.add_argument("--pattern", default="RGGB", choices=["RGGB", "BGGR", "GRBG", "GBRG"])
    parser.add_argument("--no-plot", action="store_true", help="Désactiver l'affichage")
    args = parser.parse_args()

    list_methods()

    cfg = METHODS_CONFIGS[args.method]

    bayer = cv2.imread(args.image, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if bayer is None:
        raise FileNotFoundError(f"Image introuvable : {args.image}")
    bayer = bayer.astype(np.float32) / np.iinfo(bayer.dtype).max if np.issubdtype(bayer.dtype, np.integer) else bayer.astype(np.float32)
    print(f"Image Bayer shape : {bayer.shape}  dtype : {bayer.dtype}")

    # Les méthodes RGGB uniquement n'exposent pas de paramètre `pattern`.
    kwargs = {"pattern": args.pattern} if len(cfg["patterns"]) > 1 else {}
    rgb = demosaic(bayer, method=args.method, **kwargs)

    print(f"\n{args.method} appliqué — sortie shape : {rgb.shape}\n")

    # ── Affichage ─────────────────────────────────────────────────────────────
    if not args.no_plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(bayer, cmap="gray")
        axes[0].set_title("Bayer (entrée)")
        axes[1].imshow(np.clip(rgb, 0, 1))
        axes[1].set_title(f"RGB démosaïqué — {args.method}")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()
