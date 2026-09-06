#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exemple d'utilisation — applique les 26 méthodes du catalogue sur une
image Bayer réelle (suite Kodak, voir tests/images/NOTICE.md), affiche
le temps d'exécution et un indicateur de fidélité (RMSE vs. la vérité
terrain) pour chacune. Script de démonstration, PAS un test automatisé
(voir tests/test_backends.py pour les smoke tests).

Usage
-----
    PYTHONPATH=src python examples/all_methods_overview.py
"""
__author__  = "Olivier Amram"
__version__ = "20260905"
__status__  = "Development"

# ── Imports standard ──────────────────────────────────────────────────────────
import sys
import os
from time import time

# ── Imports tiers ─────────────────────────────────────────────────────────────
import cv2
import numpy as np
from rich.console import Console
from rich.table import Table

# ── Imports eoqual.demosaicing ────────────────────────────────────────────────
# Ajoute src/ au path pour exécution sans installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eoqual.demosaicing.runner import demosaic
from eoqual.demosaicing.config import METHODS_CONFIGS


# ── Programme principal ───────────────────────────────────────────────────────
if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "tests", "images")

    ground_truth = cv2.imread(os.path.join(data_dir, "lighthouse.tif"), cv2.IMREAD_COLOR)
    ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    bayer = cv2.imread(os.path.join(data_dir, "lighthouse_bayer.tif"), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    bayer = bayer.astype(np.float32) / 255.0

    # Recadrage : les backends `edge_aware` / `malvar_bilateral` bouclent
    # pixel à pixel en Python pur — trop lents sur l'image complète.
    crop = (slice(0, 128), slice(0, 128))
    bayer = bayer[crop]
    ground_truth = ground_truth[crop]

    console = Console()
    table = Table(title="Comparaison des méthodes de démosaïçage — image Kodak 'lighthouse' (crop 128x128)")
    table.add_column("Méthode", style="bold cyan")
    table.add_column("Temps (ms)", justify="right")
    table.add_column("RMSE vs. vérité terrain", justify="right")

    for name, cfg in METHODS_CONFIGS.items():
        kwargs = {"pattern": "RGGB"} if len(cfg["patterns"]) > 1 else {}
        start = time()
        try:
            rgb = np.clip(demosaic(bayer, method=name, **kwargs), 0, 1)
            elapsed_ms = (time() - start) * 1000
            rmse = float(np.sqrt(np.mean((rgb - ground_truth) ** 2)))
            table.add_row(name, f"{elapsed_ms:.1f}", f"{rmse:.4f}")
        except Exception as exc:  # pragma: no cover — démonstration seulement
            table.add_row(name, "—", f"[red]erreur: {exc}[/red]")

    console.print(table)
