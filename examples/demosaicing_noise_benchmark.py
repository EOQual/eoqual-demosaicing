#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Démosaïqueurs face au bruit (REFERENCE_TECHNIQUE.md, §11.6) : temps de
calcul, transfert du bruit (κ_Y, κ_U, κ_V), qualité sans bruit, et
qualité après débruitage NL-Bayes pour un bruit de capteur

    var(L) = a + b·L      (L : luminance linéaire normalisée [0, 1])

avec stabilisation de variance (GAT) de paramètres (a, b) appliquée
**après** le démosaïçage, puis NL-Bayes avec σ = C (domaine stabilisé) ;
C balayé pour comparer l'optimum mesuré à la prédiction C = √κ_Y.

    python examples/demosaicing_noise_benchmark.py IMAGE.tif [--out DIR]
    python examples/demosaicing_noise_benchmark.py IMAGE.tif --plot-only   # figure depuis results.json

Même scène et mêmes dépendances que ``joint_denoising_study.py``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from joint_denoising_study import CROPS, mosaic_rggb, scores, srgb_to_linear

from eoqual.demosaicing.config import METHODS_CONFIGS
from eoqual.demosaicing.runner import demosaic

#: Même noyau qu'une autre méthode (κ identiques) : seule celle-ci est évaluée avec bruit.
DUPLICATES = {"opencv_bilinear": "bilinear", "colour_bilinear": "bilinear", "colour_malvar2004": "malvar"}
#: Trop lentes pour l'évaluation avec bruit (≥ 5 s par 512×512) : qualité sans bruit seulement.
SLOW = {"cdm", "edge_aware", "malvar_bilateral", "edge_aware_simplified"}
#: Trois niveaux de bruit : (a, b) = ((2/N)², 1/N), N photo-électrons à pleine échelle.
LEVELS = {100: ((2 / 100) ** 2, 1 / 100), 400: ((2 / 400) ** 2, 1 / 400), 1600: ((2 / 1600) ** 2, 1 / 1600)}
FACTORS = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75)
_T = np.array([[1, 1, 1], [1, -1, 0], [1, 1, -2]], float)
_T /= np.linalg.norm(_T, axis=1, keepdims=True)


def run(method, mosaic):
    """Démosaïçage RGGB ; les méthodes câblées en RGGB n'acceptent pas ``pattern``."""
    try:
        return demosaic(mosaic, method, pattern="RGGB").astype(np.float64)
    except TypeError:
        return demosaic(mosaic, method).astype(np.float64)


def noise_transfer(method, sigma=20 / 255, size=512, seed=0):
    """κ (variance après / avant) et corrélation au voisin, en Y, U, V, sur un champ uniforme bruité."""
    rng = np.random.default_rng(seed)
    base = np.full((size, size, 3), 0.5)
    e = run(method, mosaic_rggb(base) + rng.normal(0, sigma, (size, size))) - base
    yuv = e[8:-8, 8:-8] @ _T.T
    kappa = [float(yuv[..., i].var() / sigma**2) for i in range(3)]
    corr = [float(np.corrcoef(yuv[:, :-1, i].ravel(), yuv[:, 1:, i].ravel())[0, 1]) for i in range(3)]
    return kappa, corr


def timing(method, mosaic, repeat=2):
    best = np.inf
    for _ in range(repeat):
        t = time.perf_counter()
        run(method, mosaic)
        best = min(best, time.perf_counter() - t)
    return best


def noisy_mosaic(x, a, b, seed):
    """y = Poisson(L/b)·b + N(0, a) : var(y) = a + b·L."""
    rng = np.random.default_rng(seed)
    clean = mosaic_rggb(x)
    return rng.poisson(clean / b) * b + rng.normal(0, np.sqrt(a), clean.shape)


def denoise_gat(rgb, a, b, C):
    from eoqual.denoising.backends.nl_bayes.native import nl_bayes_denoise
    from eoqual.denoising.vst import generalized_anscombe, inverse_generalized_anscombe

    gain, rn = 1.0 / b, np.sqrt(a)          # var = a + b·L  ⇔  gain 1/b, bruit de lecture √a
    g = generalized_anscombe(rgb, gain, rn)
    d = nl_bayes_denoise(g, sigma=float(C), data_range=(float(g.min()), float(g.max())))
    return inverse_generalized_anscombe(d, gain, rn)


def rule_factor(kappa_y):
    """Facteur C conseillé sur σ après démosaïçage : √κ_Y + 0.2 (REFERENCE_TECHNIQUE.md §11.4)."""
    return float(np.sqrt(kappa_y)) + 0.2


def plot_rule(report, path):
    """Figure PNG : C optimal mesuré en fonction de √κ_Y (trois niveaux de bruit)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=130)
    markers = {"100": ("o", "fort (100 e⁻)"), "400": ("s", "moyen (400 e⁻)"), "1600": ("^", "faible (1600 e⁻)")}
    for n, (mk, label) in markers.items():
        xs, ys = [], []
        for m, e in report.items():
            if "noisy" in e:
                d = e["noisy"][n]
                xs.append(np.sqrt(e["kappa"][0]))
                ys.append(float(max(d, key=lambda C: d[C][0])))
        jitter = {"100": -0.012, "400": 0.0, "1600": 0.012}[n]
        ax.scatter(np.array(xs) + jitter, ys, marker=mk, label=label, alpha=0.75)
    for m in ("ha", "gbtf", "ari", "malvar", "ri"):
        if m in report:
            ax.annotate(m, (np.sqrt(report[m]["kappa"][0]), 0.38), ha="center", fontsize=8, color="dimgray")
    t = np.linspace(0.3, 1.7, 2)
    ax.plot(t, t, "k:", lw=1, label="C = √κ_Y (luminance seule)")
    ax.plot(t, t + 0.2, "k--", lw=1.2, label="C = √κ_Y + 0.2 (règle retenue)")
    ax.set_xlabel("√κ_Y (gain d'écart-type du bruit en luminance, champ uniforme)")
    ax.set_ylabel("C optimal mesuré (σ = C dans le domaine stabilisé)")
    ax.set_ylim(0.3, 1.9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Facteur sur σ après démosaïçage : mesure et prédiction", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--out", default="demosaicing_noise_benchmark")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        report = json.loads((Path(args.out) / "results.json").read_text())
        plot_rule(report, Path(args.out) / "facteur_C_vs_kappa.png")
        return
    import tifffile

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = tifffile.imread(args.image)[..., :3]
    big = mosaic_rggb(srgb_to_linear(src[:2048, :2048].astype(np.float64)))
    small = mosaic_rggb(srgb_to_linear(src[:512, :512].astype(np.float64)))
    zones = {n: srgb_to_linear(src[r : r + 512, c : c + 512].astype(np.float64)) for n, (r, c) in CROPS.items()}
    path = out_dir / "results.json"
    report = json.loads(path.read_text()) if path.exists() else {}   # reprise
    for m in METHODS_CONFIGS:
        if m in report:
            continue
        t0 = time.time()
        entry = {"seconds_512": timing(m, small, 1 if m in SLOW else 2)}
        entry["seconds_2048"] = entry["seconds_512"] * 16 if m in SLOW else timing(m, big, 1)
        entry["kappa"], entry["corr"] = noise_transfer(m)
        floors = [scores(x, run(m, mosaic_rggb(x))) for x in zones.values()]
        entry["floor"] = np.mean(floors, axis=0).tolist()
        if m not in SLOW and m not in DUPLICATES:
            entry["noisy"] = {}
            for n, (a, b) in LEVELS.items():
                per_c = {C: [] for C in FACTORS}
                for k, x in enumerate(zones.values()):
                    rgb = run(m, noisy_mosaic(x, a, b, seed=n + k))
                    for C in FACTORS:
                        per_c[C].append(scores(x, denoise_gat(rgb, a, b, C)))
                entry["noisy"][str(n)] = {str(C): np.mean(v, axis=0).tolist() for C, v in per_c.items()}
        report[m] = entry
        line = (f"{m:22s} {entry['seconds_2048']:7.2f} s/4 Mpx  κ_Y {entry['kappa'][0]:.2f}  "
                f"sans bruit {entry['floor'][0]:.2f} dB / ΔE {entry['floor'][1]:.2f}")
        if "noisy" in entry:
            for n, d in entry["noisy"].items():
                best = max(d, key=lambda C: d[C][0])
                line += f"  | {n} e⁻ : C*={best} {d[best][0]:.2f}/{d[best][1]:.2f}"
        print(line + f"  ({time.time() - t0:.0f} s)", flush=True)
        path.write_text(json.dumps(report, indent=1))
    plot_rule(report, out_dir / "facteur_C_vs_kappa.png")
    print("FIN")


if __name__ == "__main__":
    main()
