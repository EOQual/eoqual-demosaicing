#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Étude « démosaïçage et débruitage » (REFERENCE_TECHNIQUE.md, chapitre 11).

Compare, sur une orthophoto supposée sans bruit, des chaînes
démosaïçage + débruitage appliquées à une mosaïque RGGB affectée d'un
bruit poisson-gaussien :

- Hamilton-Adams (HA) ou ARI puis NL-Bayes, avec le niveau de bruit σ₀ du
  capteur ou C·σ₀ (C = 1.5 : Jin, Facciolo & Morel 2020) ;
- HA puis stabilisation de variance (GAT) puis NL-Bayes ;
- débruitage de la mosaïque (4 sous-plans, domaine GAT) puis HA ;
- HA, NL-Bayes (σ₀ ou 1.5σ₀) ou DRUNet, puis médian sur R−G et B−G
  (Freeman 1988) ;
- HA puis DRUNet couleur (deepinv), σ₀ et 1.5σ₀ ;
- Plug-and-Play conjoint (HQS) avec NL-Bayes ou DRUNet comme a priori.

Scène : extrait de l'Orthophotoplan 2024 de Toulouse Métropole (5 cm,
Licence Ouverte v2.0 Etalab), linéarisé par l'inverse du gamma sRGB.
Scores : CPSNR et ΔE2000 moyen, calculés en sRGB (espace d'affichage).

    python examples/joint_denoising_study.py IMAGE.tif [--out DIR] [--fast]

Dépendances de l'étude (hors paquet) : eoqual-denoising, eoqual-metrics,
tifffile, scikit-image ; deepinv + torch pour les chaînes DRUNet
(sautées sinon). Les vignettes PNG sont écrites dans ``--out``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from eoqual.demosaicing.runner import demosaic

#: Zones 512×512 (ligne, colonne du coin haut gauche) de l'extrait 2048×2048.
CROPS = {
    "tuiles": (400, 1040),      # toits de tuiles, lucarnes
    "végétation": (0, 1520),    # arbres, voitures, ombres
    "pavés": (1200, 0),         # place, piétons, ombres portées
    "verrière": (680, 560),     # verrière, zinc, hautes fréquences
}
#: Photo-électrons à pleine échelle (bruit fort → faible) ; bruit de lecture.
FULL_SCALE_E = (100.0, 400.0, 1600.0)
READ_NOISE_E = 2.0
FACTORS = (1.0, 1.25, 1.5, 1.75, 2.0)
#: Démosaïqueurs comparés sans bruit (plancher de chaque méthode).
FLOOR_METHODS = ("ha", "ri", "mlri", "ari", "gbtf", "dlmmse", "malvar", "lslcd")


def srgb_to_linear(s):
    s = s / 255.0
    return np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(lin):
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


def mosaic_rggb(x):
    m = np.zeros(x.shape[:2])
    m[0::2, 0::2] = x[0::2, 0::2, 0]
    m[0::2, 1::2] = x[0::2, 1::2, 1]
    m[1::2, 0::2] = x[1::2, 0::2, 1]
    m[1::2, 1::2] = x[1::2, 1::2, 2]
    return m


def cfa_mask(shape):
    M = np.zeros(shape + (3,))
    M[0::2, 0::2, 0] = M[0::2, 1::2, 1] = M[1::2, 0::2, 1] = M[1::2, 1::2, 2] = 1.0
    return M


def dm(m, method="ha"):
    return demosaic(m, method, pattern="RGGB").astype(np.float64)


def ha(m):
    return dm(m, "ha")


def scores(gt_lin, out_lin):
    from skimage.color import deltaE_ciede2000, rgb2lab

    from eoqual.metrics.fr.signal import psnr

    g, o = linear_to_srgb(gt_lin), linear_to_srgb(out_lin)
    return float(psnr(g, o)), float(np.mean(deltaE_ciede2000(rgb2lab(g), rgb2lab(o))))


def chroma_median(rgb, size=3):
    """Médian sur les différences de couleur R−G et B−G (Freeman 1988)."""
    from scipy.ndimage import median_filter

    g = rgb[..., 1]
    r = g + median_filter(rgb[..., 0] - g, size=size)
    b = g + median_filter(rgb[..., 2] - g, size=size)
    return np.stack([r, g, b], axis=-1)


def demosaicked_noise_stats(sigma=20 / 255, size=512, seed=0):
    """Variance de Y, U, V (isométrie de R³) et corrélation au voisin du bruit démosaïqué par HA."""
    rng = np.random.default_rng(seed)
    base = np.full((size, size, 3), 0.5)
    noisy = mosaic_rggb(base) + rng.normal(0, sigma, (size, size))
    e = ha(noisy) - base
    T = np.array([[1, 1, 1], [1, -1, 0], [1, 1, -2]], float)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    yuv = e @ T.T
    out = {}
    for k, name in enumerate("YUV"):
        c = yuv[8:-8, 8:-8, k]
        out[name] = {
            "variance_ratio": float(c.var() / sigma**2),
            "corr_lag1": float(np.corrcoef(c[:, :-1].ravel(), c[:, 1:].ravel())[0, 1]),
        }
    return out


class Chains:
    def __init__(self):
        from eoqual.denoising.backends.nl_bayes.native import nl_bayes_denoise
        from eoqual.denoising.vst import generalized_anscombe, inverse_generalized_anscombe

        self.nlb = lambda u, s, rng=(0.0, 1.0): nl_bayes_denoise(u, sigma=float(s), data_range=rng)
        self.gat, self.igat = generalized_anscombe, inverse_generalized_anscombe
        self.drunet = None
        try:
            import deepinv as dinv
            import torch

            self._torch = torch
            self._dev = "mps" if torch.backends.mps.is_available() else "cpu"
            self._net = dinv.models.DRUNet(in_channels=3, out_channels=3, pretrained="download",
                                           device=self._dev).eval()
            self.drunet = self._drunet
        except Exception as exc:  # pragma: no cover - dépend de l'environnement
            print(f"DRUNet indisponible ({type(exc).__name__}) : chaînes DRUNet sautées")

    def _drunet(self, u, s):
        torch = self._torch
        with torch.no_grad():
            t = torch.from_numpy(np.ascontiguousarray(u.transpose(2, 0, 1)[None])).float().to(self._dev)
            return self._net(t, sigma=float(s)).cpu().numpy()[0].transpose(1, 2, 0).astype(np.float64)

    def pnp(self, den, y, var, init, sig0, lam, start, end, n_iter):
        """HQS conjoint : x-pas pixel par pixel (attache pondérée par 1/variance), z-pas = débruiteur."""
        M = cfa_mask(y.shape)
        Y = M * y[..., None]
        W = M / np.maximum(var, 1e-12)[..., None]
        z = init.copy()
        for sk in np.geomspace(start * sig0, end * sig0, n_iter):
            alpha = lam / sk**2
            z = den((W * Y + alpha * z) / (W + alpha), sk)
        return z


def run_case(ch, x, full_e, seed, keep_images=False):
    rng = np.random.default_rng(seed)
    clean = mosaic_rggb(x)
    z = rng.poisson(clean * full_e) + rng.normal(0, READ_NOISE_E, clean.shape)   # e⁻ (gain 1)
    y = z / full_e
    var = (np.clip(y, 0, None) * full_e + READ_NOISE_E**2) / full_e**2
    sig0 = float(np.sqrt(np.mean(clean * full_e + READ_NOISE_E**2)) / full_e)   # σ gaussien équivalent
    hy, ay = ha(y), dm(y, "ari")
    out = {"HA seul": hy, "HA sans bruit (plancher)": ha(clean), "ARI sans bruit (plancher)": dm(clean, "ari")}
    scan_nlb = {C: ch.nlb(hy, C * sig0) for C in FACTORS}
    scan_ari = {C: ch.nlb(ay, C * sig0) for C in FACTORS}
    out["HA → NLB σ₀"] = scan_nlb[1.0]
    out["HA → NLB 1.5σ₀"] = scan_nlb[1.5]
    hz = np.clip(ha(z), 0, None)
    g = ch.gat(hz, 1.0, READ_NOISE_E)
    out["HA → GAT → NLB 1.5"] = ch.igat(ch.nlb(g, 1.5, (g.min(), g.max())), 1.0, READ_NOISE_E) / full_e
    gm = ch.gat(z, 1.0, READ_NOISE_E)
    gd = np.empty_like(gm)
    for i in (0, 1):
        for j in (0, 1):
            gd[i::2, j::2] = ch.nlb(gm[i::2, j::2], 1.0, (gm.min(), gm.max()))
    out["GAT → NLB(mosaïque) → HA"] = ha(ch.igat(gd, 1.0, READ_NOISE_E) / full_e)
    out["HA → NLB σ₀ → médian chroma"] = chroma_median(scan_nlb[1.0])
    out["HA → NLB 1.5σ₀ → médian chroma"] = chroma_median(scan_nlb[1.5])
    out["ARI → NLB σ₀"] = scan_ari[1.0]
    out["ARI → NLB 1.5σ₀"] = scan_ari[1.5]
    out["PnP conjoint NLB"] = ch.pnp(ch.nlb, y, var, hy, sig0, lam=0.1, start=1.7, end=0.5, n_iter=8)
    scan_dru = {}
    if ch.drunet is not None:
        scan_dru = {C: ch.drunet(hy, C * sig0) for C in FACTORS}
        out["HA → DRUNet 1.5σ₀"] = scan_dru[1.5]
        out["HA → DRUNet 1.5σ₀ → médian chroma"] = chroma_median(scan_dru[1.5])
        out["ARI → DRUNet 1.5σ₀"] = ch.drunet(ay, 1.5 * sig0)
        out["PnP conjoint DRUNet"] = ch.pnp(ch.drunet, y, var, hy, sig0, lam=0.23, start=3.4, end=0.7, n_iter=12)
    res = {k: scores(x, v) for k, v in out.items()}
    res_scan = {"HA→NLB": {C: scores(x, v) for C, v in scan_nlb.items()},
                "ARI→NLB": {C: scores(x, v) for C, v in scan_ari.items()},
                "HA→DRUNet": {C: scores(x, v) for C, v in scan_dru.items()}}
    return sig0, res, res_scan, (out if keep_images else None)


def _font(size=13):
    """Police avec accents et symboles (σ, →) ; police par défaut sinon."""
    from PIL import ImageFont

    candidates = ["/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/Library/Fonts/Arial Unicode.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    try:
        import matplotlib

        candidates.append(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"))
    except ImportError:
        pass
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


VIGNETTE_ROWS = ("HA seul", "HA → NLB σ₀", "HA → NLB 1.5σ₀ → médian chroma", "ARI → NLB 1.5σ₀",
                 "HA → DRUNet 1.5σ₀ → médian chroma", "PnP conjoint DRUNet")


def save_vignettes(images, x, path, rows=VIGNETTE_ROWS, box=(96, 96, 160), scores_by_name=None):
    """Vignettes sRGB 160×160 agrandies ×2, avec titre et scores (CPSNR, ΔE00 de la zone entière)."""
    from PIL import Image, ImageDraw

    r0, c0, n = box
    tiles = [("Vérité terrain", x)] + [(k, images[k]) for k in rows if k in images]
    w, head = n * 2, 40
    canvas = Image.new("RGB", (w * len(tiles), w + head), "white")
    draw, font = ImageDraw.Draw(canvas), _font()
    for i, (name, img) in enumerate(tiles):
        crop = (linear_to_srgb(img[r0 : r0 + n, c0 : c0 + n]) * 255).round().astype(np.uint8)
        canvas.paste(Image.fromarray(crop).resize((w, w), Image.NEAREST), (i * w, head))
        draw.text((i * w + 4, 2), name, fill="black", font=font)
        if scores_by_name and name in scores_by_name:
            p, de = scores_by_name[name]
            draw.text((i * w + 4, 20), f"{p:.2f} dB · ΔE {de:.2f}", fill="dimgray", font=font)
    canvas.save(path)


def floors(src):
    """CPSNR / ΔE00 des démosaïqueurs sans bruit, avec et sans médian de chrominance (moyenne des zones)."""
    out = {}
    for m in FLOOR_METHODS:
        a, b = [], []
        for r, c in CROPS.values():
            x = srgb_to_linear(src[r : r + 512, c : c + 512].astype(np.float64))
            rgb = dm(mosaic_rggb(x), m)
            a.append(scores(x, rgb))
            b.append(scores(x, chroma_median(rgb)))
        out[m] = {"seul": np.mean(a, axis=0).tolist(), "médian chroma": np.mean(b, axis=0).tolist()}
        print(f"  {m:8s} sans bruit {out[m]['seul'][0]:6.2f} dB / ΔE {out[m]['seul'][1]:.2f}   "
              f"+ médian chroma {out[m]['médian chroma'][0]:6.2f} dB / ΔE {out[m]['médian chroma'][1]:.2f}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--out", default="joint_denoising_study")
    parser.add_argument("--fast", action="store_true", help="une zone, un niveau de bruit")
    args = parser.parse_args()
    import tifffile

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = tifffile.imread(args.image)[..., :3]
    ch = Chains()
    crops = dict(list(CROPS.items())[:1]) if args.fast else CROPS
    levels = FULL_SCALE_E[1:2] if args.fast else FULL_SCALE_E
    report = {"noise_stats": demosaicked_noise_stats(), "cases": []}
    print("Démosaïqueurs sans bruit :")
    report["floors"] = floors(src)
    for (name, (r, c)), full_e in [(cr, n) for cr in crops.items() for n in levels]:
        x = srgb_to_linear(src[r : r + 512, c : c + 512].astype(np.float64))
        t0 = time.time()
        keep = full_e == FULL_SCALE_E[0] or args.fast
        sig0, res, scan, imgs = run_case(ch, x, full_e, seed=int(full_e), keep_images=keep)
        report["cases"].append({"crop": name, "full_e": full_e, "sigma0": sig0, "chains": res,
                                "scan": {k: {str(C): v for C, v in d.items()} for k, d in scan.items()}})
        if imgs is not None:
            save_vignettes(imgs, x, out_dir / f"vignettes_{name}_{int(full_e)}e.png", scores_by_name=res)
        print(f"{name:10s} {int(full_e):5d} e⁻  σ₀ = {sig0:.4f}  ({time.time() - t0:.0f} s)", flush=True)
        for k, (p, de) in res.items():
            print(f"    {k:34s} CPSNR {p:6.2f} dB   ΔE00 {de:5.2f}", flush=True)
    (out_dir / "results.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"Résultats : {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
