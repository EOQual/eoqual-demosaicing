#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mode opératoire « démosaïçage puis débruitage » sur données réelles
(REFERENCE_TECHNIQUE.md §11.11).

Pour une mosaïque brute et un modèle de bruit du capteur
``var = a + b·L`` (estimé en dehors), compare Hamilton-Adams (``ha``) et
GBTF (``gbtf``) suivis de NL-Bayes couleur, pour plusieurs valeurs du
facteur C (σ = C dans le domaine stabilisé ; règle ``C = √κ_Y + 0.2``) :

1. luminance linéaire normalisée ``L = (DN − offset) / (DN_max − offset)`` ;
2. démosaïçage ;
3. stabilisation de variance (GAT) avec ``(a, b)`` ;
4. NL-Bayes couleur, ``sigma = C`` ;
5. inverse exacte de la GAT.

Sans vérité terrain, les résultats sont jugés sur le **bruit de chrominance
résiduel** (écart-type de U et V dans des zones uniformes) et sur la
**préservation des détails** (énergie du gradient de Y dans des zones
texturées), plus des vignettes. En mode ``--simulate``, la mosaïque est
fabriquée à partir d'une image supposée sans bruit et le CPSNR et le ΔE2000
s'ajoutent.

Données réelles (mosaïque 2D en DN, .npy ou .tif ; a et b en DN) :

    python examples/real_data_protocol.py --raw mosaique.npy --offset 64 --dn-max 4095 \\
        --a 2.3 --b 0.41 --pattern RGGB --crop 1000 1000 1024 --out protocole/

Simulation (sans données ; 400 e⁻ à pleine échelle, 12 bits) :

    python examples/real_data_protocol.py --simulate capitole_2048_cog.tif --out protocole/

Zones imposées (coordonnées dans l'image recadrée, répétables) :

    ... --roi-flat 100 140 200 240 --roi-flat 300 340 50 90 --roi-texture 10 58 400 448

Dépendances (hors paquet) : eoqual-denoising, tifffile, Pillow ;
scikit-image et eoqual-metrics en mode simulation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from eoqual.demosaicing.runner import demosaic

#: Gain de variance du bruit en luminance (champ uniforme, §11.3).
KAPPA_Y = {"ha": 1.35, "gbtf": 2.41, "malvar": 2.05, "ari": 0.62, "mlri": 0.87, "ri": 0.79,
           "dlmmse": 2.58, "ahd": 1.62, "mrf": 1.55, "colour_menon2007": 2.50, "bilinear": 0.58}
#: Valeurs de C testées par défaut : σ du capteur (1.0), √κ_Y, règle √κ_Y + 0.2, au-delà.
DEFAULT_FACTORS = {"ha": (1.0, 1.16, 1.36, 1.5), "gbtf": (1.25, 1.55, 1.75, 2.0)}
_YUV = np.array([[1, 1, 1], [1, -1, 0], [1, 1, -2]], float)
_YUV /= np.linalg.norm(_YUV, axis=1, keepdims=True)


# --------------------------------------------------------------------------- unités
def rule_factor(method: str) -> float:
    """C conseillé : √κ_Y + 0.2 (REFERENCE_TECHNIQUE.md §11.4)."""
    return float(np.sqrt(KAPPA_Y[method])) + 0.2


def normalize_mosaic(raw: np.ndarray, offset: float, dn_max: float) -> np.ndarray:
    """Luminance linéaire normalisée L = (DN − offset) / (DN_max − offset)."""
    return (np.asarray(raw, dtype=np.float64) - offset) / (dn_max - offset)


def noise_model_to_normalized(a_dn: float, b_dn: float, offset: float, dn_max: float) -> tuple[float, float]:
    """
    (a, b) d'un modèle ``var(DN) = a_DN + b_DN · (DN − offset)`` vers la
    luminance normalisée : ``a = a_DN / S²``, ``b = b_DN / S``, ``S = DN_max − offset``.
    """
    s = dn_max - offset
    return a_dn / s**2, b_dn / s


# --------------------------------------------------------------------------- chaîne
def run_chain(L: np.ndarray, a: float, b: float, method: str, factors, pattern: str = "RGGB") -> dict:
    """Démosaïçage, GAT (a, b), NL-Bayes couleur σ = C, inverse : {C: image RGB linéaire}."""
    from eoqual.denoising.backends.nl_bayes.native import nl_bayes_denoise
    from eoqual.denoising.vst import generalized_anscombe, inverse_generalized_anscombe

    gain, read_noise = 1.0 / b, float(np.sqrt(a))
    rgb = demosaic(L, method, pattern=pattern).astype(np.float64)
    g = generalized_anscombe(rgb, gain, read_noise)
    lo, hi = float(g.min()), float(g.max())
    out = {"brut": rgb}
    for C in factors:
        d = nl_bayes_denoise(g, sigma=float(C), data_range=(lo, hi))
        out[float(C)] = inverse_generalized_anscombe(d, gain, read_noise)
    return out


# --------------------------------------------------------------------------- critères sans vérité terrain
def luminance(rgb):
    return rgb @ _YUV[0]


def auto_rois(reference_rgb: np.ndarray, block: int = 32, n_flat: int = 4, n_texture: int = 2):
    """
    Zones uniformes (énergie du gradient de Y la plus faible) et texturées (la
    plus forte), blocs ``block``×``block`` avec recouvrement de moitié, sans
    chevauchement entre zones retenues ; pixels saturés ou noirs exclus.
    """
    y = luminance(reference_rgb)
    gy, gx = np.gradient(y)
    energy = np.hypot(gx, gy)
    h, w = y.shape
    step = block // 2
    cands = []
    for r in range(step, h - block - step + 1, step):
        for c in range(step, w - block - step + 1, step):
            patch = y[r : r + block, c : c + block]
            if patch.min() <= 0.02 or patch.max() >= 0.98:
                continue
            cands.append((float(energy[r : r + block, c : c + block].mean()), (r, r + block, c, c + block)))
    if len(cands) < n_flat + n_texture:
        raise ValueError("Image trop petite pour choisir les zones : passer --roi-flat/--roi-texture ou agrandir --crop.")
    cands.sort()

    def pick(seq, n):
        chosen = []
        for _, roi in seq:
            if all(roi[1] <= q[0] or roi[0] >= q[1] or roi[3] <= q[2] or roi[2] >= q[3] for q in chosen):
                chosen.append(roi)
            if len(chosen) == n:
                break
        return chosen

    flat, texture = pick(cands, n_flat), pick(reversed(cands), n_texture)
    ratio = cands[0][0] / float(np.median([e for e, _ in cands]))
    if ratio > 0.5:
        print(f"Attention : pas de vraie zone uniforme (gradient minimal = {ratio:.0%} du médian) ; "
              "σ_U, σ_V mêlent bruit et texture — préciser --roi-flat.")
    return flat, texture


def _residual_std(channel: np.ndarray) -> float:
    """Écart-type après retrait du plan moyen (un dégradé d'éclairement n'est pas du bruit)."""
    from eoqual.core.detrend import fit_plane_residual

    yy, xx = np.indices(channel.shape)
    coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)
    return float(np.std(fit_plane_residual(channel.ravel(), coords)))


def chroma_noise(rgb: np.ndarray, rois) -> tuple[float, float]:
    """Écart-type moyen de U et V (base orthonormée, plan retiré) dans des zones uniformes."""
    vals = []
    for r0, r1, c0, c1 in rois:
        yuv = rgb[r0:r1, c0:c1] @ _YUV.T
        vals.append([_residual_std(yuv[..., 1]), _residual_std(yuv[..., 2])])
    return tuple(float(v) for v in np.mean(vals, axis=0))


def luma_noise(rgb: np.ndarray, rois) -> float:
    """Écart-type moyen de Y (plan retiré) dans des zones uniformes (bruit de luminance résiduel)."""
    return float(np.mean([_residual_std(luminance(rgb[r0:r1, c0:c1])) for r0, r1, c0, c1 in rois]))


def detail_energy(rgb: np.ndarray, rois) -> float:
    """Énergie moyenne du gradient de Y dans des zones texturées (détails conservés)."""
    out = []
    for r0, r1, c0, c1 in rois:
        gy, gx = np.gradient(luminance(rgb[r0:r1, c0:c1]))
        out.append(np.hypot(gx, gy).mean())
    return float(np.mean(out))


# --------------------------------------------------------------------------- affichage
def preview(rgb: np.ndarray, wb: np.ndarray | None = None) -> np.ndarray:
    """Aperçu sRGB 8 bits : balance des blancs (gris moyen) puis gamma sRGB. Pour l'œil seulement."""
    if wb is None:
        means = rgb.reshape(-1, 3).mean(axis=0)
        wb = means[1] / np.maximum(means, 1e-6)
    x = np.clip(rgb * wb, 0, 1)
    x = np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
    return (x * 255).round().astype(np.uint8)


def save_vignettes(results: dict, rois_flat, rois_texture, path: Path, zoom: int = 3):
    """Planche : une ligne par démosaïqueur, une colonne par C ; zone texturée puis zone uniforme."""
    from PIL import Image, ImageDraw

    from joint_denoising_study import _font

    first = next(iter(results.values()))["brut"]
    means = first.reshape(-1, 3).mean(axis=0)
    wb = means[1] / np.maximum(means, 1e-6)
    zones = [rois_texture[0], rois_flat[0]]
    side = (zones[0][1] - zones[0][0]) * zoom
    cols = max(len(v) for v in results.values())
    head, label_w = 22, 70
    canvas = Image.new("RGB", (label_w + cols * side, len(results) * (head + 2 * side)), "white")
    draw, font = ImageDraw.Draw(canvas), _font(12)
    for i, (method, imgs) in enumerate(results.items()):
        y0 = i * (head + 2 * side)
        draw.text((4, y0 + head + side - 8), method, fill="black", font=font)
        for j, (key, img) in enumerate(imgs.items()):
            x0 = label_w + j * side
            title = "sans débruitage" if key == "brut" else f"C = {key:g}" + (" (règle)" if abs(key - rule_factor(method)) < 0.01 else "")
            draw.text((x0 + 4, y0 + 4), title, fill="black", font=font)
            for k, (r0, r1, c0, c1) in enumerate(zones):
                tile = Image.fromarray(preview(img[r0:r1, c0:c1], wb)).resize((side, side), Image.NEAREST)
                canvas.paste(tile, (x0, y0 + head + k * side))
    canvas.save(path)


# --------------------------------------------------------------------------- simulation
def simulate_raw(image_path: str, electrons: float = 400.0, read_noise_e: float = 2.0, offset: float = 64.0,
                 dn_max: float = 4095.0, crop=(400, 1040, 512), pattern: str = "RGGB", seed: int = 0):
    """
    Mosaïque brute 12 bits simulée depuis une image supposée sans bruit :
    ``electrons`` à pleine échelle, bruit de lecture ``read_noise_e``. Renvoie
    (mosaïque en DN, vérité linéaire RGB, a_DN, b_DN).
    """
    import tifffile

    from joint_denoising_study import mosaic_rggb, srgb_to_linear

    if pattern != "RGGB":
        raise ValueError("La simulation ne produit qu'une mosaïque RGGB.")
    r, c, n = crop
    truth = srgb_to_linear(tifffile.imread(image_path)[r : r + n, c : c + n, :3].astype(np.float64))
    rng = np.random.default_rng(seed)
    s = dn_max - offset
    gain = electrons / s                                    # e⁻/DN
    e = rng.poisson(mosaic_rggb(truth) * electrons) + rng.normal(0, read_noise_e, (n, n))
    raw = e / gain + offset
    a_dn, b_dn = (read_noise_e / gain) ** 2, 1.0 / gain     # var(DN) = a_DN + b_DN·(DN − offset)
    return raw, truth, a_dn, b_dn


# --------------------------------------------------------------------------- programme
def run_protocol(L, a, b, pattern="RGGB", factors=None, truth=None, rois=None, out_dir=None):
    """Exécute le protocole ; renvoie (métriques, résultats, zones)."""
    factors = factors or DEFAULT_FACTORS
    results, timing = {}, {}
    for method, cs in factors.items():
        t0 = time.perf_counter()
        results[method] = run_chain(L, a, b, method, cs, pattern)
        timing[method] = time.perf_counter() - t0
    if rois is None:
        ref = results["ha"][rule_factor("ha")] if rule_factor("ha") in results.get("ha", {}) else \
            next(v for v in next(iter(results.values())).values())
        rois_flat, rois_texture = auto_rois(ref)
    else:
        rois_flat, rois_texture = rois
    metrics = {"a": a, "b": b, "rois_flat": rois_flat, "rois_texture": rois_texture, "seconds": timing,
               "factors": {m: list(map(float, cs)) for m, cs in factors.items()},
               "rule": {m: rule_factor(m) for m in factors}, "chains": {}}
    for method, imgs in results.items():
        ref_detail = detail_energy(imgs["brut"], rois_texture)
        for key, img in imgs.items():
            su, sv = chroma_noise(img, rois_flat)
            m = {"sigma_U": su, "sigma_V": sv, "sigma_Y": luma_noise(img, rois_flat),
                 "detail_ratio": detail_energy(img, rois_texture) / ref_detail}
            if truth is not None:
                from joint_denoising_study import scores

                m["cpsnr"], m["delta_e"] = scores(truth, img)
            metrics["chains"][f"{method}|{key}"] = m
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "protocol_results.json").write_text(json.dumps(metrics, indent=1))
        save_vignettes(results, rois_flat, rois_texture, out_dir / "protocol_vignettes.png")
    return metrics, results, (rois_flat, rois_texture)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--raw", help="mosaïque brute 2D en DN (.npy ou .tif)")
    src.add_argument("--simulate", metavar="IMAGE", help="image supposée sans bruit (simulation)")
    p.add_argument("--offset", type=float, default=0.0)
    p.add_argument("--dn-max", type=float, default=4095.0)
    p.add_argument("--a", type=float, help="plancher du modèle var = a + b·(DN − offset), en DN²")
    p.add_argument("--b", type=float, help="pente du modèle, en DN")
    p.add_argument("--pattern", default="RGGB")
    p.add_argument("--crop", type=int, nargs=3, metavar=("LIGNE", "COLONNE", "TAILLE"))
    p.add_argument("--electrons", type=float, default=400.0, help="simulation : e⁻ à pleine échelle")
    p.add_argument("--roi-flat", type=int, nargs=4, action="append", metavar=("L0", "L1", "C0", "C1"),
                   help="zone uniforme (répétable), coordonnées après --crop ; défaut : choix automatique")
    p.add_argument("--roi-texture", type=int, nargs=4, action="append", metavar=("L0", "L1", "C0", "C1"),
                   help="zone texturée (répétable)")
    p.add_argument("--out", default="real_data_protocol")
    args = p.parse_args()

    truth = None
    if args.simulate:
        crop = tuple(args.crop) if args.crop else (400, 1040, 512)
        raw, truth, a_dn, b_dn = simulate_raw(args.simulate, args.electrons, offset=64.0, dn_max=args.dn_max,
                                              crop=crop, pattern=args.pattern)
        offset = 64.0
    else:
        if args.a is None or args.b is None:
            p.error("--a et --b (modèle de bruit en DN) sont requis avec --raw")
        if args.raw.endswith(".npy"):
            raw = np.load(args.raw)
        else:
            import tifffile

            raw = tifffile.imread(args.raw)
        if args.crop:
            r, c, n = args.crop
            r, c = r - r % 2, c - c % 2                       # garder la phase du motif
            raw = raw[r : r + n, c : c + n]
        a_dn, b_dn, offset = args.a, args.b, args.offset
    L = normalize_mosaic(raw, offset, args.dn_max)
    a, b = noise_model_to_normalized(a_dn, b_dn, offset, args.dn_max)
    print(f"Modèle normalisé : a = {a:.3e}, b = {b:.3e}  (σ à mi-échelle = {np.sqrt(a + b * 0.5):.4f})")
    rois = None
    if args.roi_flat or args.roi_texture:
        if not (args.roi_flat and args.roi_texture):
            p.error("--roi-flat et --roi-texture vont ensemble")
        rois = ([tuple(r) for r in args.roi_flat], [tuple(r) for r in args.roi_texture])
    metrics, _, _ = run_protocol(L, a, b, args.pattern, truth=truth, rois=rois, out_dir=args.out)
    print(f"{'chaîne':16s} {'σ_U':>8s} {'σ_V':>8s} {'σ_Y':>8s} {'détails':>8s}" + ("   CPSNR    ΔE" if truth is not None else ""))
    for k, m in metrics["chains"].items():
        line = f"{k:16s} {m['sigma_U']:8.4f} {m['sigma_V']:8.4f} {m['sigma_Y']:8.4f} {m['detail_ratio']:8.2f}"
        if truth is not None:
            line += f"  {m['cpsnr']:6.2f}  {m['delta_e']:5.2f}"
        print(line)
    print(f"Temps : " + ", ".join(f"{k} {v:.1f} s" for k, v in metrics["seconds"].items()))
    print(f"Résultats : {Path(args.out) / 'protocol_results.json'}, vignettes : {Path(args.out) / 'protocol_vignettes.png'}")


if __name__ == "__main__":
    main()
