"""Protocole démosaïçage + débruitage (examples/real_data_protocol.py) sur une scène synthétique."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("eoqual.denoising")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import real_data_protocol as R  # noqa: E402


def _scene(n=192):
    y, x = np.mgrid[:n, :n] / n
    base = 0.35 + 0.25 * np.sin(6 * x) * np.cos(4 * y)
    rgb = np.stack([base, 0.9 * base + 0.05, 0.7 * base + 0.1], axis=-1)
    rgb[n // 2 :, n // 2 :] = [0.5, 0.45, 0.4]          # zone uniforme
    rgb[: n // 3, : n // 3] += 0.15 * (np.indices((n // 3, n // 3)).sum(0) % 4 < 2)[..., None]  # texture
    return np.clip(rgb, 0, 1)


def _mosaic(rgb):
    m = np.empty(rgb.shape[:2])
    m[0::2, 0::2], m[0::2, 1::2] = rgb[0::2, 0::2, 0], rgb[0::2, 1::2, 1]
    m[1::2, 0::2], m[1::2, 1::2] = rgb[1::2, 0::2, 1], rgb[1::2, 1::2, 2]
    return m


def test_units_roundtrip():
    offset, dn_max = 64.0, 4095.0
    a, b = R.noise_model_to_normalized(400.0, 10.0, offset, dn_max)
    s = dn_max - offset
    L = 0.3
    var_dn = 400.0 + 10.0 * (L * s)
    assert a + b * L == pytest.approx(var_dn / s**2)
    assert R.normalize_mosaic(np.array([offset, dn_max]), offset, dn_max).tolist() == [0.0, 1.0]
    assert R.rule_factor("ha") == pytest.approx(np.sqrt(1.35) + 0.2)
    assert R.rule_factor("gbtf") == pytest.approx(np.sqrt(2.41) + 0.2)


def test_protocol_reduces_chroma_noise():
    truth = _scene()
    a, b = (2 / 400) ** 2, 1 / 400
    rng = np.random.default_rng(0)
    L = rng.poisson(_mosaic(truth) / b) * b + rng.normal(0, np.sqrt(a), truth.shape[:2])
    factors = {"ha": (1.0, 1.36), "gbtf": (1.25, 1.75)}
    metrics, results, (flat, texture) = R.run_protocol(L, a, b, "RGGB", factors=factors)
    assert flat and texture
    for m, cs in factors.items():
        raw = metrics["chains"][f"{m}|brut"]
        lo, hi = (metrics["chains"][f"{m}|{float(C)}"] for C in cs)
        assert lo["sigma_U"] < raw["sigma_U"] and hi["sigma_U"] <= lo["sigma_U"] + 1e-4
        assert 0 < hi["detail_ratio"] <= 1.05
        assert results[m][float(cs[1])].shape == truth.shape
    assert set(metrics["rule"]) == {"ha", "gbtf"}


def test_auto_rois_finds_flat_and_textured_zones():
    flat, texture = R.auto_rois(_scene(), block=32)
    n = 192
    assert all(r0 >= n // 2 - 16 and c0 >= n // 2 - 16 for r0, _, c0, _ in flat[:1])   # zone uniforme
    assert R.luma_noise(_scene(), flat) < R.luma_noise(_scene(), texture)
