"""Cohérence des fonctions de motifs de Bayer de ``core/bayer.py``."""
import numpy as np
import pytest

from eoqual.demosaicing.core.bayer import extract_channel, get_mosaic_masks_GGRB

PATTERNS = ["RGGB", "BGGR", "GRBG", "GBRG"]


@pytest.mark.parametrize("pattern", PATTERNS)
def test_pattern_layout_matches_name(pattern):
    """Le nom du motif se lit ligne par ligne dans le bloc 2x2 en haut à gauche."""
    mosaic = np.arange(1, 5, dtype=float).reshape(2, 2)  # positions 1..4
    for channel in "RGB":
        positions = np.flatnonzero(extract_channel(mosaic, channel, pattern))
        expected = [i for i, c in enumerate(pattern) if c == channel]
        assert positions.tolist() == expected


@pytest.mark.parametrize("pattern", PATTERNS)
def test_extract_channel_matches_masks(pattern):
    image = np.random.default_rng(0).uniform(1, 2, (6, 8))
    mask_gr, mask_gb, mask_r, mask_b = get_mosaic_masks_GGRB(pattern, image.shape)
    for channel, mask in (("R", mask_r), ("G", mask_gr + mask_gb), ("B", mask_b)):
        assert np.array_equal(extract_channel(image, channel, pattern) > 0, mask > 0)
