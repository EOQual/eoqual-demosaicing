"""
eoqual.demosaicing — Collection de méthodes de démosaïçage d'images
issues d'un capteur à matrice de filtres colorés (motif de Bayer).

Public API : importer les méthodes directement depuis ce paquet.

Examples
--------
>>> import eoqual.demosaicing as d
>>> rgb = d.malvar_demosaic(bayer, pattern="RGGB")
>>> rgb = d.ari_demosaic(bayer, pattern="RGGB")

>>> from eoqual.demosaicing.runner import demosaic, list_methods
>>> list_methods()
>>> rgb = demosaic(bayer, method="malvar")

Toutes les méthodes attendent une image 2D (mosaïque Bayer brute, un
seul canal) et renvoient une image RGB ``(H, W, 3)``. Voir
``README.md`` pour le catalogue complet et ``THIRD_PARTY_LICENSES.md``
pour l'audit des licences du code adapté (``ha``, ``ari``, ``ri`` et
variantes, ``malvar``).
"""

__version__ = "0.1.0"
__title__ = "eoqual.demosaicing"
__description__ = "Collection of Bayer-pattern image demosaicing methods"
__url__ = "https://github.com/EOQual/eoqual-demosaicing"
__uri__ = __url__
__doc__ = __description__ + " <" + __uri__ + ">"
__author__ = "Olivier Amram"
__email__ = "olivier.amram@free.fr"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2026 Olivier Amram"

from .config import METHODS_CONFIGS  # noqa: F401

from .backends.interpolation.generic import interp_demosaic  # noqa: F401
from .backends.opencv.wrapper import opencv_demosaic  # noqa: F401
from .backends.colour_science.wrapper import colour_science_demosaic  # noqa: F401
from .backends.bilinear.channelwise import bilinear_demosaic  # noqa: F401
from .backends.green_edge_based.directional import green_edge_based_demosaic  # noqa: F401
from .backends.malvar.he_cutler import malvar_demosaic  # noqa: F401
from .backends.malvar_bilateral.bilateral import malvar_bilateral_demosaic  # noqa: F401
from .backends.edge_aware.full import edge_aware_demosaic  # noqa: F401
from .backends.edge_aware.simplified import edge_aware_simplified_demosaic  # noqa: F401
from .backends.ahd.homogeneity import ahd_demosaic  # noqa: F401
from .backends.mrf.diffusion import mrf_demosaic  # noqa: F401
from .backends.ha.ha import ha_demosaic  # noqa: F401
from .backends.ari.ari import ari_demosaic  # noqa: F401
from .backends.ri.ri import ri_demosaic  # noqa: F401
from .backends.dlmmse.zhang_wu import dlmmse_demosaic  # noqa: F401
from .backends.cdm.nat_cdm import cdm_demosaic  # noqa: F401
from .backends.lslcd.dubois import lslcd_demosaic  # noqa: F401

from .runner import demosaic, list_methods  # noqa: F401

__all__ = [
    "METHODS_CONFIGS",
    "interp_demosaic",
    "opencv_demosaic",
    "colour_science_demosaic",
    "bilinear_demosaic",
    "green_edge_based_demosaic",
    "malvar_demosaic",
    "malvar_bilateral_demosaic",
    "edge_aware_demosaic",
    "edge_aware_simplified_demosaic",
    "ahd_demosaic",
    "mrf_demosaic",
    "ha_demosaic",
    "ari_demosaic",
    "ri_demosaic",
    "dlmmse_demosaic",
    "cdm_demosaic",
    "lslcd_demosaic",
    "demosaic",
    "list_methods",
]
