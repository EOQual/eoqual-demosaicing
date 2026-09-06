"""
Point d'entrée unique pour appeler n'importe quelle méthode de
démosaïçage par son nom, et lister les méthodes disponibles.

Exemple
-------
>>> from eoqual.demosaicing.runner import demosaic, list_methods
>>> list_methods()
>>> rgb = demosaic(bayer, method="malvar")
"""

from __future__ import annotations

import numpy as np
from rich.console import Console
from rich.table import Table

from .config import METHODS_CONFIGS

__all__ = ["demosaic", "list_methods"]

# Import différé des fonctions publiques : "module:fonction" (+ kwargs
# fixes le cas échéant, pour les backends couvrant plusieurs méthodes du
# registre — ex. `interp_demosaic(method=...)`, `ri_demosaic(algorithm=...)`).
_DISPATCH = {
    "interp_bilinear": ("eoqual.demosaicing.backends.interpolation.generic", "interp_demosaic", {"method": "bilinear"}),
    "interp_bicubic": ("eoqual.demosaicing.backends.interpolation.generic", "interp_demosaic", {"method": "bicubic"}),
    "interp_spline": ("eoqual.demosaicing.backends.interpolation.generic", "interp_demosaic", {"method": "spline"}),
    "opencv_bilinear": ("eoqual.demosaicing.backends.opencv.wrapper", "opencv_demosaic", {"method": "bilinear"}),
    "opencv_vng": ("eoqual.demosaicing.backends.opencv.wrapper", "opencv_demosaic", {"method": "vng"}),
    "opencv_ea": ("eoqual.demosaicing.backends.opencv.wrapper", "opencv_demosaic", {"method": "ea"}),
    "colour_bilinear": ("eoqual.demosaicing.backends.colour_science.wrapper", "colour_science_demosaic", {"method": "bilinear"}),
    "colour_malvar2004": ("eoqual.demosaicing.backends.colour_science.wrapper", "colour_science_demosaic", {"method": "malvar2004"}),
    "colour_menon2007": ("eoqual.demosaicing.backends.colour_science.wrapper", "colour_science_demosaic", {"method": "menon2007"}),
    "bilinear": ("eoqual.demosaicing.backends.bilinear.amram", "bilinear_demosaic", {}),
    "green_edge_based": ("eoqual.demosaicing.backends.green_edge_based.amram", "green_edge_based_demosaic", {}),
    "malvar": ("eoqual.demosaicing.backends.malvar.he_cutler", "malvar_demosaic", {}),
    "malvar_bilateral": ("eoqual.demosaicing.backends.malvar_bilateral.amram", "malvar_bilateral_demosaic", {}),
    "edge_aware": ("eoqual.demosaicing.backends.edge_aware.full", "edge_aware_demosaic", {}),
    "edge_aware_simplified": ("eoqual.demosaicing.backends.edge_aware.simplified", "edge_aware_simplified_demosaic", {}),
    "ahd": ("eoqual.demosaicing.backends.ahd.amram", "ahd_demosaic", {}),
    "mrf": ("eoqual.demosaicing.backends.mrf.amram", "mrf_demosaic", {}),
    "ha": ("eoqual.demosaicing.backends.ha.ha", "ha_demosaic", {}),
    "ari": ("eoqual.demosaicing.backends.ari.ari", "ari_demosaic", {}),
    "ri": ("eoqual.demosaicing.backends.ri.ri", "ri_demosaic", {"algorithm": "RI"}),
    "gbtf": ("eoqual.demosaicing.backends.ri.ri", "ri_demosaic", {"algorithm": "GBTF"}),
    "mlri": ("eoqual.demosaicing.backends.ri.ri", "ri_demosaic", {"algorithm": "MLRI"}),
    "wmlri": ("eoqual.demosaicing.backends.ri.ri", "ri_demosaic", {"algorithm": "WMLRI"}),
    "dlmmse": ("eoqual.demosaicing.backends.dlmmse.zhang_wu", "dlmmse_demosaic", {}),
    "cdm": ("eoqual.demosaicing.backends.cdm.nat_cdm", "cdm_demosaic", {}),
    "lslcd": ("eoqual.demosaicing.backends.lslcd.dubois", "lslcd_demosaic", {}),
}


def _resolve(method: str):
    module_path, func_name, fixed_kwargs = _DISPATCH[method]
    module = __import__(module_path, fromlist=[func_name])
    func = getattr(module, func_name)
    if fixed_kwargs:
        import functools
        func = functools.partial(func, **fixed_kwargs)
    return func


def list_methods() -> None:
    """Affiche toutes les méthodes de démosaïçage disponibles."""
    console = Console()
    table = Table(title="Méthodes disponibles dans eoqual.demosaicing", header_style="bold magenta")
    table.add_column("Méthode", style="bold cyan", no_wrap=True)
    table.add_column("Motifs Bayer")
    table.add_column("Licence amont")
    table.add_column("Référence")

    for name, cfg in METHODS_CONFIGS.items():
        table.add_row(name, ", ".join(cfg["patterns"]), cfg["license"], cfg["reference"])

    console.print(table)


def demosaic(image: np.ndarray, method: str, **kwargs) -> np.ndarray:
    """
    Applique une méthode de démosaïçage par son nom.

    Parameters
    ----------
    image : np.ndarray
        Image Bayer 2D (mosaïque brute, un seul canal).
    method : str
        Nom de la méthode — une des clés de :data:`METHODS_CONFIGS`
        (voir :func:`list_methods`).
    **kwargs
        Transmis à l'implémentation choisie (voir sa docstring) —
        typiquement ``pattern`` (motif de Bayer).

    Returns
    -------
    np.ndarray
        Image RGB démosaïquée.

    Raises
    ------
    ValueError
        Si ``method`` n'est pas une méthode connue.
    """
    if method not in METHODS_CONFIGS:
        raise ValueError(
            f"Méthode '{method}' inconnue. Méthodes disponibles : "
            f"{list(METHODS_CONFIGS.keys())}"
        )

    func = _resolve(method)
    return func(image, **kwargs)
