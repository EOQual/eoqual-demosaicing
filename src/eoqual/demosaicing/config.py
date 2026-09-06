"""
Registre central des méthodes de démosaïçage disponibles dans
``eoqual.demosaicing``.

Chaque entrée décrit : les motifs de Bayer supportés par
l'implémentation, sa licence amont vérifiée, et une référence
bibliographique. Voir ``README.md`` (catalogue) et
``THIRD_PARTY_LICENSES.md`` (audit des licences) pour le détail.
"""

from collections import OrderedDict

METHODS_CONFIGS: OrderedDict = OrderedDict(
    {
        "interp_bilinear": {
            "patterns": ["RGGB", "BGGR", "GRBG", "GBRG"],
            "license": "MIT (auteur du dépôt)",
            "reference": "Interpolation générique canal par canal — noyau moyenneur",
        },
        "interp_bicubic": {
            "patterns": ["RGGB", "BGGR", "GRBG", "GBRG"],
            "license": "MIT (auteur du dépôt)",
            "reference": "Interpolation générique canal par canal — noyau gaussien séparable",
        },
        "interp_spline": {
            "patterns": ["RGGB", "BGGR", "GRBG", "GBRG"],
            "license": "MIT (auteur du dépôt)",
            "reference": "Interpolation générique canal par canal — spline cubique",
        },
        "opencv_bilinear": {
            "patterns": ["RGGB", "BGGR", "GRBG", "GBRG"],
            "license": "MIT (wrapper) — OpenCV Apache-2.0",
            "reference": "cv2.cvtColor(COLOR_BAYER_*2RGB)",
        },
        "opencv_vng": {
            "patterns": ["RGGB", "BGGR", "GRBG", "GBRG"],
            "license": "MIT (wrapper) — OpenCV Apache-2.0",
            "reference": "Variable Number of Gradients — cv2.cvtColor(..._VNG)",
        },
        "opencv_ea": {
            "patterns": ["RGGB", "BGGR", "GRBG", "GBRG"],
            "license": "MIT (wrapper) — OpenCV Apache-2.0",
            "reference": "Edge-Aware — cv2.cvtColor(..._EA)",
        },
        "colour_bilinear": {
            "patterns": ["RGGB", "BGGR", "GRBG", "GBRG"],
            "license": "MIT (wrapper) — colour-demosaicing BSD-3-Clause",
            "reference": "colour_demosaicing.demosaicing_CFA_Bayer_bilinear",
        },
        "colour_malvar2004": {
            "patterns": ["RGGB", "BGGR", "GRBG", "GBRG"],
            "license": "MIT (wrapper) — colour-demosaicing BSD-3-Clause",
            "reference": "Malvar, He & Cutler (2004), IEEE ICASSP",
        },
        "colour_menon2007": {
            "patterns": ["RGGB", "BGGR", "GRBG", "GBRG"],
            "license": "MIT (wrapper) — colour-demosaicing BSD-3-Clause",
            "reference": "Menon, Andriani & Calvagno (2007), IEEE TIP 16(1), 132-141 (DDFAPD)",
        },
        "bilinear": {
            "patterns": ["RGGB", "BGGR", "GRBG", "GBRG"],
            "license": "MIT (auteur du dépôt)",
            "reference": "Interpolation bilinéaire canal par canal",
        },
        "green_edge_based": {
            "patterns": ["RGGB", "BGGR", "GRBG", "GBRG"],
            "license": "MIT (auteur du dépôt)",
            "reference": "Interpolation basée sur les contours du canal vert",
        },
        "malvar": {
            "patterns": ["RGGB", "BGGR"],
            "license": "MIT (adapté de prysm, brandondube)",
            "reference": "Malvar, He & Cutler (2004), IEEE ICASSP",
        },
        "malvar_bilateral": {
            "patterns": ["RGGB"],
            "license": "MIT (auteur du dépôt)",
            "reference": "Malvar (vert) + filtre bilatéral guidé (rouge/bleu)",
        },
        "edge_aware": {
            "patterns": ["RGGB"],
            "license": "MIT (auteur du dépôt)",
            "reference": "Interpolation edge-aware (directionnelle, guidée par le vert)",
        },
        "edge_aware_simplified": {
            "patterns": ["RGGB"],
            "license": "MIT (auteur du dépôt)",
            "reference": "Interpolation edge-aware simplifiée (fenêtre 3x3)",
        },
        "ahd": {
            "patterns": ["RGGB"],
            "license": "MIT (auteur du dépôt)",
            "reference": "Inspiré de Hirakawa & Parks (2005), IEEE TIP 14(3), 360-369",
        },
        "mrf": {
            "patterns": ["RGGB"],
            "license": "MIT (auteur du dépôt)",
            "reference": "Diffusion edge-aware de la chrominance (inspiré des champs de Markov)",
        },
        "ha": {
            "patterns": ["RGGB", "BGGR", "GRBG", "GBRG"],
            "license": "BSD 2-Clause modifiée (Jin, Guo, Facciolo — adapté)",
            "reference": "Hamilton & Adams (1997), US Patent 5,629,734",
        },
        "ari": {
            "patterns": ["RGGB", "BGGR", "GRBG", "GBRG"],
            "license": "BSD 2-Clause modifiée (Jin, Guo, Facciolo — adapté)",
            "reference": "Monno et al. (2016), Sensors 17(12), 2787 (ARI)",
        },
        "ri": {
            "patterns": ["RGGB", "BGGR", "GRBG", "GBRG"],
            "license": "BSD 2-Clause modifiée (Jin, Guo, Facciolo — adapté)",
            "reference": "Monno et al. (2016), Sensors 17(12), 2787 (RI)",
        },
        "gbtf": {
            "patterns": ["RGGB", "BGGR", "GRBG", "GBRG"],
            "license": "BSD 2-Clause modifiée (Jin, Guo, Facciolo — adapté)",
            "reference": "Pekkucuksen & Altunbasak (2010), IEEE ICIP (GBTF)",
        },
        "mlri": {
            "patterns": ["RGGB", "BGGR", "GRBG", "GBRG"],
            "license": "BSD 2-Clause modifiée (Jin, Guo, Facciolo — adapté)",
            "reference": "Kiku et al. (2014), Minimized-Laplacian Residual Interpolation (MLRI)",
        },
        "wmlri": {
            "patterns": ["RGGB", "BGGR", "GRBG", "GBRG"],
            "license": "BSD 2-Clause modifiée (Jin, Guo, Facciolo — adapté)",
            "reference": "Kiku et al. (2016), variante pondérée de MLRI (WMLRI)",
        },
        "dlmmse": {
            "patterns": ["RGGB", "BGGR", "GRBG", "GBRG"],
            "license": "MIT (réimplémentation) — algorithme depuis code BSD (Getreuer)",
            "reference": "Zhang & Wu (2005), IEEE TIP 14(12), 2167-2178 (DLMMSE)",
        },
        "cdm": {
            "patterns": ["RGGB", "BGGR", "GRBG", "GBRG"],
            "license": "MIT (réimplémentation depuis l'article, aucun code tiers repris)",
            "reference": "Zhang, Wu, Buades & Li (2011), J. Electronic Imaging 20(2), 023016 (LDI-NAT)",
        },
        "lslcd": {
            "patterns": ["RGGB", "BGGR", "GRBG", "GBRG"],
            "license": "MIT (réimplémentation depuis l'article, aucun code tiers repris)",
            "reference": "Dubois (2005), IEEE SPL 12(12), 847-850 — luma-chroma demultiplexing",
        },
    }
)
