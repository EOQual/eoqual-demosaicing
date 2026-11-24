# Démosaïçage et débruitage : bruit du capteur, choix de σ, choix du démosaïqueur

Chapitre 11 de la documentation technique ; ses sections gardent leur
numéro d'un document à l'autre.

Question de départ : une chaîne « Hamilton-Adams (HA) puis NL-Bayes »
laisse des défauts chromatiques dans l'image finale. Faut-il changer
l'ordre des étapes, le niveau de bruit donné au débruiteur, le
démosaïqueur, ou passer à une méthode conjointe ? Ce chapitre pose le
modèle physique du bruit, calcule ce que le démosaïçage en fait, en
déduit le σ à donner au débruiteur, puis le vérifie et compare les
démosaïqueurs (qualité et temps de calcul) sur une orthophoto.

Le débruitage lui-même est traité dans la référence technique
d'**eoqual-denoising** (référence technique : [§4](../methodes-projet.md#4-implémentations-propres-du-projet) rôle de σ, [§5](../methodes-publiees.md#5-malvar--malvar-he--cutler) stabilisation
de variance, [§6](../methodes-publiees.md#6-famille-ipol--ha-ari-ri-gbtf-mlri-wmlri) débruiter une image démosaïquée, cas particulier du
bruit coloré ; document « Optimisation des paramètres », [§2](../methodes-generiques.md#2-interpolation-générique) : régler σ
et le facteur C). Ce chapitre-ci
en est le pendant côté démosaïçage.

Reproduction (dépendances hors paquet : eoqual-denoising,
eoqual-metrics, tifffile, scikit-image ; deepinv et PyTorch pour les
chaînes DRUNet ; matplotlib pour la figure) :

```bash
python examples/joint_denoising_study.py capitole_2048_cog.tif --out sortie/        # §11.6
python examples/demosaicing_noise_benchmark.py capitole_2048_cog.tif --out sortie/  # §11.3, §11.7
```

Résultats bruts : `docs/technique/figures/joint_denoising/results.json` (chaînes)
et `demosaicing_benchmark.json` (démosaïqueurs).

| Document | Sections | Contenu |
|---|---|---|
| [Modèle et choix de σ](modele.md) | 11.1, 11.2, 11.3, 11.4 | Littérature, modèle physique du bruit, ce que le démosaïçage en fait, σ à donner au débruiteur |
| [Études](etudes.md) | 11.5, 11.6, 11.7, 11.8 | Protocole, chaînes démosaïçage + débruitage, comparaison des démosaïqueurs, bilan |
| [Recommandations et mode opératoire](recommandations.md) | 11.9, 11.10, 11.11 | Recommandations, limites, mode opératoire sur données réelles |

```{toctree}
:hidden:
:maxdepth: 1

modele
etudes
recommandations
```
