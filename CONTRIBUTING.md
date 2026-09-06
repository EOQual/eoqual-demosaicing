# Contribuer à eoqual-demosaicing

Les contributions sont bienvenues, sous toutes leurs formes : signalement
de bug, correction, proposition de nouvelle méthode, implémentation d'une
méthode déjà proposée, amélioration de la documentation.

## Signaler un bug

Ouvrez une issue avec :
- la méthode concernée (`list_methods()` pour voir le registre complet
  dans votre environnement) ;
- un exemple minimal reproduisant le problème (image de test, code) ;
- le résultat obtenu vs. attendu.

## Proposer une méthode

Ouvrez une issue décrivant la méthode (nom, référence bibliographique,
ce qu'elle apporte par rapport aux méthodes déjà cataloguées — voir
`REFERENCE_TECHNIQUE.md` pour l'existant). Pas besoin de savoir
l'implémenter pour la proposer.

## Implémenter une méthode

Convention du projet : une méthode = une fonction
`xxx_demosaic(image: np.ndarray, pattern: str = "RGGB", ...) -> np.ndarray`,
placée dans `src/eoqual/demosaicing/backends/<méthode>/<fichier>.py`,
puis déclarée dans `config.py::METHODS_CONFIGS` (motifs de Bayer
supportés, licence, référence) et dans le dispatch de
`runner.py::_DISPATCH` + exposée dans `__init__.py`.

**Licence — condition de fusion, pas une formalité.** Toute implémentation
reprenant ou s'inspirant de près d'un code publié ailleurs (mêmes noms de
fonctions, même structure, portage quasi littéral) doit citer sa source
et la licence *vérifiée* de cette source dans le module (voir les
en-têtes existants — `backends/malvar/he_cutler.py`, `backends/ari/`,
`backends/ri/` — pour le format : vérification directe du fichier
LICENSE ou du champ `license` de l'API GitHub/PyPI, pas une supposition).

- Une implémentation dont la licence d'origine n'est pas identifiée, ou
  qui s'avère copyleft/non permissive, **ne sera pas acceptée** — voir
  le traitement de `DLMMSE` (masterAllen, aucune licence identifiée)
  comme précédent, `THIRD_PARTY_LICENSES.md` §2.2.

Voir `THIRD_PARTY_LICENSES.md` pour le principe déjà appliqué à
l'existant, et son avertissement en tête : une erreur de licence
constatée après coup sera corrigée, implémentation retirée si nécessaire.
Éviter d'y exposer le projet en amont.

## Tests

Toute nouvelle méthode doit être accompagnée :
- d'une entrée dans `tests/test_backends.py` (smoke test : l'appel sur
  une image Bayer réelle ne lève pas d'exception, renvoie une image RGB
  de forme cohérente) ;
- idéalement, d'un script dans `examples/` démontrant son usage.

## Documentation

Toute méthode nouvelle ou modifiée doit rester reflétée dans `README.md`
(catalogue) et `REFERENCE_TECHNIQUE.md` (principe, implémentation,
limites) — les docstrings seules ne suffisent pas, ces deux documents
sont la carte d'ensemble du projet.
