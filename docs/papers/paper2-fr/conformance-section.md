<!--
SPDX-License-Identifier: CC-BY-4.0
Authorship byline : dreamOfkiki project contributors
License : Creative Commons Attribution 4.0 International (CC-BY-4.0)
-->

# §4 Le Critère de Conformité en pratique (Article 2, brouillon C2.14)

**Signataires** : *contributeurs du projet dreamOfkiki*
**Licence** : CC-BY-4.0

**Cible de longueur** : ~1,5 pages markdown (≈ 1300 mots)

---

## 4.1 Les trois conditions DR-3, rappel

Le Framework C §6.2 (`docs/specs/2026-04-17-dreamofkiki-
framework-C-design.md`) définit le **Critère de Conformité**
comme trois conditions indépendamment vérifiables qu'un
substrat candidat doit satisfaire pour hériter des garanties
du framework :

- **C1 — typage de signature (Protocols typés).** Les 8
  primitives (entrées α, β, γ, δ + sorties canal-1..4) sont
  déclarées comme `Protocol`s Python typés. Un substrat est
  conforme à C1 en exposant des handlers / factories dont les
  signatures sont assignables à ces Protocols. Le registre
  d'opérations (`kiki_oniric.dream.registry`) doit être
  complet pour les profils que le substrat exerce.
- **C2 — tests de propriétés d'axiomes passent.** DR-0..DR-4
  disposent de suites de tests fondés sur les propriétés dans
  `tests/conformance/axioms/`. Chaque suite est paramétrée
  sur le type d'état du substrat. Un substrat est conforme à
  C2 si chaque suite d'axiome applicable est verte sous sa
  représentation d'état native.
- **C3 — invariants BLOCKING applicables.** Les gardes de la
  famille S (S1 non-régression retenue, S2 finitude, S3
  topologie, S4 attention_budget) doivent être appelables sur
  l'état du substrat et doivent *refuser* les valeurs mal
  formées en levant la classe d'exception documentée. C3 est
  vérifié par des tests négatifs dans
  `tests/conformance/invariants/` qui câblent des états
  délibérément mal formés à travers les gardes.

DR-3 est **existentiel, pas universel** : il ne prétend pas
que tout substrat est conforme ; il prétend que tout substrat
*conforme* hérite des garanties du framework. L'Article 1 a
exhibé un substrat conforme (MLX) ; l'Article 2 en exhibe un
second (squelette E-SNN), et réserve une troisième ligne pour
le cycle 3.

## 4.2 Substrat 1 — `mlx_kiki_oniric` (référence cycle 1)

Le substrat de référence MLX est l'implémentation canonique
du cycle 1. Chaque cellule ci-dessous est étayée par un
artefact de test concret en arbre :

| Condition | Verdict | Preuve |
|-----------|---------|--------|
| C1 typage de signature | PASS | les 8 primitives déclarées comme Protocols + registre complet (`tests/conformance/axioms/test_dr3_substrate.py`) |
| C2 tests de propriétés d'axiomes | PASS | suites DR-0, DR-1, DR-3, DR-4 vertes sur MLX (`tests/conformance/axioms/`) |
| C3 invariants BLOCKING | PASS | gardes S2 finitude + S3 topologie applicables sur arrays MLX (`tests/conformance/invariants/`) |

La ligne MLX ne porte pas de drapeau de substitution
synthétique au niveau de la *conformité* : les trois
conditions concernent la surface du framework (typage,
axiomes, gardes), pas les données qui y circulent. La
précaution de substitution synthétique pour MLX s'active
seulement quand MLX est utilisé comme **prédicteur** au §7
(où le prédicteur est partagé entre les étiquettes de
substrats).

## 4.3 Substrat 2 — `esnn_thalamocortical` (ajout cycle 2, substitution synthétique)

Le substrat E-SNN est un **squelette numpy LIF de taux de
spikes**. Il expose les mêmes 4 factories d'opérations
(replay, downscale, restructure, recombine) et consomme les
mêmes 8 primitives que MLX, mais son état interne est une
représentation de population de type leaky-integrate-and-fire
(`LIFState`) dont l'évolution est simulée numériquement — pas
déployée sur un matériel Loihi-2.

**(substitution synthétique — pas de matériel Loihi-2.)** Les
lignes C2 et C3 ci-dessous portent le drapeau explicite de
substitution synthétique hérité de
`docs/proofs/dr3-substrate-evidence.md` :

| Condition | Verdict | Preuve |
|-----------|---------|--------|
| C1 typage de signature | PASS | 4 factories d'op appelables + registre noyau partagé avec MLX (squelette numpy LIF de taux de spikes, substitution synthétique) (`tests/conformance/axioms/test_dr3_esnn_substrate.py`) |
| C2 tests de propriétés d'axiomes | PASS *(substitution synthétique — pas de matériel Loihi-2)* | la suite de conformité DR-3 E-SNN passe sur le squelette numpy LIF (`tests/conformance/axioms/test_dr3_esnn_substrate.py`) |
| C3 invariants BLOCKING | PASS *(substitution synthétique — numpy LIF taux de spikes)* | gardes S2 finitude + S3 topologie applicables sur LIFState (`tests/conformance/axioms/test_dr3_esnn_substrate.py`) |

La ligne E-SNN est **l'artefact nouveau clé du cycle 2** pour
DR-3. Passer C1..C3 sur une implémentation structurellement
indépendante — dynamique de taux de spikes plutôt que mises à
jour par gradient sur des matrices denses — est la preuve
architecturale que les surfaces de typage + axiomes + gardes
du framework ne dépendent pas secrètement d'internals MLX. Un
mapping Loihi-2 réel renforcerait cela en une revendication
matérielle neuromorphique ; en attendant, le squelette de
taux de spikes est une **seconde implémentation structurelle**
adéquate pour valider la forme du Critère de Conformité, pas
ses conséquences matérielles.

## 4.4 Une troisième ligne : `hypothetical_cycle3` (placeholder, pas preuve)

La matrice de conformité réserve une troisième ligne pour un
futur substrat (cycle 3) :

| Condition | Verdict | Preuve |
|-----------|---------|--------|
| C1 typage de signature | N/A | pas encore implémenté |
| C2 tests de propriétés d'axiomes | N/A | pas encore implémenté |
| C3 invariants BLOCKING | N/A | pas encore implémenté |

La ligne est explicitement marquée `N/A` et **ne doit pas
être lue comme passant, échouant, ou même testable**. Son
rôle est double : (i) garder la forme de la matrice stable
pour le cycle 3 afin que les diffs dans le rapport de clôture
cycle-3 soient minimaux, (ii) rappeler visuellement aux
lecteurs que les preuves DR-3 sont une *collection*, pas un
binaire ; deux substrats est un commencement, pas un
plafond. Les candidats substrats cycle-3 envisagés incluent
des instances basées sur transformers, des mappings
SpiNNaker / Norse, ou un déploiement Loihi-2 réel si le
partenariat Intel NRC se matérialise
(`docs/milestones/g9-cycle2-publication.md` § agenda cycle-3).

## 4.5 La matrice comme artefact réutilisable

La matrice de conformité n'est pas une revendication unique ;
c'est un artefact régénérable. Exécuter :

```bash
uv run python scripts/conformance_matrix.py
```

re-dérive la matrice à partir de la suite de tests et écrit à
la fois les dumps Markdown (`docs/milestones/conformance-
matrix.md`) et JSON (`docs/milestones/conformance-
matrix.json`). Le JSON est structuré pour l'automatisation
aval : un futur `test_conformance_matrix_regression` pourrait
diffuser le JSON contre un snapshot commité pour attraper
les régressions silencieuses.

La suite de tests de support (`tests/conformance/`) est
paramétrée par substrat. Un reviewer ou un utilisateur avec
un nouveau substrat candidat ajoute une ligne d'enregistrement
(`kiki_oniric/substrates/__init__.py`), écrit les 4 factories
d'op + le type d'état, et toute la batterie C1..C3 s'exécute
automatiquement. C'est la promesse opérationnelle de DR-3 :
**un framework indépendant du substrat doit expédier une
suite de tests indépendante du substrat**, pas une batterie
monolithique par substrat réécrite à chaque fois.

## 4.6 Ce que la Conformité certifie et ne certifie pas

Deux lectures doivent être distinguées :

**Ce que la Conformité certifie (C1..C3 tous PASS) :**

- La surface typée du substrat est compatible avec le
  framework.
- La représentation d'état du substrat satisfait les tests
  de propriétés d'axiomes DR-0..DR-4 pertinents pour son
  profil.
- L'état du substrat peut être *refusé* par les gardes
  BLOCKING quand il est mal formé.
- La promesse centrale du framework — que deux substrats
  conformes sont interchangeables au niveau du Protocol —
  est opérationnellement tenue par au moins ces deux
  substrats.

**Ce que la Conformité ne certifie PAS :**

- Aucune **revendication de performance empirique** sur des
  données réelles. Les résultats inter-substrats au §7 sont
  par substitution synthétique ; C1..C3 est une propriété
  architecturale, pas une propriété d'efficacité sur
  données.
- Aucune **revendication de coût énergie ou de latence
  matérielle**. E-SNN sur Loihi-2 serait une revendication
  matérielle ; E-SNN sur numpy est une revendication
  architecturale. Les deux sont distinctes et l'Article 2
  ne fait que la seconde.
- **Fidélité biologique** des paramètres LIF E-SNN. Les
  mappings de piliers (Article 1 §3) sont hérités de
  l'Article 1 ; l'Article 2 ne les ré-argumente pas.

Références croisées : `docs/proofs/dr3-substrate-evidence.md`
est l'enregistrement de preuves par substrat ;
`docs/milestones/conformance-matrix.md` est la matrice
régénérable ; `docs/milestones/g7-esnn-conformance.md` est
le rapport de gate G7 qui a verrouillé le verdict de
Conformité E-SNN.

---

## Notes pour révision

- Insérer la matrice comme figure de table rendue une fois
  le style-file NeurIPS choisi.
- Référencer §5 architecture (qui décrit les ops derrière le
  registre) une fois §5 livrée.
- Resserrer à ≤ 1200 mots à la passe de pré-soumission
  NeurIPS.
