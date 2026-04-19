<!--
SPDX-License-Identifier: CC-BY-4.0
Authorship byline : dreamOfkiki project contributors
License : Creative Commons Attribution 4.0 International (CC-BY-4.0)
-->

# Introduction (Article 2, brouillon C2.13)

**Signataires** : *contributeurs du projet dreamOfkiki*
**Licence** : CC-BY-4.0

**Cible de longueur** : ~1 page markdown (≈ 900 mots)

---

## 1. D'un substrat unique à une revendication de Conformité

L'Article 1 [dreamOfkiki, cycle 1] a introduit le Framework C —
un framework formel exécutable pour la consolidation mnésique
basée sur le rêve, avec les axiomes DR-0..DR-4, un semi-groupe
libre de 4 opérations oniriques canoniques (replay, downscale,
restructure, recombine), 8 primitives typées (entrées α, β, γ,
δ + 4 canaux de sortie), et trois profils (P_min, P_equ, P_max)
chaînés par l'inclusion DR-4. La contribution théorique la plus
structurante de l'Article 1 fut **DR-3** : un Critère de
Conformité qui spécifie les trois conditions (typage de
signature, tests de propriétés d'axiomes, invariants BLOCKING
applicables) que tout substrat candidat doit satisfaire pour
hériter des garanties du framework.

DR-3 est le pont porteur entre la théorie et l'ingénierie.
L'Article 1 a prouvé son existence en tant que dispositif formel
et l'a exercé sur un substrat unique : MLX `kiki_oniric` sur
Apple Silicon. Un seul substrat, cependant, est une preuve
d'existence — pas une indépendance du substrat. Un *deuxième*
substrat est le moment où la revendication transite de
théorique à défendable.

## 2. L'objectif cycle 2 : réplication inter-substrats

Le cycle 2 poursuit exactement ce deuxième substrat et
l'infrastructure de réplication qui l'entoure. La contribution
d'ingénierie de l'Article 2 est quadruple :

1. **Un deuxième enregistrement de substrat.** Nous câblons
   `esnn_thalamocortical`, un squelette numpy LIF de taux de
   spikes d'un E-SNN thalamocortical, exposant les mêmes 4 op
   factories et consommant les mêmes 8 primitives que MLX
   `kiki_oniric`. Le squelette n'est explicitement *pas* un
   matériel Loihi-2 ; c'est une seconde implémentation
   structurelle des Protocols du framework, ce qui est ce que
   DR-3 exige.
2. **Une matrice de Conformité DR-3.** Nous exhibons la matrice
   substrat × condition (3 conditions × 2 substrats réels +
   1 ligne placeholder pour un futur troisième substrat) et
   nous étayons chaque cellule avec un artefact de test concret
   (`tests/conformance/axioms/` et
   `tests/conformance/invariants/`). La matrice vit à
   `docs/milestones/conformance-matrix.md` et est régénérée de
   manière déterministe par `scripts/conformance_matrix.py`.
3. **Une réplication H1-H4 inter-substrats.** La chaîne
   statistique pré-enregistrée du cycle 1 (Welch / TOST /
   Jonckheere / t une-échantillon sous Bonferroni
   α = 0,0125) est re-exécutée *par substrat* par
   `scripts/ablation_cycle2.py`, produisant la table
   comparative dans `docs/milestones/cross-substrate-
   results.md`. Le runner, le pont de prédicteurs et le module
   statistique sont partagés entre les étiquettes de substrats
   — ce qui est précisément la manière dont un framework
   indépendant du substrat est censé se comporter.
4. **Une architecture d'ingénierie adaptée à la réplication.**
   L'Article 2 documente le worker de rêve asynchrone (C2.17),
   le pipeline 3-profils (P_min / P_equ / P_max), le
   protocole de swap avec les gardes S1 / S2 / S3 / S4, le
   registre de runs avec le contrat de déterminisme R1 32-hex,
   et le tag DualVer `C-v0.6.0+PARTIAL` enregistrant le bump
   d'axe formel pour l'extension de Conformité E-SNN.

## 3. Honnêteté de portée : substitution synthétique partout

L'Article 2 est un **article de méthodologie / réplication**,
pas un article de nouvelle revendication empirique. Ce cadrage
est non négociable : les deux lignes de substrats dans la
matrice inter-substrats partagent le même prédicteur mock
Python. Un vrai prédicteur spécifique au substrat est un
livrable du cycle 3 (`docs/milestones/g9-cycle2-publication.md`
§ amorçage cycle-3 ; l'utilisateur a cadré la Phase 3 + 4
comme explicitement synthétique-seulement pour ce cycle).
Par conséquent :

- Toutes les p-values H1-H4 au §7 sont étiquetées
  *(substitution synthétique)*.
- Le verdict d'accord inter-substrats est trivialement OUI par
  construction — prédicteur identique → échantillon identique
  → p-value identique. La **valeur de test** est que le
  pipeline (runner → stats → dump → Markdown) s'exécute de
  bout en bout sur deux enregistrements de substrats
  structurellement distincts, ce qui est la moitié
  *architecturale* de la Conformité DR-3.
- Aucune revendication biologique, neuromorphique ou de
  performance matérielle n'est faite dans l'Article 2. Ces
  revendications appartiennent au cycle 3 (mapping Loihi-2
  réel, cohorte IRMf réelle, réplication à prédicteur
  divergent).

La règle de discipline de recherche §3 du `CLAUDE.md` du dépôt
— *ne jamais rapporter des résultats synthétiques comme des
revendications empiriques* — est le garde-fou. Chaque légende
de table au §7, chaque cellule de p-value, chaque ligne de
verdict doit porter l'étiquette `(substitution synthétique)` ou
un drapeau équivalent en ligne. Les lecteurs qui élident cette
étiquette lisent l'article de travers.

## 4. Différenciation par rapport à l'Article 1 et feuille de route

L'Article 1 était théorique : axiomes + preuves + définition
formelle de la Conformité. L'Article 2 est d'ingénierie :
substrats opérationnels + suite de tests réutilisable + runner
de réplication + preuves par substitution synthétique que le
framework est *prêt* à être répliqué sur des substrats réels
une fois de vrais prédicteurs câblés. Les deux articles
partagent le corps d'auteurs, les licences (code MIT, docs
CC-BY-4.0), le pré-enregistrement (OSF, hérité de l'Article 1)
et la lignée DualVer.

Le reste de l'article est organisé comme suit. §3 de contexte
situe l'Article 2 par rapport à l'Article 1 et aux quatre
piliers (bref, car l'Article 1 a déjà fait le gros œuvre). §4
rapporte les preuves du Critère de Conformité sur les deux
substrats avec citations de matrice. §5 présente
l'architecture d'ingénierie (2 substrats, 3 profils, 4 ops,
8 primitives, worker asynchrone). §6 détaille la méthodologie
(chaîne H1-H4 pré-enregistrée, Bonferroni, détails du
prédicteur de substitution synthétique, contrat R1). §7
rapporte les résultats inter-substrats avec chaque légende
étiquetée. §8 discute les limitations honnêtement — ce que les
données par substitution synthétique impliquent et n'impliquent
pas. §9 esquisse la suite du cycle 3 : Loihi-2 réel, cohorte
IRMf, chemin d'émergence de l'Article 3.

---

## Notes pour révision

- Insérer les citations bibtex une fois `references.bib`
  étendu depuis le stub de l'Article 1 (inter-cycle S20-S22).
- Référencer §3..§9 les numéros de ligne une fois le brouillon
  complet mis en page dans le style-file NeurIPS.
- Resserrer à ≤ 900 mots avant soumission NeurIPS
  (actuellement ~ 820).
