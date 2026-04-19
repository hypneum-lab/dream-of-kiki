<!--
SPDX-License-Identifier: CC-BY-4.0
Authorship byline : dreamOfkiki project contributors
License : Creative Commons Attribution 4.0 International (CC-BY-4.0)
-->

# §8 Discussion (Article 2, brouillon C2.16)

**Signataires** : *contributeurs du projet dreamOfkiki*
**Licence** : CC-BY-4.0

**Cible de longueur** : ~1 page markdown (≈ 900 mots)

---

## 8.1 Ce que la convergence inter-substrats signifie dans l'Article 2

Le titre de l'Article 2 est une **convergence structurelle**,
pas une surprise numérique. Les deux substrats — MLX
`kiki_oniric` et E-SNN `thalamocortical` — passent les trois
conditions de Conformité DR-3 (C1 Protocols typés, C2 tests
de propriétés d'axiomes, C3 invariants BLOCKING applicables,
§4). Les deux substrats sont exercés via le même runner
d'ablation, la même chaîne H1-H4 pré-enregistrée, et le même
Bonferroni α = 0,0125 (§6). Les deux émettent les mêmes
verdicts 4 / 4 d'hypothèses (§7.3).

La lecture correcte est : *le Critère de Conformité du
framework est opérationnel sur deux implémentations
structurellement indépendantes de ses 8 primitives.* La
lecture erronée est : *le framework est empiriquement
indépendant du substrat.* Les deux lectures sont distinctes,
et la distinction repose sur le prédicteur par substitution
synthétique documenté au §6.4.

## 8.2 Ce que la substitution synthétique implique pour les revendications

**(substitution synthétique — pas de revendication
empirique.)**  Les deux lignes de substrats partagent le même
prédicteur mock Python. Concrètement, cela signifie :

- Les p-values H1-H4 (§7.2) sont trivialement identiques entre
  substrats dans la limite d'un prédicteur partagé parfait.
- Le verdict d'accord (§7.3, 4 / 4 d'accord) est trivialement
  OUI par construction.
- Le *pattern* de verdicts (3 rejet, 1 échec à rejeter)
  reflète l'Article 1 §7.7 car le prédicteur mock est le même.
  Il n'y a pas de signal E-SNN indépendant dans les chiffres.

Ce que cela signifie pour les revendications :

- Nous **ne prétendons pas** que MLX et E-SNN produisent un
  comportement équivalent sur des données réelles. Nous
  prétendons que le pipeline émet des verdicts équivalents
  quand le prédicteur est partagé, ce qui est une condition
  nécessaire à la Conformité DR-3 mais pas une condition
  suffisante pour une efficacité empirique indépendante du
  substrat.
- Nous **ne prétendons pas** un avantage matériel
  neuromorphique. Le substrat E-SNN est un squelette numpy
  LIF ; le matériel Loihi-2 n'a pas été exercé.
- Nous **prétendons** la moitié *architecturale* de DR-3 : les
  surfaces typées et les suites d'axiomes / gardes du framework
  sont génériques par rapport au substrat, comme attesté par
  deux enregistrements de substrats structurellement
  indépendants passant la même batterie de tests sans
  duplication de code.

Les reviewers évaluant l'Article 2 doivent lire l'accord
inter-substrats comme **preuve de pipeline**, pas comme
preuve **biologique** ou **neuromorphique**. Tout lecteur qui
élide le drapeau `(substitution synthétique)` lit l'Article 2
de travers.

## 8.3 Limitations (énumération honnête)

Quatre limitations bornent la contribution cycle-2 :

**(i) Prédicteur partagé entre substrats.** C'est la
limitation dominante. Une réplication à prédicteur divergent
— inférence spécifique au substrat sur l'état natif de chaque
substrat — est la cible cycle-3. Jusqu'à son atterrissage,
l'Article 2 ne peut pas porter de revendication de
*performance* indépendante du substrat.

**(ii) Seulement deux substrats réels + un placeholder.** La
Conformité DR-3 est une revendication inductive ; deux est un
commencement, pas un plafond. La ligne `hypothetical_cycle3`
de la matrice est explicitement marquée `N/A`. Les candidats
cycle-3 incluent des instances basées sur transformers, des
mappings SpiNNaker / Norse, et — selon le partenariat Intel
NRC — un déploiement Loihi-2 réel.

**(iii) Benchmark mega-v2 synthétique.** Le benchmark retenu
500-items (§6.5) est un fallback stratifié synthétique
hérité de l'Article 1. L'inférence mega-v2 réelle est
repoussée à un cycle ultérieur. Par conséquent, les p-values
du §7 ne portent aucune revendication sur la consolidation
linguistique réelle.

**(iv) Risque de séquencement de publication de l'Article 1.**
La narration de l'Article 2 dépend de l'Article 1 (framework +
axiomes + DR-3). Si l'acceptation de l'Article 1 est retardée
ou si une révision majeure est requise, l'Article 2 devrait
soit citer le préprint arXiv soit adopter une contingence
Pivot B (un récapitulatif autonome du framework, re-compressé
depuis l'Article 1 §4 + §5). Ni l'ID arXiv ni le dépôt HAL FR
ne sont actuellement verrouillés
(`docs/milestones/g9-cycle2-publication.md` § actions
externes).

## 8.4 Le pivot T-Col vers des données IRMf réelles

L'Article 1 §6.4 a verrouillé Studyforrest comme fallback IRMf
cycle-1 (Branche A). L'outreach T-Col (Tower-Cologne) cycle-2
a poursuivi un partenariat actif de laboratoire — Huth Lab
(UT Austin), Norman Lab (Princeton), Gallant Lab (UC Berkeley)
— pour débloquer des stimuli linguistiques
**contrôlés-par-tâche** au-delà de la compréhension narrative
que fournit Studyforrest. Le partenariat renforcerait H3
(alignement représentationnel monotone), qui a atteint une
significativité seulement borderline au cycle-1 et a échoué à
Bonferroni au cycle 2 (§7.2) en raison de la dispersion
constante du prédicteur mock.

Le pivot T-Col est un **livrable cycle-3**. Pour les besoins
de l'Article 2, le pivot est le mécanisme par lequel les
cycles futurs remplacent les données par substitution
synthétique par un signal biologique réel ; cet article ne
prétend à aucune telle donnée.

## 8.5 Comparaison avec la discussion §8.5 de l'Article 1

L'Article 1 §8.5 a ajouté un paragraphe rétroactif de
réplication préliminaire inter-substrats pointant vers les
artefacts cycle-2. L'Article 2 est la narration complète de ce
paragraphe. Trois choses sont cohérentes entre les deux
articles :

- L'Article 1 §8.5 signale le verdict inter-substrats comme
  *trivialement d'accord par construction* ; l'Article 2 §7.3
  et §8.2 redoublent sur le même cadrage.
- L'Article 1 §8.5 note que les preuves par substitution
  synthétique *renforcent mais ne substituent pas* aux H1-H4
  cycle-1 ; l'Article 2 §8.1 reformule cela comme la
  distinction architecturale-vs-empirique.
- L'Article 1 §8.3 (iii) signale le squelette P_max ;
  l'Article 2 §5.2 enregistre que P_max est désormais
  entièrement câblé (G8) et §7.2 rapporte les premières
  exécutions H2 et H3 sous une vraie structure à trois groupes
  — même si le verdict numérique est limité par le prédicteur
  partagé.

Aucune contradiction n'existe entre les revendications des
Articles 1 et 2 ; les deux articles sont cohérents sous le
cadrage par substitution synthétique.

## 8.6 Arbitrages d'ingénierie (MLX vs E-SNN)

Le cycle 2 exerce MLX et E-SNN sous le même pipeline mais ne
les benchmark pas tête-à-tête sur l'énergie ou la latence. Ce
benchmark n'est pas possible sous le setup à prédicteur
partagé : tout chiffre de temps d'horloge ou de ratio
d'énergie serait dominé par le prédicteur, pas par le
substrat. Quand une réplication à prédicteur divergent
atterrit au cycle 3, l'espace d'arbitrage devient mesurable :

- **MLX** optimise pour des mises à jour séquentielles de
  type gradient et est la cible naturelle pour les
  déploiements Apple Silicon / CUDA.
- **E-SNN** (sur matériel Loihi-2 réel) optimise pour le
  calcul sparse événementiel et est la cible naturelle pour
  les déploiements de recherche neuromorphique.

Un framework indépendant du substrat accepte les deux sans
exiger que l'un domine. La valeur de l'Article 2 ici est qu'il
**rend la comparaison possible** — le registre, le runner
d'ablation, la matrice de conformité, le worker asynchrone
sont tous en place. La comparaison elle-même est un article
cycle-3 (émergence Article 3, selon
`docs/milestones/g9-cycle2-publication.md` § amorçage
cycle-3).

---

## Notes pour révision

- Resserrer à ≤ 800 mots à la passe de pré-soumission NeurIPS.
- Référencer §9 travaux futurs une fois §9 livrée.
- Si l'ID arXiv de l'Article 1 atterrit avant soumission,
  remplacer les références `Article 1 §X.Y` par des appels
  `\cite{}` propres.
