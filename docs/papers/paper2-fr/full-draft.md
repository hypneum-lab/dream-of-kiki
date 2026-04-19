---
title: "Article 2 dreamOfkiki : conformité inter-substrats d'un framework de consolidation mnésique basée sur le rêve (réplication par substitution synthétique)"
author: "contributeurs du projet dreamOfkiki"
contact: "Clement Saillant <clement@saillant.cc>"
affiliation: "L'Electron Rare, France"
date: "2026"
draft: "v0.1 (cycle-2, assemblage C2.16)"
---

# Article 2 — Assemblage complet du brouillon (version française)

⚠️ **Statut** : assemblage de brouillon. Les fichiers `.md` de
section restent les sources de vérité ; ce fichier intègre
leur contenu comme source assemblée pour le rendu pandoc de
`build/full-draft.tex`.

⚠️ **Substitution synthétique — article de méthodologie /
réplication.**  Chaque table, figure et verdict empirique
rapporté ci-dessous est produit par un prédicteur mock Python
partagé, câblé à travers deux enregistrements de substrats
(MLX + squelette numpy LIF E-SNN). Aucun matériel Loihi-2 ni
cohorte IRMf ne participe à aucun résultat. L'Article 2
documente le pipeline de réplication inter-substrats, pas une
nouvelle revendication empirique.

---

## 1. Résumé

L'Article 1 introduisait le Framework C-v0.6.0+PARTIAL
(axiomes DR-0..DR-4, 8 primitives typées, 4 opérations
canoniques, 3 profils) et son Critère de Conformité
exécutable (DR-3). L'Article 2 pose la question d'ingénierie
que l'Article 1 ne pouvait résoudre : *un seul substrat
conforme constitue-t-il une preuve d'indépendance du
substrat ?*

Nous répondons en répliquant la chaîne pré-enregistrée H1-H4
du cycle 1 sur deux substrats enregistrés — MLX kiki-oniric
sur Apple Silicon et un squelette E-SNN thalamocortical
numpy LIF de taux de spikes — et en exhibant une matrice de
Conformité DR-3 à deux substrats (3 conditions × 2 substrats,
tous PASS ; une troisième ligne `hypothetical_cycle3` reste
N/A). Sous Bonferroni α = 0,0125, les deux substrats
s'accordent sur 4 / 4 hypothèses (H1 rejeté, H2 rejeté, H3
échec à rejeter, H4 rejeté) — *substitution synthétique ;
l'accord est trivial par construction à prédicteur partagé et
valide le pipeline, pas l'efficacité empirique du framework
sur des données réelles*. Tout le code, le pré-enregistrement
et les run_ids sont MIT / CC-BY-4.0.

---

## 2. Introduction

### 2.1 D'un substrat unique à une revendication de Conformité

L'Article 1 [dreamOfkiki, cycle 1] a introduit le Framework C
avec les axiomes DR-0..DR-4, un semi-groupe libre de 4
opérations oniriques canoniques, 8 primitives typées, et 3
profils chaînés par DR-4. La contribution la plus
structurante fut **DR-3** : un Critère de Conformité
exécutable. Un seul substrat est une preuve d'existence ;
un deuxième substrat est où la revendication transite de
théorique à défendable.

### 2.2 L'objectif cycle 2 : réplication inter-substrats

Quatre contributions : (1) un deuxième enregistrement de
substrat (`esnn_thalamocortical`, squelette numpy LIF, pas
de Loihi-2) ; (2) une matrice de Conformité DR-3 à 3 lignes
(2 réelles + 1 placeholder) ; (3) une réplication
inter-substrats H1-H4 via `scripts/ablation_cycle2.py` ; (4)
une architecture d'ingénierie adaptée à la réplication
(worker asynchrone C2.17, registre avec R1 32-hex, DualVer
`C-v0.6.0+PARTIAL`).

### 2.3 Honnêteté de portée : substitution synthétique partout

L'Article 2 est un article de méthodologie / réplication. Les
deux lignes de substrats partagent le même prédicteur mock
Python. Un vrai prédicteur spécifique au substrat est un
livrable cycle-3. Par conséquent : toutes les p-values H1-H4
au §7 sont étiquetées *(substitution synthétique)* ; le
verdict d'accord inter-substrats est trivialement OUI par
construction ; aucune revendication biologique,
neuromorphique ou de performance matérielle n'est faite. La
moitié architecturale de la Conformité DR-3 est ce que
l'Article 2 gagne.

### 2.4 Différenciation par rapport à l'Article 1 et feuille de route

L'Article 1 était théorique ; l'Article 2 est d'ingénierie.
§3 contexte, §4 conformité, §5 architecture, §6 méthodologie,
§7 résultats, §8 discussion, §9 cycle-3.

---

## 3. Contexte

L'Article 1 a défini le Framework C (8 primitives, 4 ops, 3
profils, DR-0..DR-4). L'Article 2 hérite du mapping
quatre-piliers (Walker-replay [@walker2004sleep;
@stickgold2005sleep], Tononi-downscale [@tononi2014sleep],
Hobson-recombine [@hobson2009rem; @solms2021revising],
Friston-restructure [@friston2010free]).

DR-3 spécifie C1 (typage de signature via Protocols typés),
C2 (tests de propriétés d'axiomes passent), C3 (invariants
BLOCKING applicables). Le critère est exécutable dans
`tests/conformance/` paramétré par le type d'état du
substrat.

La réplication inter-substrats en apprentissage continu est
sous-représentée dans l'art antérieur : [@vandeven2020brain]
sur une seule architecture, [@kirkpatrick2017overcoming]
comme régularisateur adjacent SHY, [@rebuffi2017icarl] et
[@shin2017continual] spécifiques à l'architecture. Aucun
travail antérieur ne livre une suite de tests de conformité
réutilisable sur des modèles de matériel qualitativement
distincts (MLX dense vs LIF spiking).

L'Article 2 est un article de méthodologie / réplication ;
les lecteurs qui élident le drapeau de substitution
synthétique lisent l'article de travers.

---

## 4. Le Critère de Conformité en pratique

### 4.1 Les trois conditions DR-3, rappel

**C1** typage de signature (Protocols typés), **C2** tests de
propriétés d'axiomes passent, **C3** invariants BLOCKING
(S1..S4) applicables. DR-3 est existentiel, pas universel.

### 4.2 Substrat 1 — `mlx_kiki_oniric`

| Condition | Verdict | Preuve |
|-----------|---------|--------|
| C1 | PASS | `tests/conformance/axioms/test_dr3_substrate.py` |
| C2 | PASS | `tests/conformance/axioms/` |
| C3 | PASS | `tests/conformance/invariants/` |

### 4.3 Substrat 2 — `esnn_thalamocortical` (substitution synthétique)

**(substitution synthétique — pas de matériel Loihi-2.)**

| Condition | Verdict | Preuve |
|-----------|---------|--------|
| C1 | PASS | 4 factories appelables + registre partagé *(substitution synthétique)* |
| C2 | PASS *(substitution synthétique — pas de Loihi-2)* | suite DR-3 E-SNN passe sur numpy LIF |
| C3 | PASS *(substitution synthétique — numpy LIF taux de spikes)* | gardes S2 + S3 applicables sur LIFState |

Fichier : `tests/conformance/axioms/test_dr3_esnn_substrate.py`.

### 4.4 Troisième ligne : `hypothetical_cycle3` (placeholder)

| Condition | Verdict | Preuve |
|-----------|---------|--------|
| C1 | N/A | pas encore implémenté |
| C2 | N/A | pas encore implémenté |
| C3 | N/A | pas encore implémenté |

Candidats cycle-3 : basés sur transformers, SpiNNaker
[@furber2014spinnaker] / Norse, ou Loihi-2 réel
[@davies2018loihi].

### 4.5 La matrice comme artefact réutilisable

`scripts/conformance_matrix.py` régénère la matrice (Markdown
+ JSON). Suite de tests paramétrée par substrat.

### 4.6 Ce que la Conformité certifie et ne certifie pas

Certifie : compatibilité de surface typée + axiomes + gardes.
Ne certifie PAS : performance empirique, énergie matérielle,
fidélité biologique.

---

## 5. Architecture d'ingénierie

### 5.1 Deux substrats, mêmes Protocols

`mlx_kiki_oniric` [@mlx2023] + `esnn_thalamocortical`
(substitution synthétique) partagent les 4 factories d'op.

### 5.2 Trois profils

P_min (replay, canal-1 — cycle 1), P_equ (+ downscale,
restructure ; canaux 1+2+3 — cycle 1), P_max (+ recombine,
canal-4 ATTENTION_PRIOR — cycle 2 G8).

### 5.3 Quatre opérations canoniques

replay (Walker), downscale (Tononi, commutatif
non-idempotent), restructure (Friston, garde S3), recombine
(Hobson, VAE-light → VAE complet cycle 2). 12 / 16 paires
non-commutatives.

### 5.4 Huit primitives + protocole de swap

Entrées α / β / γ / δ. Sorties canal-1..4. Chaque sortie
gardée par S1 (non-régression retenue), S2 (finitude), S3
(topologie), S4 (attention_budget ≤ 1,5).

### 5.5 Worker de rêve asynchrone (C2.17)

`asyncio`-based (`018fd05`), remplace le squelette Future-API
cycle-1.

### 5.6 Registre de runs + déterminisme R1

R1 : préfixe SHA-256 32-hex. `cycle2_batch_id` =
`3a94254190224ca82c70586e1f00d845`. `ablation_runner_run_id`
= `45eccc12953e758440fca182244ddba2`.

### 5.7 Lignée DualVer : `C-v0.6.0+PARTIAL`

Bump MINEUR axe formel (extension E-SNN). `+PARTIAL` axe
empirique — le `+STABLE` requiert une preuve cycle-3 à
prédicteur divergent.

---

## 6. Méthodologie

### 6.1 Hypothèses pré-enregistrées (OSF, héritées)

H1 oubli (Welch unilatéral), H2 équivalence (TOST ε=0,05), H3
monotonicité (Jonckheere-Terpstra), H4 budget énergétique
(t un-échantillon vs 2,0). Verrouillées OSF en cycle 1 (DOI
en attente [@osf]).

### 6.2 Pipeline statistique + α de Bonferroni

`α_par_hypothèse = 0,0125`. Module
`kiki_oniric.eval.statistics` [@virtanen2020scipy] inchangé
cycle 2.

### 6.3 Matrice d'ablation : 2 × 3 × 3 = 18 cellules

Générée par `scripts/ablation_cycle2.py`. Seeds
`[42, 123, 7]`.

### 6.4 Prédicteur par substitution synthétique (la précaution critique)

**(substitution synthétique — pas de revendication
empirique.)**  Les deux lignes de substrats partagent le même
prédicteur mock Python. Conséquences : p-values trivialement
identiques dans la limite à prédicteur partagé ; verdict
d'accord trivialement OUI ; la revendication
*architecturale* (pipeline s'exécute sur deux enregistrements
sans duplication) est validée ; la revendication *empirique*
n'est pas faite. Réplication à prédicteur divergent = cycle-3.

### 6.5 Reproductibilité : contrat R1 + intégrité du benchmark

Run_ids au §5.6. R3 via fichier SHA-256. Cycle-2 utilise le
fallback synthétique `synthetic:c8a0712000b641...` en
attendant le vrai mega-v2. Tag DualVer `C-v0.6.0+PARTIAL`
attaché à chaque artefact.

### 6.6 Ce qui a changé vs méthodologie Article 1

Mêmes hypothèses, même module stats, mêmes seeds, même
Bonferroni, même prédicteur. **Nouveau** : dimension
substrat (2 lignes), verdict d'accord inter-substrats,
artefact de matrice de conformité.

---

## 7. Résultats

> ⚠️ **Précaution (substitution synthétique — pas de
> revendication empirique).**  Chaque revendication
> quantitative est produite par un prédicteur mock Python
> partagé. L'accord inter-substrats est trivialement OUI par
> construction ; les valeurs valident le pipeline de
> réplication inter-substrats, pas l'efficacité empirique sur
> des données réelles.

### 7.1 Provenance (substitution synthétique — pas de revendication empirique)

| champ | valeur |
|-------|--------|
| harness_version | `C-v0.6.0+PARTIAL` |
| cycle2_batch_id | `3a94254190224ca82c70586e1f00d845` |
| ablation_runner_run_id | `45eccc12953e758440fca182244ddba2` |
| benchmark | mega-v2 stratifié, 500 items synthétiques |
| benchmark_hash | `synthetic:c8a0712000b641...` |
| seeds | `[42, 123, 7]` |
| substrats | `mlx_kiki_oniric`, `esnn_thalamocortical` |
| data_provenance | synthétique — mocks partagés |

### 7.2 Table comparative inter-substrats H1-H4 (substitution synthétique)

**Table 7.2 — MLX vs E-SNN à Bonferroni α = 0,0125
(substitution synthétique — pas de revendication empirique).**

| hypothèse | test | p MLX | verdict MLX | p E-SNN | verdict E-SNN | accord |
|-----------|------|-------|-------------|---------|---------------|--------|
| H1 oubli | Welch unilatéral | 0,0000 | rejet H0 | 0,0000 | rejet H0 | OUI |
| H2 auto-équivalence | TOST (ε=0,05) | 0,0000 | rejet H0 | 0,0000 | rejet H0 | OUI |
| H3 monotonicité | Jonckheere-Terpstra | 0,0248 | échec à rejeter | 0,0248 | échec à rejeter | OUI |
| H4 budget énergie | t un-échantillon (sup) | 0,0101 | rejet H0 | 0,0101 | rejet H0 | OUI |

MLX 3 / 4 significatifs, E-SNN 3 / 4 significatifs,
cohérence complète *(substitution synthétique — trivialement
OUI par construction)*. H3 échoue à α = 0,0125 en raison de
la dispersion constante du prédicteur mock — propriété du
mock, pas du framework.

### 7.3 Matrice d'accord (substitution synthétique)

| hypothèse | verdicts égaux ? | rejet MLX | rejet E-SNN |
|-----------|------------------|-----------|-------------|
| H1 oubli | OUI | true | true |
| H2 auto-équivalence | OUI | true | true |
| H3 monotonicité | OUI | false | false |
| H4 budget énergie | OUI | true | true |

4 / 4 d'accord.

### 7.4 Matrice de Conformité DR-3 (référence croisée §4)

| substrat | C1 | C2 | C3 |
|----------|----|----|----|
| `mlx_kiki_oniric` | PASS | PASS | PASS |
| `esnn_thalamocortical` | PASS *(substitution synthétique)* | PASS *(substitution synthétique — pas de Loihi-2)* | PASS *(substitution synthétique — numpy LIF taux de spikes)* |
| `hypothetical_cycle3` | N/A | N/A | N/A |

Source : `docs/milestones/conformance-matrix.md` (commit
`fd54df7`).

### 7.5 Ce que le §7 établit et n'établit pas

Établit : pipeline inter-substrats de bout en bout ; verdicts
identiques sur entrées identiques ; matrice DR-3 PASS sur 2
lignes ; contrat R1 tient. N'établit PAS : performance
biologique / neuromorphique ; efficacité empirique de
consolidation ; réplication à prédicteur divergent ; énergie
matérielle.

### 7.6 Résumé de gate (substitution synthétique)

H1 PASS, H2 PASS (fumée d'auto-équivalence), H3 ÉCHEC
Bonferroni / PASS α = 0,05, H4 PASS. Verdict agrégé cycle-2
G9 : **CONDITIONAL-GO / PARTIAL** selon
`docs/milestones/g9-cycle2-publication.md`.

---

## 8. Discussion

### 8.1 Ce que la convergence inter-substrats signifie

Les deux substrats passent DR-3 (C1 + C2 + C3) et émettent
les mêmes verdicts 4 / 4. Lecture correcte : Critère de
Conformité opérationnel sur deux implémentations
structurellement indépendantes. Lecture erronée :
indépendance empirique du substrat. Le prédicteur par
substitution synthétique est la distinction.

### 8.2 Ce que la substitution synthétique implique

Nous **ne prétendons pas** à un comportement équivalent sur
données réelles ni à un avantage matériel neuromorphique.
Nous **prétendons** la moitié architecturale de DR-3.

### 8.3 Limitations (énumération honnête)

**(i)** Prédicteur partagé — limitation dominante.
**(ii)** Seulement 2 substrats réels + 1 placeholder.
**(iii)** Benchmark mega-v2 synthétique. **(iv)** Risque de
séquencement de publication de l'Article 1 — si retard
> 6 mois, Pivot B (§9.5).

### 8.4 Le pivot T-Col vers des données IRMf réelles

Outreach T-Col cible Huth / Norman / Gallant Labs pour des
données IRMf contrôlées-par-tâche. Livrable cycle-3.

### 8.5 Comparaison avec la discussion §8.5 de l'Article 1

Aucune contradiction sous le cadrage par substitution
synthétique.

### 8.6 Arbitrages d'ingénierie (MLX vs E-SNN)

Benchmark énergie / latence impossible sous prédicteur
partagé. Sous prédicteur divergent (cycle 3), MLX optimise
les mises à jour tensorielles denses, E-SNN sur Loihi-2
[@davies2018loihi] optimiserait le calcul sparse événementiel.

---

## 9. Travaux futurs — Cycle 3

### 9.1 Terminer Phase 3 — réplication à prédicteur divergent

Remplacer le mock partagé par inférence spécifique au
substrat. Débloque `C-v0.7.0+STABLE`.

### 9.2 Mapping matériel Loihi-2 réel

Si partenariat Intel NRC se matérialise
[@davies2018loihi]. Fallbacks : SpiNNaker via Norse
[@furber2014spinnaker], simulation Lava SDK, ou statu quo.

### 9.3 Cohorte IRMf réelle (pivot T-Col)

Peut glisser au cycle 4.

### 9.4 Émergence de l'Article 3

Provisoire selon `docs/papers/paper3/outline.md`.

### 9.5 Séquencement Phase 4 avec acceptation Article 1

**Préprint-d'abord** ou **Pivot B** si Article 1 retardé
> 6 mois.

---

## 10. Références

Voir `references.bib`. Bibliographie rendue via pandoc
`--citeproc` ou `--biblatex` au moment du rendu.

---

## Notes sur l'assemblage

- Tous les fichiers de section restent individuellement
  éditables. Les révisions futures devraient mettre à jour à
  la fois le fichier de section et cet assemblage.
- Chaque légende de table au §7 porte un drapeau
  `(substitution synthétique)`. Ne pas supprimer lors des
  passes de resserrement.
- Miroir EN : `docs/papers/paper2/full-draft.md`.
