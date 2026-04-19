---
title: "dreamOfkiki : un framework formel indépendant du substrat pour la consolidation mnésique basée sur le rêve dans les systèmes cognitifs artificiels"
author: "contributeurs du projet dreamOfkiki"
contact: "Clement Saillant <clement@saillant.cc>"
affiliation: "L'Electron Rare, France"
date: "2026"
draft: "v0.2 (cycle-1 closeout, miroir EN d6866f3, retarget PLOS CB)"
---

# Paper 1 — Assemblage complet du brouillon (version française)

⚠️ **Statut** : assemblage du brouillon, miroir français de la
v0.2 anglaise (commit `d6866f3`). Les fichiers .md de section
étaient la source de vérité originale ; ce fichier intègre désormais
leur contenu pour devenir la source assemblée en vue du rendu pandoc.

⚠️ Le §7 a été réécrit en *validation pipeline-et-framework* (pas
de p-values synthétiques rapportées). L'évaluation empirique des
hypothèses H1–H4 sur des bancs de test d'apprentissage continu
réels est l'objet du Paper 2.

---

## 1. Résumé

**Note de portée.** Cet article rapporte un *framework formel
exécutable* pour la consolidation onirique dans les systèmes
cognitifs artificiels ; les contributions principales sont les
axiomes, le Critère de Conformité et un parcours de conformité
inter-substrats. L'évaluation empirique au niveau des hypothèses
sur des bancs de test d'apprentissage continu et d'alignement
IRMf est différée au papier compagnon (Paper 2).

L'oubli catastrophique demeure un obstacle central pour les systèmes
cognitifs artificiels apprenant séquentiellement une succession de
tâches. La consolidation mnésique inspirée du sommeil a été proposée
comme remède, les travaux antérieurs ayant exploré la réactivation
(Walker, van de Ven), l'homéostasie synaptique (Tononi), la
recombinaison créative (Hobson) et le codage prédictif (Friston) —
mais aucun framework formel unifié n'intègre ces quatre piliers en
opérations composables et indépendantes du substrat.

Nous introduisons **dreamOfkiki**, un framework formel à axiomes
exécutables. **La compositionnalité DR-2** est ici prouvée comme
théorème de semi-groupe engendré sur quatre opérations canoniques
(fermeture + additivité du budget + composition fonctionnelle +
associativité ; preuve dans
`docs/proofs/dr2-compositionality.md`). Les axiomes restants sont
DR-0 redevabilité, DR-1 conservation épisodique, DR-3 indépendance
du substrat via un Critère de Conformité exécutable (typage des
signatures ∧ tests de propriété axiomatiques ∧ applicabilité des
invariants BLOCKING), et DR-4 inclusion en chaîne des profils
(preuve dans `docs/proofs/dr4-profile-inclusion.md`). Le framework
définit 8 primitives typées (entrées α, β, γ, δ ; 4 canaux de
sortie), 4 opérations canoniques (replay, downscale, restructure,
recombine) et une ontologie de Dream Episode en quintuplet.

**Conformité inter-substrats (§5.6).** Deux substrats indépendants
— une implémentation *kiki-oniric* à base de gradients MLX et un
*E-SNN thalamocortical* numpy-LIF — satisfont tous deux les trois
conditions du Critère de Conformité, avec 9 tests de propriété
axiomatique et d'application d'invariants passants sur chacun
(porte G7 LOCKED, voir `docs/milestones/g7-esnn-conformance.md`).
Cela étaye l'assertion d'indépendance du substrat à la portée du
Paper 1, plutôt que de la différer au cycle 2.

**Validation du pipeline (§7).** Nous validons la chaîne de mesure
et statistique (quatre tests pré-enregistrés sous correction de
Bonferroni), l'injection de fautes sur les gardes des invariants
BLOCKING S1–S3, et le déterminisme de reproductibilité R1 sur des
tuples appariés du registre. Nous rapportons en complément des
mesures de portabilité inter-substrats issues du projet sœur
Nerve-WML (§7.4) qui confirment indépendamment l'indépendance du
substrat sur les tâches linéairement séparables (écart < 5 %) et
divulguent l'écart en régime non linéaire (12,1 %) comme résultat
négatif honnête. **Aucune décision d'hypothèse d'apprentissage
continu sur H1–H4 n'est annoncée dans cet article** ; l'évaluation
empirique de H1–H4 est rapportée dans le Paper 2.

Le pré-enregistrement est verrouillé sur l'Open Science Framework
(DOI 10.17605/OSF.IO/Q6JYN). L'ensemble du code, des spécifications
et du pré-enregistrement est ouvert sous MIT / CC-BY-4.0 ; les
implémentations de référence et les suites de tests sont disponibles
sur `github.com/genial-lab/dream-of-kiki`.

---

## 2. Introduction

### 2.1 L'oubli catastrophique et la lacune de consolidation

Les systèmes cognitifs artificiels modernes excellent dans
l'apprentissage mono-tâche, mais se dégradent rapidement lorsqu'ils
sont entraînés séquentiellement sur plusieurs tâches — un phénomène
connu sous le nom d'**oubli catastrophique** [@mccloskey1989catastrophic;
@french1999catastrophic]. Malgré deux décennies de stratégies
d'atténuation (elastic weight consolidation [@kirkpatrick2017overcoming],
réactivation générative [@shin2017continual], mémoire par
répétition [@rebuffi2017icarl]), le champ manque toujours d'une
*théorie unifiée* expliquant pourquoi ces mécanismes fonctionnent
et quand ils doivent se composer.

La cognition biologique résout ce problème pendant le **sommeil**.
La réactivation hippocampique durant le NREM, la régulation à la
baisse synaptique, la restructuration prédictive des représentations
corticales et la recombinaison créative pendant le REM forment
ensemble un pipeline de consolidation multi-étapes
[@diekelmann2010memory; @tononi2014sleep]. Pourtant, les travaux
existants en IA n'ont intégré que des fragments de cette biologie,
en se concentrant généralement sur un mécanisme unique (p. ex. la
réactivation seule) sans théorie raisonnée de la manière dont les
mécanismes interagissent.

### 2.2 Quatre piliers de la consolidation mnésique basée sur le rêve

Nous identifions quatre piliers théoriques que tout framework
complet de consolidation en IA inspirée du rêve doit adresser :

- **A — consolidation Walker/Stickgold** : transfert épisodique-
  vers-sémantique via la réactivation [@walker2004sleep;
  @stickgold2005sleep].
- **B — SHY de Tononi** : homéostasie synaptique renormalisant les
  poids pendant le sommeil [@tononi2014sleep].
- **C — rêve créatif Hobson/Solms** : recombinaison et abstraction
  pendant le REM [@hobson2009rem; @solms2021revising].
- **D — FEP de Friston** : minimisation de l'énergie libre comme
  théorie unificatrice de l'inférence et de la consolidation
  [@friston2010free].

Les travaux antérieurs en IA ont implémenté A [@vandeven2020brain],
B [@kirkpatrick2017overcoming as a SHY-adjacent regularization] et
des éléments de D [@rao1999predictive; @whittington2017approximation],
mais **aucun travail n'a formalisé la manière dont les quatre
piliers se composent** de façon indépendante du substrat, propice à
l'ablation et à la preuve.

### 2.3 La lacune compositionnelle

Pourquoi la composition importe-t-elle ? Empiriquement, l'ordre
dans lequel s'appliquent les opérations de consolidation modifie
l'état cognitif résultant — la réactivation avant la régulation à
la baisse préserve la spécificité épisodique, tandis que la
régulation à la baisse avant la restructuration peut effacer les
représentations mêmes que la restructuration est censée affiner.
Notre analyse (`docs/proofs/op-pair-analysis.md`) énumère les 16
paires d'opérations et constate que 12 paires croisées sont
non-commutatives, confirmant que *l'ordre fait partie du framework*
et non d'un détail d'implémentation.

Un framework formel digne de ce nom doit donc (i) spécifier les
opérations comme primitives composables à types bien définis, (ii)
expliciter quelles compositions sont valides, (iii) fournir une
théorie **exécutable** que tout substrat conforme peut implémenter
et (iv) supporter l'ablation empirique comparant différents
profils d'opérations. Aucun des travaux antérieurs ne satisfait
les quatre critères.

### 2.4 Feuille de route des contributions

Dans cet article, nous présentons **dreamOfkiki**, le premier
framework formel pour la consolidation mnésique basée sur le rêve
dans les systèmes cognitifs artificiels, avec les contributions
suivantes :

1. **Framework C-v0.5.0+STABLE** : 8 primitives typées, 4 opérations
   canoniques formant un semi-groupe libre, 4 OutputChannels,
   ontologie de Dream Episode en quintuplet, axiomes DR-0..DR-4 avec
   Critère de Conformité exécutable (§4). Les éléments 2–4 ci-
   dessous sont rapportés dans le Paper 2 (compagnon empirique) ;
   le Paper 1 se limite aux contributions formelles et à la feuille
   de route de conformité.
2. **Feuille de route** vers la généralisation multi-substrats
   (substrats supplémentaires au-delà de l'implémentation de
   référence du cycle 1) et vers l'alignement représentationnel
   IRMf réel (partenariat de laboratoire poursuivi via la campagne
   T-Col).

Le reste de l'article est organisé comme suit : §3 passe en revue
les quatre piliers en profondeur ; §4 développe le Framework
C-v0.5.0+STABLE avec axiomes et preuves ; §5 esquisse l'approche
de validation du Critère de Conformité (les résultats empiriques
spécifiques au substrat résident dans le Paper 2) ; §6 détaille la
méthodologie ; §7 rapporte les résultats de validation du pipeline
synthétique ; §8 discute les implications et limites ; §9 esquisse
les travaux futurs du cycle 2.

---

## 3. Contexte théorique — quatre piliers

### 3.1 Pilier A — Consolidation Walker / Stickgold

La consolidation mnésique dépendante du sommeil désigne le
phénomène établi empiriquement selon lequel les souvenirs
nouvellement encodés sont sélectivement renforcés, abstraits et
intégrés au stockage à long terme pendant le sommeil
[@walker2004sleep; @stickgold2005sleep]. La réactivation
hippocampique durant le sommeil lent NREM est le substrat neural
le plus directement impliqué. Le propos fonctionnel est que la
réactivation effectue des **mises à jour de type gradient** sur
les représentations corticales, biaisées vers la rétention des
épisodes rejoués — ce qui équivaut dans notre framework à
l'opération `replay` : échantillonner des épisodes du tampon β,
les propager en avant à travers les paramètres courants, appliquer
des mises à jour par gradient contre un objectif de rétention.

### 3.2 Pilier B — Homéostasie synaptique SHY de Tononi

L'Hypothèse d'Homéostasie Synaptique (SHY) postule que l'éveil
entraîne une potentiation synaptique nette, et que le sommeil
impose une régulation à la baisse synaptique globale qui restaure
le rapport signal-sur-bruit sans effacer le motif de renforcement
différentiel [@tononi2014sleep]. La régulation à la baisse est
soutenue empiriquement par des preuves ultrastructurales
(réductions de taille des synapses pendant le sommeil) et par des
preuves comportementales (amélioration dépendante du sommeil sur
les tâches préalablement entraînées). Dans notre framework, SHY
correspond à l'opération `downscale` : rétrécissement
multiplicatif des poids par un facteur dans (0, 1]. Comme établi
dans notre analyse des paires d'opérations (voir
`docs/proofs/op-pair-analysis.md`, axiomes DR-2 + invariants S2),
downscale est **commutative mais non idempotente** (shrink_f ∘
shrink_f donne facteur², pas facteur) — propriété qui contraint
les choix d'ordonnancement canonique.

### 3.3 Pilier C — Rêve créatif Hobson / Solms

Le rêve en REM est associé à la recombinaison créative, à la
génération de scénarios contrefactuels et à l'intégration de
matériel émotionnellement significatif [@hobson2009rem;
@solms2021revising]. Le mécanisme est hypothéquement un
échantillonnage de style modèle génératif à partir d'une
représentation latente des expériences récentes, produisant des
combinaisons nouvelles qui sondent les frontières de la structure
apprise. Dans notre framework, ceci se projette sur l'opération
`recombine` : échantillonner les latents du snapshot δ, appliquer
un VAE allégé ou un noyau d'interpolation, émettre de nouveaux
échantillons latents sur le canal 2.

### 3.4 Pilier D — Principe d'Énergie Libre de Friston

Le Principe d'Énergie Libre (FEP) [@friston2010free] encadre la
perception, l'action et l'apprentissage comme la minimisation de
l'énergie libre variationnelle sous un modèle génératif
hiérarchique. Au sein du FEP, le sommeil est interprété comme
une phase hors ligne qui **restructure** le modèle génératif pour
mieux minimiser l'énergie libre attendue sur la distribution des
entrées d'éveil. Dans notre framework, ceci correspond à
l'opération `restructure` : modifier la topologie du modèle
hiérarchique (ajouter une couche, retirer une couche, rerouter la
connectivité) afin de réduire l'erreur prédictive sur les épisodes
retenus. La garde topologique S3 (validate_topology) assure que
les opérations restructure préservent les invariants de niveau
framework S3 (connectivité d'espèces, absence de boucles
autoréférentes, bornes sur le nombre de couches — voir
`docs/invariants/registry.md` pour les définitions canoniques et
la référence de garde S3 dans
`kiki_oniric/dream/guards/topology.py`).

### 3.5 La lacune compositionnelle

Les travaux existants en IA ont implémenté un ou deux des quatre
piliers (notamment A via @vandeven2020brain replay génératif et B
via @kirkpatrick2017overcoming EWC, traité comme régulateur
adjacent à SHY). Cependant, aucun travail antérieur n'a
**formalisé la composition** des quatre opérations comme structure
algébrique unifiée à propriétés prouvables.

La lacune compositionnelle importe empiriquement, car notre
analyse des paires d'opérations
(`docs/proofs/op-pair-analysis.md`) établit que 12 des 16 paires
croisées (op_i, op_j) sont **non-commutatives** — c'est-à-dire
qu'appliquer replay puis downscale produit un état cognitif
différent de l'application de downscale puis replay. L'ordre
canonique choisi dans
`docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §4.3
(replay → downscale → restructure ; recombine en parallèle) est
donc une décision de conception portante, non un choix arbitraire
d'implémentation.

Un framework formel digne de ce nom doit donc (i) spécifier les
opérations comme primitives composables à types bien définis,
(ii) expliciter quelles compositions sont valides, (iii) fournir
une théorie exécutable que tout substrat conforme peut implémenter,
et (iv) supporter l'ablation empirique comparant différents
profils d'opérations. Aucun des travaux antérieurs ne satisfait
les quatre critères. Notre Framework C-v0.5.0+STABLE (§4) est le
premier à y parvenir, cartographiant les quatre piliers sur le
framework axiomatique canonique : pilier A → DR-1 conservation
épisodique, pilier B → DR-2 compositionnalité (contrainte d'ordre
sur downscale), pilier D → DR-3 indépendance du substrat (la
garde topologique restructure S3 vit sur cet axe), pilier C → DR-4
inclusion en chaîne des profils qui maintient les profils riches
en recombine au sommet. L'axiome de compositionnalité en
semi-groupe libre DR-2 (prouvé dans
`docs/proofs/dr2-compositionality.md`) est la propriété
fondationnelle, et le Critère de Conformité DR-3 le contrat
exécutable pour l'indépendance du substrat.

---

## 4. Framework C

⚠️ **Source** : `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`
couvre cette section. La version papier ci-dessous est une narration
condensée de cette spécification, structurée selon le plan §4 de
outline.md.

### 4.1 Primitives — 8 Protocoles typés

Canaux Awake → Dream :
- α (traces brutes, P_max uniquement) — tampon circulaire firehose
- β (tampon épisodique curaté) — journal append SQLite avec
  insertion gatée par saillance (les enregistrements ne passent
  que lorsque leur score de saillance dépasse un seuil top-k
  adaptatif)
- γ (snapshot des poids) — repli pointeur de checkpoint
- δ (latents hiérarchiques) — tampon circulaire N=256
  multi-espèces

Canaux Dream → Awake :
- 1 (delta de poids) — appliqué via le protocole de basculement
- 2 (échantillons latents) — file de replay génératif
- 3 (diff de hiérarchie) — application atomique au basculement
  avec garde S3
- 4 (attention prior) — guidage méta-cognitif (P_max uniquement)

### 4.2 Profils — inclusion en chaîne DR-4

| Profil | Canaux entrée | Canaux sortie | Opérations |
|---------|-------------|--------------|------------|
| P_min   | β | 1 | replay, downscale |
| P_equ   | β + δ | 1 + 3 + 4 | replay, downscale, restructure, recombine_light |
| P_max   | α + β + δ | 1 + 2 + 3 + 4 | replay, downscale, restructure, recombine_full |

DR-4 (prouvé dans `docs/proofs/dr4-profile-inclusion.md`) :
ops(P_min) ⊆ ops(P_equ) ⊆ target_ops(P_max), et de même pour les
canaux. P_max est en squelette uniquement au cycle 1.

### 4.3 Ontologie du Dream-episode (quintuplet)

Chaque dream-episode (DE) est un quintuplet :
`(trigger, input_slice, operation_set, output_channels, budget)`.
Triggers ∈ {SCHEDULED, SATURATION, EXTERNAL}. Les opérations sont
un tuple non-vide de {REPLAY, DOWNSCALE, RESTRUCTURE, RECOMBINE}.
BudgetCap impose non-négativité finie (FLOPs, wall_time_s,
energy_j) par invariant K1.

### 4.4 Opérations — semi-groupe d'étapes de consolidation

L'ensemble d'opérations forme un semi-groupe libre
non-commutatif sous la composition `∘` avec budget additif (DR-2
compositionnalité, brouillon de preuve dans
`docs/proofs/dr2-compositionality.md`). Ordre canonique : replay →
downscale → restructure (séquentiel, ordre A-B-D des piliers) ;
recombine en parallèle (pilier C). L'analyse des paires
d'opérations (`docs/proofs/op-pair-analysis.md`) énumère les 16
paires, trouvant 12 paires croisées non-commutatives.

### 4.5 Axiomes DR-0..DR-4

- **DR-0 (redevabilité)** : chaque DE exécutée produit une
  `EpisodeLogEntry`, même en cas d'exception dans le handler
  (garantie try/finally).
- **DR-1 (conservation épisodique)** : chaque enregistrement β
  est consommé avant purge (τ_max borné).
- **DR-2 (compositionnalité — prouvée, ce papier)** : l'ensemble
  des opérations `Op = {replay, downscale, restructure,
  recombine}` sous composition `∘` avec budget additif forme un
  semi-groupe non commutatif engendré par les quatre primitives.
  Le théorème est établi en quatre étapes (fermeture du typage,
  additivité composante par composante du triplet de budget
  `(FLOPs, wall_time, energy_J)`, composition fonctionnelle de
  l'application `effect`, et associativité de la composition de
  fonctions) ; l'analyse de cas complète et les exemples de
  non-commutativité par paires d'opérations sont fournis dans
  `docs/proofs/dr2-compositionality.md`. Nous ne revendiquons pas
  la propriété universelle de liberté (absence de relations non
  triviales au-delà de l'associativité) — cette propriété reste
  ouverte et est orthogonale aux trois conjoints qui constituent
  DR-2 tel que formulé.
- **DR-3 (indépendance du substrat)** : Critère de Conformité
  exécutable = typage des signatures ∧ tests de propriété
  axiomatiques passants ∧ invariants BLOCKING applicables. Deux
  substrats indépendants satisfont les trois conditions (§5.6).
- **DR-4 (inclusion en chaîne des profils)** : P_min ⊆ P_equ ⊆ P_max
  pour les opérations et les canaux ; un lemme de monotonie
  (DR-4.L) donne l'ordre faible des résultats métriques attendus
  en espérance sur les classes métriques monotones en capacité.
  Preuve dans `docs/proofs/dr4-profile-inclusion.md`.

### 4.6 Invariants — I/S/K avec matrice d'application

- **I1** conservation épisodique (BLOCKING)
- **I2** traçabilité de la hiérarchie (BLOCKING)
- **I3** dérive distributionnelle des latents (WARN)
- **S1** non-régression du retenu (BLOCKING, garde de basculement)
- **S2** poids finis sans NaN/Inf (BLOCKING, garde de basculement)
- **S3** topologie valide (BLOCKING, garde de basculement)
- **S4** attention prior bornée (P_max uniquement)
- **K1** budget dream-episode (BLOCKING)
- **K3** latence de basculement bornée (WARN)
- **K4** couverture matrice d'évaluation au bump MAJOR (BLOCKING)

### 4.7 Versionnage DualVer formel+empirique

`C-vX.Y.Z+{STABLE,UNSTABLE}` — axe formel (FC) et axe empirique
(EC) bumpent indépendamment. Actuel : C-v0.5.0+STABLE
(cible post-G3 : C-v0.7.0+STABLE).

---

## 5. Approche de validation du Critère de Conformité

⚠️ **Indépendant du substrat par conception.** Le Paper 1 se limite
au contrat de conformité abstrait que toute implémentation conforme
doit satisfaire. Une instanciation empirique (l'implémentation de
référence du cycle 1) est rapportée dans le Paper 2.

### 5.1 Graphe de compilation déterministe

Un substrat conforme expose un graphe de compilation déterministe
pour chacune des quatre opérations, de sorte que la réexécution
avec la même graine produit des sorties bit-stables (contrat R1).
C'est la pré-condition la plus difficile pour que le run registry
puisse enregistrer un lot comme reproductible.

### 5.2 Ordonnanceur monothread avec registre de handlers

La redevabilité DR-0 exige que chaque dream-episode exécutée
produise une `EpisodeLogEntry` même en cas d'exception dans le
handler. Un ordonnanceur monothread avec un registre de handlers
par opération et un motif try/except/finally est la réalisation
canonique ; les variantes multithread doivent démontrer des
garanties de journalisation équivalentes.

### 5.3 Basculement atomique avec gardes d'invariants

La promotion de l'état awake doit être atomique et doit avorter
sur tout invariant BLOCKING violé (S1 non-régression du retenu, S2
finitude des poids, S3 validité topologique). Les substrats
conformes exposent une sortie de secours de style `SwapAborted`
clé par l'identifiant de l'invariant violé.

### 5.4 Inclusion en chaîne des profils

DR-4 exige que tout ensemble conforme de profils (P_min ⊆ P_equ
⊆ P_max) hérite des opérations et des canaux par inclusion. La
suite de tests de conformité livre des vérifications génériques
d'appartenance ; le câblage spécifique au substrat est rapporté
dans le Paper 2.

### 5.5 Pointeur d'implémentation de référence

Voir Paper 2 pour une instanciation empirique (implémentation de
référence basée MLX du cycle 1). Le Paper 1 ne prétend à aucune
implémentation spécifique au-delà du contrat formel ci-dessus.

### 5.6 Parcours de conformité inter-substrats — deux substrats satisfont le critère

Pour étayer l'assertion d'indépendance du substrat de DR-3 *à la
portée du Paper 1*, nous rapportons le résultat de l'application
du Critère de Conformité à deux substrats structurellement
divergents. L'objectif n'est pas une comparaison de benchmark
mais une démonstration que le critère est non vide : il accepte
des substrats qui diffèrent dans leurs primitives mathématiques
(descente de gradient sur tenseurs denses vs. dynamique de taux
de spike sur événements épars), tout en rejetant les
implémentations incorrectes via les invariants BLOCKING.

**Substrat 1 — MLX kiki-oniric (Track A, à base de gradients).**
Une implémentation de référence utilisant les opérations
tensorielles denses Apple MLX, des mises à jour de poids adaptées
LoRA et un tampon épisodique adossé à SQLite. Canonique pour le
cycle 1 ; source à `kiki_oniric/substrates/mlx_kiki_oniric.py`.

**Substrat 2 — E-SNN thalamocortical (numpy-LIF, à base de
spikes).** Un réseau neuronal spiking à intégration-et-tir avec
fuite et connectivité thalamocorticale, primitives codées en
taux et bascule de backend NORSE / nxNET. Développé spécifiquement
pour mettre à l'épreuve DR-3 sur un substrat *formellement
distant* du substrat 1 (pas de gradients dans les handlers
d'opérations ; l'état est constitué de trains de spikes, pas de
tenseurs de poids). Source à
`kiki_oniric/substrates/esnn_thalamocortical.py` ; rapport de
conformité de la porte G7 à
`docs/milestones/g7-esnn-conformance.md`.

**Tableau des conditions du critère.** Chaque condition de DR-3
est exercée par un ensemble discret de tests ; chaque substrat
reçoit un pass/fail indépendant par condition.

| Condition DR-3 | MLX kiki-oniric | E-SNN thalamocortical |
|----------------|-----------------|-----------------------|
| (1) Typage des signatures (4 handlers d'opérations, conformes au Protocol, constantes d'identité exportées) | ✅ 4 tests passent | ✅ 4 tests passent (`test_dr3_esnn_substrate.py`) |
| (2) Tests de propriété axiomatique (DR-0 redevabilité sur chaque DE exécutée ; DR-2 propriété non idempotente du downscale ; déterminisme R1 du recombine) | ✅ 3 tests passent | ✅ 3 tests passent |
| (3) Invariants BLOCKING applicables (garde S2 poids finis, garde S3 topologie, garde S1 non-régression du retenu) | ✅ 2 tests passent (S2, S3) | ✅ 2 tests passent (S2 sur LIF `state.v`, S3 sur la chaîne d'espèces) |
| **Global** | **Conforme** | **Conforme** |

DR-1 (conservation épisodique) et DR-4 (inclusion en chaîne des
profils) sont des propriétés indépendantes du substrat vérifiées
une fois au niveau du framework ; chaque substrat conforme en
hérite par construction.

**Interprétation.** Les deux substrats passent le Critère de
Conformité sous 9 tests de propriété axiomatique et d'application
d'invariants chacun (27 assertions au total). Le critère est donc
non vide au sens minimal — il distingue les implémentations
conformes des non-conformes (confirmé par les expériences
d'injection de fautes rapportées au §7.2). Le critère reste un
test *nécessaire* de conformité au framework ; une validation
empirique *suffisante* de l'assertion d'indépendance du substrat
sur une classe plus large de substrats (variantes SNN,
instanciations à base de transformers, matériel neuromorphique
analogique) est poursuivie dans le papier compagnon.

### 5.7 Esquisses de preuves — DR-0..DR-4

DR-0 prouvé par l'invariant try/finally du registre de handlers ;
DR-1 prouvé par la comptabilité de drainage du tampon β ; DR-2
prouvé comme théorème de semi-groupe engendré (preuve complète
dans `docs/proofs/dr2-compositionality.md`) ; DR-3 validé par le
Critère de Conformité exécutable sur deux substrats (§5.6) ;
DR-4 prouvé dans `docs/proofs/dr4-profile-inclusion.md` comme
inclusion en chaîne des opérations et canaux plus le lemme de
monotonie DR-4.L.

---

## 6. Méthodologie

### 6.1 Hypothèses pré-enregistrées (OSF)

Quatre hypothèses ont été pré-enregistrées sur l'Open Science
Framework (OSF) avant toute exécution empirique, selon le gabarit
Standard Pre-Data Collection. Le pré-enregistrement a été
verrouillé à S3 du cycle (référence calendaire) ; le DOI OSF est
cité dans les pages liminaires de l'article et se résout en un
enregistrement horodaté immuable.

- **H1 — Réduction de l'oubli** : `mean(forgetting_P_equ) <
  mean(forgetting_baseline)`. Test : t de Welch, unilatéral.
- **H2 — Équivalence P_max** : `|mean(acc_P_max) -
  mean(acc_P_equ)| < 0.05`. Test : deux tests unilatéraux (TOST).
  *Statut cycle 1* : test de fumée d'auto-équivalence uniquement
  (P_max en squelette).
- **H3 — Alignement monotone** : `mean(acc_P_min) <
  mean(acc_P_equ) < mean(acc_P_max)`. Test :
  Jonckheere-Terpstra. *Statut cycle 1* : deux groupes
  (P_min ↔ P_equ) uniquement.
- **H4 — Budget énergétique** : `mean(energy_dream / energy_awake)
  < 2.0`. Test : t à un échantillon contre seuil.

### 6.2 Tests statistiques + correction de Bonferroni

Tous les tests d'hypothèses utilisent un seuil de significativité
corrigé par Bonferroni : `α_par_hypothèse = 0.05 / 4 = 0.0125`.
Les quatre tests sont implémentés dans le module statistique de
l'implémentation de référence (qui enveloppe des bibliothèques
statistiques standard ; voir Paper 2 pour le chemin de code
spécifique au substrat) :

- **`welch_one_sided`** (H1) : `scipy.stats.ttest_ind` avec
  `equal_var=False`, p-value divisée par deux pour interprétation
  unilatérale.
- **`tost_equivalence`** (H2) : deux tests t unilatéraux manuels
  (borne inférieure `diff <= -ε` et borne supérieure
  `diff >= +ε`), rejet de H0 lorsque les deux passent à α (règle
  du max-p de TOST).
- **`jonckheere_trend`** (H3) : somme des comptes appariés de
  Mann-Whitney U à travers les groupes ordonnés, approximation z
  pour la p-value (pas de natif scipy).
- **`one_sample_threshold`** (H4) : `scipy.stats.ttest_1samp`
  contre `popmean=seuil`, p-value ajustée pour unilatéral
  (échantillon sous le seuil).

Tous les tests retournent un uniforme `StatTestResult(test_name,
p_value, reject_h0, statistic)` pour traitement en aval.

### 6.3 Banc de test mega-v2

Les exécutions empiriques utilisent le jeu de données **mega-v2**
(498 k exemples répartis sur 25 domaines : phonologie, lexique,
syntaxe, sémantique, pragmatique, etc.). Le cycle 1 stratifie un
**sous-ensemble retenu de 500 items** (20 items par domaine) et le
fige via un hash SHA-256 pour le contrat de reproductibilité R1.

Le banc de test retenu figé est chargé via
`harness.benchmarks.mega_v2.adapter.load_megav2_stratified()`, qui
bascule sur une substitution synthétique déterministe si le chemin
mega-v2 réel est indisponible. **Tous les résultats du cycle 1 au
§7 utilisent le repli synthétique ; l'intégration mega-v2 réelle
intervient en clôture du cycle 1 (S20+) ou au cycle 2.**

### 6.4 Alignement RSA IRMf (Studyforrest)

L'hypothèse H3 d'alignement représentationnel monotone est
évaluée par Analyse de Similarité Représentationnelle (RSA) entre
les activations de kiki-oniric et les réponses IRMf. Le cycle 1
utilise le jeu de données **Studyforrest** (Branche A verrouillée
à G1 — voir `docs/feasibility/studyforrest-rsa-note.md`) :

- **Format** : BIDS, distribué par DataLad, licence PDDL (ouvert).
- **Annotations** : 16 187 mots horodatés, 2 528 phrases, 66 611
  phonèmes ; vecteurs de mots STOP 300-d. Cartographiables sur
  les ortho-espèces (rho_phono / rho_lex / rho_syntax / rho_sem).
- **ROIs** : extraites via parcellations FreeSurfer + Shen-268
  pour STG, IFG, AG (le réseau langagier canonique).
- **Pipeline** : `nilearn` en mode CPU déterministe pour la
  reproductibilité R1. Ablation réelle différée à S20+ (inférence
  modèle réelle) ; le cycle 1 ne rapporte que la validation
  d'infrastructure.

### 6.5 Contrat de reproductibilité R1 + R3

La reproductibilité est appliquée par deux contrats :

- **R1 (run_id déterministe)** : chaque exécution est clé par un
  préfixe SHA-256 de 16 caractères de `(c_version, profile, seed,
  commit_sha)`. Réexécuter avec la même clé produit un `run_id`
  identique (vérifié par `harness.storage.run_registry`). La
  largeur a été portée de 16 → 32 caractères hex dans le commit
  `df731b0` après qu'une revue de code a signalé un risque de
  collision 64 bits à grande échelle.
- **R3 (adressabilité d'artefact)** : tous les bancs de test sont
  livrés avec un fichier d'intégrité `.sha256` apparié. Le
  chargeur `RetainedBenchmark` rejette tout fichier items dont le
  hash ne correspond pas à la référence figée, levant
  `RetainedIntegrityError`.

Le schéma de versionnage DualVer (axe formel FC + axe empirique
EC) tague chaque artefact avec la version du framework sous
laquelle il a été produit. Les résultats empiriques ne sont
valides que contre le `c_version` déclaré ; un bump FC-MAJOR
invalide EC et nécessite de réexécuter la matrice affectée.

---

## 7. Validation pipeline-et-framework

Cette section rapporte des résultats de validation au **niveau du
framework** : (i) le pipeline de mesure et statistique s'exécute
de bout en bout sur un jeu de données d'entrée scripté ; (ii) les
invariants BLOCKING S1–S3 rejettent les tentatives de basculement
volontairement faussées ; (iii) le contrat de reproductibilité R1
produit des hash de `run_id` déterministes pour des tuples de
paramètres appariés ; et (iv) les mesures de portabilité
inter-substrats d'un projet sœur sont résumées comme test
indépendant de l'assertion d'indépendance du substrat. **Aucune
décision d'hypothèse sur H1–H4 n'est annoncée ici** ;
l'évaluation empirique des hypothèses sur des bancs de test
d'apprentissage continu (mega-v2) et l'alignement IRMf
(Studyforrest / partenariats de laboratoire) est l'objet du
Paper 2.

### 7.1 Exécution de bout en bout du pipeline statistique

Les quatre tests pré-enregistrés (Welch unilatéral, équivalence
TOST, tendance Jonckheere–Terpstra, seuil à un échantillon) sont
encapsulés dans une interface `StatTestResult` uniforme dans
`harness.statistics.tests`. Sur une entrée stratifiée contrôlée
de 500 items aux niveaux de précision scriptés (50 % / 70 % /
85 % à travers baseline / P_min / P_equ ; `run_id
syn_s15_3_g4_synthetic_pipeline_v1`,
`docs/milestones/ablation-results.json`) :

- les quatre tests s'exécutent sans erreur ;
- chacun retourne une statistique au signe attendu, cohérente
  avec la direction scriptée du signal ;
- le booléen `StatTestResult.reject_h0` est calculé à partir du
  seuil α = 0,0125 corrigé par Bonferroni et est interprétable
  en aval par la logique de décision de porte.

*Interprétation.* Cela établit la chaîne statistique comme
opérationnelle sous la politique de comparaisons multiples
pré-enregistrée. Aucune p-value n'est rapportée ici : les
p-values calculées à partir de précisions scriptées sont non
informatives sur l'efficacité du framework et seraient trompeuses
si affichées. Les p-values réelles, avec des prédicteurs non
scriptés sur mega-v2, sont rapportées dans le Paper 2.

### 7.2 Injection de fautes sur les gardes d'invariants

Les trois invariants BLOCKING S1 (non-régression du retenu), S2
(poids finis, pas de NaN/Inf dans les poids scratch) et S3
(validité topologique) sont exercés par trois injections de
fautes dédiées sur chacun des deux substrats conformes :

| Invariant | Faute | Issue attendue | Observée (MLX) | Observée (E-SNN) |
|-----------|-------|------------------|----------------|------------------|
| S1 | Régression de poids sur le sous-ensemble retained-bench | `SwapAborted(S1)` + log dans `aborted-swaps/` | ✅ | ✅ |
| S2 | NaN injecté dans `W_scratch` | `SwapAborted(S2)` pré-S1 | ✅ | ✅ (sur `state.v`) |
| S3 | Boucle topologique auto-référente insérée | `SwapAborted(S3)` post-S1 | ✅ | ✅ (chaîne d'espèces) |

Les six injections de fautes ont avorté le basculement, journalisé
le bon ID d'invariant et laissé l'état d'éveil non promu. Voir
`tests/invariants/test_swap_guards.py` et le rapport de conformité
G7 pour les variantes spécifiques au E-SNN.

### 7.3 Déterminisme du registre des runs (R1)

Le contrat R1 attribue un préfixe SHA-256 déterministe de 32
caractères comme `run_id` clé sur `(c_version, profile, seed,
commit_sha, benchmark_version)`. Nous avons généré 1000 tuples
appariés à paramètres fixés et 1000 tuples avec un composant
unique perturbé ; les 1000 tuples appariés ont produit des hash
de `run_id` identiques, et les 1000 tuples perturbés ont produit
des hash distincts. Le registre applique l'intégrité d'artefact
R3 en rejetant tout `RetainedBenchmark` dont le SHA-256 ne
correspond pas à la référence figée ; cette garde s'est
déclenchée comme attendu lors d'une mutation délibérée du fichier
(`test_r3_integrity.py`). La largeur a été portée de 16 → 32
caractères hex dans le commit `df731b0` après une revue de code
ayant signalé un risque de collision 64 bits à l'échelle attendue
du cycle 2.

### 7.4 Portabilité inter-substrats — corroboration indépendante

Un projet sœur à `github.com/genial-lab/nerve-wml` (même byline,
même discipline de typage Protocol) mesure le polymorphisme
inter-substrats sur une interface Nerve séparée avec deux
substrats concrets : `MlpWML` (MLP dense) et `LifWML` (SNN à
intégration-et-tir avec fuite, gradient de substitution). La
mesure de la porte W de nerve-wml est un test du même *principe
d'indépendance du substrat* articulé ici comme DR-3. Nous
résumons les chiffres rapportés — détails complets dans le
preprint cité.

**Tâche linéairement séparable (FlowProxyTask 4 classes, pool
N = 4, multi-graine).** Les pools `MlpWML` et `LifWML` saturent
tous deux à une précision de 1,000 ; écart relatif 0,000. Les
deux substrats satisfont le protocole Nerve partagé.

**Tâche non linéaire (HardFlowProxyTask 12 classes, étiquette
projetée par XOR).** `MlpWML` atteint une précision de 0,547 ;
`LifWML` atteint une précision de 0,480. Écart absolu 0,067,
écart relatif **12,1 %** — au-dessus de la cible de 5 %. Cela
est divulgué dans le papier nerve-wml comme un résultat négatif
honnête et reflète le retard du décodeur actuel par appariement
de motifs cosinus dans la variante LIF sur les tâches non
linéaires.

**Porte M (test de fusion).** Un `MlpWML` entraîné contre un
Nerve mock et déployé contre un Nerve réel retient **1,000** de
sa précision baseline mock (critère ≥ 0,95), confirmant
l'interopérabilité inter-substrats de bout en bout.

*Interprétation.* Lu à la portée du Paper 1 : (a) le principe
d'indépendance du substrat de DR-3 est empiriquement traitable
et déjà démontré sur des tâches linéairement séparables dans un
système sœur ; (b) l'écart non linéaire de 12,1 % borne
l'enveloppe actuelle de portabilité inter-substrats et fixe une
cible cycle-2 explicite ; (c) la discipline de rapport honnête —
divulguer l'écart plutôt que le masquer — est préservée. Ces
chiffres ne constituent pas des preuves H1–H4 ; ils constituent
une preuve *méthodologique* que l'assertion d'indépendance du
framework est opérationnelle plutôt qu'asserte.

### 7.5 Synthèse

La validation du §7 établit quatre propriétés opérationnelles du
framework : le pipeline statistique pré-enregistré s'exécute
correctement sous correction de Bonferroni ; les invariants
BLOCKING S1–S3 imposent le comportement d'avortement attendu sous
injection de fautes sur les deux substrats conformes ; le contrat
de reproductibilité R1 produit des hash de `run_id`
déterministes à la largeur requise ; et les mesures
inter-substrats indépendantes dans un système sœur corroborent
le principe d'indépendance du substrat dans les régimes où il
est mesurable aujourd'hui. Aucune décision d'hypothèse
d'apprentissage continu sur H1–H4 n'est annoncée dans le
Paper 1 ; ces décisions sont l'objet du Paper 2.

---

## 8. Discussion

### 8.1 Contribution théorique

Notre framework C-v0.5.0+STABLE est, à notre connaissance, le
premier framework formel exécutable pour la consolidation mnésique
basée sur le rêve dans les systèmes cognitifs artificiels. En
axiomatisant les quatre piliers (replay (DR-1), downscaling
(DR-2), restructuring (DR-3), recombination (DR-4)) comme
opérations composables sur un semi-groupe libre à budget additif
(voir DR-2 dans `docs/proofs/dr2-compositionality.md`), nous
explicitons ce que les travaux antérieurs laissaient implicite :
l'**ordre et la composition** des opérations de consolidation
importent, et raisonner sur leurs interactions exige davantage que
des choix d'ingénierie ad hoc.

Le Critère de Conformité (DR-3) opérationnalise l'indépendance
du substrat : tout substrat qui satisfait le typage des signatures
+ les tests de propriété axiomatiques + l'applicabilité des
invariants BLOCKING hérite des garanties du framework. Ceci diffère
qualitativement des frameworks antérieurs qui lient la théorie à
une implémentation spécifique [@kirkpatrick2017overcoming;
@vandeven2020brain] — les détails d'implémentation sont discutés
dans le Paper 2. L'inclusion en chaîne des profils DR-4
(P_min ⊆ P_equ ⊆ P_max) structure en outre l'espace d'ablation de
telle sorte que les assertions expérimentales sur des profils plus
riches ne reposent pas par inadvertance sur des invariants de
profils plus faibles.

### 8.2 Contribution empirique

Le pipeline d'ablation synthétique (S15.3, run_id
`syn_s15_3_g4_synthetic_pipeline_v1`, dump
`docs/milestones/ablation-results.json`) démontre que la chaîne
d'évaluation statistique (Welch / TOST / Jonckheere / test t à un
échantillon sous correction de Bonferroni) est opérationnelle de
bout en bout sur un banc de test stratifié de 500 items. Trois des
quatre hypothèses pré-enregistrées passent à α = 0,0125 (H1
réduction de l'oubli, H4 conformité au budget énergétique, test
de fumée d'auto-équivalence H2), H3 tendance monotone atteignant
le seuil conventionnel 0,05 mais limite au niveau corrigé.

Bien que les valeurs rapportées soient des substitutions
synthétiques en attente de l'intégration des prédicteurs réels
mega-v2 + inférés par MLX (S20+), l'**infrastructure de mesure**
est elle-même validée : le chargeur RetainedBenchmark avec
intégrité SHA-256, le pont prédicteur `evaluate_retained`, le
harness AblationRunner et les quatre enveloppes statistiques
interopèrent proprement. Le lot synthétique ci-dessus est
enregistré sous le profil `G4_ablation` dans le registre projet
afin que le dump JSON reste traçable. Le contrat de
reproductibilité R1 (`run_id` déterministe depuis (c_version,
profile, seed, commit_sha)) est appliqué par le run registry.

### 8.3 Limites

Trois limites bornent la contribution du Paper 1 :

**(i) Aucune décision empirique au niveau des hypothèses.** Le
Paper 1 rapporte la validation du framework (pipeline,
invariants, reproductibilité, conformité inter-substrats) mais
**ne revendique pas** d'efficacité H1–H4 au niveau des hypothèses
sur des bancs de test d'apprentissage continu réels. Les chiffres
du §7 sont au niveau du framework et n'entraînent pas de p-values
par hypothèse sur des données d'apprentissage continu — celles-ci
sont différées au Paper 2, où les prédicteurs mega-v2 réels sur
le substrat de référence sont évalués sous le pré-enregistrement
verrouillé.

**(ii) Indépendance du substrat testée sur deux substrats, pas
encore sur la classe plus large.** Deux substrats indépendants
(MLX kiki-oniric et E-SNN thalamocortical) satisfont le Critère
de Conformité (§5.6), ce qui constitue un test *nécessaire* de
l'indépendance. Une validation *suffisante* sur la classe plus
large de substrats (instanciations à base de transformers,
variantes SNN profondes sur Loihi-2, matériel neuromorphique
analogique) relève du cycle 2.

**(iii) P_max en squelette uniquement.** Le profil P_max est
déclaré via des métadonnées (opérations cibles, canaux cibles)
mais ses handlers ne sont pas pleinement câblés. L'hypothèse H2
(équivalence P_max vs P_equ dans ±5 %) est donc une évaluation
cycle-2 ; dans le Paper 1, P_max n'est utilisé que pour
structurer l'inclusion en chaîne des profils DR-4, sans
revendiquer de parité empirique.

### 8.4 Comparaison avec l'état de l'art

| Travail antérieur | Contribution | Apport dreamOfkiki |
|-----------|--------------|----------------------|
| @vandeven2020brain (replay inspiré du cerveau) | Replay génératif pour CL | Replay est l'une des quatre opérations composées, avec budget explicite + non-commutativité avec downscale, restructure, recombine |
| @kirkpatrick2017overcoming (EWC) | Régularisation pondérée par Fisher protégeant les poids importants | Complémentaire plutôt que subsumée : EWC réalise un *schéma de pondération* au sein de la classe d'opérations `downscale`, pas la même primitive que la régulation à la baisse multiplicative SHY |
| @tononi2014sleep (SHY) | Thèse théorique de l'homéostasie synaptique | Opérationnalisée comme opération `downscale` à propriété empiriquement vérifiée commutative-mais-non-idempotente (§3.2) |
| @friston2010free (FEP) | Principe d'Énergie Libre | Opérationnalisé comme opération `restructure` avec garde topologique S3 ; interface étroite avec les comptes rendus d'inférence active (voir ci-dessous) |
| @hobson2009rem / @solms2021revising | Rêve créatif REM | Opérationnalisé comme `recombine` avec noyau VAE-allégé / d'interpolation et borne de dérive I3 |
| @mcclelland1995complementary (CLS) | Système dual hippocampe + néocortex | Intégré dans l'inclusion en chaîne des profils P_min ⊆ P_equ ⊆ P_max (DR-4) |
| Réseaux progressifs, PathNet [Rusu 2016, Fernando 2017] | Allocation de sous-réseaux spécifiques à la tâche | Orthogonal : ce sont des dispositifs CL *architecturaux* ; notre framework compose des *opérations* et admet de telles architectures comme substrats |
| MERLIN, MEMO, Gated Linear Networks [Wayne 2018, Banino 2020, Veness 2021] | CL multi-mécanismes avec mémoire explicite | Candidats substrats pour notre framework ; aucun ne satisfait un critère de conformité exécutable à travers les backends à base de gradients et de spikes |
| Dark Experience Replay, iCaRL, A-GEM [Buzzega 2020, Rebuffi 2017, Chaudhry 2019] | Baselines CL modernes basées sur la répétition | Tous réalisent l'opération `replay` seule ; DR-2 fournit une sémantique de composition formelle dont ils héritent par instanciation |
| Inférence active / agents de style PEARL [Tschantz 2020, Millidge 2021] | Modèles d'agents enracinés FEP composant perception + action | Complément : l'inférence active est compatible au niveau du framework avec notre opération `restructure` et le Critère de Conformité ; la conformité sur un tel agent est un travail futur explicite |
| Modèles du monde (Dreamer, DreamerV2/V3) [Hafner 2020–2023] | Agents à modèle du monde latent avec replay et apprentissage structurel | Recouvrement sur replay + restructure, mais pas de `recombine` ou `downscale` explicite dans leur ontologie ; substrat conforme possible avec ces deux primitives laissées en `no-op` ou absorbées dans l'objectif d'apprentissage |
| Famille JEPA [LeCun 2022 « A Path Towards Autonomous Machine Intelligence » ; Assran et al. 2023 I-JEPA ; Bardes et al. 2024 V-JEPA ; AMI Labs 2026] | Entraînement de modèles du monde prédictifs auto-supervisés sur des latents par embeddings joints ; « modèles du monde qui apprennent du réel, pas du langage » | Candidat substrat complémentaire : JEPA réalise la primitive `restructure` (Pilier D / FEP) à grande échelle sur des représentations latentes, sans contrepartie native explicite de `replay` ou `downscale`. Un parcours de conformité sur un substrat de classe JEPA (p. ex. encodeur vidéo V-JEPA avec ajout d'un tampon épisodique β et d'un downscale homéostatique) relève explicitement du cycle 2 ; le motif de portabilité inter-substrats démontré ici sur MLX + E-SNN est destiné à généraliser vers une telle configuration. |

Nos traits distinctifs par rapport à ce qui précède : **(a)** un
framework *formel* unifié couvrant les quatre piliers avec un
théorème de compositionnalité **prouvé** (DR-2), **(b)** un
Critère de Conformité *exécutable* empiriquement validé sur deux
substrats structurellement divergents (MLX dense + LIF spiking),
**(c)** une méthodologie d'évaluation pré-enregistrée avec bancs
de test figés et tuples `run_id` déterministes (contrat R1),
**(d)** des artefacts de science ouverte (code MIT, docs
CC-BY-4.0, DOI OSF, DOI Zenodo planifié pour les artefacts).
Nous ne revendiquons pas un nouvel *algorithme* d'apprentissage
continu ; la revendication est un *framework* contre lequel les
algorithmes peuvent être spécifiés, composés et comparés avec
des garanties au niveau du contrat.

---

## 9. Travaux futurs

### 9.1 Substrat E-SNN (thalamocortical Loihi-2)

L'extension la plus directe du cycle 1 consiste à valider le
Critère de Conformité DR-3 sur un second substrat : un réseau
neuronal spiking thalamocortical (E-SNN) déployé sur le matériel
neuromorphique Intel Loihi-2. Ceci a été différé du cycle 1 selon
la décision de SCOPE-DOWN (master spec §7.3) pour assurer que le
cycle 1 se clôture à temps avec une validation sur substrat
unique.

Le substrat E-SNN testerait si les axiomes exécutables du framework
restent opérationnels lorsque les opérations sont réalisées comme
dynamiques de taux de spike plutôt que comme mises à jour par
gradient sur matrices denses. Une conformité réussie apporterait
la preuve d'indépendance du substrat que le Paper 1 revendique
comme propriété théorique mais ne démontre pas encore
empiriquement sur deux substrats.

### 9.2 Câblage réel du profil P_max

Le cycle 1 n'implémente P_max qu'en squelette (`status="skeleton"`,
`unimplemented_ops=["recombine_full"]`). Le cycle 2 câblera les
composants restants :

- **Traces brutes du flux α** canal d'entrée (actuellement déclaré
  P_max-uniquement mais non consommé) — requiert un tampon
  circulaire firehose avec rétention bornée
- **Canal de sortie canal-4 ATTENTION_PRIOR** — requiert
  l'invariant de bornage de l'attention prior (S4) et le câblage
  en aval vers les modules consommateurs
- **Variante d'opération `recombine_full`** — paire complète
  d'encodeur / décodeur VAE au-delà du squelette d'interpolation
  allégée C-Hobson

Avec P_max réellement câblé, l'hypothèse H2 (équivalence P_max vs
P_equ dans ±5 %) devient une comparaison réelle plutôt que le test
de fumée d'auto-équivalence du cycle 1.

### 9.3 Partenariat réel avec un laboratoire IRMf

Le cycle 1 verrouille Studyforrest comme repli IRMf (G1 Branche
A). Le cycle 2 poursuit un partenariat actif avec un ou plusieurs
laboratoires IRMf identifiés via la campagne de recrutement de
relecteurs T-Col :

- **Huth Lab** (UT Austin) : jeu de données Narratives
- **Norman Lab** (Princeton) : études de mémoire épisodique
- **Gallant Lab** (UC Berkeley) : BOLD piloté par stimuli
  naturalistes

Un partenariat réel avec un laboratoire permettrait la RSA sur
des stimuli linguistiques **contrôlés par tâche** plutôt que sur
le repli de compréhension narrative fourni par Studyforrest. Ceci
renforcerait H3 (alignement représentationnel monotone) qui n'a
atteint qu'une significativité limite dans la validation du
pipeline synthétique du cycle 1 (run_id
`syn_s15_3_g4_synthetic_pipeline_v1`, dump
`docs/milestones/ablation-results.json`).

### 9.4 Validation multi-substrat du Critère de Conformité

L'assertion théorique la plus forte du Framework C-v0.5.0+STABLE
— l'indépendance du substrat via le Critère de Conformité DR-3 —
nécessite une validation empirique sur plus de deux substrats pour
être défendable en relecture par les pairs. Le cycle 2 établit la
matrice de validation : pour chaque substrat candidat
(implémentation de référence du cycle 1 ✅, E-SNN, instance
hypothétique basée sur transformer), vérifier les trois conditions
de conformité (typage des signatures, tests de propriété
axiomatiques passants, invariants BLOCKING applicables).

Une suite de tests de conformité réutilisable (brouillonnée au
cycle 1 sous `tests/conformance/`) constitue le fondement. Le
cycle 2 l'étendra avec des adaptateurs spécifiques au substrat et
exécutera la suite complète contre chaque substrat candidat,
produisant un rapport de conformité publiable comme artefact
supplémentaire pour le Paper 1 (ou comme contribution principale
de l'article d'ablation ingénierie du Paper 2).

---

## 10. Références

→ Voir `references.bib` (16 entrées stub cycle-1, sera étendue à
~30-40 en S20-S22 au fur et à mesure du rendu du brouillon
complet). Intégration BibTeX via `\bibliography{references}` dans
le rendu LaTeX.

Citations clés (alphabétique) :
- Diekelmann & Born 2010 (mémoire du sommeil)
- French 1999 (oubli catastrophique)
- Friston 2010 (FEP)
- Hobson 2009 (rêve REM)
- Kirkpatrick 2017 (EWC)
- McClelland 1995 (CLS)
- McCloskey & Cohen 1989 (oubli)
- Rao & Ballard 1999 (codage prédictif)
- Rebuffi 2017 (iCaRL)
- Shin 2017 (replay génératif)
- Solms 2021 (conscience)
- Stickgold 2005 (consolidation)
- Tononi & Cirelli 2014 (SHY)
- van de Ven 2020 (replay inspiré du cerveau)
- Walker & Stickgold 2004 (consolidation)
- Whittington & Bogacz 2017 (codage prédictif)

---

## Récapitulatif du compte de mots (cible : ~5000 mots corps + supp)

| Section | Cible | Statut |
|---------|--------|--------|
| §1 Résumé | ≤250 | rédigé (~265, à resserrer) |
| §2 Introduction | ≤1500 | rédigé (~1200) |
| §3 Contexte théorique | ≤1500 | rédigé (~1500) |
| §4 Framework | condensé en corps + réf spec | fait |
| §5 Implémentation | condensé | fait |
| §6 Méthodologie | ≤1500 | rédigé (~1500) |
| §7 Résultats | ≤2000 | rédigé (placeholder) |
| §8 Discussion | ≤1500 | rédigé (~1500) |
| §9 Travaux futurs | ≤700 | rédigé (~700) |
| §10 Références | s.o. | 16 entrées stub |

**Total estimé** : ~10000 mots (nécessite un resserrement agressif
pour la discipline Nature HB 5000-mots corps principal ; le
supplément peut absorber le dépassement).

---

## Notes pour révision

- Rendu via Quarto / pandoc en PDF + LaTeX pour soumission arXiv
  (S21.1)
- Insérer le DOI OSF au §6.1 une fois le verrouillage OSF terminé
- Remplacer les substitutions synthétiques au §7 par les valeurs
  d'ablation réelles post S20+
- Resserrer le §1 résumé à ≤250 mots
- Resserrer §3 + §6 + §8 pour tenir dans le budget global du corps
  principal
- Ajouter les Figures (1 schéma d'architecture, 2 boxplot
  résultats, 3 tendance Jonckheere, 4 conceptuelle des quatre
  piliers)
- Rendu BibTeX avec appels `\cite{}` appropriés
