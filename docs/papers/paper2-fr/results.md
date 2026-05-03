<!--
SPDX-License-Identifier: CC-BY-4.0
Authorship byline : Saillant, Clément
License : Creative Commons Attribution 4.0 International (CC-BY-4.0)
-->

# §7 Résultats (Article 2, brouillon C2.15)

**Signataires** : *Saillant, Clément*
**Licence** : CC-BY-4.0

**Cible de longueur** : ~2 pages markdown (≈ 1700 mots)

---

> ⚠️ **Précaution (substitution synthétique — pas de
> revendication empirique).**  Chaque revendication
> quantitative dans cette section est produite par un
> prédicteur mock Python partagé, canalisé à travers deux
> enregistrements de substrats (MLX + squelette numpy LIF
> E-SNN). L'accord inter-substrats est trivialement OUI
> **par construction** ; les valeurs valident le pipeline
> de réplication inter-substrats, pas l'efficacité
> empirique du framework sur des données réelles. Le vrai
> câblage vers l'inférence spécifique au substrat est un
> livrable du cycle 3. Source :
> `docs/milestones/cross-substrate-results.md` (commit
> `fa0f26e`), régénérée par
> `scripts/ablation_cycle2.py`.

---

## 7.1 Provenance (substitution synthétique — pas de revendication empirique)

La matrice inter-substrats cycle-2 est enregistrée sous les
identifiants suivants (verbatim de
`docs/milestones/cross-substrate-results.md`) :

| champ | valeur |
|-------|--------|
| harness_version | `C-v0.6.0+PARTIAL` |
| cycle2_batch_id | `3a94254190224ca82c70586e1f00d845` |
| ablation_runner_run_id | `45eccc12953e758440fca182244ddba2` |
| benchmark | mega-v2 stratifié, 500 items synthétiques |
| benchmark_hash | `synthetic:c8a0712000b641...` |
| seeds | `[42, 123, 7]` |
| substrats | `mlx_kiki_oniric`, `esnn_thalamocortical` |
| data_provenance | synthétique — prédicteurs mock partagés entre étiquettes de substrats |

Reproductibilité : exécuter
`uv run python scripts/ablation_cycle2.py` depuis
`C-v0.6.0+PARTIAL` doit régénérer un JSON identique
(byte-stable) et un Markdown identique (modulo la
sérialisation déterministe) sous la grille de seeds
fixée.

## 7.1.1 Pilote G4 (première évidence empirique — 2026-05-03)

Le pilote G4 est le premier résultat **non synthétique** de la
§7. Le balayage est `4 bras × 5 graines = 20 cellules` sur
l'apprentissage continu Split-FMNIST 5 tâches, substrat MLX
(`kiki_oniric.substrates.mlx_kiki_oniric`), pilote
`experiments/g4_split_fmnist/run_g4.py`. Pré-enregistrement :
[`docs/osf-prereg-g4-pilot.md`](../../osf-prereg-g4-pilot.md).
Dépôt du jalon :
[`docs/milestones/g4-pilot-2026-05-03.{json,md}`](../../milestones/g4-pilot-2026-05-03.md).

Trois hypothèses pré-enregistrées :

- **H1** : Hedges' g observé de `(rétention[P_equ] vs
  rétention[baseline])` ≥ borne inférieure de l'IC 95 % Hu 2020
  (0,21). Observé `g_h1 = 0.0000` ; dans l'IC 95 % Hu 2020 :
  `False` ; p Welch unilatérale
  (α/3 = 0,0167) `0.5000` → reject_h0 = `False`.

- **H3** : |Hedges' g| observé de `(rétention[P_min] vs
  rétention[baseline])` ≥ borne inférieure de l'IC 95 %
  Javadi 2024 (0,13), signe décrément (g ≤ -0,13). Observé
  `g_h3 = 0.0000` ; rejet côté décrément = `False`.

- **H_DR4** : ordre monotone `rétention moyenne[P_max] ≥
  rétention moyenne[P_equ] ≥ rétention moyenne[P_min]`
  (Jonckheere-Terpstra). Monotonie observée = `True` (moyennes
  égales à 0,5988 sur les trois profils) ; p unilatérale =
  `0.5000` → reject_h0 = `False`.

Le wrapper `dream_episode()` dispatche le jeu d'opérations de
chaque profil via `profile.runtime.execute(...)` pour la
traçabilité DR-0 ; dans ce pilote minimal il **ne mute pas** les
poids du classifieur, ce qui explique que les quatre bras
produisent une rétention identique par graine. Le pilote établit
donc le plancher empirique : effet nul sous un wrapper de pure
journalisation DR-0, avec p Welch = 0,5 pour tous les tests
appariés. Un couplage mutant les poids (par ex. delta LoRA
piloté par les rêves ou buffer de replay alimentant
l'optimiseur) constitue le premier suivi.

À N = 5 / bras, ce pilote est **exploratoire** pour les
magnitudes absolues de g (g minimal détectable à 80 % de
puissance ≈ 1,4). Franchir ou rester sous la borne inférieure
de l'IC Hu / Javadi dans ce pilote est traité, conformément au
pré-enregistrement §4, comme un **signal d'ordonnancement**
pour un suivi confirmatoire N ≥ 30, et non comme une
confirmation / falsification empirique finale.

Traçabilité R1 : chaque cellule porte un `run_id` déterministe
de 32 chiffres hex enregistré dans
`harness/storage/run_registry.RunRegistry`. Ré-exécuter
`experiments/g4_split_fmnist/run_g4.py` sur le même tuple
`(c_version, profile, seed, commit_sha)` est stable au bit près
sur Apple Silicon M3 Ultra (vérifié 2026-05-03, identifiants
identiques sur deux balayages consécutifs).

## 7.1.2 Pilote G4-bis (re-exécution avec couplage de replay — 2026-05-03)

Le pilote G4 est ré-exécuté après que le wrapper
`dream_episode()` soit câblé pour modifier les poids du
classifieur (Plan G4-bis). Le tampon β se remplit de 32 paires
image-classe brutes par tâche complétée (capacité 256) ; entre
tâches, les bras dream-actifs exécutent un replay (n=32
enregistrements × 1 pas SGD à lr=0.01) et un downscale SHY
(facteur 0.95). Le balayage des bras, des graines et les
hypothèses pré-enregistrées sont inchangés. Dump du jalon :
[`docs/milestones/g4-pilot-2026-05-03-bis.{json,md}`](../../milestones/g4-pilot-2026-05-03-bis.md).

Trois hypothèses pré-enregistrées (réévaluées sous couplage) :

- **H1** : `g_h1 observé = -2.3067`. Dans l'IC à 95 % de
  Hu 2020 : `False` ; p unilatéral de Welch (α/3 =
  0.0167) `0.9973` → rejet_h0 = `False`.

- **H3** : `g_h3 observé = -2.3067`. Rejet côté décrément
  (g ≤ -0.13) : `True`.

- **H_DR4** : `rétention moyenne[P_max] = 0.5609`,
  `rétention moyenne[P_equ] = 0.5609`,
  `rétention moyenne[P_min] = 0.5609`. Ordre monotone :
  `True (égalité dégénérée)` ; p unilatéral de Jonckheere =
  `0.5000` → rejet_h0 = `False`.

Les `run_id` du dump bis diffèrent du dump original
2026-05-03 parce que la sémantique de `dream_episode()` a
changé — le couplage fait partie du tuple d'entrée *en
esprit*, même si la clé R1 formelle reste `(c_version,
profile, seed, commit_sha)`. Le dump original 2026-05-03
est conservé comme référence du baseline-spectateur.

Mise en garde importante sur la distribution des cellules
bis : les trois bras dream-actifs (P_min, P_equ, P_max) ont
produit des vecteurs de rétention bit-identiques sur les cinq
graines. La tête MLP binaire n'expose que les canaux de
couplage REPLAY + DOWNSCALE ; les opérations RESTRUCTURE et
RECOMBINE enregistrées pour P_equ / P_max restent
spectateur-seulement sur ce substrat (pas de hiérarchie ni
de latents VAE à muter). Le drapeau monotone de H_DR4 est
donc une égalité *dégénérée*, pas un ordonnancement
substantiel — H_DR4 reste non-testable sur ce substrat à
cette échelle.

Avec N = 5 / bras, ce pilote reste exploratoire pour les
amplitudes absolues de g ; le pré-enregistrement §4
déclenche toujours un suivi confirmatoire N ≥ 30 avant toute
promotion STABLE.

## 7.1.3 Pilote G5 cross-substrat (E-SNN thalamo-cortical — 2026-05-03)

Premier pilote empirique cross-substrat du framework C : le
balayage 20-cellules du §7.1.2 a été reproduit sur le substrat
E-SNN thalamo-cortical (`kiki_oniric.substrates.esnn_thalamocortical`,
fallback LIF en numpy ; aucune dépendance à `norse`).
Pré-enregistrement :
[`docs/osf-prereg-g5-cross-substrate.md`](../../osf-prereg-g5-cross-substrate.md)
verrouillé au commit `1411228` avant l'exécution du pilote au
`5fb36f0`. Jalon associé :
[`docs/milestones/g5-cross-substrate-2026-05-03.md`](../../milestones/g5-cross-substrate-2026-05-03.md).

Trouvaille intra-substrat (E-SNN, ancres [@hu2020tmr ;
@javadi2024sleeprestriction]) :
- rétention moyenne `P_min = P_equ = P_max = 0,5119` à travers
  les trois profils oniriques, reproduisant le motif spectateur
  du §7.1.2 ;
- `g_h1 = 0`, `g_h3 = 0`, `H_DR4` trivialement monotone (égalité
  dégénérée).

Agrégat cross-substrat (Welch bilatéral par bras au seuil
Bonferroni α/4 = 0,0125, jalon associé
[`g5-cross-substrate-aggregate-2026-05-03.md`](../../milestones/g5-cross-substrate-aggregate-2026-05-03.md)) :

| Bras | g (MLX − E-SNN) | p Welch bilatéral | rejet H₀ ? |
|------|-----------------|---------------------|-------------|
| baseline | 9,98 | 5,2 × 10⁻⁵ | OUI |
| P_min | 3,49 | 3,5 × 10⁻³ | OUI |
| P_equ | 3,49 | 3,5 × 10⁻³ | OUI |
| P_max | 3,49 | 3,5 × 10⁻³ | OUI |

`dr3_cross_substrate_consistency_ok = False`. Les deux substrats
**divergent en niveau absolu de rétention** : la baseline MLX
atterrit à 0,5988 (G4-bis), la baseline E-SNN à 0,5119 — un
écart de rétention de ≈ 0,087 qui est lui-même substantivement
plus grand que l'effet spectateur intra-substrat.

**Lecture honnête**. Les deux substrats exhibent *le même motif
spectateur qualitatif* à l'intérieur de chacun (P_min ≡ P_equ ≡
P_max), donc la conformité DR-3 au niveau axiomatique (DR-0 /
DR-1 / DR-2' / DR-4) tient sur les deux. Le verdict
cross-substrat sur la rétention *absolue* est divergence : la
dynamique de taux-de-spike E-SNN produit une rétention absolue
uniformément plus basse que la tête MLP MLX, donc les valeurs
moyennes ne correspondent pas à travers les substrats. Cela
exclut une mise à niveau simple de
`docs/proofs/dr3-substrate-evidence.md` de « substitut
synthétique » à « preuve empirique réel-substrat » *au niveau
de la rétention absolue*. Une lecture plus grossière au niveau
*motif qualitatif* (les deux substrats exhibent le motif
spectateur) est cohérente avec la substrat-agnosticité, mais ce
n'est pas le test que le pré-enregistrement spécifiait.

La trouvaille positive G4-ter sur tête riche (g_h2 = + 2,77 sur
MLX hiérarchique, §7.1.5) n'est pas encore portée vers E-SNN ;
un futur G5-bis ré-exécuterait ce protocole sur un classifieur
E-SNN hiérarchique et demanderait si l'effet onirique lui-même
transfère à travers les substrats.

## 7.1.4 Pilote G6 Voie B (LLM réel, flux d'apprentissage continu — 2026-05-03)

Première exposition de
`kiki_oniric.substrates.micro_kiki.MicroKikiSubstrate` à un flux
d'apprentissage continu sur 5 sous-domaines MMLU (anatomy →
astronomy → business_ethics → clinical_knowledge → college_biology)
sous les quatre bras (`baseline`, `P_min`, `P_equ`, `P_max`). Le
pilote a tourné sur un hôte Apple M1 Max / 32 Go où
`KIKI-Mac_tunner` est absent ; conformément au document de
décisions verrouillé
(`docs/milestones/g6-pilot-decisions-2026-05-03.md`), la
**Voie B (mutation inference-only des tenseurs LoRA)** a été
retenue. La Voie A (Studio + `mlx_lm.lora` fine-tuning LoRA réel)
est planifiée comme travail futur.

**Pré-enregistrement** : `docs/osf-prereg-g6-pilot.md` (verrouillé
le 2026-05-03). Le pré-enregistrement intègre un amendement
explicite de H_NEW étant donné l'inversion de signe de G4-bis
(g_h1 = −2,31) : H_NEW est reformulé comme validation
d'infrastructure exploratoire, non comme test de marge de
non-infériorité. Conformément au pré-enregistrement §6, la Voie B
ne déclenche jamais une promotion STABLE / UNSTABLE de l'axe EC,
quel que soit le résultat.

**Cellules** : 4 bras × 3 graines = 12 séquences, chacune
touchant 5 sous-domaines (60 mesures d'oubli). Temps de calcul :
0,02 s sur M1 Max (le proxy inference-only est un calcul de
norme L2 numpy, pas une vraie inference Qwen).

**Scalaires de verdict** (cf.
`docs/milestones/g6-pilot-pathB-2026-05-03.json`,
commit épinglé dans le jalon) :

| Hypothèse | Direction | Ancrage | Observé | Rejet H0 |
|---|---|---|---|---|
| H1' P_equ vs baseline | g ≥ 0,21 | Hu 2020 borne inf CI | g_h1' = 0,0 | Faux |
| H3' P_min vs baseline | g ≤ −0,13 | Javadi 2024 borne inf CI | g_h3' = 0,0 | Faux |
| H_DR4' Jonckheere | mono(P_min, P_equ, P_max) | DR-4 | Vrai (dégénéré, moyennes égales) | Faux |
| H_NEW (amendé) | toute différence non-nulle | seuil infrastructure 1e-6 | max\|Δ\| = 0,0 | Faux (aucun effet) |

**Schéma de spectateur observé** : les quatre handlers oniriques
opèrent sur des charges utiles synthétiques construites par
`experiments.g6_mmlu_stream.dream_wrap.build_episode_payload`, et
non sur le tampon vivant `InferenceOnlyAdapter._deltas`. Les
tenseurs retournés par les handlers ne sont **pas** réinjectés
dans le delta de l'adaptateur — les estampilles DR-0 / DR-1
atterrissent sur les dataclasses `_recombine_state` /
`_restructure_state` du substrat, mais la surface d'évaluation
au runtime (proxy d'exactitude Voie B piloté par norme L2) voit
un delta d'adaptateur identique à travers les quatre bras pour
chaque graine. Le g de Hedges s'effondre à 0,0 ; le test de
monotonicité Jonckheere est dégénéré (moyennes égales).

Cela reproduit le schéma de spectateur G4 (pré-couplage, cf.
§7.1.1) et constitue le résultat attendu lorsque les handlers
oniriques opèrent sur des charges synthétiques disjointes de la
surface d'évaluation. La leçon renforce le pivot G4 → G4-bis :
un différentiel d'oubli authentique exige que les tenseurs
retournés par les handlers mutent la surface d'entraînement
(fine-tune LoRA réel Voie A ; ou extension Voie B où les sorties
des handlers alimentent `adapter.set_delta`).

**Verdict G6 Voie B** : la forme du pipeline est validée
(60 mesures d'oubli, 12 `run_id` R1 enregistrés dans
`.run_registry.sqlite`, déterminisme inter-exécutions). Les
handlers du substrat sont honnêtement spectateurs uniquement sur
l'adaptateur inference-only. EC reste PARTIAL ; pas de bump FC.
La Voie A sur Studio reste le chemin G6 publiable.

Référence : `docs/superpowers/plans/2026-05-03-g6-micro-kiki-mmlu-cl.md`
+ `docs/osf-prereg-g6-pilot.md`
+ `docs/milestones/g6-pilot-pathB-2026-05-03.{json,md}`.

## 7.1.5 Pilote G4-ter (balayage HP + substrat enrichi — 2026-05-03)

Le pilote G4-ter est le suivi confirmatoire N≥30 prévu par la règle
d'évidence-positive-exploratoire de G4-bis
(`docs/osf-prereg-g4-pilot.md` §4) et pré-enregistré dans
`docs/osf-prereg-g4-ter-pilot.md`. Il distingue trois explications
concurrentes du résultat nul de G4-bis (`g_h1 = -2.31`, `H_DR4`
moyennes égales dégénérées) :

- **H1 — artefact HP** : la combinaison HP de G4-bis
  (`replay_lr=0,01`, `replay_n_steps=1`, `downscale_factor=0,95`)
  est trop agressive. Un balayage de sous-grille curée à 10 combos
  sur la tête MLP binaire (300 cellules, N=10 par cellule) donne
  un meilleur Hedges' g de `+11,81` à la combinaison `C9`
  (`downscale_factor=0,99`, `replay_batch=64`,
  `replay_n_steps=10`, `replay_lr=0,05`) avec `g_hp_best > 0,21`
  (au-dessus de la borne basse de l'IC Hu 2020). H0_HP **rejetée**
  (criblage, N=10 par combo).
- **H2 — limitation au niveau du substrat** : la tête binaire
  expose uniquement les canaux de couplage REPLAY + DOWNSCALE.
  Une tête hiérarchique (entrée → 32 → 16 → sortie) qui expose
  `_l2.weight` comme site RESTRUCTURE et les activations
  hidden_2 comme source d'échantillonnage Gaussienne-MoG pour
  RECOMBINE produit `g_h2 = +2,77` à la combinaison ancre C5 sur
  120 cellules (N=30 par bras), Welch unilatéral
  p = 4,9e-14 ≪ α/4 = 0,0125. H0_substrat **rejetée**.
- **H_DR4-ter — monotonicité** : le substrat enrichi
  structurellement-distingué brise partiellement le tie dégénéré
  de G4-bis — la rétention de référence chute à 0,5869 (n=30)
  tandis que les trois bras de rêve se regroupent autour de
  0,7046–0,7065. Rétention moyenne `P_min = 0,7065`,
  `P_equ = 0,7046`, `P_max = 0,7046` (Jonckheere J = 1335,0,
  p = 0,544 à α/4 = 0,0125). L'ordre prédit
  `P_max ≥ P_equ ≥ P_min` n'est **pas observé** (P_min dépasse
  marginalement les autres). H_DR4-ter est **non concluante** —
  ni monotonicité-rejetée, ni falsifiée.

Le verdict verrouille l'axe EC à `PARTIAL` selon la table DualVer
de la pré-enregistration §7 (H_DR4-ter non concluante, même si H1
et H2 rejettent H0). Les profils de rêve à tête enrichie battent la
référence sans rêve par une large marge (g ≈ 2,8 ≫ borne ancre Hu
2020 = 0,21), mais l'**ordre entre profils de rêve** est lessivé à
l'ancre C5. Un suivi confirmatoire N≥95 (G4-quater) est prévu pour
tester si substrat enrichi × `hp_best=C9` retrouve l'ordre prédit
P_max ≥ P_equ ≥ P_min.

Les clés de profil du registre de runs
`g4-ter/{richer,hp}/<bras>/<combo>` identifient chaque cellule pour
satisfaire R1.

Provenance :
- Pré-enregistration : `docs/osf-prereg-g4-ter-pilot.md`
- Jalon (md) : `docs/milestones/g4-ter-pilot-2026-05-03.md`
- Jalon (json) : `docs/milestones/g4-ter-pilot-2026-05-03.json`
- Pilote : `experiments/g4_ter_hp_sweep/run_g4_ter.py`
- Substrat : `experiments.g4_ter_hp_sweep.dream_wrap_hier.G4HierarchicalClassifier`
- Grille HP : `experiments.g4_ter_hp_sweep.hp_grid.HP_COMBOS`

## 7.2 Table comparative inter-substrats H1-H4 (substitution synthétique — pas de revendication empirique)

**Table 7.2 — MLX vs E-SNN hypothèses à Bonferroni
α = 0,0125 (substitution synthétique — pas de
revendication empirique).**

| hypothèse | test | p-value MLX | verdict MLX | p-value E-SNN | verdict E-SNN | accord |
|-----------|------|-------------|-------------|---------------|---------------|--------|
| H1 oubli | Welch unilatéral | 0,0000 | rejet H0 | 0,0000 | rejet H0 | OUI |
| H2 auto-équivalence | TOST (ε = 0,05) | 0,0000 | rejet H0 | 0,0000 | rejet H0 | OUI |
| H3 monotonicité | Jonckheere-Terpstra | 0,0248 | échec à rejeter | 0,0248 | échec à rejeter | OUI |
| H4 budget énergie | t un-échantillon (sup) | 0,0101 | rejet H0 | 0,0101 | rejet H0 | OUI |

- Nombre significatif MLX : **3 / 4**.
- Nombre significatif E-SNN : **3 / 4**.
- **Entièrement cohérent entre substrats** : **OUI**
  *(substitution synthétique — trivialement OUI par
  construction à prédicteur partagé)*.

### Guide de lecture de légende (substitution synthétique — pas de revendication empirique)

- H3 échoue à α = 0,0125 sur les deux substrats car le
  prédicteur mock émet une accuracy constante par profil
  (pas de dispersion par seed). La statistique de tendance
  Jonckheere-Terpstra ne peut pas franchir le seuil de
  Bonferroni sur un échantillon dégénéré. C'est une
  **propriété du prédicteur mock**, pas du framework.
- H2 passant comme `rejet H0` est une **passe de fumée
  d'auto-équivalence** héritée de l'Article 1 §7.6 ; le
  fait que P_max soit câblé en cycle 2 ne change pas le
  résultat à prédicteur partagé car le prédicteur ne
  distingue pas P_equ de P_max sous le mock.
- H1 et H4 rejetant H0 est la conception scriptée du
  prédicteur mock : le mock émet une accuracy plus grande
  sous P_equ que sous la baseline (donc
  `forgetting_P_equ < forgetting_baseline` tient) et émet
  un ratio d'énergie sous-seuil (donc le t un-échantillon
  rejette à la hausse).

## 7.3 Matrice d'accord (substitution synthétique — pas de revendication empirique)

**Table 7.3 — Accord de verdicts par hypothèse entre
substrats (substitution synthétique — pas de
revendication empirique).**

| hypothèse | verdicts égaux ? | rejet MLX | rejet E-SNN |
|-----------|------------------|-----------|-------------|
| H1 oubli | OUI | true | true |
| H2 auto-équivalence | OUI | true | true |
| H3 monotonicité | OUI | false | false |
| H4 budget énergie | OUI | true | true |

Les quatre hypothèses s'accordent entre substrats
(4 / 4 d'accord). *(substitution synthétique.)*  C'est le
résultat attendu quand le prédicteur sous-jacent est
identique entre les lignes de substrats — cela montre que
le chemin stats par substrat est correctement câblé et
émet des verdicts identiques sur des entrées identiques.
Une réplication à prédicteur divergent est la cible de
câblage réel cycle-3.

## 7.4 Matrice de Conformité DR-3 (référence croisée vers §4)

**Table 7.4 — 3 conditions × 2 substrats réels + 1
placeholder.**

| substrat | C1 typage signature | C2 tests propriétés axiomes | C3 invariants BLOCKING |
|----------|--------------------|-----------------------------|------------------------|
| `mlx_kiki_oniric` | PASS | PASS | PASS |
| `esnn_thalamocortical` | PASS *(substitution synthétique)* | PASS *(substitution synthétique — pas de matériel Loihi-2)* | PASS *(substitution synthétique — numpy LIF taux de spikes)* |
| `hypothetical_cycle3` | N/A | N/A | N/A |

Légende : matrice de Conformité DR-3 (source :
`docs/milestones/conformance-matrix.md`, commit `fd54df7`).
La ligne E-SNN porte le drapeau de substitution
synthétique selon §4.3. La ligne `hypothetical_cycle3`
est explicitement placeholder-seulement et ne doit pas
être lue comme passant / échouant. Cette matrice est la
preuve *architecturale* pour DR-3 ; la table H1-H4 du §7.2
est la preuve du *pipeline de réplication*. Les deux sont
nécessaires pour la revendication composite de
l'Article 2.

## 7.5 Ce que le §7 établit et ne établit pas

**Ce que le §7 établit :**

- Le pipeline de réplication inter-substrats s'exécute de
  bout en bout sur deux enregistrements de substrats
  structurellement distincts (§7.1 provenance + §7.2
  parité de p-values).
- Le chemin statistique par substrat émet des verdicts
  identiques sur des entrées identiques (§7.3 matrice
  d'accord).
- La matrice de Conformité DR-3 exhibe deux lignes de
  substrats PASSant (§7.4).
- Le contrat de reproductibilité R1 tient : chaque run_id
  au §7.1 circule aller-retour vers un artefact du dépôt.

**Ce que le §7 n'établit PAS :**

- Aucune **revendication de performance biologique ou
  neuromorphique** sur la dynamique E-SNN. Le substrat
  E-SNN est un squelette numpy LIF, pas Loihi-2.
- Aucune **revendication d'efficacité empirique de
  consolidation** sur des données linguistiques ou
  sensorielles réelles. Le prédicteur est un mock Python
  partagé ; le vrai câblage mega-v2 ou IRMf est repoussé.
- Aucune **réplication à prédicteur divergent**. L'accord
  inter-substrats est trivialement OUI par construction
  à prédicteur partagé.
- Aucun **ratio matériel d'énergie ou de latence**. Le
  `ratio d'énergie` de H4 est une valeur mock scriptée,
  pas une trace mesurée en temps d'horloge.

## 7.6 Résumé de gate (substitution synthétique — pas de revendication empirique)

Sur les 4 hypothèses pré-enregistrées (OSF, héritées de
l'Article 1), les verdicts de gate par substrat sont :

- **H1 oubli** : PASS *(substitution synthétique)*,
  rejetée sur les deux substrats.
- **H2 auto-équivalence** : PASS *(substitution
  synthétique, fumée d'auto-équivalence)*, rejetée sur les
  deux substrats.
- **H3 monotonicité** : ÉCHEC à α = 0,0125, PASS à
  α = 0,05 *(substitution synthétique, limitée par la
  dispersion du mock)*, échec à rejeter sur les deux
  substrats.
- **H4 budget énergie** : PASS *(substitution
  synthétique)*, rejetée sur les deux substrats.

**Verdict agrégé cycle-2 G9** : **CONDITIONAL-GO /
PARTIAL** selon
`docs/milestones/g9-cycle2-publication.md`. La fondation
d'ingénierie (conformité DR-3 E-SNN G7, P_max fonctionnel
G8, worker asynchrone C2.17) est complète ; le pipeline
de réplication inter-substrats est câblé ; la narration
Article 2 est brouillonnée ; mais la force empirique de la
revendication attend une réplication à prédicteur
divergent du cycle-3.

## 7.7 Ligne de base Wake-Sleep CL (Alfarano 2024)

En plus de la grille 18-cellules substrat × profil × seed, la
table de résultats inclut une quatrième ligne autonome tirée
de [@alfarano2024wakesleep] (IEEE TNNLS, arXiv 2401.08623) —
l'analogue NREM/REM dual-phase publié le plus proche
(Article 1 §3, introduction.md L94, L108). La ligne est
générée par `scripts/baseline_wake_sleep_cl.py` et dumpée
vers `docs/milestones/wake-sleep-baseline-2026-05-03.json`.
**(placeholder synthétique — variante c, valeurs de référence
publiées.)**

| seed | run_id | forgetting_rate | avg_accuracy |
|------|--------|-----------------|--------------|
| 42  | `60a86e833d477c69745393df9b1b8af1` | 0,0820 | 0,8470 |
| 123 | `4b6b475e9efbeb38363cd4be2e8ddf6e` | 0,0820 | 0,8470 |
| 7   | `fcd2873d5efc5141178a4173e667485b` | 0,0820 | 0,8470 |

Les valeurs numériques ci-dessus sont des PLACEHOLDERS en
attente de vérification PDF des Tables 2-3 d'Alfarano 2024 ;
le dump Markdown porte la même précaution. L'identité du
seed-round-trip (mêmes nombres sur tous les seeds) est
**attendue** sous variante c — les variantes a/b
produiraient des lignes dépendantes du seed.

Une tentative de vérification du 2026-05-03 contre
arXiv 2401.08623v1
(`docs/milestones/wake-sleep-baseline-verify-2026-05-03.md`)
a identifié deux discordances : (i) Alfarano 2024 §4.1 évalue
WSCL sur Split CIFAR-10, Tiny-ImageNet1/2 et Split FG-ImageNet
— *pas* sur Split-FMNIST ; (ii) les Tables 2-3 rapportent des
pourcentages et la paire placeholder `(0,082, 0,847)` ne
correspond à aucune cellule ER-ACE+WSCL pour aucune taille de
buffer ni aucun des trois benchmarks rapportés. Le drapeau
PLACEHOLDER reste donc actif en attente d'une décision du
mainteneur (re-clé sur un benchmark Alfarano, ou bascule de
l'ancre comparateur vers un papier qui rapporte effectivement
Split-FMNIST).

Un test d'équivalence TOST de style H2 contre P_equ est
**délibérément omis** : la précaution prédicteur (§6.4) se
compose avec la précaution variant-c valeurs-de-référence ;
un TOST significatif requiert la réplication à prédicteur
divergent du cycle-3 sur le même jeu de données. La ligne
de base sert d'**ancre référence-publiée**, pas de test de
significativité.

---

## Notes pour révision

- Ajouter Figure 7.1 : boîtes à moustaches côte à côte
  d'accuracy par profil sur MLX vs E-SNN (montrera des
  distributions identiques sous prédicteur partagé,
  confirmant la précaution synthétique visuellement).
- Ajouter Figure 7.2 : matrice de conformité en heat-map
  (cellules PASS / N/A).
- Référencer §6.4 prédicteur par substitution synthétique
  et §8.3 limitations une fois §8 livrée.
- Resserrer à ≤ 1500 mots à la passe de pré-soumission
  NeurIPS.
- Chaque légende de table doit conserver au moins un
  drapeau `(substitution synthétique)` ; ne pas supprimer
  lors des passes de resserrement.
