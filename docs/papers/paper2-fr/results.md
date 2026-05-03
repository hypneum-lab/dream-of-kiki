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

## 7.1.5 Pilote G4-ter — le couplage REPLAY+DOWNSCALE sur substrat enrichi dépasse la référence sans rêve (2026-05-03)

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
  *en plus* `_l2.weight` comme site RESTRUCTURE et les activations
  hidden_2 comme source d'échantillonnage Gaussienne-MoG pour
  RECOMBINE produit `g_h2 = +2,77` à la combinaison ancre C5 sur
  120 cellules (N=30 par bras), Welch unilatéral
  p = 4,9e-14 ≪ α/4 = 0,0125. H0_substrat **rejetée**. *Lecture
  honnête* (voir « Ce que dit et ne dit pas l'effet +2,77 »
  ci-dessous) : le rejet de H0_substrat établit que **le couplage
  REPLAY+DOWNSCALE sur un substrat hôte plus riche dépasse la
  référence sans rêve** ; il n'établit **pas** que RESTRUCTURE
  ou RECOMBINE ont contribué de manière mesurable au-dessus du
  bruit.
- **H_DR4-ter — monotonicité** : rétention moyenne
  `P_min = 0,7065`, `P_equ = 0,7046`, `P_max = 0,7046`
  (Jonckheere J = 1335,0, p = 0,544 à α/4 = 0,0125 ; rétention
  référence 0,5869, n=30). L'ordre prédit
  `P_max ≥ P_equ ≥ P_min` n'est **pas observé** : les trois bras
  de rêve sont à égalité à ±0,002 près et `P_min` dépasse (et non
  pas dépassé par) les autres. Ce n'est *pas* une monotonicité
  non-concluante — c'est l'**inverse** de la prédiction DR-4
  (`P_min > P_equ = P_max` au lieu de `P_max ≥ P_equ ≥ P_min`)
  et cela constitue une réfutation partielle de la revendication
  framework-C « profil plus riche → consolidation plus riche »
  **à cette échelle, sur ce substrat**.

### Ce que dit et ne dit pas l'effet +2,77 (post-pilote)

Le résultat g=+2,77 se lit au mieux comme **une confirmation
directionnelle qu'un effet de couplage de rêve sur la rétention
existe**, et non comme une preuve calibrée de magnitude. Trois
mises en garde sont structurantes pour le verdict EC et pour
toute citation externe de ce nombre :

- *Hu 2020 est une ancre directionnelle, pas un calibrateur de
  magnitude.* Hu et al. 2020 g=0,29 est une estimation
  méta-analytique sur **la consolidation mnésique
  sommeil-dépendante humaine** ; g=+2,77 est sur une tête MLP
  jouet dérivée de CIFAR-100 avec N=30 seeds. Les deux tailles
  d'effet appartiennent à des classes de référence différentes
  (cohorte biologique vs. pilote numérique seedé) et une
  comparaison de magnitude inter-classes est une erreur de
  catégorie. La pré-enregistration utilise Hu 2020 pour fixer
  le **signe** de l'hypothèse alternative (`g > 0` plutôt que
  `g > 0,29` comme seuil littéral) ; présenter g=+2,77 comme
  un « swing à +9σ par rapport à Hu » ou « effet de rêve dix
  fois plus fort que le sommeil humain » mésinterpréterait
  l'ancre.
- *Ordre des profils DR-4 NON soutenu.* L'ordre empirique
  `P_min > P_equ = P_max` (0,7065 > 0,7046 = 0,7046) est
  l'**inverse** de la prédiction DR-4 du framework-C
  `P_max ≥ P_equ ≥ P_min`. RESTRUCTURE et RECOMBINE — les deux
  canaux exposés par P_equ/P_max mais pas par P_min — n'ont
  rien apporté de mesurable au-dessus du plancher de bruit
  intra-bras à l'ancre C5. La revendication framework-C
  « ops plus riches → consolidation plus riche » est donc
  **partiellement réfutée par ce pilote à cette échelle**, et
  pas seulement « non concluante ». C'est consigné comme une
  réfutation explicite de la prédiction monotonique DR-4 et
  non comme un échec de Type-II à détecter.
- *Attribution des canaux actifs.* Le substrat enrichi expose
  4 canaux mais seuls 2 sont démontrablement actifs. Le
  résultat honnête que le +2,77 soutient est : « le couplage
  REPLAY + DOWNSCALE sur un hôte dont la géométrie cachée
  autorise déjà un signal de replay non trivial dépasse la
  référence sans rêve par une large marge » — RESTRUCTURE et
  RECOMBINE restent des canaux spectateurs à cette échelle,
  reflétant le motif spectateur de G4-bis un cran plus haut
  sur l'échelle de complexité du substrat.

### Verdict

Le verdict **renforce** plutôt qu'il n'affaiblit le verrouillage
EC=PARTIAL selon la table DualVer de la pré-enregistration §7.
H1+H2 rejettent fortement H0 (l'effet
REPLAY+DOWNSCALE-sur-hôte-enrichi existe et est substantiel),
mais H_DR4-ter est **partiellement réfutée** (P_min dépasse
P_equ = P_max ; la prédiction de monotonicité centrale du
framework ne tient pas). PARTIAL est le verdict honnête pour
cette double issue — la promotion vers STABLE exigerait à la
fois H1+H2 et un ordre soutenant DR-4 sur un pilote
confirmatoire. Un suivi N≥95 (G4-quater) est prévu pour tester
si substrat enrichi × `hp_best=C9` retrouve l'ordre prédit sur
une cellule à seaux distingués, et pour épingler le N auquel
l'égalité P_min ≈ P_equ ≈ P_max se résout dans l'ordre prédit
ou se solidifie en réfutation réelle.

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

## 7.1.6 Pilote G4-quater — vacuité empirique de RESTRUCTURE+RECOMBINE confirmée (2026-05-03)

À la suite du résultat de G4-ter §7.1.5 selon lequel
RESTRUCTURE+RECOMBINE restaient des canaux spectateurs à
l'échelle 3 couches, G4-quater a exécuté un pilote
confirmatoire séquentiel en 3 étapes pour distinguer entre
trois sous-hypothèses pré-enregistrées dans
[`docs/osf-prereg-g4-quater-pilot.md`](../../osf-prereg-g4-quater-pilot.md) :
H4-A profondeur du substrat, H4-B calibration HP, H4-C
vacuité théorique. Chaque étape produit son propre jalon
daté avec des `run_id` R1 ; un agrégateur émet un verdict
unique sur les trois.

### Table de verdicts

| sous-hypothèse | test | observé | verdict |
|---|---|---|---|
| H4-A profondeur du substrat (tête 5 couches 64-32-16-8, 380 cellules, N=95) | Jonckheere α = 0,0167 | J = 13511,5, p = 0,514 ; moyennes P_min = 0,5959, P_equ = 0,5958, P_max = 0,5958 (à 1e-4 près) ; monotonic_observed = False | **NON confirmée** — approfondir le substrat seul ne récupère pas l'ordre DR-4 prédit à ce N. L'inversion H_DR4 observée à G4-ter persiste à profondeur 5 couches. |
| H4-B calibration HP (tête 3 couches × facteur RESTRUCTURE ∈ {0,85, 0,95, 0,99}, 360 cellules, N=30) | Jonckheere par facteur α = 0,0056 (3 facteurs × 3 hypothèses) | tous les facteurs : `mean P_equ = mean P_max < mean P_min` ; J ∈ {1034, 1076, 1094}, p ∈ {0,971, 0,979, 0,990} ; monotonic_observed = False aux trois facteurs | **NON confirmée** — la calibration HP de RESTRUCTURE seule ne récupère pas l'ordre DR-4 prédit ; RESTRUCTURE *dégrade* en fait la rétention par rapport à P_min REPLAY+DOWNSCALE seul à chaque facteur échantillonné. |
| H4-C vacuité empirique de RECOMBINE (tête 3 couches × stratégie ∈ {mog, ae, none}, 1140 cellules, N=95) | Welch bilatéral P_max(mog) vs P_max(none) à α = 0,0167 | mean P_max(mog) = 0,7007, mean P_max(none) = 0,7006 ; Welch t = 0,014, p = 0,989 ; Hedges' g (mog vs none) = 0,002 ; fail_to_reject_h0 = True. AE secondaire : mean P_max(ae) = 0,7006, Welch p (ae vs none) = 1,000. | **CONFIRMÉE** — Welch échoue à rejeter H0 entre mog-RECOMBINE et le bras placebo none au seuil α multiplicité-ajusté. La revendication empirique positive « RECOMBINE n'apporte rien de mesurable au-delà de REPLAY+DOWNSCALE à cette échelle » tient. |

Temps mur agrégé : Étape 1 13,2 min + Étape 2 10,8 min +
Étape 3 34,1 min ≈ 58 min sur M1 Max.

### Lecture honnête

- **L'échec de Welch à rejeter est le résultat prédit sous
  H4-C.** Le cadrage pré-enregistré de H4-C est « le canal
  RECOMBINE est structurellement vide à cette échelle »,
  opérationnalisé comme « Welch échoue à rejeter H0 entre
  RECOMBINE=mog et RECOMBINE=none ». Le p = 0,989 ≫ α = 0,0167
  observé, avec des moyennes à 1e-4 près et un Hedges' g =
  0,002, est donc une **revendication empirique positive** que
  le canal est vide — et non un échec de détection de Type-II.
  C'est la distinction absence-d'évidence vs évidence-d'absence
  résolue du côté absence-d'évidence : à N = 95 par bras, l'IC
  95 % sur la différence mog-vs-none exclut tout effet
  pratiquement signifiant.
- **La réfutation partielle de DR-4 se renforce.** G4-ter
  §7.1.5 rapportait `P_min > P_equ = P_max` comme une
  réfutation partielle de la revendication framework-C
  « ops plus riches → consolidation plus riche ». G4-quater
  élimine deux des trois clauses d'échappement restantes
  (profondeur du substrat et calibration HP de RESTRUCTURE) et
  confirme positivement la troisième (RECOMBINE empiriquement
  vide). La revendication framework-C, **à cette échelle de
  benchmark**, est désormais une hypothèse empiriquement
  réfutée plutôt qu'une hypothèse insuffisamment testée. Le
  lemme DR-4 (métriques capacité-monotones non-décroissantes)
  n'est **pas** réfuté — les écarts de rétention intra-bras
  sont à ±0,001 près — mais la prédiction « profil plus riche
  retient *plus* » perd son support empirique à cette échelle.
- **Ce que le verdict ne dit pas.** G4-quater ne teste qu'un
  benchmark (Split-FMNIST, tête MLP). Les travaux futurs
  pré-enregistrés dans `docs/osf-prereg-g4-quater-pilot.md` §6 —
  test sur CIFAR-10 / ImageNet / E-SNN hiérarchique — pourraient
  en principe inverser le verdict à plus haute capacité ;
  jusqu'à ces tests, aucune promotion STABLE de la revendication
  framework-C « ops plus riches → consolidation plus riche »
  ne peut survenir.

### Impact DualVer

Selon la pré-enregistration `docs/osf-prereg-g4-quater-pilot.md`
§6 et §7, EC reste **PARTIAL** sous l'issue H4-C-confirmée ; FC
reste à **C-v0.12.0** (pas de bump axe formel). Le fichier
d'évidence empirique `docs/proofs/dr4-profile-inclusion.md` est
amendé avec un addendum G4-quater.

Les clés de profil du registre de runs
`g4-quater/{step1,step2,step3}/<bras>/<combo>/<seed>`
identifient chaque cellule pour satisfaire R1.

Provenance :
- Pré-enregistration : [docs/osf-prereg-g4-quater-pilot.md](../../osf-prereg-g4-quater-pilot.md)
- Jalon Étape 1 : `docs/milestones/g4-quater-step1-2026-05-03.{json,md}`
- Jalon Étape 2 : `docs/milestones/g4-quater-step2-2026-05-03.{json,md}`
- Jalon Étape 3 : `docs/milestones/g4-quater-step3-2026-05-03.{json,md}`
- Agrégat : `docs/milestones/g4-quater-aggregate-2026-05-03.{json,md}`
- Pilote Étape 1 : `experiments/g4_quater_test/run_step1_deeper.py`
- Pilote Étape 2 : `experiments/g4_quater_test/run_step2_restructure_sweep.py`
- Pilote Étape 3 : `experiments/g4_quater_test/run_step3_recombine_strategies.py`
- Substrat (5 couches) : `experiments.g4_quater_test.deeper_classifier.G4HierarchicalDeeperClassifier`
- Substrat (3 couches) : `experiments.g4_ter_hp_sweep.dream_wrap_hier.G4HierarchicalClassifier`
- Stratégies RECOMBINE : `experiments.g4_quater_test.recombine_strategies.sample_synthetic_latents`

## 7.1.7 Pilote G4-quinto — universalité de la vacuité de RECOMBINE sur FMNIST + CIFAR-CNN (2026-05-03)

À la suite du résultat de G4-quater §7.1.6 selon lequel
RECOMBINE était empiriquement vide à l'échelle Split-FMNIST
3 couches MLP, la pré-enregistration G4-quater §6 ligne 5
flaggait explicitement « travaux futurs : tester des
benchmarks plus profonds (CIFAR-10, ImageNet) avant toute
promotion STABLE ». G4-quinto exécute ce suivi sur le
benchmark suivant de l'échelle d'escalade (Split-CIFAR-10)
avec deux substrats (MLP 5 couches sur CIFAR et un petit CNN)
afin de tester si le verdict H4-C est **lié à l'échelle**
(artefact FMNIST) ou **universel**.

Le pilote est pré-enregistré dans
[`docs/osf-prereg-g4-quinto-pilot.md`](../../osf-prereg-g4-quinto-pilot.md)
(commit `a02b82c`, verrouillé avant toute exécution de
pilote). Un amendement §9.1 a été déposé dans le même
fichier de verrouillage lorsque le miroir canonique
`www.cs.toronto.edu/~kriz` a renvoyé HTTP 503 sur l'ensemble
de l'arbre `~kriz` le 2026-05-03 ; le chargeur a gagné un
fallback SHA-256-épinglé vers le dataset Hugging Face
`uoft-cs/cifar10` (commit `0b2714987...`), qui est un
ré-encodage PNG de la même publication Krizhevsky 2009. Le
chemin d'acquisition change ; le contrat expérimental ne
change pas.

Trois étapes séquentielles miroir de la disposition G4-quater :

### Table de verdicts

| sous-hypothèse | test | observé | verdict |
|---|---|---|---|
| H5-A échelle benchmark (MLP 5 couches sur CIFAR, hidden 256-128-64-32, 120 cellules, N = 30) | Jonckheere α = 0,0167 | J = 1362,0, p = 0,4646 ; moyennes P_min = 0,8713, P_equ = 0,8754, P_max = 0,8754 (P_equ = P_max à 1e-4 près) ; monotonic_observed = True ; reject_h0 = False | **NON confirmée** — l'ordre DR-4 prédit est observé en rétention moyenne (monotone) mais Jonckheere échoue à rejeter H0 au seuil α multiplicité-ajusté ; la simple montée en échelle de benchmark (FMNIST → CIFAR-10) sur un MLP plus large ne récupère pas statistiquement l'ordre. |
| H5-B échelle architecture (G4SmallCNN, Conv2d×2 + MaxPool2d×2 + Linear×2, latent_dim = 64, 120 cellules, N = 30) | Jonckheere α = 0,0167 | J = 1356,0, p = 0,4823 ; moyennes P_min = 0,9841, P_equ = 0,9842, P_max = 0,9842 (P_equ = P_max à 1e-4 près) ; monotonic_observed = True ; reject_h0 = False | **NON confirmée** — même schéma que H5-A : monotone en moyenne, ne rejette pas statistiquement H0. Ajouter une structure conv hiérarchique échoue également à récupérer l'ordre prédit à ce N. |
| H5-C universalité de la vacuité de RECOMBINE (G4SmallCNN × stratégie ∈ {mog, ae, none}, 360 cellules, N = 30) | Welch bilatéral P_max(mog) vs P_max(none) à α = 0,0167 | mean P_max(mog) = 0,9842, mean P_max(none) = 0,9845 ; Welch t = -0,0104, p = 0,9918 ; Hedges' g (mog vs none) = -0,0026 ; fail_to_reject_h0 = True. AE secondaire : mean P_max(ae) = 0,9840, Welch p (ae vs none) = 0,9857. | **CONFIRMÉE** — Welch échoue à rejeter H0 entre mog-RECOMBINE et le bras placebo none au seuil α multiplicité-ajusté. Le résultat G4-quater H4-C de vacuité de RECOMBINE **universalise** entre substrats : FMNIST 3 couches MLP (g = 0,002, p = 0,989) et CIFAR-CNN (g = -0,0026, p = 0,9918) échouent tous deux à détecter une différence mog vs none. |

Temps mur agrégé : Étape 1 3,8 min + Étape 2 17,3 min + Étape 3
50,9 min ≈ 72 min sur M1 Max (bien en dessous de l'enveloppe
9-15 h Option A de la pré-enregistration ; le budget était
conservateur).

### Lecture honnête

- **L'échec de Welch à rejeter est le résultat prédit sous
  H5-C.** H5-C est opérationnalisé exactement comme H4-C :
  « le canal RECOMBINE est structurellement vide à cette
  échelle », avec « Welch bilatéral échoue à rejeter H0 entre
  RECOMBINE = mog et RECOMBINE = none ». Le p = 0,9918 ≫
  α = 0,0167 observé, avec des moyennes à 3e-4 près et un
  Hedges' g = -0,0026, est donc une **revendication empirique
  positive** que le canal est vide sur le substrat CNN à
  l'échelle CIFAR-10 — et non un échec de détection de Type-II.
  Combiné avec la confirmation H4-C de G4-quater (g = 0,002,
  p = 0,989), le drapeau d'universalité s'allume sur deux
  benchmarks × deux substrats.
- **La réfutation partielle de DR-4 universalise.** G4-ter
  §7.1.5 a partiellement réfuté « ops plus riches → consolidation
  plus riche » sur le MLP 3 couches. G4-quater §7.1.6 a
  renforcé cela en revendication empirique positive que
  RECOMBINE n'apporte rien de mesurable sur FMNIST. G4-quinto
  étend maintenant cette revendication à CIFAR-CNN : le canal
  est vide à chaque combinaison benchmark × substrat testée
  dans l'échelle d'escalade. La revendication framework-C
  « ops plus riches » est désormais une hypothèse empiriquement
  réfutée au niveau du canal RECOMBINE sur **deux benchmarks
  × deux substrats**, et non un seul. Le lemme DR-4.L lui-même
  n'est **pas** réfuté — les écarts intra-bras restent
  ≤ 0,001 — mais le contenu prédictif de DR-4 est sévèrement
  réduit en portée.
- **Ce que le verdict ne dit pas.** G4-quinto teste
  Split-CIFAR-10 avec deux substrats ; il ne teste pas
  ImageNet, une tête transformer complète, ni un E-SNN
  hiérarchique. La pré-enregistration §6 ligne 6 exige
  toujours ces tests avant toute promotion STABLE. Les nuls
  H5-A et H5-B (« monotone mais non significatif ») sont
  rapportés ici comme **pas d'évidence à ce N**, et non comme
  évidence d'absence : l'ordre prédit pourrait émerger à plus
  grand N ou sous des grilles HP différentes.

### Impact DualVer

Selon la pré-enregistration `docs/osf-prereg-g4-quinto-pilot.md`
§6 ligne 4, EC reste **PARTIAL** sous l'issue H5-C-confirmée ;
FC reste à **C-v0.12.0** (pas de bump axe formel). Le fichier
d'évidence empirique `docs/proofs/dr4-profile-inclusion.md`
est amendé avec un addendum §7.1.7 G4-quinto qui universalise
le résultat FMNIST.

Les clés de profil du registre de runs
`g4-quinto/{step1,step2,step3}/<bras>/<combo>[/<stratégie>]`
identifient chaque cellule pour satisfaire R1.

Provenance :
- Pré-enregistration : [docs/osf-prereg-g4-quinto-pilot.md](../../osf-prereg-g4-quinto-pilot.md)
- Jalon Étape 1 : `docs/milestones/g4-quinto-step1-2026-05-03.{json,md}`
- Jalon Étape 2 : `docs/milestones/g4-quinto-step2-2026-05-03.{json,md}`
- Jalon Étape 3 : `docs/milestones/g4-quinto-step3-2026-05-03.{json,md}`
- Agrégat : `docs/milestones/g4-quinto-aggregate-2026-05-03.{json,md}`
- Pilote Étape 1 : `experiments/g4_quinto_test/run_step1_mlp_cifar.py`
- Pilote Étape 2 : `experiments/g4_quinto_test/run_step2_cnn_cifar.py`
- Pilote Étape 3 : `experiments/g4_quinto_test/run_step3_cnn_recombine.py`
- Substrat (MLP sur CIFAR) : `experiments.g4_quinto_test.cifar_mlp_classifier.G4HierarchicalCIFARClassifier`
- Substrat (CNN) : `experiments.g4_quinto_test.small_cnn.G4SmallCNN`
- Stratégies RECOMBINE : `experiments.g4_quater_test.recombine_strategies.sample_synthetic_latents`

## 7.1.9 Pilote G5-bis — tête riche portée sur E-SNN, verdict artefact MLX-only (2026-05-03)

La trouvaille G4-ter §7.1.5 (g_h2 = + 2,77 sur substrat MLX
hiérarchique) soulevait la question cross-substrat : l'effet
positif REPLAY+DOWNSCALE survit-il au port vers un substrat
non-MLX ? Le pilote G5-bis y répond. Pré-enregistrement
[`docs/osf-prereg-g5-bis-richer-esnn.md`](../../osf-prereg-g5-bis-richer-esnn.md)
verrouillé au commit `ae640a5` ; exécution du pilote au commit
`5168400`. Jalon associé :
[`docs/milestones/g5-bis-richer-esnn-2026-05-03.md`](../../milestones/g5-bis-richer-esnn-2026-05-03.md).

**Substrat** : `EsnnG5BisHierarchicalClassifier`, un SNN
LIF rate-coded à 3 couches (32 → 16 → 2 unités de taux de
décharge) avec rétropropagation par estimateur straight-through
(Wu 2018), implémenté en numpy pur (sans dépendance `norse`).

**Trouvaille intra-substrat (E-SNN richer, N=10 graines)** :
- rétention moyenne `baseline = 0,5127`, `P_min = 0,5129`,
  `P_equ = 0,5131`, `P_max = 0,5131` ;
- `g_h7a = 0,1043` observé (E-SNN richer P_equ vs baseline) ;
- p Welch unilatéral = 0,4052 à α/4 = 0,0125 → échec à rejeter H0 ;
- seuil pré-enregistré `H7B_G_THRESHOLD = 0,5` non atteint.

**Agrégat cross-substrat vs G4-ter MLX richer head**
([`g5-bis-aggregate-2026-05-03.md`](../../milestones/g5-bis-aggregate-2026-05-03.md)) :

| Bras | g (MLX − E-SNN) | p Welch bilatéral | rejet H₀ ? |
|------|-----------------|---------------------|-------------|
| baseline | 3,23 | 6,4 × 10⁻¹⁶ | OUI |
| P_min | 4,20 | 7,4 × 10⁻¹⁹ | OUI |
| P_equ | 4,02 | 2,3 × 10⁻¹⁸ | OUI |
| P_max | 4,02 | 2,3 × 10⁻¹⁸ | OUI |

**Classification : H7-B (artefact MLX-only à cette échelle).**

**Lecture honnête**. L'agrégateur émet `h7_classification =
"H7-B"` — l'effet positif G4-ter (g = +2,77) **ne transfère
pas** à la tête riche E-SNN à ce N. La quantification de
spike-rate, la non-linéarité LIF, et l'approximation
STE-backward du substrat E-SNN washent apparemment l'effet
onirique qui émergeait sur MLX. L'ancre Hu 2020 est une
référence directionnelle (signe de l'hypothèse alternative), pas
un calibrateur de magnitude ; le verdict H7-B est rapporté
comme l'issue prédite sous le pré-enregistrement si
l'universalité cross-substrat H7-C ne tient pas, et **n'est
pas** adouci en "E-SNN fonctionne à plus petite échelle".

**Implications pour DR-3.** La substrat-agnosticité DR-3 reste
formellement valide au niveau des tests-de-propriété
axiomatiques (les vérifications DR-0/1/2'/4 passent sur les
substrats MLX et E-SNN ; cf. §4.1). Mais la
substrat-agnosticité *empirique* est désormais réfutée à deux
niveaux distincts : (a) la rétention absolue diverge
[@g5-cross-substrate, §7.1.3 binary-head] ; (b) l'effet positif
onirique lui-même est MLX-bound à ce N. Le fichier de preuve DR-3
`docs/proofs/dr3-substrate-evidence.md` doit être révisé pour
noter que la garantie de substrat-agnosticité ne couvre que les
vérifications du Critère de Conformité, et non la transférabilité
empirique des tailles d'effet.

Un suivi confirmatoire N=30 (`Option A`) au seuil H7B_G_THRESHOLD
= 0,5 resserrerait le verdict ; une étape 2 plus profonde dans
G5-ter portant `G4SmallCNN` vers un CNN à spikes testerait si
la non-linéarité spiking est le mécanisme load-bearing du
washout. Les deux planifiés en travaux futurs par pré-reg §6
ligne 6.

## 7.1.10 Pilote G5-ter — verdict cross-substrat H8 spiking-CNN (2026-05-03)

Pour désambiguïser le verdict H7-B en échec-à-rejeter du pilote
G5-bis §7.1.9 (`g_h7a = +0,1043`, artefact MLX-only pour le
MLP LIF à 3 couches), nous avons porté l'architecture small-CNN
G4-quinto étape 2 sur le substrat E-SNN sous forme d'un CNN à
spikes à 4 couches (`EsnnG5TerSpikingCNN` : Conv2d(3→16) →
taux LIF → Conv2d(16→32) → taux LIF → avg-pool 4×4 → flatten +
Linear(2048→64) → taux LIF → Linear(64→2), STE-backward sur les
trois étages LIF, Conv2d en numpy pur via im2col).
Pré-enregistrement
[`docs/osf-prereg-g5-ter-spiking-cnn.md`](../../osf-prereg-g5-ter-spiking-cnn.md)
verrouillé au commit `f18030b` ; exécution du pilote au commit
`5174220`. Jalons associés :
[`g5-ter-spiking-cnn-2026-05-03.md`](../../milestones/g5-ter-spiking-cnn-2026-05-03.md)
+
[`g5-ter-aggregate-2026-05-03.md`](../../milestones/g5-ter-aggregate-2026-05-03.md).
Un amendement budget-de-calcul documenté sous-échantillonne le
shard d'entraînement CIFAR-10 à 1500 exemples par tâche par
cellule (shard de test intact) selon
[`docs/osf-deviations-g5-ter-2026-05-03.md`](../../osf-deviations-g5-ter-2026-05-03.md);
le design pré-enregistré 4 bras × N=10 × combo HP C5 est
préservé verbatim.

La règle de décision pré-enregistrée (seuils VERROUILLÉS
0,5 / 1,0 / 2,0) associe le triplet observé (`g_h8`, issue Welch
intra-substrat, `g_p_equ_cross`) à l'une des trois hypothèses :
H8-A (la non-linéarité LIF est le mécanisme de washout
load-bearing), H8-B (le mismatch architectural était le problème
; la structure CNN récupère le signal), H8-C (partiel — les
deux contribuent).

**Trouvaille intra-substrat (E-SNN spiking-CNN, N=10 graines,
36 min wall sur M1 Max)** :

- rétention moyenne `baseline = 0,8537`, `P_min = 0,8463`,
  `P_equ = 0,8417`, `P_max = 0,8417` ;
- `g_h8 = -0,1093` observé (E-SNN spiking-CNN P_equ vs
  baseline) — c.-à-d. la moyenne du bras onirique est
  marginalement **inférieure** au baseline ;
- p Welch unilatéral = 0,5992 à α/4 = 0,0125 → échec à rejeter
  H₀ ;
- seuil intra-substrat pré-enregistré `H7B_G_THRESHOLD = 0,5`
  non atteint (et non franchi non plus dans le sens négatif).

**Agrégat cross-substrat vs G4-quinto étape 2 small-CNN MLX**
([`g5-ter-aggregate-2026-05-03.md`](../../milestones/g5-ter-aggregate-2026-05-03.md)) :

| Bras | g (MLX − E-SNN) | p Welch bilatéral | rejet H₀ ? |
|------|-----------------|---------------------|-------------|
| baseline | +1,21 | 3,4 × 10⁻³ | OUI |
| P_min | +1,32 | 4,3 × 10⁻⁴ | OUI |
| P_equ | +1,31 | 1,2 × 10⁻³ | OUI |
| P_max | +1,31 | 1,2 × 10⁻³ | OUI |

`g_p_equ_cross = +1,31` se situe **entre** le plancher H8-A
(2,0) et le plafond H8-B (1,0).

**Classification : H8-C (partiel — architecture et
non-linéarité LIF contribuent toutes deux).**

**Lecture honnête.** L'agrégateur émet `h8_classification =
"H8-C"`. Deux observations empiriques cadrent le verdict :

1. *La non-linéarité LIF contribue au washout* : `g_h8 =
   -0,1093` intra-substrat et l'échec-à-rejeter Welch
   (p = 0,60) confirment que le canal positif du cycle 3 ne
   transfère **pas** à travers la rate-coding E-SNN **même
   lorsque l'architecture est convolutionnelle**. Le washout
   du MLP G5-bis est reproduit par le CNN à spikes G5-ter.
2. *Le mismatch architectural n'est pas le seul moteur* : mais
   l'écart cross-substrat à P_equ (`g_p_equ_cross = +1,31`)
   est nettement plus petit que l'écart G5-bis MLP (`+4,02`).
   Passer d'un MLP LIF à 3 couches à un CNN à spikes à
   4 couches ferme environ deux tiers de l'écart de niveau
   de rétention cross-substrat, ce qui est cohérent avec une
   contribution partielle du biais inductif architectural.

Selon le précédent du Critic, l'échec-à-rejeter se lit comme
absence-de-preuve à ce N (plancher de détection Option B
`g ≈ 1,27`), pas preuve-d'absence ; l'estimation ponctuelle
négative est de petite magnitude (|g| = 0,11) et son intervalle
bootstrap à 95 % chevauche zéro. L'ancre Hu 2020 reste une
référence directionnelle (signe de l'hypothèse alternative), pas
un calibrateur de magnitude. H8-C est l'issue prédite sous le
pré-enregistrement si ni l'extrême pure-LIF-washout (H8-A) ni
l'extrême récupération-architecturale (H8-B) ne correspond aux
données.

**Implications pour DR-3.** La substrat-agnosticité DR-3 reste
formellement valide au niveau des tests-de-propriété
axiomatiques. La garantie de substrat-agnosticité empirique sur
le canal positif du cycle 3 est désormais réfutée à *deux*
profondeurs architecturales (MLP LIF à 3 couches : G5-bis H7-B ;
CNN à spikes à 4 couches : G5-ter H8-C avec `g_h8 = -0,11`),
et l'écart cross-substrat résiduel survit même après la
correction du biais architectural. Le fichier de preuve DR-3
`docs/proofs/dr3-substrate-evidence.md` est complété par une
ligne H8-C consignant la contribution partielle des deux
mécanismes. Un suivi confirmatoire N=30 Option A est planifié
pour resserrer la lecture H8-C (le plancher de détection
Option B aurait déjà fait apparaître un effet `g ≥ 1,27` à
l'α choisi). EC reste PARTIAL ; pas de bump FC.

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
vers `docs/milestones/wake-sleep-baseline-rekey-2026-05-03.json`
(supersède le dump gelé `wake-sleep-baseline-2026-05-03.json`).
**(variante c — valeurs de référence publiées, re-clé du
2026-05-03 vers CIFAR-10 pour correspondre au benchmark
réellement rapporté par Alfarano 2024.)**

| seed | run_id | forgetting_rate | avg_accuracy |
|------|--------|-----------------|--------------|
| 42  | `38ad694fe99c2dfbb3f8ca4c312852b7` | 0,1069 | 0,7418 |
| 123 | `2cdef3880915543654c81205fe4edf9a` | 0,1069 | 0,7418 |
| 7   | `16f511205877c190074790f094309316` | 0,1069 | 0,7418 |

Les valeurs numériques sont importées directement des
Tables 2-3 d'Alfarano 2024, ligne « ER-ACE+WSCL (Ours) »,
Split CIFAR-10 (5 tâches binaires, class-incremental),
buffer = 500 : final-average-accuracy 74,18 ± 1,28 % et
oubli 10,69 %, ré-échelonnés des pourcentages vers l'intervalle
unité. L'identité du seed-round-trip (mêmes nombres sur tous
les seeds) est **attendue** sous variante c — les variantes
a/b produiraient des lignes dépendantes du seed.

La tentative de vérification du 2026-05-03 contre
arXiv 2401.08623v1
(`docs/milestones/wake-sleep-baseline-verify-2026-05-03.md`)
avait identifié deux discordances dans la paire placeholder
`split_fmnist_5tasks` antérieure `(0,082, 0,847)` : (i)
Alfarano 2024 §4.1 évalue WSCL sur Split CIFAR-10,
Tiny-ImageNet1/2 et Split FG-ImageNet — *pas* sur Split-FMNIST ;
(ii) les Tables 2-3 rapportent des pourcentages et cette paire
placeholder ne correspondait à aucune cellule ER-ACE+WSCL pour
aucune taille de buffer ni aucun des trois benchmarks rapportés.
L'entrée superséante
`docs/milestones/wake-sleep-baseline-rekey-2026-05-03.md`
enregistre le chemin de résolution retenu (Option 1 : re-clé
sur un benchmark Alfarano) ; le dump parent gelé est préservé
selon la discipline append-only des jalons.

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
