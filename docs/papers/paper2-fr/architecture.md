<!--
SPDX-License-Identifier: CC-BY-4.0
Authorship byline : dreamOfkiki project contributors
License : Creative Commons Attribution 4.0 International (CC-BY-4.0)
-->

# §5 Architecture d'ingénierie (Article 2, brouillon C2.14)

**Signataires** : *contributeurs du projet dreamOfkiki*
**Licence** : CC-BY-4.0

**Cible de longueur** : ~1,5 pages markdown (≈ 1300 mots)

---

## 5.1 Deux substrats, mêmes Protocols

L'Article 1 a défini la surface du framework comme un ensemble
de `Protocol`s Python typés : 8 primitives (entrées α, β, γ,
δ + sorties canal-1..4) et 4 opérations (replay, downscale,
restructure, recombine). L'Article 2 exerce cette surface sur
deux substrats :

- **`mlx_kiki_oniric`** (référence cycle 1) — arrays MLX sur
  Apple Silicon, mises à jour de type gradient sur tenseurs
  denses, source de vérité pour les 4 handlers d'op dans
  `kiki_oniric/substrates/mlx_kiki_oniric.py`.
- **`esnn_thalamocortical`** (ajout cycle 2, substitution
  synthétique) — squelette numpy LIF taux de spikes dans
  `kiki_oniric/substrates/esnn_thalamocortical.py`. Les 4
  factories d'op émettent des transitions d'état en taux de
  spikes plutôt que des mises à jour par gradient. Aucun
  matériel Loihi-2 n'est impliqué ; le squelette existe pour
  exercer la forme du Protocol, pas pour prétendre à une
  efficacité matérielle neuromorphique.

L'API d'enregistrement
(`kiki_oniric/substrates/__init__.py`) est un point d'entrée
unique retournant un tuple nommé avec factories d'op + type
d'état ; le runner d'ablation + la matrice de conformité
consomment l'enregistrement uniformément, ce qui est la
manière dont DR-3 se manifeste opérationnellement dans le
codebase.

## 5.2 Trois profils : P_min, P_equ, P_max

L'Article 1 a défini la chaîne de profils (inclusion DR-4)
comme `P_min ⊆ P_equ ⊆ P_max`. Le cycle 2 câble les trois
de bout en bout :

- **P_min** — replay seul, canal-1 seul. Plus petit profil
  fonctionnel qui satisfait encore la suite d'axiomes (DR-0
  redevabilité, R1 déterminisme). Spec :
  `kiki_oniric/profiles/p_min.py`. Déjà câblé cycle 1.
- **P_equ** — `{replay, downscale, restructure}` sur canaux
  1 + 2 + 3. Le profil *équi*- correspondant au récit de
  consolidation quatre piliers (Walker / Tononi / Friston),
  moins le bras de recombinaison Hobson. Spec :
  `kiki_oniric/profiles/p_equ.py`. Déjà câblé cycle 1.
- **P_max** — 4 ops (+ `recombine`) sur 4 canaux (+ entrée
  α-stream + sortie canal-4 ATTENTION_PRIOR). Précédemment
  squelette seul en fin de cycle 1 (Article 1 §9.2) ;
  **entièrement câblé en cycle 2** via G8
  (`docs/milestones/g8-p-max-functional.md`). Avec P_max
  câblé réel, l'hypothèse H2 (équivalence P_max vs P_equ
  à ±5 %) devient une vraie comparaison plutôt qu'un test
  de fumée d'auto-équivalence.

L'inclusion de chaîne de profils DR-4 est appliquée
structurellement : l'ensemble d'ops de P_equ est un
sur-ensemble de celui de P_min, celui de P_max est un
sur-ensemble de celui de P_equ, et la même inclusion tient
au niveau des canaux. Les violations seraient attrapées par
la suite d'axiomes DR-4
(`tests/conformance/axioms/test_dr4_*`).

## 5.3 Quatre opérations canoniques sur un semi-groupe libre

Les 4 opérations sont, selon DR-2 (compositionnalité,
`docs/proofs/dr2-compositionality.md`), des générateurs d'un
**semi-groupe libre** de séquences d'opérations oniriques :

- **`replay`** — pilier de consolidation Walker / Stickgold.
  Échantillonne les épisodes du buffer β, propage à travers
  les paramètres actuels, applique des mises à jour de
  rétention de type gradient. Sur MLX : ops tensorielles
  denses. Sur E-SNN : ré-injection de taux de spikes sur la
  population LIF.
- **`downscale`** — pilier Tononi SHY. Rétrécissement
  multiplicatif des poids dans (0, 1]. Commutatif mais
  **pas** idempotent : `downscale_f ∘ downscale_f` donne
  facteur², pas facteur.
- **`restructure`** — pilier Friston FEP. Modification de
  topologie (ajout / suppression de couche, reroutage) sous
  la garde topologie S3. Toute émission doit passer
  `validate_topology` avant le swap.
- **`recombine`** — pilier Hobson / Solms. Ré-échantillonnage
  génératif depuis le snapshot δ. Sur MLX : VAE-light
  (squelette cycle 1) puis VAE complet avec upgrade KL
  (cycle 2 `9906520`). Sur E-SNN : interpolation sur codes
  latents de taux de spikes. L'émission de recombine cible
  canal-2 RECOMBINED_LATENTS (et canal-4 ATTENTION_PRIOR en
  P_max).

Sur les 16 paires croisées `(op_i, op_j)`, 12 sont non-
commutatives (`docs/proofs/op-pair-analysis.md`) — ce qui
signifie que l'ordre compte et que l'ordre canonique
(replay → downscale → restructure ; recombine en parallèle)
est un choix de conception porteur, pas incident.

## 5.4 Huit primitives : 4 entrées + 4 canaux de sortie

La frontière du framework est typée en 8 points :

- **Entrées (awake → dream)** :
  - α — flux de traces brutes (P_max seul en cycle 2 :
    anneau 1024, FIFO, câblé `8ee452b`).
  - β — buffer épisodique (tous profils).
  - γ — snapshot sémantique (P_equ + P_max).
  - δ — snapshot latent (P_max pour recombine).
- **Sorties (dream → awake)** :
  - canal-1 UPDATED_WEIGHTS (tous profils).
  - canal-2 RECOMBINED_LATENTS (P_equ + P_max).
  - canal-3 RESTRUCTURED_TOPOLOGY (P_equ + P_max).
  - canal-4 ATTENTION_PRIOR (P_max seul ; garde S4 ≤ 1,5
    via `63af87d`).

Chaque canal de sortie est gardé par le **protocole de swap** :

1. Calculer l'émission candidate sous l'état de rêve actuel.
2. Appliquer S1 (non-régression retenue — candidat ne doit
   pas régresser sur le bench retenu sous l'évaluation du
   prédicteur).
3. Appliquer S2 (finitude — pas de NaN / Inf sur le
   candidat).
4. Appliquer S3 (topologie — connectivité d'espèces ortho,
   pas d'auto-boucles, bornes de comptage de couches).
5. Appliquer S4 (attention_budget ≤ 1,5) sur canal-4.
6. Si toutes les gardes PASSent, commit du swap et
   enregistrement de l'artefact ; sinon, abandon avec
   l'exception de refus de la garde.

Le protocole de swap est indépendant du substrat : les arrays
MLX et le LIFState traversent tous les deux la même séquence
de gardes.

## 5.5 Le worker de rêve asynchrone (C2.17)

Le cycle 1 a livré le worker de rêve comme **squelette
Future-API** — l'interface du worker existait mais
l'exécution était séquentielle. Le C2.17 du cycle 2
(`018fd05`) livre le **worker asynchrone réel** (basé sur
`asyncio`, exécution concurrente, plus un squelette). Le
worker :

- Accepte des 5-uplets `DreamEpisode` via une file.
- Dépêche la séquence d'ops de l'épisode à travers les
  factories d'op du substrat enregistré.
- Attend l'application des gardes + le commit du swap de
  manière asynchrone.
- Enregistre le run dans le registre de runs sous la clé R1.

La concurrence achète deux propriétés d'ingénierie qui
intéressent l'Article 2 : (i) le runner inter-substrats peut
émettre les cellules MLX + E-SNN en parallèle à la couche
d'orchestration (même si l'inférence par cellule est encore
synthétique à prédicteur partagé), et (ii) l'architecture est
prête pour les prédicteurs spécifiques au substrat du cycle 3
où le coût d'inférence par cellule diverge.

## 5.6 Registre de runs + déterminisme R1

Le registre de runs (`harness/storage/run_registry.py`)
applique le **contrat de reproductibilité R1** : chaque run
est indexé par un **préfixe SHA-256 32-hex** de `(c_version,
profile, seed, commit_sha)`. Ré-exécuter la même clé doit
produire un `run_id` identique. La largeur a été bumpée de
16 → 32 hex en cycle 1 commit `df731b0` après un flag de
revue de code sur le risque de collision 64-bit à grande
échelle.

Chaque artefact rapporté au §7 résout vers un run_id
enregistré :

- `ablation_runner_run_id` : `45eccc12953e758440fca182244ddba2`
  (entrée du runner multi-substrats).
- `cycle2_batch_id` : `3a94254190224ca82c70586e1f00d845`
  (le wrapper du batch cycle-2).
- `harness_version` : `C-v0.6.0+PARTIAL`.

Les artefacts qui ne résolvent pas vers un run_id ou un
fichier de preuve ne sont pas rapportés dans l'Article 2.
C'est la forme forte du contrat de reproductibilité : les
tables de l'article et les artefacts du dépôt doivent
circuler aller-retour.

## 5.7 Lignée DualVer : `C-v0.6.0+PARTIAL`

L'Article 1 a été expédié sous `C-v0.5.0+STABLE`. Le cycle 2
bumpe l'**axe formel** à `0.6` (extension de Conformité
E-SNN — une extension formelle de la surface du framework,
pas un changement cassant, d'où MINEUR) et laisse l'**axe
empirique** à `+PARTIAL` (l'ablation inter-substrats est une
substitution synthétique par portée, pas une preuve
empirique stable — d'où le qualifieur `PARTIAL` selon les
règles DualVer §12).

Le tag `C-v0.6.0+PARTIAL` est l'étiquette honnête pour
l'Article 2 : ingénierie-complet sur 2 substrats + 3 profils
+ 4 ops + 4 canaux, empirique-partiel parce que les lignes
inter-substrats partagent un prédicteur. Un qualifieur
`+STABLE` requiert une preuve à prédicteur réel du cycle 3.

---

## Notes pour révision

- Envisager une Figure 1 : diagramme de pipeline (entrées
  α/β/γ/δ → 4 ops sous gardes S1..S4 → canaux 1-4).
- Référencer §4 Conformité une fois les deux sections dans
  le brouillon complet.
- Resserrer à ≤ 1200 mots à la passe de pré-soumission
  NeurIPS.
