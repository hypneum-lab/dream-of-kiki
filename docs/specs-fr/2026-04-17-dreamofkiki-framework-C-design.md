# dreamOfkiki — Spécification de conception formelle du framework C

**Version** : C-v0.5.0+STABLE
**Date** : 2026-04-17
**Auteur** : Clément Saillant (L'Electron Rare)
**Statut** : Brouillon pour relecture utilisateur
**Companion** : `2026-04-17-dreamofkiki-master-design.md` (vision maître)

Ce spec est le **cœur formel** du programme de recherche dreamOfkiki. Il définit les primitives, profils, invariants, axiomes, protocole d'évaluation, discipline de test et interfaces cross-track. Conçu pour être indépendant du substrat ; l'instanciation courante est kiki-oniric (Track A), la future instanciation cycle 2 est E-SNN thalamocortical.

---

## 1. Portée de ce spec

**Dans la portée** :
- Définition formelle des primitives (α, β, γ, δ, 1, 2, 3, 4)
- Définition formelle des profils (P_min, P_equ, P_max) avec chaîne d'inclusion
- Ontologie de l'épisode de rêve (DE = 5-uplet)
- Invariants (I information, S sûreté, K calcul) avec application
- Axiomes (DR-0..DR-4) avec formalisation mathématique stricte + cibles de preuve
- Protocole d'évaluation (8 métriques, matrice stratifiée, reproductibilité bit-exact)
- Critères du mode de maturité publication-ready
- Pyramide de tests + cibles de couverture
- Contrats d'interface cross-track

**Hors portée** :
- Détails d'implémentation (voir conception Track A dans `kiki-oniric/`)
- Positionnement business/commercial (voir spec maître)
- Substrat E SNN cycle 2 (voir spec maître Section 2.2)

---

## 2. Primitives

### 2.1 Signatures

Les primitives sont des communications typées entre le processus éveil et le processus rêve. Chaque primitive a une signature formelle avec précondition, postcondition et borne de complexité.

#### Éveil → Rêve

**α — Traces brutes**
- Type : `Stream<ForwardPassTrace>` où `ForwardPassTrace = (tokens, activations, attention_patterns, prediction_errors)`
- But : flux massif des données d'inférence de l'éveil
- Activation : P_max uniquement
- Complexité : O(tokens × model_size) par passe avant
- Stockage : ring buffer mmap, rotation LIFO
- Précondition : processus éveil actif
- Postcondition : trace ajoutée au ring dans une latence bornée

**β — Buffer épisodique curaté**
- Type : `AppendLog<EpisodicRecord>` où `EpisodicRecord = (context, outcome, saillance_score, timestamp, consumed_by)`
- But : sélection hippocampique des épisodes saillants
- Activation : tous les profils
- Complexité : O(1) append quand saillance > seuil
- Stockage : SQLite avec index sur saillance + consumed_by
- Précondition : émetteur de saillance actif dans l'éveil
- Postcondition : record persisté atomiquement

**γ — Snapshot weights-only**
- Type : `Pointer<CheckpointHandle>`
- But : état minimal, pour downscaling SHY pur
- Activation : repli / diagnostic
- Complexité : O(1) accès pointeur
- Précondition : checkpoint existe
- Postcondition : vue read-only valide jusqu'au prochain checkpoint

**δ — Snapshots latents hiérarchiques**
- Type : `RingBuffer<HierarchicalSnapshot>` où `HierarchicalSnapshot = Dict[SpeciesName, LatentTensor]`
- But : représentation multi-échelle pour restructuration thalamocorticale
- Activation : P_equ, P_max
- Complexité : O(Σ species_dim) par snapshot
- Stockage : ring buffer N=256
- Précondition : hooks d'activation des species installés
- Postcondition : snapshot valide pendant au moins N×τ_snapshot wall-clock

#### Rêve → Éveil

**1 — Delta de poids**
- Type : `WeightUpdate` où `WeightUpdate = (LoRAdelta, FisherBump)`
- But : sortie de consolidation paramétrique
- Appliqué via : swap de worktree (Section 6)
- Contrainte : doit satisfaire l'invariant S1 (non-régression sur benchmark retenu) + S2 (valeurs finies)

**2 — Échantillons latents**
- Type : `Queue<LatentSample>` où `LatentSample = (species, latent_vector, provenance)`
- But : entrée de replay génératif / data augmentation pour l'éveil
- Appliqué via : le data augmenter de l'éveil consomme la file
- Contrainte : doit satisfaire l'invariant I3 (dérive distributionnelle bornée)

**3 — Changement de hiérarchie**
- Type : `TopologyDiff` où `TopologyDiff = List[TopoOp]`, `TopoOp ∈ {Add(layer), Remove(layer), Reroute(src, dst)}`
- But : sortie d'apprentissage structurel
- Appliqué via : application atomique au moment du swap
- Contrainte : doit satisfaire l'invariant S3 (topologie valide)

**4 — Prior d'attention**
- Type : `Tensor[species, attention_weight]`
- But : sortie de guidance méta-cognitive
- Appliqué via : copie au swap ou lecture seule en direct
- Contrainte : doit satisfaire l'invariant S4 (attention bornée)

### 2.2 Matrice de couplage des canaux

| Paire de canaux | Couplage | Rationale |
|-----------------|----------|-----------|
| β + δ | Synergique | Saillance + structure = hippocampe + thalamocortex cohérent |
| γ seul | Anti-couplage | Tue A (réactivation), C (recombinaison), D (restructure) — garder γ uniquement en complément |
| α + tous | Coûteux | Le flux massif rend tout le reste plus coûteux |
| 1 + 3 | Synergique mais risqué | Changement cohérent, nécessite la garde S3 |
| 2 seul | Isolé | L'éveil doit activement consommer (pas de mise à jour paramétrique) |
| 4 + 1 | Synergique | Les priors d'attention guident les prochaines mises à jour de poids |

---

## 3. Profils

### 3.1 Définitions formelles

```
P_min = { primitives_in: {β},     primitives_out: {1},     ops: {replay, downscale} }
P_equ = { primitives_in: {β, δ},  primitives_out: {1,3,4}, ops: {replay, downscale, restructure, recombine_light} }
P_max = { primitives_in: {α,β,δ}, primitives_out: {1,2,3,4}, ops: {replay, downscale, restructure, recombine_full} }
```

### 3.2 Chaîne d'inclusion (setup axiome DR-4)

```
ops(P_min) ⊆ ops(P_equ) ⊆ ops(P_max)
channels_in(P_min) ⊆ channels_in(P_equ) ⊆ channels_in(P_max)
channels_out(P_min) ⊆ channels_out(P_equ) ⊆ channels_out(P_max)
```

Cette inclusion n'est pas simplement cosmétique : elle garantit que toute exécution sous P_min est un cas particulier valide de P_equ (avec les canaux inutilisés ignorés), ce qui permet un raisonnement basé sur la monotonicité en ablation.

---

## 4. Ontologie de l'épisode de rêve (DE)

### 4.1 Définition

Un **épisode de rêve** est un 5-uplet :

```
DE = (trigger, input_slice, operation_set, output_delta, budget)
```

où :
- `trigger ∈ { scheduled(Δt), saturation(signal_type, threshold), external(event) }`
- `input_slice = snapshot(β_t) ∪ snapshot(δ_t)` (optionnellement ∪ `snapshot(α_t)` si P_max)
- `operation_set ⊆ {replay, downscale, restructure, recombine}` (composées)
- `output_delta ∈ {WeightUpdate, LatentSample, TopologyDiff, AttentionPrior}` (canaux 1-4)
- `budget = max(FLOPs, wall_time, energy_J)` (plafond de ressources)

### 4.2 Opérations

Quatre opérations canoniques, typées `Op : State × Budget → State × Output` :

**replay** — échantillonne depuis β, propage en avant via W courant, met à jour via gradient sur objectif de rétention
- Source : consolidation mnésique A-Walker/Stickgold
- Entrée : tranche β
- Sortie : WeightUpdate (canal 1)

**downscale** — applique un régulariseur homéostatique (SHY-style), rétrécit les poids vers un prior, réduit le bruit
- Source : homéostasie synaptique B-Tononi
- Entrée : γ ou W directement
- Sortie : WeightUpdate (canal 1)

**restructure** — applique une mise à jour de codage prédictif, minimise l'énergie libre sur la hiérarchie, ajoute/supprime/reroute des couches si nécessaire
- Source : D-Friston FEP
- Entrée : snapshot δ
- Sortie : WeightUpdate + TopologyDiff (canaux 1+3) + optionnellement AttentionPrior (canal 4)

**recombine** — échantillonne des latents, interpole/mixe entre species, génère de nouveaux candidats (type VAE ou diffusion)
- Source : génération créative C-Hobson/Solms
- Entrée : δ ou latents combinés
- Sortie : LatentSample (canal 2), optionnellement WeightUpdate

### 4.3 Composition canonique

Un épisode de rêve P_equ typique a un operation_set avec composition ordonnée :

```
DE_canonical = replay → downscale → restructure (parallèle : recombine)
```

Remarque : l'ordre canonique A→B→D en série est motivé thermodynamiquement (préserver d'abord, réguler ensuite, restructurer enfin). Recombine (C) tourne en parallèle comme branche séparée.

---

## 5. Invariants

Tous les invariants suivent la structure : `nom, énoncé formel, motivation, exemple kiki, test de réfutation, application, criticité`.

### 5.1 Famille I (Information)

**I1 — Conservation épisodique jusqu'à consolidation**
- Formel : `∀ e ∈ β_t, ∃ t' ≤ t + τ_max, e ∈ inputs(DE_{t'})` avant purge
- Motivation : rien de saillant n'est oublié sans chance d'être consolidé (cœur A-Walker)
- Kiki : chaque record β a un flag `consumed_by_DE_id` ; purge ssi flag posé
- Test : `SELECT COUNT(*) FROM beta WHERE consumed_by IS NULL AND created_at < now() - τ_max` == 0
- Application : cron horaire T-Ops
- Criticité : **BLOCKING**

**I2 — Traçabilité de hiérarchie**
- Formel : chaque `TopologyDiff δ` est enregistré avec `(DE_id, C_version, before_hash, after_hash, applied_at)` et le diff enregistré égale le delta topologique effectif
- Motivation : reproductibilité + forensique
- Kiki : table `hierarchy_diffs` avec contraintes FK
- Test : le diff entre topologies pré/post swap égale le diff enregistré
- Application : validateur post-swap T-Ops
- Criticité : **BLOCKING**

**I3 — Dérive distributionnelle latente bornée**
- Formel : `KL(p_recombined || p_awake) ≤ ε_drift` par version C, mesurée sur fenêtre glissante de 1000 échantillons
- Motivation : empêcher le rêve d'empoisonner la distribution de l'éveil
- Kiki : estimation KL appariée sur échantillons matchés
- Test : intégré à la métrique M4.a
- Application : vérification dans le runner d'expérience A.5
- Criticité : **WARN** (détecter, pas bloquer)

### 5.2 Famille S (Sûreté)

**S1 — Non-régression sur tâche retenue**
- Formel : `acc(W_post_swap, retained) ≥ acc(W_pre_swap, retained) − δ_regression` (défaut δ = 2 %)
- Motivation : prévenir l'oubli catastrophique via la consolidation mnésique
- Kiki : retained = sous-ensemble gelé de 500 items de mega-v2
- Test : intégré à l'étape 3 du protocole de swap
- Application : le protocole de swap avorte si échec, log vers `aborted-swaps/`
- Criticité : **BLOCKING**

**S2 — Pas de NaN/Inf dans W_scratch**
- Formel : `∀ w ∈ W_scratch, isfinite(w) ∧ |w| ≤ w_max`
- Motivation : éviter l'explosion de gradient via recombine mal calibré
- Kiki : scan de norme numpy/MLX + détection NaN pré-swap
- Application : étape 2 du protocole de swap, avant la garde S1
- Criticité : **BLOCKING**

**S3 — Garde de hiérarchie**
- Formel : `validate_topology(G_post) == True` où validate vérifie (i) connexité des species, (ii) pas de cycles indésirables, (iii) compte de couches dans les bornes
- Motivation : restructure peut déconnecter ρ_sem et rendre kiki muet
- Kiki : fonction `validate_topology(G)` avec liste de règles explicite
- Application : intégration à la garde S1 du swap + vérification pré-émission dans A.4
- Criticité : **BLOCKING**

**S4 — Prior d'attention borné**
- Formel : `∀ i, prior[i] ∈ [0, 1] ∧ Σ prior ≤ budget_attention` (défaut 1.5 × uniforme)
- Motivation : empêcher le rêve obsessionnel (rêver 90 % sur ρ_phono bloque la syntaxe)
- Kiki : clamp + normalisation avant application
- Application : émission du runtime de rêve A.4
- Criticité : **WARN** (auto-correction + log)

### 5.3 Famille K (Calcul)

**K1 — Budget de l'épisode de rêve respecté**
- Formel : `∀ DE, FLOPs_actual ≤ budget.FLOPs ∧ wall_time_actual ≤ budget.wall ∧ energy_actual ≤ budget.energy`
- Motivation : prévenir les DE qui s'emballent
- Kiki : wrapper en context manager avec compteur FLOPs (profil MLX) + proxy d'énergie
- Application : wrapper runtime A.4 ; en cas de dépassement → avorter le DE, jeter la sortie partielle
- Criticité : **BLOCKING** (pour ce DE, pas pour le pipeline)

**K3 — Latence de swap bornée**
- Formel : `wall_clock(swap_atomic) ≤ K3_max` (défaut 1 s, configurable par topologie)
- Motivation : glitch perçu si trop long
- Kiki : mesure avant/après swap
- Application : monitoring de swap, alerte si dépassé
- Criticité : **WARN**

**K4 — Couverture de la matrice d'évaluation**
- Formel : pour tout bump MAJOR de la version C, la matrice stratifiée est entièrement exécutée avant le tag
- Motivation : prévenir la publication sur version C non validée
- Kiki : hook CI T-Ops refuse le tag sans tous les runs présents
- Application : CI T-Ops
- Criticité : **BLOCKING** (pour le tagging)

---

## 6. Axiomes (DR — Dream-episode Rules, strictement formalisés)

### 6.1 Cadre formel

Soit :
- **State** = `(W, H, M)` où W = poids, H = topologie de hiérarchie, M = mémoire épisodique (buffer β)
- **Op** = monoïde d'opérations {replay, downscale, restructure, recombine} typées `Op : State × Budget → State × Output`
- **DE** = composition ordonnée d'éléments de Op avec budget additif

### 6.2 Axiomes

#### DR-0 (Redevabilité)

```
∀ δ ∈ dream_output_channels,
∃ DE ∈ History : budget(DE) < ∞ ∧ δ ∈ outputs(DE)
```

**Interprétation** : pas de « rêve ambiant » — chaque contribution du rêve est traçable à un DE borné.

**Application** : I1 + I2 + K1 combinés.

#### DR-1 (Conservation épisodique, formalise I1)

```
∀ e ∈ β_t, ∃ t' ∈ [t, t + τ_max] : e ∈ inputs(DE_{t'})
```

**Interprétation** : chaque épisode bufferisé a une chance de consolidation.

**Application** : vérification runtime I1 + porte de purge β.

#### DR-2 (Compositionnalité — À PROUVER)

```
∀ op_1, op_2 ∈ Op,
  op_2 ∘ op_1 ∈ Op
  ∧ budget(op_2 ∘ op_1) = budget(op_1) + budget(op_2)
  ∧ effect(op_2 ∘ op_1, s) = effect(op_2, effect(op_1, s))
```

**Cible de preuve** : par cas sur les 4 opérations + lemme d'associativité.

**Esquisse de preuve** (à compléter dans la sortie Track C `formal-proofs.md`) :

1. **Clôture** : montrer que l'application de op_2 après op_1 préserve le type State et produit une Output valide. Par règles de typage.
2. **Additivité du budget** : par définition du budget comme compteur de ressources (FLOPs, wall-time, énergie), additif par construction.
3. **Composition fonctionnelle** : par définition de `effect` comme fonction sur State.
4. **Associativité** : montrer `(op_3 ∘ op_2) ∘ op_1 = op_3 ∘ (op_2 ∘ op_1)` par analyse de cas.

**La commutativité n'est PAS revendiquée**. En particulier `recombine ∘ downscale ≠ downscale ∘ recombine` en général. Le monoïde est non commutatif.

**Repli DR-2'** : si la preuve stricte de compositionnalité échoue, adopter :

```
DR-2' (Compositionnalité avec ordre canonique)
∀ op_1, op_2 ∈ Op_canonical_order,
  op_2 ∘ op_1 ∈ Op
  (ordre canonique = replay < downscale < restructure < recombine)
```

Ceci est plus faible mais suffisant pour la composition canonique des DE définie en 4.3.

#### DR-3 (Indépendance du substrat)

```
∀ substrat S, si S satisfait le Critère de Conformité ci-dessous,
alors DR-0, DR-1, DR-2 (ou DR-2') sont vérifiés sur S.
```

**Interprétation** : kiki-oniric et E-SNN hypothétique sont tous deux des instanciations valides **ssi ils passent le Critère de Conformité**. Le typage de signature pur n'est **pas** suffisant — la conformité comportementale est requise.

**Critère de Conformité (exécutable)** : un substrat S instancie le framework ssi les trois conditions tiennent :

1. **Typage de signature** — S implémente les signatures Protocol typées des primitives α, β, γ, δ, 1, 2, 3, 4 telles que définies au §2.1 (Python Protocol types, interfaces TypeScript, ou équivalent dans le langage cible).

2. **Tests de propriétés d'axiomes** — la suite de tests de propriétés pour DR-0, DR-1 et DR-2 (ou DR-2') passe sur S avec ≥100 % de couverture sur les cas BLOCKING. La suite de tests est versionnée dans T-Ops sous `tests/conformance/axioms/` et exécutée via `dream-harness conformance --substrate <S>`.

3. **Invariants BLOCKING applicables** — les invariants S1 (non-régression retenue), S2 (pas de NaN/Inf), S3 (hiérarchie valide) et I1 (conservation épisodique) sont implémentés comme des vérifications runtime sur S avec application automatisée (abort-on-violation + log vers `aborted-swaps/` ou équivalent).

**Énoncé formel** :

```
conforms(S) ≜ typed(S) ∧ axiom_tests_pass(S) ∧ invariants_enforced(S, {S1,S2,S3,I1})
∀ S, conforms(S) ⟹ (DR-0 ∧ DR-1 ∧ DR-2) holds on S
```

**Esquisse de preuve** : étant donné `conforms(S)`, chacune des trois conditions active directement une partie des axiomes :
- `typed(S)` → les opérations sont compositionnelles en type (partie clôture de DR-2)
- `axiom_tests_pass(S)` → les axiomes comportementaux sont empiriquement validés (partie composition fonctionnelle de DR-2, redevabilité DR-0, conservation DR-1)
- `invariants_enforced(S)` → les garanties runtime préservent les axiomes sous exécution concurrente (swap de worktree, processus rêve asynchrone)

Ensemble, le substrat conforme est validé à la fois **statiquement** (typage + tests de propriété) et **dynamiquement** (application des invariants), pas simplement asserté par construction.

**Statut cycle 1** : la conformité kiki-oniric est établie incrémentalement — typage de signature verrouillé en S2 (Story 0 expose-primitives), tests d'axiomes passant en S6 (aligné sur jalon G3-draft), invariants applicables en S8 (runtime Track A P_equ complet). Le substrat cycle-2 E-SNN doit passer les trois mêmes conditions avant de pouvoir être revendiqué comme instanciation.

#### DR-4 (Inclusion de chaîne de profils)

```
ops(P_min) ⊆ ops(P_equ) ⊆ ops(P_max)
∧ channels(P_min) ⊆ channels(P_equ) ⊆ channels(P_max)
```

**Lemme DR-4.L** : si P_min est valide (invariants satisfaits) sur substrat S, alors P_equ ne dégrade pas strictement les métriques monotones en capacité.

**Esquisse de preuve** : P_equ étend P_min avec canaux et opérations supplémentaires ; un run P_equ peut simuler un run P_min en ignorant les canaux ajoutés ; donc best(P_equ) ≥ best(P_min) en espérance sur les classes de métriques appropriées.

### 6.3 Vérification des axiomes en implémentation

Chaque axiome doit être **empiriquement testable** via un test mécanique :

| Axiome | Forme du test | Track | Fréquence |
|--------|---------------|-------|-----------|
| DR-0 | Requête de tous les deltas de sortie → trace vers DE avec budget fini | test de propriété T-Ops | CI nocturne |
| DR-1 | Requête des records β → tous consommés dans τ_max | cron horaire I1 | Runtime |
| DR-2 | Test de propriété : paires d'ops aléatoires, vérifier types de composition + additivité | CI T-Ops | Chaque PR |
| DR-3 | Pour chaque substrat, exécuter la suite d'invariants complète → tous passent | T-Ops | Par substrat |
| DR-4 | Comparer les ensembles ops/canaux → inclusion tient | test de contrat T-Ops | Chaque changement de profil |

---

## 7. Protocole runtime (swap de worktree)

### 7.1 Copies d'état

- `W_awake` : poids actifs utilisés par le processus éveil (lecture/écriture)
- `W_dream` : snapshot gelé lu par le processus rêve (lecture seule)
- `W_scratch` : copie de travail modifiée par le processus rêve (lecture/écriture, rêve uniquement)

### 7.2 Protocole de swap (atomique)

Quand le rêve signale `ready_to_commit` :

```
1. awake.pause(max=500ms)                      # budget K3
2. validate_scratch_finite(W_scratch)          # garde S2
3. acc_post = eval(W_scratch, retained_bench)  # garde S1
4. if acc_post < acc_pre - δ_regression:
      abort_swap()
      log_to('aborted-swaps/')
      return
5. if topology_changed:
      validate_topology(H_scratch)             # garde S3
      if fail: abort_swap() ; log ; return
6. atomic_swap:
      W_awake ← W_scratch
      W_dream ← W_awake  (fresh baseline)
      W_scratch ← W_awake  (nouvelle copie de travail)
7. apply_channel_4_if_present(attention_prior)
8. awake.resume()
```

### 7.3 Nettoyage post-swap

- Mettre à jour le tableau de bord (T-Ops.3) avec le résultat du swap
- Logger le swap vers `runs/<C-ver>/<profile>/<seed>/<run-id>/swaps/`
- Avancer la machine d'état de l'épisode de rêve
- Réinitialiser `W_scratch` prêt pour le prochain cycle DE

---

## 8. Protocole d'évaluation

### 8.1 Métriques (8 au total, E3 cognitif + E4 ingénierie)

| Code | Nom | Formule/Procédure | Stratégie déterministe |
|------|-----|-------------------|------------------------|
| M1.a | Taux d'oubli | `(acc_task_1_initial - acc_task_1_after_task_N) / acc_task_1_initial` | RNG numpy seedé |
| M1.b | Exactitude moyenne | `mean(acc_task_i for i in 1..N after full sequence)` | RNG numpy seedé |
| M2.b | Alignement RSA IRMf | Corrélation de Pearson entre RDM(représentations kiki) et RDM(activité IRMf) sur stimuli matchés | Ordre des stimuli seedé ; nilearn mode CPU |
| M3.a | Ratio FLOPs | `FLOPs(dream) / FLOPs(awake)` sur fenêtre comparable | Profil MLX statique |
| M3.b | Gain hors-ligne | `Δ(M1.b, post-dream) - Δ(M1.b, no-dream)` normalisé par wall-clock FLOPs-équivalent | Wall-clock simulé depuis les FLOPs |
| M3.c | Énergie par épisode | `energy_proxy = f(FLOPs, model_size, precision)` — fonction calibrée | Fonction déterministe |
| M4.a | Qualité de recombinaison | Le scorer enseignant (Qwen3.5-9B Q4_K_M SHA épinglé) évalue la plausibilité + diversité des échantillons latents | Scorer temp=0, seed=0 |
| M4.b | Découverte de structure | Test de permutation sur invariants de hiérarchie appris vs baseline | Permutation seedée |

### 8.2 Matrice stratifiée

| Type de bump | Cellules obligatoires |
|--------------|-----------------------|
| **PATCH** (C-vX.Y.z+1) | Métrique de l'axe touché × P_equ × 1 seed |
| **MINOR** (C-vX.y+1) | 8 métriques × P_equ × 3 seeds |
| **MAJOR** (C-vx+1) | **Grille complète** : 8 métriques × 3 profils × 3 seeds = 72 runs |
| **Changement EC** (STABLE → DIRTY) | Re-run des métriques publiées uniquement |

### 8.3 Contrats de reproductibilité

**R1 (bit-exact)** : chaque `MetricResult` est reproductible bit-identique depuis `(c_version, profile, seed, run_id, commit_sha, benchmark_version)`. S'applique aux 8 métriques via les stratégies du §8.1.

**R3 (adressabilité d'artefact)** : tous les artefacts sont adressables par checksum SHA-256 stocké dans `metadata.yaml`, schéma de stockage selon spec maître §5.4.

(R2 supprimé — toutes les métriques sont désormais déterministes.)

---

## 9. Mode de maturité publication-ready

Pipeline de maturité à trois niveaux, avec transitions explicites :

| Mode | Critères | Autorise |
|------|----------|----------|
| **RED** | ≥1 BLOCKING violé OU ≥3 WARN en 24 h | Pas d'action, escalade Dream-sync |
| **GREEN** | Tous les BLOCKING respectés, WARN sous seuil | Développement, expériences |
| **PUBLICATION-READY** | GREEN + critères additionnels ci-dessous | Soumission d'article |

### Critères additionnels pour PUBLICATION-READY

| Critère | Seuil |
|---------|-------|
| Couverture de la matrice d'évaluation | 100 % des cellules stratifiées (MAJOR/MINOR/PATCH) |
| Reproductibilité | ≥3 seeds par (profil, métrique), variance documentée |
| Banc de test retenu | Pas de régression δ > 1 % sur les 10 derniers swaps |
| Zéro BLOCKING | 7 jours consécutifs sans violation |
| Statut DualVer | `+STABLE` (pas DIRTY, pas INVALIDATED) |
| Relecture pré-soumission | ≥1 retour positif du réseau T-Col.4 |
| Axiomes DR | DR-0..DR-4 formalisés, DR-2 prouvé (ou DR-2' adopté) |
| Ablation complète | P_min, P_equ, P_max testés avec significativité |
| Brouillon d'article | Complet, peer-reviewé en interne ≥1×, pas de TODO |

**Application** : la CI T-Ops bloque le commit vers le dossier `papers/*/submitted/` sauf si mode == PUBLICATION-READY.

**Transitions** :
- RED → GREEN : résoudre le blocker + 24 h de monitoring
- GREEN → PUBLICATION-READY : vérification CI T-Ops déclenchée manuellement (décision Dream-sync)
- PUBLICATION-READY → GREEN : tout bump MAJOR de version C efface le statut

---

## 10. Discipline de test

### 10.1 Pyramide

```
      Tests E2E (peu, lents, niveau scénario)
     Tests d'intégration (contrats cross-track)
    Tests unitaires (par module, rapides)
   Tests de propriété (invariants, Hypothesis)
```

### 10.2 Cibles de couverture (mode strict)

| Famille | Cible de couverture | Application |
|---------|---------------------|-------------|
| Tests unitaires | ≥90 % ligne par module | Porte CI bloquante |
| Tests de propriété | Invariants BLOCKING + WARN 100 % | Porte CI bloquante |
| Tests d'axiomes | DR-0..DR-4 tous testés mécaniquement | Porte CI bloquante |
| Tests d'intégration | Chaque interface cross-track ≥1× | Porte CI bloquante |
| Tests de banc de test | Banc retenu re-exécuté par merge | Porte CI bloquante |
| Tests de repro | Runs en or re-testés hebdomadairement | Alerte nocturne |

### 10.3 Discipline TDD

- **Invariants I/S/K et axiomes DR** : TDD strict — test d'abord, code jusqu'à ce que le test passe.
- **Pipelines de données + métriques** : TDD strict — doivent être testés avant production.
- **Code applicatif** : TDD recommandé mais non appliqué — la porte de couverture 90 % garantit le résultat.

---

## 11. Interfaces cross-track

Chaque interface est un **artefact versionné de première classe** (taggé semver avec C-DualVer), testé par tests de contrat dans T-Ops.

| Interface | Artefact | Propriétaires | Date de verrouillage |
|-----------|----------|---------------|---------------------|
| C ↔ A | `interfaces/primitives.md` + Python Protocol types | Track C + Track A | S2 |
| C ↔ T-Ops | `interfaces/eval-matrix.yaml` | Track C + T-Ops | S5 |
| A ↔ T-Ops | `interfaces/experiment-contract.md` | Track A + T-Ops | S6 |
| C ↔ T-Col | `interfaces/proposal-template.md` | Track C + T-Col | S3 |
| A ↔ T-Col | `interfaces/fmri-schema.yaml` | Track A + T-Col | S4 |

Les changements d'interface nécessitent un bump minor du DualVer des **deux** tracks concernées.

---

## 12. Schéma de versionnage DualVer

### 12.1 Format

`C-v<FC-MAJOR>.<FC-MINOR>.<FC-PATCH>+<EC-STATE>`

### 12.2 Axe FC (cohérence formelle — type SemVer)

- **FC-MAJOR** : changement sémantique d'axiome / primitive / signature d'invariant
- **FC-MINOR** : ajout d'un nouvel axiome / nouvelle primitive optionnelle / nouvelle contrainte dérivée
- **FC-PATCH** : clarification / typo / reformulation équivalente / fix de pseudo-code

### 12.3 Axe EC (cohérence empirique)

- **STABLE** : tous les résultats empiriques mesurés sous FC courant, cohérents avec les axiomes
- **DIRTY** : au moins un résultat est orphelin (mesuré sous un FC plus ancien, non re-vérifié)
- **INVALIDATED** : un résultat publié/soumis est invalidé par un FC-MAJOR courant

### 12.4 Flux de bump

1. Le propriétaire Track C propose le bump via PR
2. T-Ops exécute la **suite de compatibilité stratifiée** (par §8.2) contre le nouveau C
3. Si les tests de contrat passent → EC reste STABLE
4. Si les tests cassent → EC auto-mis à DIRTY, tests morts taggés, issue ouverte
5. Merge de la PR → bump officiel, glossaire mis à jour, T-Col notifié

---

## 13. Questions formelles ouvertes (à résoudre S1-S8)

1. **Preuve DR-2** — la compositionnalité stricte est-elle prouvable, ou devons-nous nous replier sur DR-2' ? (décision G3 S8)
2. **Frontière de commutativité** — pour quelles paires d'ops `op_2 ∘ op_1 = op_1 ∘ op_2` ? (informe l'ordre canonique)
3. **Additivité du budget sous branche C parallèle** — comment l'additivité se compose-t-elle quand recombine tourne en parallèle de la branche série A→B→D ? (nécessite extension monoïde parallèle)
4. **Critère de conformité du substrat** — RÉSOLU (voir §6.2 DR-3) : la conformité requiert typage de signature ∧ tests de propriétés d'axiomes ∧ invariants BLOCKING applicables. Le typage seul est insuffisant.
5. **Classe de métriques de monotonicité pour DR-4.L** — caractériser formellement quelles métriques sont monotones en capacité pour que le lemme tienne strictement.

Ces questions alimentent le travail de preuve de la section C.4 (annexe formal-proofs), semaines S5-S8.

---

## 14. Annexes

### A. Glossaire canonique (extrait)

| Terme | Définition |
|-------|-----------|
| `dreamOfkiki` | Nom du programme, identifiant logique (camelCase). Nom technique de dépôt : `dream-of-kiki` (kebab-case pour compat filesystem) |
| `kiki-oniric` | Fork dédiée de `kiki-flow-core` pour les expériences Track A |
| Épisode de rêve (DE) | 5-uplet (trigger, input_slice, operation_set, output_delta, budget) — unité atomique de rêve |
| Processus éveil | Processus d'inférence/entraînement kiki tournant en temps réel |
| Processus rêve | Processus de consolidation hors-ligne asynchrone tournant sur calcul séparé (par topologie de profil) |
| ortho species | 4 niveaux linguistiques : ρ_phono, ρ_lex, ρ_syntax, ρ_sem (Baddeley + Levelt) |
| Swap | Promotion atomique de W_scratch vers W_awake (remplace le merge du design original) |
| PUBLICATION-READY | Mode de maturité du pipeline autorisant la soumission d'article |
| DualVer | Schéma de versionnage `C-v<FC>+<EC>` avec axes formel + empirique séparés |

### B. Référence au spec maître

Voir `2026-04-17-dreamofkiki-master-design.md` pour :
- Intégration business/calendrier
- Détails de la stratégie de publication
- Registre de risques (sections 7-8)
- Détails opérationnels Track A/T-Ops/T-Col
- Topologie de calcul par profil
- Calendrier S1-S28

---

**Fin du spec de conception formelle du framework C.**

Étape suivante après relecture utilisateur : invoquer le skill `writing-plans` pour générer le plan d'implémentation par track.
