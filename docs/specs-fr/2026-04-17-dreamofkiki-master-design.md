# dreamOfkiki — Spécification de conception maître

**Version** : v0.1.0-draft
**Date** : 2026-04-17
**Auteur** : Clément Saillant (L'Electron Rare)
**Statut** : Brouillon pour relecture utilisateur
**Specs liées** :
- `2026-04-17-dreamofkiki-framework-C-design.md` (framework formel)
**Projets liés** :
- `kiki-flow-research` (3 stories en cours — indépendantes de ce programme)
- `kiki-flow-core` (source de la fork `kiki-oniric`)

---

## 0. Résumé

dreamOfkiki est un programme de recherche de 6-7 mois (cycle 1) visant à concevoir une architecture cognitive inspirée du sommeil humain pour des systèmes d'IA. Le programme produit deux articles complémentaires — un article théorique (framework formel C, cible *Nature Human Behaviour*) et un article empirique (ablation sur substrat kiki-oniric, cible NeurIPS/ICML/TMLR) — avec publication séquentielle stricte.

L'architecture repose sur quatre piliers théoriques (consolidation mnésique Walker/Stickgold, homéostasie synaptique Tononi SHY, codage prédictif Friston FEP en série + recombinaison créative Hobson/Solms en parallèle), instanciée sur kiki-oniric (fork dédiée de kiki-flow-core) via un runtime asynchrone à swap de worktree atomique, évaluée sur une matrice stratifiée de 8 métriques (E3 cognitif + E4 ingénierie) avec reproductibilité bit-exact.

Le programme adopte une **approche 5 tracks** (C framework, A implémentation kiki, T-Ops infrastructure, T-Col collaboration externe, plus un track E substrat SNN thalamocortical **différé au cycle 2**), un versionnage hybride **DualVer** (formel + empirique), et un modèle de livraison scientifique **ouvert par défaut** (code MIT, pré-enregistrement OSF, DOI Zenodo, tableau de bord public en lecture seule).

---

## 1. Motivation & positionnement scientifique

### 1.1 Question de recherche centrale

*Comment une IA peut-elle apprendre, mémoriser et organiser son savoir par le rêve ?*

Le rêve n'est pas traité comme métaphore mais comme **fonction computationnelle** : un processus hors-ligne/asynchrone dédié à consolider, réguler et restructurer le savoir acquis pendant l'éveil. Le programme formalise cette intuition en un framework indépendant du substrat et le valide empiriquement sur un système linguistique hiérarchique existant (kiki).

### 1.2 Piliers théoriques

**En série (séquence canonique de l'épisode de rêve)** :
- **A — Consolidation mnésique** *(Walker, Stickgold, McClelland 1995)* : la réactivation hippocampique transfère l'épisodique vers le sémantique
- **B — Homéostasie synaptique SHY** *(Tononi, Cirelli)* : la régulation à la baisse réduit le bruit, économise l'énergie, améliore le rapport signal/bruit
- **D — Restructuration prédictive** *(Friston principe d'énergie libre, Clark codage prédictif)* : minimisation de l'énergie libre globale, abstraction d'invariants

**En parallèle (branche créative)** :
- **C — Recombinaison & simulation** *(Hobson, Solms, Revonsuo)* : génération de scénarios nouveaux, exploration de l'espace des possibles

L'ordre A→B→D est justifié thermodynamiquement (dissipation progressive : consolider d'abord, dégrader ensuite, restructurer enfin). La branche C en parallèle capture la branche REM créative qui coexiste avec la consolidation NREM.

### 1.3 Positionnement par rapport à la littérature

| Travaux | Contribution | Apport dreamOfkiki |
|---------|--------------|--------------------|
| van de Ven 2020 (replay inspiré du cerveau) | Replay génératif via VAE latent | Framework formel + ablation 3 profils + alignement IRMf |
| Kirkpatrick 2017 (EWC) / OPLoRA | Consolidation synaptique Fisher | Intégration comme instance de B-Tononi dans une taxonomie unifiée |
| Complementary Learning Systems | Analogie hippocampe + néocortex | Opérationnalisation en épisode de rêve comptable |
| Diba & Buzsáki 2007 | Replay en veille calme | Justifie le choix Q6-C entrelacé continu |
| Tononi SHY | Régulation à la baisse pendant le sommeil | Axiomatisé dans le framework C (opération downscale) |
| Friston FEP | Minimisation de l'énergie libre | Opération restructure |

**Différenciation** : le framework C est le premier (à notre connaissance) à **axiomatiser** formellement l'ensemble des opérations oniriques comme un monoïde d'opérations composables avec budget borné (DR-2 compositionnalité).

---

## 2. Portée & hors-portée

### 2.1 Portée cycle 1 (6-7 mois, S1-S28)

- **Track C** : framework formel (40-80 pages) avec axiomes DR-0..DR-4 formalisés strictement, preuve de compositionnalité DR-2 (ou repli DR-2')
- **Track A** : implémentation 3 profils (P_min, P_equ, P_max) sur fork `kiki-oniric` avec runtime asynchrone à swap de worktree
- **Track T-Ops** : infrastructure CI/CD, banc de test d'évaluation stratifié, tableau de bord public, scorer enseignant gelé, reproductibilité bit-exact
- **Track T-Col** : approche des labos IRMf (Gallant/Norman/Huth), repli Studyforrest pré-verrouillé, réseau de pré-soumission
- **2 articles** : Article 1 (Nature HB) soumis S20, Article 2 (NeurIPS/ICML/TMLR) brouillon complet S24, soumis **après acceptation Article 1** (S1-séquentielle stricte)
- **Infrastructure ouverte** : GitHub public, modèles HuggingFace, DOI Zenodo, pré-enregistrement OSF

### 2.2 Hors-portée cycle 1 (déplacé au cycle 2)

- **Track E — substrat SNN thalamocortical** : co-conception clean-room sur Loihi-2 ou MLX-SNN personnalisé. Positionné comme **Travaux futurs** dans l'Article 1, avec amorce d'approche Intel NRC T-Col.5 dès S9+
- **Intégrations en aval** : consolidation mascarade, rêves Professor Zacus, agents Factory 4 Life — tout cela relève de projets séparés
- **Applications grand public / commerciales** : formations Moodle, conseil, levier business L'Electron Rare — gérés hors du programme scientifique
- **Extensions multimodales** : perception + action incarnée — réservé aux cycles ultérieurs

---

## 3. Architecture d'ensemble (5 tracks)

```
┌──────────────────────────────────────────────────────────┐
│           COUCHE DE MÉTA-COORDINATION                     │
│  Invariants I/S/K · Glossaire · DualVer · Dream-sync HITL │
└────┬────────┬────────────┬───────────┬───────────────────┘
     │        │            │           │
  ┌──▼──┐  ┌──▼──┐      ┌──▼───┐   ┌───▼──┐
  │  C  │  │  A  │      │T-Ops │   │T-Col │
  │Frmwk│  │kiki-│      │infra │   │ext   │
  │     │  │oniric│      │trans │   │trans │
  │ 8s  │  │ 12s │      │ 28s  │   │ 28s  │
  │     │  │     │      │      │   │      │
  │P1   │  │P2   │      │(rap. │   │(labo │
  │NatHB│  │NeuR │      │tech) │   │ IRMf)│
  │     │  │IPS  │      │      │   │      │
  └─────┘  └─────┘      └──────┘   └──────┘
     ▲        ▲
     └────────┴────┐
                   ▼
         BANC DE TEST D'ÉVALUATION PARTAGÉ
         (matrice stratifiée, repro bit-exact)

ROADMAP cycle 2 (2027+) :
└─ E — co-conception SNN thalamocortical (Loihi-2 ou MLX-SNN)
   Documenté comme Travaux futurs Section 7 de l'Article 1.
```

### 3.1 Méta-coordination

- **Registre d'invariants** : `invariants.md` numéroté (I1..In information, S1..Sn sûreté, K1..Kn calcul, DR-0..DR-4 axiomes théoriques). Chaque track y fait référence explicitement.
- **Glossaire canonique** : `glossary.md`, termes définis une seule fois. Mention du nom logique `dreamOfkiki` (camelCase) vs nom de dépôt technique `dream-of-kiki` (kebab-case).
- **Sync HITL** : « Dream-sync Monday » chaque lundi, 1 h chrono, paquet de sync par track, décisions prises par vous. Budget d'attention ≤15 h/sem avec cut-gate à 2 semaines.
- **DualVer** : format `C-v<FC-MAJOR>.<FC-MINOR>.<FC-PATCH>+<EC-STATE>`. FC = cohérence formelle (SemVer), EC = cohérence empirique {STABLE, DIRTY, INVALIDATED}.

### 3.2 Tracks scientifiques

| Track | Durée | Livrables clés | Responsable |
|-------|-------|---------------|-------------|
| **C** (framework) | S1-S8 | `framework-C-v1.0.md`, axiomes DR formalisés, preuve DR-2, protocole d'évaluation | Vous |
| **A** (kiki-oniric) | S1-S12 | Fork `kiki-oniric`, Story 0 expose-primitives, 3 profils P_min/P_equ/P_max, runtime à swap de worktree | Vous + sous-agents |

### 3.3 Tracks de services transverses

| Track | Durée | Livrables clés |
|-------|-------|----------------|
| **T-Ops** | S1-S28 | Monorepo CI/CD, banc de test d'évaluation stratifié, tableau de bord public `dream.saillant.cc`, application de la reproductibilité bit-exact, scorer enseignant gelé (Qwen3.5-9B Q4_K_M SHA épinglé) |
| **T-Col** | S1-S28 (chargement amont S1-S8) | Repli Studyforrest verrouillé S2, propositions 3 labos IRMf S4, réseau de pré-soumission S14-S18, amorçage cycle-2 Intel NRC S9+ |

### 3.4 Calendrier 6-7 mois (28 semaines)

| Phase | Semaines | Activités dominantes |
|-------|----------|----------------------|
| **Mise en place** | S1-S2 | Verrouillage repli T-Col, infra T-Ops, démarrage C, Story 0 A expose-primitives |
| **Formalisation + Fondation** | S3-S6 | C v0.3→v0.5, A P_min fonctionnel, pré-enregistrement OSF H1-H4 |
| **Implémentation cœur** | S7-S12 | C v0.7→v0.9, A P_equ, jalon G2 (P_min viable S8), G3 (preuve DR-2 S8), G4 (P_equ fonctionnel S12) |
| **Ablation + Expériences** | S13-S18 | C v1.0, A P_max, évaluation E3+E4 complète, porte G5 PUBLICATION-READY S18 |
| **Soumission Article 1** | S18-S20 | Article 1 arXiv S18, relecture pré-soumission T-Col.4, soumission Nature HB S20 |
| **Brouillon Article 2 + tampon** | S20-S24 | Article 2 brouillon complet S24 (gelé en attente acceptation Article 1), lancement business L'Electron Rare |
| **Tampon** | S25-S28 | Maintenance, réponse aux relectures Article 1 si rapide, porte G6 décision cycle-2 S28 |

---

## 4. Architecture runtime (éveil + rêve asynchrones)

### 4.1 Topologie variable par profil

| Profil | Éveil | Rêve | État partagé |
|--------|-------|-------|--------------|
| **P_min** (β → 1) | Studio M3 Ultra | Studio M3 Ultra (thread worker + worktree isolé) | mmap local Arrow IPC |
| **P_equ** (β+δ → 1+3+4) | Studio M3 Ultra | KXKM-AI RTX 4090 (GPU dédié rêve) | NFS `/mnt/shared` + sync par lot |
| **P_max** (α+β+δ → 1+2+3+4) | Studio M3 Ultra | KXKM-AI + GrosMac M5 (VAE de recombinaison) | pipeline 3 machines + cache LRU |

### 4.2 Swap de worktree atomique (pas de merge actif)

Le processus rêve tient `W_scratch` (copie de travail modifiable) tandis que l'éveil opère sur `W_awake`. `W_dream` est un snapshot gelé. Quand l'épisode de rêve signale `ready_to_commit` (budget atteint / N DE complétés / validation interne OK) :

1. L'éveil se met en pause ~500 ms
2. Test de garde S2 (NaN/Inf) sur W_scratch
3. Test de garde S1 (non-régression sur banc de test retenu) sur W_scratch
4. Test de garde S3 (topologie valide) si canal 3 présent
5. Si tous OK → **swap atomique** : `W_awake ← W_scratch`, re-baseline `W_dream ← W_awake`, nouveau `W_scratch ← W_awake`
6. L'éveil reprend

Avantages par rapport à EASGD : conflits éliminés par construction, rollback trivial (swap avorté), reproductibilité préservée.

### 4.3 Canaux d'information (profils)

**Éveil → Rêve** :
- α traces brutes (P_max uniquement) : ring buffer disque 10 Go max
- β épisodique curaté : log append SQLite avec signature de saillance
- δ latents hiérarchiques : ring 256 snapshots × 5 niveaux ortho species
- γ weights-only : pointeur vers le checkpoint courant

**Rêve → Éveil** :
- 1 weight_delta : via swap
- 2 latent_samples : file FIFO consommée par le data augmenter de l'éveil
- 3 hierarchy_chg : appliqué atomiquement au swap, avec garde S3
- 4 attention_prior : copié au swap ou lu en direct en lecture seule

---

## 5. Évaluation (E3 cognitif + E4 ingénierie)

### 5.1 Métriques (8 au total, M3.b pivot)

| Code | Nom | Famille | Substrat cible |
|------|-----|---------|----------------|
| **M1.a** | Taux d'oubli | Apprentissage continu | kiki sur SplitNLP + mega-v2 |
| **M1.b** | Exactitude moyenne inter-tâches | Apprentissage continu | kiki |
| **M2.b** | Alignement RSA IRMf | Représentationnel | kiki + IRMf (repli Studyforrest ou labo) |
| **M3.a** | Ratio FLOPs rêve/éveil | Ingénierie | profil MLX kiki |
| **M3.b** ★ | Gain hors-ligne (wall-clock FLOPs-équivalent) | Ingénierie (pivot E3/E4) | kiki |
| **M3.c** | Énergie par épisode (proxy MLX) | Ingénierie | kiki (Loihi natif en cycle 2) |
| **M4.a** | Qualité de recombinaison (scorer enseignant gelé) | Émergence | kiki + Qwen3.5-9B Q4_K_M SHA épinglé |
| **M4.b** | Découverte de structure (test de permutation) | Émergence | kiki |

### 5.2 Matrice d'évaluation stratifiée

| Type de bump | Cellules obligatoires |
|--------------|-----------------------|
| **PATCH** | Métrique(s) de l'axe touché × P_equ × 1 seed |
| **MINOR** | 8 métriques × P_equ × 3 seeds |
| **MAJOR** | **Grille complète** : 8 × 3 profils × 3 seeds = 72 runs |
| **Changement EC** | Re-run des métriques publiées uniquement |

### 5.3 Reproductibilité bit-exact (8 métriques, contrat R1 étendu)

Tous les résultats sont bit-identiques pour le même `(c_version, profile, seed, run_id, commit_sha, benchmark_version)`. Stratégies par métrique : RNG seedé, mode déterministe MLX ou repli CPU, wall-clock FLOPs-équivalent (pas wall-clock réel), scorer enseignant Qwen3.5-9B Q4_K_M gelé par SHA256.

### 5.4 Hypothèses pré-enregistrées (OSF, S3)

- **H1** : P_equ améliore M1.a de ≥10 % vs baseline sans-rêve sur mega-v2
- **H2** : P_max marginal ≤5 % vs P_equ (rendements décroissants)
- **H3** : M2.b augmente monotoniquement P_min < P_equ < P_max
- **H4** : M3.c proxy ≤2× éveil pour P_equ (déploiement viable)

---

## 6. Stratégie de publication (S1-séquentielle stricte)

### 6.1 Article 1 — Nature Human Behaviour / PLoS Comp Bio

*"dreamOfkiki : A Substrate-Agnostic Formal Framework for Dream-Based Knowledge Consolidation in Artificial Cognitive Systems"*

- Contribution théorique (framework formel) + validation empirique (ablation kiki-oniric) + alignement cognitif (RSA IRMf)
- Principal : 8-10 pages ; Supplémentaire : 30-50 pages (preuves formelles, détails)
- Jalon : arXiv S18, soumission S20

### 6.2 Article 2 — NeurIPS / ICML / TMLR

*"Ablating Dream Channels : Engineering Trade-offs of Memory Consolidation in Large Language Models"*

- Contribution ingénierie (swap worktree, runtime asynchrone) + ablation empirique 3 profils + publication open-source
- Principal : 9 pages ; Supp : illimité
- Jalon : brouillon complet S24, soumission **après acceptation Article 1** (probablement 2027)

### 6.3 Paternité & affiliation

**Signataires** : « dreamOfkiki project contributors »

```
Clément Saillant¹ (correspondant),
[collaborateur labo IRMf]² (affiliation de courtoisie, si applicable),
et Collaborateurs IA³ (remerciés séparément)

¹ L'Electron Rare, France
² [Nom du labo], [Université]
³ Voir CONTRIBUTORS.md pour la liste complète
```

À vérifier avant S18 : directives de soumission Nature HB. Repli : auteur unique « Saillant, C. on behalf of dreamOfkiki project contributors ».

### 6.4 Engagements d'open science

| Artefact | Plateforme | Timing |
|----------|-----------|--------|
| Code | `github.com/electron-rare/dreamOfkiki` (MIT) | S1 |
| Modèles | `huggingface.co/clemsail/kiki-oniric-{P_min,P_equ,P_max}` | S22 |
| Dataset d'évaluation | banc de test retenu 500 items | S20 |
| Harness | `dreamOfkiki.harness` installable via pip | S22 |
| Tableau de bord | `dream.saillant.cc` public en lecture seule | S1 (croissant) |
| Pré-enregistrement | OSF | S3 |
| DOI artefacts | Zenodo | S20-S22 |
| Conformité FAIR | supp Article 1 | S18 |

---

## 7. Risques & portes go/no-go

### 7.1 Risques HIGH consolidés

| ID | Description | Atténuation |
|----|-------------|-------------|
| R-EXT-01 | Approche labo IRMf échoue | Repli Studyforrest pré-verrouillé S2 |
| R-CHA-01 | Charge cognitive > 15 h/sem | Cut-gate 2 sem, délégation agressive |
| R-FRM-01 | Preuve DR-2 échoue | Repli DR-2' ordre canonique |
| R-IMP-01 | Gardes de swap trop strictes | Seuils configurables, permissif au début |
| R-CAL-01 | Rejet chronique Article 1 | Repli PLoS CB / Cognitive Science |

### 7.2 Portes go/no-go

| Porte | S | Décision | Critère go |
|-------|---|----------|------------|
| G1 | S2 | Repli T-Col verrouillé | Adaptateur Studyforrest testé |
| G2 | S8 | P_min viable | Exactitude ≥ baseline − 2 %, runtime stable 48 h |
| G3 | S8 | Preuve DR-2 OK | Preuve complétée et peer-reviewée en interne |
| G4 | S12 | P_equ fonctionnel | > P_min sur ≥2 métriques significatives, invariants verts 7 j |
| G5 | S18 | PUBLICATION-READY | Tous critères Section 4.6 du spec framework |
| G6 | S28 | Amorcer cycle 2 | Article 1 soumis, bande passante dispo |

### 7.3 Pivots préparés

**Pivot A** (si G2 ou G4 échoue) : article unique TMLR/ICLR workshop, durée 4-5 mois, budget libéré → L'Electron Rare
**Pivot B** (si G3 échoue) : Article 1 semi-formel, cible Neural Computation / Cognitive Systems Research, calendrier S18 préservé

### 7.4 Critères de sortie cycle 1 (seuil 5/8 succès complet)

1. Article 1 soumis
2. Article 2 brouillon complet
3. Harness open-source publié (GitHub + DOI Zenodo)
4. 3 snapshots de modèles publics HuggingFace
5. Tableau de bord `dream.saillant.cc` en ligne
6. Pré-enregistrement OSF H1-H4 clos avec résultats
7. `CONTRIBUTORS.md` à jour
8. Zéro violation BLOCKING non résolue

---

## 8. Ressources & dépendances

### 8.1 Calcul

- **Studio M3 Ultra 512 Go** (MLX) : éveil primaire, peut absorber le rêve P_min
- **KXKM-AI RTX 4090 24 Go** : GPU rêve dédié P_equ/P_max
- **GrosMac M5 16 Go** : VAE de recombinaison P_max uniquement
- **Tower 31 Go** : services (miroir Grafana, Langfuse, Piper TTS non critique)

### 8.2 Données

- **mega-v2** : 498 723 exemples, 25 domaines (existant)
- **banc de test retenu** : 500 items gelés (à créer S1-S3)
- **IRMf Studyforrest** (repli) ou **données labo partenaire** (cible S8)

### 8.3 Dépendances externes

- Labo IRMf partenaire (Gallant UC Berkeley / Norman Princeton / Huth UT Austin) — approche T-Col S3-S4
- Approche Intel NRC cycle-2 (T-Col.5 S9+, non bloquant cycle 1)
- Pré-enregistrement OSF (S3)
- DOI Zenodo (S20-S22)
- Hébergement modèles HuggingFace (existant, compte `clemsail`)

### 8.4 Infrastructure existante réutilisée

- Plugins Claude Code (superpowers, OMC, oh-my-claude statusline)
- Sous-agents : `general-purpose`, `critic`, `validator`, `explore`, `planner`, `oh-my-claude:librarian`
- Baseline de consolidation OPLoRA (instance de B-Tononi dans C)
- Source `kiki-flow-core` (forkée en `kiki-oniric`)
- Grist + Keycloak (suivi des tâches optionnel)
- Tunnels autossh cross-machines

---

## 9. Relations avec les projets existants

| Projet | Relation | Impact |
|--------|----------|--------|
| `kiki-flow-research` (3 stories lancées 2026-04-16) | **Indépendant** — Q2.3 (c) : fork dédiée `kiki-oniric`, pas de blocage | Aucun sur kiki-flow-research |
| `kiki-flow-core` | Source de la fork, rebase jalonné (r3) S1/S8/S18 | Rebase consomme ~1 semaine chaque fois |
| `micro-kiki` (triple-hybride SNN+quantique+classique) | **Précurseur conceptuel** pour E cycle 2 | Acquis SNN réutilisés au cycle 2 |
| OPLoRA | Instance B-Tononi (opération downscale) dans framework C | Cité formellement dans framework §4.2 opérations |
| mascarade | Consommatrice potentielle cycle 3+ | Hors portée cycle 1 |
| Factory 4 Life | Consommatrice potentielle cycle 3+ | Hors portée cycle 1 |
| Lancement L'Electron Rare mai 2026 | **Coïncide S4-S8** cycle 1 | Budget d'attention 15 h/sem inclus |

---

## 10. Questions ouvertes & suites

Questions non bloquantes pour la phase writing-plans, à traiter en S1-S3 :

1. **Vérification des signataires** : les directives Nature HB autorisent-elles le style « project contributors » ? (T-Col vérifie S2)
2. **Format de banc de test standardisé** : la communauté apprentissage continu a-t-elle une convention pour l'ablation multi-profils à adopter ? (revue T-Ops S2)
3. **Validation du scorer enseignant** : Qwen3.5-9B Q4_K_M gelé par SHA produit-il des scores RSA-alignables ? (test A.5 S3)
4. **Labo IRMf préférentiel** : ordre de préférence à affiner avec critères (compatibilité des données, réponse reviewer potentielle, disponibilité co-auteur) (T-Col S3)
5. **Template de pré-enregistrement OSF** : template à adapter à l'hybride ML/cognitif plutôt que cognitif seul (S3)
6. **Outillage DualVer** : script de bump semi-automatique dans T-Ops.1 (S2-S3)

---

## 11. Annexes (placeholder — renvoi au spec framework)

Les annexes détaillées (invariants formels, axiomes DR, plan de tests, interfaces cross-track) sont dans le spec framework C : `2026-04-17-dreamofkiki-framework-C-design.md`.

---

**Fin du spec de conception maître.**

Étape suivante après relecture utilisateur : invoquer le skill `writing-plans` pour générer le plan d'implémentation.
