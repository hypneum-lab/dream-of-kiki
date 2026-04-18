# Paper 1 — Plan (brouillon cycle 1)

**Revue cible** : Nature Human Behaviour (primaire)
**Format** : 8-10 pages corps principal + 30-50 pages supplément
**Objectif de mots** : ~5000 mots corps principal + supplément libre

---

## 1. Résumé (250 mots)

→ `abstract.md`

Structure :
- Phrase 1 : contexte / problème ouvert (oubli catastrophique,
  lacune de la consolidation mnésique en IA)
- Phrase 2 : annonce de contribution (framework formel avec
  axiomes exécutables DR-0..DR-4 + Critère de Conformité)
- Phrase 3 : synthèse de la validation du pipeline (ablation de
  substitution synthétique, pipeline statistique de bout en bout ;
  chiffres empiriques rapportés dans le Paper 2)
- Phrase 4 : portée (indépendance du substrat, science ouverte
  reproductible, pré-enregistré sur OSF)
- 1-2 phrases : résultat clé + pointeur vers les travaux futurs
  (E-SNN cycle 2)

---

## 2. Introduction (~1,5 pages)

→ `introduction.md`

Sous-sections :
- 2.1 L'oubli catastrophique et la lacune de consolidation
  mnésique en IA
- 2.2 Consolidation inspirée du sommeil : état de l'art (van de Ven
  2020, Kirkpatrick 2017 EWC, Tononi SHY, Friston FEP, Hobson/Solms)
- 2.3 Les quatre piliers (A consolidation, B homéostasie, C
  créatif, D prédictif) et pourquoi il manque un framework formel
  unifié
- 2.4 Feuille de route des contributions : framework C-v0.5.0+STABLE
  avec axiomes exécutables + Critère de Conformité ; implémentation
  de référence rapportée dans le Paper 2

---

## 3. Contexte théorique — quatre piliers

Sous-sections :
- 3.1 Pilier A : consolidation mnésique Walker/Stickgold
- 3.2 Pilier B : homéostasie synaptique SHY de Tononi
- 3.3 Pilier C : rêve créatif Hobson/Solms
- 3.4 Pilier D : Principe d'Énergie Libre de Friston
- 3.5 La lacune compositionnelle : pourquoi une intégration
  fragmentaire échoue

---

## 4. Framework C-v0.5.0+STABLE

Sous-sections :
- 4.1 Primitives : 8 Protocoles typés (α, β, γ, δ, canaux 1-4).
  Chaque opération doit énumérer sa **référence de garde**
  (replay → S1, downscale → S2, restructure → S3, recombine → I3
  WARN) et se lier à un test de conformité sous
  `tests/conformance/`. Règle : **ne jamais ajouter une opération
  sans référence de garde accompagnatrice et sans test de
  conformité.**
- 4.2 Profils : P_min, P_equ, P_max avec inclusion en chaîne DR-4
- 4.3 Ontologie du Dream Episode (quintuplet)
- 4.4 Opérations : replay, downscale, restructure, recombine
- 4.5 Axiomes DR-0..DR-4
  - 4.5.1 DR-0 Redevabilité
  - 4.5.2 DR-1 Conservation épisodique
  - 4.5.3 DR-2 Compositionnalité (semi-groupe libre) — preuve
    différée (voir brouillon `docs/proofs/dr2-compositionality.md`)
  - 4.5.4 DR-3 Indépendance du substrat + Critère de Conformité
  - 4.5.5 DR-4 Inclusion en chaîne des profils
- 4.6 Invariants I/S/K avec matrice d'application
- 4.7 Versionnage DualVer formel+empirique
  (actuel : C-v0.5.0+STABLE selon CHANGELOG.md)

---

## 5. Approche de validation du Critère de Conformité

Indépendant du substrat. Les détails d'implémentation spécifiques
(MLX, runtime, pipeline d'ablation) résident dans le Paper 2.

Sous-sections :
- 5.1 Graphe de compilation déterministe
- 5.2 Ordonnanceur monothread avec registre de handlers
  (garantie de journalisation DR-0)
- 5.3 Basculement atomique avec gardes d'invariants (S1 + S2 + S3 + I3)
- 5.4 Inclusion en chaîne des profils DR-4
- 5.5 Pointeur vers le Paper 2 pour une instanciation empirique

---

## 6. Méthodologie

Sous-sections :
- 6.1 Hypothèses H1-H4 (DOI OSF du pré-enregistrement cité).
  H1-H4 cartographiées vers les identifiants d'axiomes canoniques :
  H1 → DR-1 (la conservation épisodique réduit l'oubli), H2 → DR-4
  (équivalence P_max ⊆ P_equ à la frontière), H3 → DR-2 (monotonie
  compositionnelle), H4 → K1 (conformité au budget).
- 6.2 Tests statistiques : Welch (H1), TOST (H2), Jonckheere (H3),
  t à un échantillon (H4) ; Bonferroni α = 0,0125
- 6.3 Banc de test : mega-v2 (498 k exemples, 25 domaines, sous-
  ensemble stratifié retenu de 500 items avec intégrité SHA-256 ;
  le drapeau `is_synthetic=true` est positionné par l'adaptateur
  lorsque le chemin réel mega-v2 est indisponible, afin que les
  outils en aval puissent filtrer les run_ids enregistrés par
  source). Les rapports du cycle 1 utilisent le run_id de
  substitution synthétique `syn_s15_3_g4_synthetic_pipeline_v1` ;
  le dump sous `docs/milestones/ablation-results.json` enregistre
  la source comme `synthetic-placeholder`.
- 6.4 RSA IRMf : Studyforrest (Branche A verrouillée G1,
  parcellation FreeSurfer STG/IFG/AG ; les rapports du cycle 1 ne
  font état que de la validation d'infrastructure — substitution
  synthétique, RSA réelle différée au cycle 2 avec un run_id
  fraîchement enregistré)
- 6.5 Reproductibilité : contrat R1 (run_id déterministe),
  épinglage d'artefact par DOI Zenodo

---

## 7. Résultats (substitution synthétique, pilote G2/G4)

→ `results-section.md` (brouillon S18.2)

Chaque entrée empirique ci-dessous cite le run_id enregistré dans
`harness.storage.run_registry.RunRegistry` et le fichier de dump
sous `docs/milestones/`. Les substitutions synthétiques sont
explicitement signalées par `(substitution synthétique, pilote
G2)` / `(substitution synthétique, pilote G4)` selon la directive
projet.

Sous-sections :
- 7.1 Viabilité de P_min (G2, run_id `syn_g2_pmin_pipeline_v1`,
  dump `docs/milestones/g2-pmin-report.md`)
- 7.2 Ablation fonctionnelle de P_equ (G4, run_id
  `syn_s15_3_g4_synthetic_pipeline_v1`, dump
  `docs/milestones/ablation-results.json`)
- 7.3 H1 réduction de l'oubli (substitution synthétique, pilote G4)
- 7.4 H3 alignement représentationnel monotone (substitution
  synthétique, pilote G4)
- 7.5 H4 conformité au budget énergétique (substitution
  synthétique, pilote G4)
- 7.6 H2 équivalence P_max (différé au cycle 2, test de fumée
  partiel)

---

## 8. Discussion

Sous-sections :
- 8.1 Contribution théorique : premier framework formel exécutable
  pour la consolidation mnésique basée sur le rêve. Chaque assertion
  cite les identifiants d'axiomes pertinents (DR-0..DR-4) et
  d'invariants (I1, I2, I3, S1, S2, S3, K1) depuis
  `docs/invariants/registry.md` et les sections de spécification
  dans `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`.
- 8.2 Contribution empirique : preuves de validation du pipeline
  (substitution synthétique, pilote G2/G4 ; preuves sur données
  réelles rapportées dans le Paper 2 via les références croisées
  axiomatiques DR-1/DR-2/DR-4 et les run_ids enregistrés)
- 8.3 Limites : précautions liées aux données synthétiques,
  validation sur substrat unique au cycle 1, P_max en squelette
  uniquement
- 8.4 Comparaison avec l'état de l'art (van de Ven, EWC, etc.)

---

## 9. Travaux futurs — Cycle 2

Sous-sections :
- 9.1 Substrat E-SNN (thalamocortical Loihi-2, différé du cycle 1
  SCOPE-DOWN)
- 9.2 Câblage complet de P_max + flux α + canal-4 ATTENTION_PRIOR
- 9.3 Partenariat réel avec un laboratoire IRMf (campagne de
  recrutement de relecteurs T-Col, S3-S5)
- 9.4 Validation multi-substrat du Critère de Conformité

---

## 10. Références (placeholder)

Citations clés à intégrer :
- Walker MP, Stickgold R (2004). Apprentissage dépendant du sommeil.
- Kirkpatrick J et al (2017). EWC.
- Tononi G, Cirelli C (2014). SHY.
- Friston K (2010). FEP.
- Hobson JA (2009). Rêve en REM.
- van de Ven GM, Tuytelaars T, Tolias AS (2020). Replay inspiré du
  cerveau.
- McClelland JL et al (1995). Systèmes d'apprentissage
  complémentaires.

---

## Artefacts

- Code, modèles, tableaux de bord, épinglage de jeux de données :
  rapportés dans le Paper 2 (spécifiques au substrat). Le Paper 1
  ne livre que les artefacts formels :
- Pré-enregistrement : DOI OSF (à déterminer après dépôt)
- Preuves : `docs/proofs/` (DR-2, DR-4, op-pair-analysis,
  pivot-b-decision)
- Contrat de run-registry : `harness/storage/run_registry.py`
  (R1 ; version de framework C-v0.5.0+STABLE)
