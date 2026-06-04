# Article 2 — Plan (post-pivot PLOS CB, 2026-05-10)

**Revue cible** : NeurIPS (principale) / ICML / TMLR (recours).
Cognitive Science conservée comme piste secondaire si l'angle
empirique fMRI domine.
**Format** : 9 pages corps + supplémentaire libre.
**Cible mots** : ~5500 mots corps.
**Justification du re-spec** : le plan cycle-2 d'origine (S27.1) est
caduc. Le pivot PLOS CB du 2026-04-20
(`docs/milestones/cycle3-plan-adaptation-2026-04-20.md`) a déplacé
le travail empirique multi-échelle, le cross-substrat Norse, la
fMRI Studyforrest et le verdict Gate D depuis la clôture du cycle-3
vers le backlog Article 2. Ce fichier réécrit le plan en
conséquence.

---

## 1. Résumé (cible 200 mots)

Contribution ingénierie-et-empirique : cadre substrate-agnostic de
consolidation onirique avec Critère de Conformité exécutable,
démontré sur **quatre substrats** (MLX `kiki_oniric`, E-SNN
thalamocortical, `micro_kiki` MoE-LoRA, Norse spiking) et **trois
échelles** (Qwen3.5-1.5B, Qwen3.5-7B, Qwen3.6-35B-A3B).
Contributions clés : (1) Protocol `SubstrateAdapter` unifiant des
ABI substrats divergents derrière un seul runtime ; (2) matrice de
validation Critère de Conformité cross-substrats ; (3) loi
d'échelle empirique cross-scale sur les hypothèses H1–H4 avec
contrôle Bonferroni ; (4) validation cross-modale contre la fMRI
BOLD Studyforrest via alignement état-onirique HMM + CCA. La
supervision ground-truth utilise le protocole pré-enregistré OSF du
cycle 1 (amendement 2026-04-23 couvrant l'extension multi-échelle).

## 2. Introduction (~1 page)

- Problème ingénierie : consolidation IA reproductible à travers
  substrats matériels **et** échelles de modèles
- Problème empirique : les effets de chaîne de profil de
  `kiki_oniric` survivent-ils à l'échelle, au substrat et à la
  validation cross-modale biologique ?
- Motivation pratique : permettre aux chercheurs de valider les
  affirmations du cadre sur leurs propres substrats sans
  re-dériver la théorie
- Feuille de route des contributions (4 items numérotés cf. §1)

## 3. Contexte (~0,75 page)

- Référencer l'Article 1 (soumission PLOS CB, framework C, axiomes
  DR-0..DR-4 avec amendement d'affaiblissement DR-2 du
  2026-04-21) comme fondation formelle. L'Article 2 cite l'Article
  1 par tag, jamais par contenu.
- Citer SHY (Tononi 2014), FEP (Friston 2010), CLS (McClelland
  1995), brain-inspired replay (van de Ven 2020).
- **Paragraphe « travaux concurrents »** (§3.x) : RecursiveMAS
  (Yang et al., arXiv:2604.25917, 2026-04-28) — dérivation
  indépendante d'une architecture à état latent partagé pour
  systèmes multi-composants hétérogènes. Citer comme évidence
  convergente ; différencier par substrat / gating /
  falsifiabilité. Voir `rebuttal-pocket-recursivemas.md` pour la
  position prête à l'emploi et le tableau de différenciation à
  trois axes.

## 4. Critère de Conformité en pratique (~1,5 page)

- 4.1 Rappel des trois conditions (typage de signature via
  Protocol `SubstrateAdapter` ; tests de propriétés axiomatiques
  DR-0..DR-4 ; invariants BLOQUANTS S1/S2/S3/famille-I
  exécutables au runtime).
- 4.2 Vérification de conformité du substrat MLX `kiki_oniric`.
- 4.3 Vérification de conformité du substrat E-SNN thalamocortical
  (G7 LOCKED).
- 4.4 Vérification de conformité du substrat `micro_kiki` (G8
  LOCKED ; handler `recombine` MoE-LoRA via TIES-merge).
- 4.5 Vérification de conformité du substrat Norse spiking
  (wrappers cycle-3 C3.11/C3.12 DONE).
- 4.6 Artefact de test réutilisable : `tests/conformance/` comme
  contrat ciblable par tout substrat tiers.

## 5. Architecture ingénierie (~1,5 page)

- 5.1 Protocol `SubstrateAdapter` unifiant quatre ABI divergentes
  derrière un point d'entrée unique `DreamRuntime.execute()`
  (commits `9852817` + `428d7a2`).
- 5.2 Opérations comme handlers composables (replay, downscale,
  restructure, recombine — implémentations spécifiques aux
  substrats partageant la signature canonique).
- 5.3 Protocole de swap avec garde S1 retained-eval, garde S2
  finitude, garde S3 topologie.
- 5.4 Worker onirique concurrent (asyncio réel, promotion
  cycle-2).
- 5.5 Registre des runs avec contrat R1 déterministe (préfixe
  SHA-256 32-hex, support multi-artefacts, enregistrement de
  hash par artefact via `register_output_hash`).

## 6. Méthodologie (~1 page)

- 6.1 Pré-enregistrement OSF (hypothèses H1-H4 cycle 1
  conservées ; amendements 2026-04-21 affaiblissement DR-2 +
  2026-04-23 extension multi-échelle).
- 6.2 Pipeline statistique (Welch / TOST / Jonckheere /
  t-une-échantillon avec Bonferroni ; pipeline de test conjoint
  cross-scale).
- 6.3 Benchmark stratifié retenu mega-v2 (500 items, SHA-256
  figé, commun aux échelles et substrats).
- 6.4 Protocole de mesure cross-substrat via
  `SubstrateAdapter.execute_profile`.
- 6.5 Protocole de mesure cross-scale (1.5B → 7B → 35B famille
  Qwen + ajustement de loi d'échelle cf.
  `scaling-law-analysis-2026-04-20.md`).
- 6.6 Validation cross-modale : loader BOLD Studyforrest (C3.15
  DONE), alignement état-onirique HMM (C3.16 DONE), CCA entre
  trajectoires de consolidation artificielles et biologiques.

## 7. Résultats (~2 pages)

- 7.1 Ablation substrat MLX `kiki_oniric` (Phase B 1.5B sanity GO
  3/3 depuis `pilot-cycle3-real-1p5b.json` + Phase B 7B et 35B
  depuis runs C3.8 multi-échelle).
- 7.2 Ablation E-SNN thalamocortical (données clôture cycle-2 G9).
- 7.3 Ablation substrat `micro_kiki` (données clôture G8 +
  validation handler recombine cf. `[C-v0.10.0+PARTIAL]`).
- 7.4 Pilote cross-substrat Norse spiking (réactivation C3.13 via
  `pilot_phase2b_neuromorph.py`, driver 624 LOC préservé avec
  en-tête deferred-note).
- 7.5 Matrice de cohérence cross-substrats (taille d'effet et
  direction de la chaîne de profils sur les quatre substrats).
- 7.6 Analyse de loi d'échelle cross-scale (ajustement
  taille-d'effet 1.5B → 7B → 35B, sorties H1–H4 corrigées
  Bonferroni).
- 7.7 Résultat cross-modal Studyforrest (rapport C3.18 G10c).
- 7.8 **Verdict Gate D** sur H1–H4 (la décision multi-échelle
  reportée que le plan cycle-3 a promise).
- 7.9 **Benchmark Q1 GammaThetaMultiplexer** (HardFlowProxyTask
  N=2, 5 graines {0,17,42,73,101}, t-test de Welch par paires
  GTM contre 3 baselines × 3 métriques = 9 comparaisons,
  Bonferroni α=0,05/9≈0,0056). **Verdict : `tied`** selon la
  pré-enregistration — GTM gagne 3/9 (mi_h contre RecursiveLink
  et MLP, round-trip contre CrossAttn), perd 5/9 (rtf et bw_eff
  contre les deux baselines à espace latent, bw_eff contre
  CrossAttn), 1 égalité (mi_h contre CrossAttn). Narratif
  reformulé : **évidence convergente** — GTM égale les baselines
  à espace latent en agrégat tout en préservant la plausibilité
  biologique du couplage de phase, avec une information mutuelle
  par unité de code supérieure (mi_h ≈1,20 contre ≈0,40 pour
  MLP/RecursiveLink). Selon la soft-gate Q2
  (`ge_3_FP_reformulate`, dream-of-kiki commit `6ff8320`),
  Paper 2 s'élargit au **Critère de Conformance C+** (étendu :
  invariants structurels + tests de propriétés d'axiomes
  spécifiques au substrat C2). L'ablation T14 partielle (sous-
  bandes γ-seul et θ-seul) reste dans la SEM du GTM complet,
  suggérant que la structure de multiplexage γ⊗θ n'est pas
  porteuse sur cette tâche. L'effet du schedule de plasticité sur
  les résultats est sous le seuil de bruit pour cette tâche et ce
  nombre de paramètres : la constellation représente 128 sur
  17 888 paramètres entraînables (0,7 %), donc moduler son gradient
  par un schedule cosinus ne produit pas de différence
  comportementale mesurable (Δ < 0,005 RTF, inférieur à l'écart-type
  par graine ; Δ observé GTM vs NoPlasticity = 0,0033 RTF ≈ 0,05
  seed-σ). C'est un **résultat négatif** cohérent avec le cadrage
  évidence-convergente : sur HardFlowProxyTask, l'ordonnancement
  de la plasticité **n'est pas porteur** — la structure de canal
  discret (alphabet PSK + quantification Gumbel) pilote l'avantage
  `mi_h`, pas la modulation temporelle de l'apprentissage de la
  constellation. Une fraction plus large du modèle devrait
  participer au schedule pour que l'ablation discrimine ; nous
  laissons cela comme observation méthodologique pour travaux
  futurs (e.g. embeddings appris par symbole, ou ablation
  `constellation_lock_after` avec constellation gelée en milieu
  d'entraînement). Figure :
  `nerve-wml v1.X.X papers/paper2/figures/multiplexer_benchmark.png`.
  Artefacts : `nerve-wml commit a6ddcba` (analyse.py + results.json
  + ablation_results.json + q1_verdict.json). Référence croisée
  milestone : `nerve-wml docs/milestones/q1-multiplexer-benchmark-2026-05-10.md`.

  **Robustesse cross-condition (extensions Q1+/Q1++, 2026-05-11).**
  Nous avons testé la robustesse du verdict `tied` selon deux axes :
  (i) mise à l'échelle des classes — ré-exécution du même balayage
  4 architectures × 5 graines sur HardFlowProxyTask N=16 (vs N=2 du
  §7.9 base) ; (ii) généralisation à la difficulté de tâche —
  ré-exécution sur la FlowProxyTask canonique 4-classes (régime
  linéairement séparable, plus facile). Les deux extensions donnent
  **le même verdict `tied`** (Q1+ à N=16 : 3W/5L/1T identique à N=2 ;
  Q1++ à FlowProxyTask : **3W/1L/2T sur 6 comparaisons effectives** —
  3 comparaisons `bw_eff` sont exclues comme statistiquement
  dégénérées car les quatre architectures s'effondrent à une
  bandwidth-efficiency constante par-graine sur le régime plus
  facile, laissant le t-test de Welch indéfini ; les moyennes
  descriptives GTM=0,125 vs baselines=0,1875 sont rapportées mais
  pas testées en significativité). Le cadrage évidence-convergente
  généralise : GTM domine systématiquement `mi_h` (information par
  unité de code) contre MLP et RecursiveLink dans les trois
  conditions, et sous-performe systématiquement sur `bw_eff` (rang
  effectif du code) à cause de l'effondrement des clusters PSK
  dans le bottleneck discret — une signature mécanique de la
  quantization, pas un artefact spécifique à la tâche (en Q1++
  cette signature est à valeur constante donc non testable, d'où
  l'exclusion des métriques dégénérées). Le corollaire de
  N-invariance (l'avantage PAC est piloté par la propriété de la
  tâche, pas par l'échelle) est donc empiriquement supporté. Le
  dénominateur Bonferroni est fixé au nombre **effectif** de
  comparaisons après exclusion dégénérée (Q1/Q1+ : α=0,05/9≈0,0056 ;
  Q1++ : α=0,05/6≈0,0083). Les tailles d'effet de Cohen d sont
  rapportées par test non dégénéré dans les JSONs de verdict.
  Artefacts de reproduction :
  `nerve-wml/experiments/benchmark_multiplexer_vs_baselines/results_n16.json`
  et `results_q1plusplus.json` ; figures
  `multiplexer_benchmark_n16.png`, `multiplexer_benchmark_q1plusplus.png` ;
  verdicts par condition `q1_verdict_n16.json`,
  `q1_verdict_q1plusplus.json` (portant `degenerate_metrics`,
  `n_effective`, et `cohens_d` par test).

## 8. Discussion (~1 page)

- 8.1 Reproductibilité validée à travers quatre substrats et
  trois échelles.
- 8.2 Compromis ingénierie (vitesse MLX vs énergie E-SNN vs
  fidélité biologique Norse vs déployabilité MoE micro_kiki).
- 8.3 Interprétation du résultat cross-modal Studyforrest (la
  consolidation artificielle s'aligne avec quels clusters HMM
  d'état-onirique BOLD ?).
- 8.4 Comparaison avec les claims théoriques de l'Article 1
  (citer l'Article 1 par tag publié une fois l'arXiv ID en main).
- 8.5 Position vs RecursiveMAS — convergence sur l'intuition
  d'état latent partagé, divergence sur substrat / gating /
  falsifiabilité.
- 8.6 Limites (transformer-only ; pas de régime continual
  on-line ; Studyforrest est un dataset unique).

## 9. Travaux futurs (~0,5 page)

- 9.1 Substrats additionnels (vrai RWKV, state-space S6,
  Foundation Models on-device).
- 9.2 Sélection dynamique de profil au runtime guidée par le
  signal HMM cross-modal.
- 9.3 Régime continual-learning (le pipeline actuel est
  batch offline).
- 9.4 Validation Studyforrest multi-sujets (le plan C3.18 actuel
  cible le sous-ensemble BOLD public).

## 10. Références

Hérite de `references.bib` de l'Article 1 plus :
- `recursivemas2026` (travaux concurrents, ajouté 2026-05-10)
- `nerve-wml` v1.8.0 Zenodo version-DOI (mintage en cours)
- Citation du framework Norse spiking
- Citation du dataset Studyforrest
- Citations méthodologiques HMM + CCA
- Citation LoRA-stack micro-kiki (tag projet parent)

---

## Différenciation par rapport à l'Article 1

| Aspect | Article 1 (PLOS CB) | Article 2 (Eng+Empirique) |
|--------|--------------------|---------------------------|
| Périmètre | Cadre formel C | Réalisation ingénierie+empirique |
| Revue cible | PLOS Computational Biology | NeurIPS / ICML / TMLR |
| Audience | Chercheurs en sciences cognitives + théoriciens | Ingénieurs ML + chercheurs systèmes + neuroscientifiques computationnels |
| Substrats couverts | 1 (kiki-oniric MLX, esquissé) | 4 (MLX + E-SNN + micro_kiki + Norse) |
| Échelles de modèle | 1 (1.5B sanity uniquement) | 3 (1.5B + 7B + 35B famille Qwen) |
| Longueur | ~5000 mots | ~5500 mots |
| Preuves formelles | DR-0..DR-4 dans le corps, DR-2 amendé | Référence Article 1 par tag |
| Focus reproductibilité | Pré-enregistrement + contrat R1 | Matrice de validation cross-substrat × cross-scale |
| Validation cross-modale | Hors périmètre | Dans le périmètre via Studyforrest BOLD + HMM + CCA |
| Emphase open-source | Conceptuelle | Opérationnelle (Protocol `SubstrateAdapter` + suite Conformance 4-substrats) |

---

## Backlog reporté de l'Article 2 (source : adaptation plan cycle-3)

Selon `docs/milestones/cycle3-plan-adaptation-2026-04-20.md`, six
items déplacés depuis la clôture du cycle-3 vers l'Article 2 :

1. **C3.8** — ablation multi-échelle complète 1.5B + 7B + 35B
   (PARTIEL : 1.5B fait comme Phase B, commit `22c58c9`).
2. **C3.9** — verdict Gate D (multi-échelle H1–H4).
3. **C3.13** — pilote cross-substrat Norse vs MLX (driver
   `scripts/pilot_phase2b_neuromorph.py`, 624 LOC).
4. **C3.14** — milestone G10a (ordering cross-substrat).
5. **C3.18** — rapport fMRI G10c (pipeline Studyforrest prêt).
6. **C3.19–C3.22** — rédaction du récit Article 2 (ce fichier +
   sections).

La fenêtre de re-spec s'ouvre **après** que l'Article 1 v0.2
décroche son arXiv ID. Le tag `paper-1-v0.2-arxiv` existe déjà
côté distant — la fenêtre est donc conceptuellement ouverte au
2026-05-10.

---

## Calendrier de réactivation (estimation, post submission Article 1)

À compter du jour où la soumission Article 1 v0.2 est déposée :

- **Semaine 1** : finalisation paper2/outline.md + abstract.md +
  réconciliation miroir paper2-fr/.
- **Semaines 2-4** : runs C3.8 multi-échelle 7B + 35B sur
  MacStudio (budget wallclock : vérifier vs allocation Studio
  actuelle pour l'inférence eu-kiki + Zacus).
- **Semaine 5** : rapport verdict Gate D C3.9.
- **Semaine 6** : réactivation + run pilote Norse C3.13.
- **Semaines 7-8** : run + rapport fMRI Studyforrest C3.18 G10c.
- **Semaines 9-12** : rédaction sections 4 → 8 résultats en main.
- **Semaine 13** : revue interne pré-soumission (agent critic sur
  le full draft cf. `feedback_critic_before_ship.md`).
- **Semaines 14-15** : fenêtre de soumission (NeurIPS / ICML /
  TMLR selon calendrier).

---

## Dépendances transversales

- **Tag Article 1 v0.2** doit être sur arXiv avec un ID stable
  avant que l'Article 2 le cite (cf. règle de pinning de
  `docs/papers/CLAUDE.md` — pas de `HEAD`, uniquement des tags).
- **Version-DOI Zenodo `nerve-wml v1.8.0`** doit être minté avant
  que l'Article 2 cite la mesure cross-substrat (actuellement en
  cours ; concept-DOI 10.5281/zenodo.19656342 en repli).
- **Benchmark `bouba_sens`** : si le récit cross-modal du §6.6
  référence la courbe dose-réponse Amedi B-1, citer par
  version-DOI Zenodo `bouba_sens` (mintage planifié en parallèle
  de la soumission TMLR).
- **Amendement OSF** pour l'extension multi-échelle cycle-3 déjà
  déposé sous `osf-amendment-bonferroni-cycle3.md` (correction
  Bonferroni adaptée au test conjoint multi-échelle).

---

## Notes

- Ce plan supersede la version cycle-2 amorçage (S27.1).
  L'original est préservé dans l'historique git au commit
  précédent de ce fichier.
- Plan sujet à révision selon les retours reviewer Article 1
  v0.2 (PLOS CB) et l'issue du verdict Gate D.
- L'Article 3 (bandeau provisional cf. `docs/papers/CLAUDE.md`)
  est intentionnellement hors périmètre ici ; ne pas étoffer
  avant kickoff cycle-3 et clôture Article 2.
- Propagation EN→FR : tout changement à ce fichier requiert une
  mise à jour correspondante de `docs/papers/paper2/outline.md`
  dans la même PR (cf. `CONTRIBUTING.md`).
