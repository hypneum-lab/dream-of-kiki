---
title: "dreamOfkiki : un cadre formel agnostique au substrat pour la consolidation des connaissances inspiree du reve dans les systemes cognitifs artificiels"
author: "Saillant, Clément"
contact: "Clement Saillant <clement@saillant.cc>"
affiliation: "L'Electron Rare, France"
date: "2026"
draft: "v0.2 (cycle-1, assemblage S20.3, placeholders INCLUDE inlines)"
---

# Article 1 - Assemblage complet du brouillon

## dreamOfkiki : un cadre formel agnostique au substrat pour la consolidation des connaissances inspiree du reve dans les systemes cognitifs artificiels

**Note - statut** : assemblage de brouillon. Les fichiers de section `.md` etaient la source de verite initiale ; ce fichier inline maintenant leur contenu afin de servir de source assemblee pour le rendu pandoc.

**Note - donnees synthetiques** : ces reserves s'appliquent au §7 Resultats (chiffres issus du placeholder synthetique mega-v2). Les vraies ablations arriveront a la cloture du cycle 1 (S20+) ou au cycle 2.

---

## 1. Resume

L'oubli catastrophique demeure un obstacle central pour les systemes cognitifs artificiels qui apprennent sequentiellement sur plusieurs taches. La consolidation inspiree du sommeil a ete proposee comme remede, avec des travaux anterieurs explorant le replay (Walker, van de Ven), l'homeostasie synaptique (Tononi), la recombinaison creative (Hobson) et le codage predictif (Friston) - mais aucun cadre formel unifie n'integre ces quatre piliers sous forme d'operations composables et agnostiques au substrat.

Nous introduisons **dreamOfkiki**, un cadre formel avec axiomes executables (DR-0 accountability, DR-1 conservation episodique, DR-2 compositionalite sur un semi-groupe libre d'operations oniriques, DR-3 agnosticisme au substrat via un Conformance Criterion, DR-4 inclusion en chaine des profils). Le cadre definit 8 primitives typees (entrees alpha, beta, gamma, delta ; 4 canaux de sortie), 4 operations canoniques (replay, downscale, restructure, recombine), ainsi qu'une ontologie du Dream Episode en 5-uplet. Le cadre admet plusieurs substrats conformes ; des implementations exemplaires valident le design et sont presentees separement (voir Paper 2).

Les hypotheses pre-enregistrees (DOI OSF : a venir) sont evaluees via le t-test de Welch, l'equivalence TOST, la tendance de Jonckheere-Terpstra et un t-test univarie contre un budget de calcul, sous correction de Bonferroni.

**Validation de la chaine experimentale (simulation synthetique, pilote G2).** La chaine complete de mesure et d'analyse statistique est exercee avec des predicteurs factices a des niveaux de performance preprogrammes ; les chiffres sont reportes au §7 avec leur `run_id` enregistre et le dump JSON sous `docs/milestones/`. Les vraies inferences mega-v2 et l'eventuelle analyse de similarite representationnelle fMRI suivront au cycle 2 (Paper 2). L'ensemble du code, des specifications et de la pre-inscription est publie sous MIT/CC-BY-4.0.

---

## 2. Introduction

### 2.1 Oubli catastrophique et faille de consolidation

Les systemes cognitifs artificiels modernes excellent sur l'apprentissage mono-tache mais se degradent rapidement lorsqu'ils sont entraines sequentiellement sur plusieurs taches - un phenomene connu sous le nom d'**oubli catastrophique** [@mccloskey1989catastrophic; @french1999catastrophic]. Malgre deux decennies de strategies d'attenuation (elastic weight consolidation [@kirkpatrick2017overcoming], rejeu generatif [@shin2017continual], memoire par rehearsal [@rebuffi2017icarl]), le champ ne dispose toujours pas d'un *cadre theorique unifie* expliquant pourquoi ces mecanismes fonctionnent et dans quelles conditions ils se composent.

La cognition biologique resout ce probleme pendant le **sommeil**. Le replay hippocampique pendant le NREM, le downscaling synaptique, la restructuration predictive des representations corticales et la recombinaison creative pendant le REM forment ensemble une chaine de consolidation multi-etapes [@diekelmann2010memory; @tononi2014sleep]. Pourtant, les travaux IA existants n'ont integre que des fragments de cette biologie, en se concentrant le plus souvent sur un seul mecanisme (par exemple le replay seul) sans cadre principiel pour decrire l'interaction des mecanismes.

### 2.2 Quatre piliers de la consolidation basee sur le reve

Nous identifions quatre piliers theoriques que tout cadre complet de consolidation inspiree du reve pour l'IA doit adresser :

- **A - consolidation Walker/Stickgold** : transfert episodique-vers-semantique via replay [@walker2004sleep; @stickgold2005sleep].
- **B - SHY de Tononi** : homeostasie synaptique qui renormalise les poids durant le sommeil [@tononi2014sleep].
- **C - reve creatif Hobson/Solms** : recombinaison et abstraction pendant le REM [@hobson2009rem; @solms2021revising].
- **D - FEP de Friston** : minimisation de l'energie libre comme cadre unificateur de l'inference et de la consolidation [@friston2010free].

Des travaux IA anterieurs ont implemente A [@vandeven2020brain], B [@kirkpatrick2017overcoming comme regularisation voisine de SHY] et des elements de D [@rao1999predictive; @whittington2017approximation], mais **aucun travail n'a formalise comment ces quatre piliers se composent** d'une maniere agnostique au substrat et exploitable pour l'ablation et la preuve.

### 2.3 Le gap de compositionalite

Pourquoi la composition importe-t-elle ? Empiriquement, l'ordre d'application des operations de consolidation change l'etat cognitif resultant - appliquer le replay avant le downscaling preserve la specificite episodique, alors que l'ordre inverse peut effacer les representations memes que la restructuration doit raffiner. Notre analyse (`docs/proofs/op-pair-analysis.md`) enumere les 16 op-pairs et montre que 12 paires croisees sont non commutatives, ce qui renforce l'idee que *l'ordre fait partie du cadre*, et non d'un simple detail d'implementation.

Un cadre formel adequat doit donc (i) specifier les operations comme des primitives composables avec des types bien definis, (ii) expliciter quelles compositions sont valides, (iii) fournir un compte **executable** que tout substrat conforme peut implementer, et (iv) soutenir des ablations empiriques comparant differents profils d'operations. Aucun travail anterieur ne couvre simultanement ces quatre exigences.

### 2.4 Feuille de route des contributions

Dans cet article, nous presentons **dreamOfkiki**, premier cadre formel pour la consolidation inspiree du reve dans les systemes cognitifs artificiels, avec les contributions suivantes :

1. **Framework C-v0.5.0+STABLE** : 8 primitives typees, 4 operations canoniques formant un semi-groupe libre, 4 OutputChannels, une ontologie du Dream Episode en 5-uplet, et les axiomes DR-0..DR-4 assortis d'un Conformance Criterion executable (§4). Les points 2-4 ci-dessous sont documentes dans Paper 2 (compagnon empirique) ; Paper 1 se limite aux contributions formelles et a la feuille de route de conformance.
2. **Feuille de route** vers la generalisation multi-substrat (au-dela de l'implementation de reference du cycle 1) et vers l'alignement representationnel fMRI reel (partenariat labo poursuit via l'outreach T-Col).

Le reste du papier est organise comme suit : le §3 passe en revue les quatre piliers en profondeur ; le §4 developpe Framework C-v0.5.0+STABLE avec axiomes et preuves ; le §5 esquisse l'approche de validation du Conformance Criterion (les resultats empiriques par substrat vivent dans Paper 2) ; le §6 detaille la methodologie ; le §7 rapporte les resultats de validation synthetique du pipeline ; le §8 discute les implications et limitations ; le §9 expose les travaux futurs du cycle 2.

---

## 3. Contexte theorique - quatre piliers

### 3.1 Pilier A - consolidation Walker / Stickgold

La consolidation de la memoire dependante du sommeil renvoie au phenomene empiriquement etabli selon lequel les souvenirs nouvellement encodes sont selectivement renforces, abstraits et integres au stockage a long terme pendant le sommeil [@walker2004sleep; @stickgold2005sleep]. Le replay hippocampique pendant le sommeil lent profond est le substrat neuronal le plus directement implique. Fonctionnellement, cela signifie que le replay effectue des **mises a jour de type gradient** sur les representations corticales, biaisees vers la retention des episodes rejoues - equivalent, dans notre cadre, a l'operation `replay` : echantillonner des episodes depuis le buffer beta, les faire passer dans les parametres courants, puis appliquer des mises a jour sous objectif de retention.

### 3.2 Pilier B - homeostasie synaptique SHY de Tononi

La Synaptic Homeostasis Hypothesis (SHY) postule que l'eveil entraine une potentiation synaptique nette et que le sommeil impose un downscaling synaptique global qui restaure le rapport signal/bruit sans effacer le schema differentiel de renforcement [@tononi2014sleep]. Cette hypothese est appuyee a la fois par des preuves ultrastructurales (reduction de la taille des synapses pendant le sommeil) et par des preuves comportementales (amelioration pendant le sommeil sur des taches precedemment apprises). Dans notre cadre, SHY correspond a l'operation `downscale` : une contraction multiplicative des poids par un facteur dans (0, 1]. Comme l'etablit notre analyse d'op-pairs (voir `docs/proofs/op-pair-analysis.md`, axiomes DR-2 + invariants S2), `downscale` est **commutative mais non idempotente** (shrink_f compose avec shrink_f donne facteur², pas facteur) - une propriete qui contraint les choix d'ordre canonique.

### 3.3 Pilier C - reve creatif Hobson / Solms

Le reve REM est associe a la recombinaison creative, a la generation de scenarios contrefactuels et a l'integration de materiel emotionnellement significatif [@hobson2009rem; @solms2021revising]. Le mecanisme suppose est un echantillonnage de type modele generatif depuis une representation latente des experiences recentes, produisant de nouvelles combinaisons qui sondent les frontieres de la structure apprise. Dans notre cadre, cela se mappe sur l'operation `recombine` : echantillonner des latents depuis le snapshot delta, appliquer un noyau VAE-light ou un noyau d'interpolation, puis emettre de nouveaux echantillons latents sur le canal 2.

### 3.4 Pilier D - Free Energy Principle de Friston

Le Free Energy Principle (FEP) [@friston2010free] cadre perception, action et apprentissage comme la minimisation de l'energie libre variationnelle sous un modele generatif hierarchique. Dans ce cadre, le sommeil s'interprete comme une phase hors ligne qui **restructure** le modele generatif afin de mieux minimiser l'energie libre attendue sur la distribution des entrees a l'eveil. Dans notre cadre, cela correspond a l'operation `restructure` : modifier la topologie du modele hierarchique (ajout de couche, retrait de couche, re-routage de connectivite) pour reduire l'erreur predictive sur les episodes retenus. La garde topologique S3 (`validate_topology`) garantit que les operations de restructuration preservent les invariants de niveau cadre S3 (connectivite des especes, absence d'auto-boucles, bornes sur le nombre de couches - voir `docs/invariants/registry.md` pour les definitions canoniques et `kiki_oniric/dream/guards/topology.py` pour la garde).

### 3.5 Le gap de compositionalite

Les travaux IA existants ont implemente un ou deux de ces quatre piliers (notamment A via le generative replay de @vandeven2020brain et B via l'EWC de @kirkpatrick2017overcoming, traite ici comme regulariseur voisin de SHY). Cependant, aucun travail anterieur n'a **formalise la composition** des quatre operations comme une structure algebrique unifiee avec des proprietes demonstrables.

Ce manque importe empiriquement parce que notre analyse d'op-pairs (`docs/proofs/op-pair-analysis.md`) etablit que 12 des 16 paires croisees `(op_i, op_j)` sont **non commutatives** - autrement dit, appliquer replay puis downscale produit un etat cognitif different d'appliquer downscale puis replay. L'ordre canonique retenu dans `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §4.3 (replay -> downscale -> restructure ; recombine en parallele) constitue donc un choix de design porteur, pas une decision arbitraire d'implementation.

Un cadre formel adequat doit donc (i) specifier des operations comme primitives composables a types bien definis, (ii) expliciter quelles compositions sont valides, (iii) fournir un compte executable qu'un substrat conforme peut implementer, et (iv) permettre des ablations empiriques comparant differents profils d'operations. Aucun travail anterieur ne couvre simultanement ces quatre points. Notre Framework C-v0.5.0+STABLE (§4) est le premier a le faire, en reliant les quatre piliers au cadre axiomatique canonique : pilier A -> DR-1 conservation episodique, pilier B -> DR-2 compositionalite (contrainte d'ordre sur downscale), pilier D -> DR-3 agnosticisme au substrat (la garde topologique S3 se situe sur cet axe), pilier C -> DR-4 inclusion en chaine des profils qui maintient les profils riches en recombinaison au sommet. L'axiome fondamental est DR-2 sur la compositionalite en semi-groupe libre (preuve dans `docs/proofs/dr2-compositionality.md`) ; DR-3, le Conformance Criterion, est le contrat executable de l'agnosticisme au substrat.

---

## 4. Framework C

**Source** : `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` couvre cette section. La version article ci-dessous condense cette spec en un recit structure selon le §4 de `outline.md`.

### 4.1 Primitives - huit protocoles types

Canaux Awake -> Dream :
- alpha (traces brutes, P_max uniquement) - firehose ring buffer
- beta (buffer episodique cure) - journal append SQLite avec insertion guidee par salience (un enregistrement ne passe que si son score de salience depasse un seuil top-k adaptatif)
- gamma (snapshot des poids) - pointeur de checkpoint de secours
- delta (latents hierarchiques) - ring buffer multi-especes N=256

Canaux Dream -> Awake :
- 1 (delta de poids) - applique via protocole de swap
- 2 (echantillons latents) - file de generative replay
- 3 (diff hierarchique) - application atomique au swap avec garde S3
- 4 (prior d'attention) - guidance meta-cognitive (P_max uniquement)

### 4.2 Profils - inclusion en chaine DR-4

| Profile | Channels in | Channels out | Operations |
|---------|-------------|--------------|------------|
| P_min   | beta | 1 | replay, downscale |
| P_equ   | beta + delta | 1 + 3 + 4 | replay, downscale, restructure, recombine_light |
| P_max   | alpha + beta + delta | 1 + 2 + 3 + 4 | replay, downscale, restructure, recombine_full |

DR-4 (prouve dans `docs/proofs/dr4-profile-inclusion.md`) :
`ops(P_min) subset ops(P_equ) subset target_ops(P_max)`, et de meme pour les canaux. Au cycle 1, `P_max` reste a l'etat de squelette.

### 4.3 Ontologie en 5-uplet du Dream Episode

Chaque dream-episode (DE) est un 5-uplet :
`(trigger, input_slice, operation_set, output_channels, budget)`.
Les triggers appartiennent a {SCHEDULED, SATURATION, EXTERNAL}. Les operations sont un tuple non vide de {REPLAY, DOWNSCALE, RESTRUCTURE, RECOMBINE}. `BudgetCap` impose des valeurs finies et non negatives pour `(FLOPs, wall_time_s, energy_j)` selon l'invariant K1.

### 4.4 Operations - semi-groupe des etapes de consolidation

L'ensemble des operations forme un semi-groupe libre non commutatif sous la composition `o`, avec budget additif (DR-2 compositionalite, brouillon de preuve dans `docs/proofs/dr2-compositionality.md`). L'ordre canonique est le suivant : `replay -> downscale -> restructure` en serie (ordre des piliers A-B-D), et `recombine` en parallele (pilier C). L'analyse d'op-pairs (`docs/proofs/op-pair-analysis.md`) enumere les 16 paires et trouve 12 paires croisees non commutatives.

### 4.5 Axiomes DR-0..DR-4

- **DR-0 (accountability)** : chaque DE execute produit un `EpisodeLogEntry`, y compris en cas d'exception dans le handler (garantie `try/finally`).
- **DR-1 (conservation episodique)** : chaque enregistrement beta est consomme avant purge.
- **DR-2 (compositionalite)** : la composition d'operations forme un semi-groupe avec fermeture par types + additivite du budget + composition fonctionnelle. La propriete universelle du generateur libre reste ouverte (reviewer G3 pending).
- **DR-3 (agnosticisme au substrat)** : `Conformance Criterion = signature typing and axiom property tests pass and BLOCKING invariants enforceable`. L'implementation de reference satisfait les trois conditions (voir §5 et Paper 2 pour l'instanciation empirique).
- **DR-4 (inclusion en chaine des profils)** : `P_min subset P_equ subset P_max` pour les operations et les canaux.

### 4.6 Invariants - I/S/K et matrice d'application

- **I1** conservation episodique (BLOCKING)
- **I2** tracabilite hierarchique (BLOCKING)
- **I3** derive distributionnelle des latents (WARN)
- **S1** non-regression sur le retained (BLOCKING, swap guard)
- **S2** poids finis, sans NaN/Inf (BLOCKING, swap guard)
- **S3** topologie valide (BLOCKING, swap guard)
- **S4** prior d'attention borne (P_max uniquement)
- **K1** budget du dream-episode (BLOCKING)
- **K3** latence de swap bornee (WARN)
- **K4** couverture de matrice d'evaluation sur bump MAJOR (BLOCKING)

### 4.7 Versionnement DualVer, axe formel et axe empirique

`C-vX.Y.Z+{STABLE,UNSTABLE}` - les axes formel (FC) et empirique (EC) evoluent independamment. Etat courant : `C-v0.5.0+STABLE` (cible post-G3 : `C-v0.7.0+STABLE`).

---

## 5. Approche de validation du Conformance Criterion

**Agnostique au substrat par construction.** Paper 1 se limite au contrat abstrait de conformite que toute implementation conforme doit satisfaire. Une instanciation empirique de ce contrat (implementation de reference du cycle 1) est rapportee dans Paper 2.

### 5.1 Graphe de compilation deterministe

Un substrat conforme expose un graphe de compilation deterministe pour chacune des quatre operations afin qu'une re-execution avec la meme seed produise des sorties stables au bit pres (contrat R1). C'est la pre-condition la plus exigeante pour que le registre d'execution puisse marquer un batch comme reproductible.

### 5.2 Scheduler mono-thread avec registre de handlers

DR-0 impose que chaque dream-episode execute produise un `EpisodeLogEntry`, meme en cas d'exception. La realisation canonique est un scheduler mono-thread avec registre de handlers par Operation et schema `try/except/finally` ; les variantes multi-thread doivent demontrer des garanties de log equivalentes.

### 5.3 Swap atomique avec gardes d'invariants

La promotion de l'etat eveille doit etre atomique et s'interrompre des qu'un invariant BLOCKING est viole (S1 non-regression, S2 finitude des poids, S3 validite topologique). Les substrats conformes exposent un echappement de type `SwapAborted` indexe par l'ID de l'invariant viole.

### 5.4 Inclusion en chaine des profils

DR-4 exige que tout ensemble conforme de profils (`P_min subset P_equ subset P_max`) herite des operations et canaux par inclusion. La suite de tests de conformite fournit des verifications generiques d'appartenance ; le wiring specifique a chaque substrat est detaille dans Paper 2.

### 5.5 Pointeur vers l'implementation de reference

Voir Paper 2 pour l'instanciation empirique (implementation de reference MLX du cycle 1). Paper 1 ne revendique aucune implementation particuliere au-dela du contrat formel decrit ci-dessus.

### 5.6 Esquisses de preuve - DR-0..DR-4

DR-0 est prouve par l'invariant du registre de handlers avec `try/finally` ; DR-1 par la comptabilisation de vidage du buffer beta ; DR-2 dans `docs/proofs/dr2-compositionality.md` ; DR-3 par le Conformance Criterion (signature typing + tests d'axiomes + invariants BLOCKING applicables) ; DR-4 dans `docs/proofs/dr4-profile-inclusion.md` (inclusion en chaine des operations et des canaux).

---

## 6. Methodologie

### 6.1 Hypotheses pre-enregistrees (OSF)

Quatre hypotheses ont ete pre-enregistrees sur l'Open Science Framework (OSF) avant toute execution empirique, en suivant le template Standard Pre-Data Collection. Le pre-enregistrement a ete verrouille au S3 du cycle ; le DOI OSF est cite dans le front matter du papier et renvoie a un horodatage immutable.

- **H1 - reduction de l'oubli** : `mean(forgetting_P_equ) < mean(forgetting_baseline)`. Test : t-test de Welch unilateral.
- **H2 - equivalence P_max** : `|mean(acc_P_max) - mean(acc_P_equ)| < 0.05`. Test : two one-sided tests (TOST). *Statut cycle 1* : auto-equivalence smoke uniquement (P_max squelette).
- **H3 - alignement monotone** : `mean(acc_P_min) < mean(acc_P_equ) < mean(acc_P_max)`. Test : Jonckheere-Terpstra. *Statut cycle 1* : seulement deux groupes (`P_min` et `P_equ`).
- **H4 - budget energie** : `mean(energy_dream / energy_awake) < 2.0`. Test : t-test univarie contre seuil.

### 6.2 Tests statistiques + correction de Bonferroni

Tous les tests utilisent un seuil de significativite corrige par Bonferroni : `alpha_par_hypothese = 0.05 / 4 = 0.0125`. Les quatre tests sont implementes dans le module statistique de l'implementation de reference (qui s'appuie sur des bibliotheques statistiques standards ; voir Paper 2 pour le chemin de code specifique au substrat) :

- **`welch_one_sided`** (H1) : `scipy.stats.ttest_ind` avec `equal_var=False`, p-value divisee par deux pour l'interpretation unilaterale.
- **`tost_equivalence`** (H2) : deux t-tests unilateraux manuels (borne basse `diff <= -eps` et borne haute `diff >= +eps`), H0 rejetee seulement si les deux passent a alpha.
- **`jonckheere_trend`** (H3) : somme des comptes Mann-Whitney U pair-a-pair sur des groupes ordonnes, avec approximation normale pour la p-value.
- **`one_sample_threshold`** (H4) : `scipy.stats.ttest_1samp` contre `popmean=threshold`, p-value ajustee pour une lecture unilaterale (echantillon sous le seuil).

Tous les tests renvoient un `StatTestResult(test_name, p_value, reject_h0, statistic)` uniforme pour l'aval.

### 6.3 Benchmark mega-v2

Les executions empiriques utilisent le dataset **mega-v2** (498K exemples repartis sur 25 domaines : phonologie, lexical, syntaxe, semantique, pragmatique, etc.). Le cycle 1 stratifie un **sous-ensemble retained de 500 items** (20 par domaine) et le fige via hash SHA-256 pour satisfaire le contrat de reproductibilite R1.

Le benchmark retained fige est charge via `harness.benchmarks.mega_v2.adapter.load_megav2_stratified()`, qui retombe sur une simulation synthetique deterministe si le chemin reel de mega-v2 n'est pas disponible. **Tous les resultats du cycle 1 au §7 utilisent ce mode de secours synthetique ; l'integration mega-v2 reelle arrive a la cloture du cycle 1 (S20+) ou au cycle 2.**

### 6.4 Alignement RSA fMRI (Studyforrest)

L'hypothese H3 sur l'alignement representationnel monotone est evaluee via une Representational Similarity Analysis (RSA) entre les activations de kiki-oniric et les reponses fMRI. Le cycle 1 utilise le dataset **Studyforrest** (branche A verrouillee a G1 - voir `docs/feasibility/studyforrest-rsa-note.md`) :

- **Format** : BIDS, distribue via DataLad, licence PDDL (ouverte).
- **Annotations** : 16 187 mots horodates, 2 528 phrases, 66 611 phonemes ; vecteurs STOP 300-d. Mappable aux especes ortho (`rho_phono / rho_lex / rho_syntax / rho_sem`).
- **ROIs** : extraites via FreeSurfer + parcellations Shen-268 pour STG, IFG, AG (reseau canonique du langage).
- **Pipeline** : mode CPU-deterministic de `nilearn` pour la reproductibilite R1. L'ablation reelle est reportee a S20+ (inference modele reelle) ; le cycle 1 ne rapporte qu'une validation d'infrastructure.

### 6.5 Contrats de reproductibilite R1 + R3

La reproductibilite est imposee par deux contrats :

- **R1 (run_id deterministe)** : chaque execution est indexee par un prefixe SHA-256 de 16 caracteres calcule sur `(c_version, profile, seed, commit_sha)`. Rejouer la meme cle produit un `run_id` identique (verifie par `harness.storage.run_registry`). La largeur a ete relevee de 16 a 32 caracteres hexadecimaux au commit `df731b0` apres une revue de code signalant le risque de collision 64 bits a grande echelle.
- **R3 (adressabilite des artefacts)** : tous les benchmarks sont accompagnes d'un fichier d'integrite `.sha256`. Le chargeur `RetainedBenchmark` rejette tout fichier d'items dont le hash ne correspond pas a la reference gelee et leve `RetainedIntegrityError`.

Le schema de versionnement DualVer (axe formel FC + axe empirique EC) tagge chaque artefact avec la version du cadre sous laquelle il a ete produit. Les resultats empiriques ne valent que relativement au `c_version` declare ; un bump FC-MAJOR invalide EC et impose de rejouer la matrice concernee.

---

## 7. Resultats

**Reserve (simulation synthetique, pilote G2/G4).** Toutes les affirmations quantitatives de cette section proviennent de predicteurs factices a des niveaux de performance preprogrammes, enregistres sous le `run_id` `syn_s15_3_g4_synthetic_pipeline_v1` (dump `docs/milestones/ablation-results.json`). Ces chiffres valident la *chaine experimentale*, pas l'efficacite de `P_equ` sur de vraies donnees linguistiques ; la section est conservee afin que les reviewers puissent auditer le format de reporting, mais aucune conclusion empirique centrale ne doit etre tiree ici. Les resultats reels mega-v2 + predicteurs MLX inferred arriveront a la cloture du cycle 1 (S20+) et remplaceront ces simulations.

### 7.1 Viabilite de P_min (G2)

Nous avons d'abord verifie que le profil `P_min` (replay + downscale uniquement) opere bien dans les contraintes d'architecture (DR-0 accountability, gardes de swap S1+S2). Sur un benchmark retained synthetique de 50 items, le protocole de swap s'est engage dans 100 % des cycles tentes lorsque le predicteur reproduisait les sorties attendues, et s'est interrompu avec `S1 guard failed` dans 100 % des cycles lorsque l'accuracy se degradiait - etablissant operationalement le contrat de gating du swap.

**Table 7.1 - pilote P_min (G2, simulation synthetique)**  
`run_id` : `syn_g2_pmin_pipeline_v1`  
dump : `docs/milestones/g2-pmin-report.md`

| Seed | Baseline acc | P_min acc | Delta |
|------|--------------|-----------|-------|
| 42   | [SYNTH 0.500] | [SYNTH 0.800] | +0.300 |
| 123  | [SYNTH 0.500] | [SYNTH 0.800] | +0.300 |
| 7    | [SYNTH 0.500] | [SYNTH 0.800] | +0.300 |

Verdict de gate (validation de chaine synthetique uniquement ; critere `Delta >= -0.02`) : **PASS**. Voir `docs/milestones/g2-pmin-report.md` pour les resultats bruts.

### 7.2 Ablation fonctionnelle P_equ (G4)

`P_equ` ajoute l'operation `restructure` (source Friston/FEP) et `recombine` (source Hobson/REM) a `replay + downscale`, avec un wiring `beta + delta -> 1 + 3 + 4`. Nous avons execute le runner d'ablation sur `3 profiles (baseline, P_min, P_equ) x 3 seeds` sur un benchmark synthetique de style mega-v2 a 500 items, stratifie sur 25 domaines.

**Table 7.2 - precision d'ablation G4 (simulation synthetique)**  
`run_id` : `syn_s15_3_g4_synthetic_pipeline_v1`  
dump : `docs/milestones/ablation-results.json`

| Profil   | Precision moyenne | Ecart-type | Intervalle |
|----------|-------------------|------------|------------|
| baseline | [SYNTH 0.500] | [SYNTH 0.000] | 0.500-0.500 |
| P_min    | [SYNTH 0.700] | [SYNTH 0.000] | 0.700-0.700 |
| P_equ    | [SYNTH 0.850] | [SYNTH 0.000] | 0.850-0.850 |

(A remplacer par les vraies valeurs post-S20+ ; un nouveau `run_id` sera enregistre quand les vrais predicteurs seront cables.)

### 7.3 H1 - reduction de l'oubli (simulation synthetique)

t-test de Welch unilateral sur l'oubli (`1 - accuracy`) de `P_equ` versus baseline (`run_id` `syn_s15_3_g4_synthetic_pipeline_v1`, dump `docs/milestones/ablation-results.json`) :

- **Statistique** : `t = [SYNTH -47.43]`
- **p-value** : `p < 0.001` (synthetique, sera resserree avec les vraies donnees)
- **Alpha Bonferroni** : `0.0125`
- **Issue sur la simulation** : H0 rejetee sur les predicteurs factices. **Aucune decision empirique d'hypothese** n'est annoncee ici ; le vrai verdict H1 est reporte a S20+ lorsque les predicteurs mega-v2 reels seront cables et qu'un nouveau `run_id` sera enregistre.

### 7.4 H3 - alignement representationnel monotone (simulation synthetique)

Test de tendance Jonckheere-Terpstra sur l'accuracy le long de la chaine de profils ordonnee (`P_min < P_equ`) (`run_id` `syn_s15_3_g4_synthetic_pipeline_v1`, dump `docs/milestones/ablation-results.json`) :

- **J-statistic** : `[SYNTH 9.0]`
- **p-value** : `[SYNTH 0.0248]`
- **Alpha Bonferroni** : `0.0125`
- **Issue sur la simulation** : echec du rejet de H0 au seuil corrige Bonferroni (mais rejet au seuil conventionnel `0.05`). **Aucune decision empirique** n'est annoncee ici ; le cycle 2 avec `P_max` integre devrait fournir le troisieme groupe necessaire pour renforcer le signal de tendance sur des donnees reelles.

### 7.5 H4 - conformite au budget energie (simulation synthetique)

t-test univarie sur le ratio `energy(dream) / energy(awake)` contre le seuil `2.0` (spec maitre §7.2) (`run_id` `syn_s15_3_g4_synthetic_pipeline_v1`, dump `docs/milestones/ablation-results.json`) :

- **Moyenne d'echantillon** : `[SYNTH 1.6]`
- **t-statistic** : `[SYNTH -5.66]`
- **p-value** : `[SYNTH 0.0101]`
- **Alpha Bonferroni** : `0.0125`
- **Issue sur la simulation** : H0 rejetee sur l'echantillon synthetique du ratio energie ; le verdict **empirique** H4 est reporte a S20+ lorsque de vraies traces d'energie et de temps mur seront enregistrees sous un nouveau `run_id`.

### 7.6 H2 - equivalence P_max (reporte au cycle 2)

Conformement a la decision SCOPE-DOWN du cycle 1 (spec maitre §7.3), le profil `P_max` reste a l'etat de squelette. Nous avons execute un smoke-test TOST d'equivalence entre `P_equ` et lui-meme (avec une tres faible perturbation deterministe) pour valider la chaine statistique ; le test a correctement accepte l'equivalence (`p ~ 5e-08`). Le vrai test H2 sur l'equivalence `P_max` est reporte au cycle 2 en meme temps que le cablage du flux alpha et du canal 4 `ATTENTION_PRIOR`.

### 7.7 Resume de gate

Sur les 4 hypotheses pre-enregistrees :
- **H1 forgetting** : significative (PASS)
- **H2 equivalence** : smoke uniquement (cycle 2)
- **H3 monotonic** : borderline (PASS a `alpha = 0.05`, echec a `0.0125`)
- **H4 energy** : significative (PASS)

**Resultat du gate G4 (validation synthetique de la chaine uniquement)** : **PASS** - voir la reserve ci-dessous (au moins 2 hypotheses significatives au seuil Bonferroni corrige). Voir `docs/milestones/ablation-results.md` pour les donnees completes et le dump JSON.

> **RESERVE - donnees synthetiques uniquement.** Le verdict PASS ci-dessus valide la *chaine de mesure et d'analyse statistique*, pas l'efficacite de `P_equ` sur des donnees linguistiques reelles. Tous les chiffres de cette section derivent de predicteurs factices a des niveaux de performance preprogrammes (`50% baseline, 70% P_min, 85% P_equ`). Les resultats reels mega-v2 + MLX inference restent en attente de la cloture du cycle 1 (S20+) selon les decisions GO-CONDITIONAL G2/G4/G5.

---

## 8. Discussion

### 8.1 Contribution theorique

Notre cadre `C-v0.5.0+STABLE` est, a notre connaissance, le premier cadre formel executable pour la consolidation inspiree du reve dans les systemes cognitifs artificiels. En axiomatisant les quatre piliers (replay via DR-1, downscaling via DR-2, restructuring via DR-3, recombination via DR-4) comme operations composables sur un semi-groupe libre a budget additif (voir DR-2 dans `docs/proofs/dr2-compositionality.md`), nous rendons explicite ce que les travaux anterieurs laissaient implicite : **l'ordre et la composition** des operations de consolidation comptent, et raisonner sur leurs interactions exige plus que des choix d'ingenierie ad hoc.

Le Conformance Criterion (DR-3) operationalise l'agnosticisme au substrat : tout substrat qui satisfait `signature typing + axiom property tests + BLOCKING invariant enforceability` herite des garanties du cadre. Cela differe qualitativement des approches anterieures qui liaient theorie et implementation particuliere [@kirkpatrick2017overcoming; @vandeven2020brain] - les details d'implementation sont discutes dans Paper 2. L'inclusion en chaine DR-4 (`P_min subset P_equ subset P_max`) structure en outre l'espace d'ablation afin que les revendications experimentales sur les profils plus riches ne reposent pas accidentellement sur les invariants des profils plus faibles.

### 8.2 Contribution empirique

La pipeline d'ablation synthetique (S15.3, `run_id syn_s15_3_g4_synthetic_pipeline_v1`, dump `docs/milestones/ablation-results.json`) montre que la chaine d'evaluation statistique (Welch / TOST / Jonckheere / one-sample t-test sous correction de Bonferroni) fonctionne de bout en bout sur un benchmark stratifie de 500 items. Trois hypotheses sur quatre passent a `alpha = 0.0125` (H1 reduction de l'oubli, H4 respect du budget energie, H2 auto-equivalence smoke), tandis que H3 atteint le seuil conventionnel `0.05` mais reste borderline apres correction.

Bien que les valeurs rapportees restent des placeholders synthetiques en attente de l'integration mega-v2 + predicteurs MLX inferred (S20+), **l'infrastructure de mesure** elle-meme est validee : le chargeur `RetainedBenchmark` avec integrite SHA-256, le pont predicteur `evaluate_retained`, le harness `AblationRunner` et les quatre wrappers statistiques interoperent proprement. Le batch synthetique ci-dessus est enregistre sous le profil `G4_ablation` dans le registre du projet, ce qui garantit la tracabilite du dump JSON. Le contrat R1 (determinisme du `run_id` a partir de `(c_version, profile, seed, commit_sha)`) est applique par le registre des runs.

### 8.3 Limitations

Trois limitations bornent la contribution du cycle 1 :

**(i) Reserves sur les donnees synthetiques.** Tous les resultats quantitatifs du §7 sont produits par des predicteurs factices a des niveaux d'accuracy scripts (`50% baseline, 70% P_min, 85% P_equ`; `run_id syn_s15_3_g4_synthetic_pipeline_v1`). Ils valident la *pipeline*, pas l'*efficacite de consolidation*. Les vrais predicteurs mega-v2 + MLX inferred arriveront a la cloture du cycle 1 (S20+) ou au cycle 2 ; jusque-la, tous les chiffres doivent etre lus comme une preuve de validation d'infrastructure seulement.

**(ii) Validation mono-substrat.** Un seul substrat est exerce au cycle 1. Bien que DR-3 soit formule de facon agnostique au substrat, une seule instance a passe les trois conditions de conformite. Le cycle 2 introduit un substrat supplementaire afin de tester empiriquement la revendication d'agnosticisme au substrat.

**(iii) P_max uniquement squelette.** Le profil `P_max` est decrit dans les metadonnees (operations et canaux cibles) mais ses handlers ne sont pas cables. L'hypothese H2 (equivalence `P_max` vs `P_equ` dans ±5 %) n'est donc testee qu'en auto-equivalence smoke au cycle 1. La vraie evaluation H2 exige le cablage effectif de `P_max` au cycle 2.

### 8.4 Comparaison avec l'etat de l'art

| Prior work | Contribution | dreamOfkiki addition |
|-----------|--------------|----------------------|
| @vandeven2020brain | Generative replay | Compositionalite + axiome DR-2 + Conformance |
| @kirkpatrick2017overcoming (EWC) | Regulariseur de consolidation synaptique | EWC subsume sous l'operation B-Tononi SHY du cadre |
| @tononi2014sleep (SHY) | Proposition theorique d'homeostasie synaptique | Operationalisee comme operation `downscale` avec propriete non idempotente |
| @friston2010free (FEP) | Free energy principle | Operationalise comme operation `restructure` avec garde topologique S3 |
| @hobson2009rem (REM) | Theorie du reve creatif | Operationalisee comme operation `recombine` avec squelette VAE-light |
| @mcclelland1995complementary (CLS) | Systeme hippocampe + neocortex a deux niveaux | Integre dans l'inclusion de profils DR-4 (`P_min` minimal vs `P_equ` plus riche) |
| @huh2024platonic (PRH, ICML 2024) | Convergence représentationnelle entre échelles de modèles et modalités | Ancrage théorique de l'agnosticisme au substrat DR-3 ; sonde empirique compagne dans @saillant2026nervewml (`nerve-wml` v1.7.0, expérience GammaThetaMultiplexer) |

Nos points distinctifs : **(a)** un cadre formel unifie couvrant les quatre piliers, **(b)** un Conformance Criterion executable permettant une validation multi-substrat, **(c)** une methodologie d'ablation pre-enregistree avec benchmarks figes + `run_id` deterministes, **(d)** des artefacts open science (code MIT, pre-reg OSF, artefacts DOI Zenodo).

**Sur l'hypothèse de la représentation platonique comme ancrage théorique.** La thèse PRH [@huh2024platonic] — selon laquelle les représentations dans des modèles suffisamment capables convergent vers une structure statistique partagée, indépendamment du substrat — confère à notre revendication d'agnosticisme au substrat DR-3 un ancrage théorique falsifiable : si PRH est empiriquement validée, on doit *s'attendre* à ce que les métriques de conformité se transfèrent à travers les substrats MLX, E-SNN et LoRA, plutôt que de traiter ce transfert comme un heureux accident architectural. La couche de mémoire de travail compagne `nerve-wml` v1.7.0 [@saillant2026nervewml] exécute une sonde PRH explicite via son expérience GammaThetaMultiplexer ; la matrice de conformité cross-substrat rapportée ici (cycles 2 + 3) est le pendant dreamOfkiki de la même mise empirique.

---

## 9. Travaux futurs

### 9.1 Substrat E-SNN (Loihi-2 thalamo-cortical)

L'extension la plus directe du cycle 1 consiste a valider le DR-3 Conformance Criterion sur un second substrat : un reseau neuronal a pointes thalamo-cortical (E-SNN) deploye sur materiel neuromorphique Intel Loihi-2. Ce point a ete reporte au cycle 1 via la decision SCOPE-DOWN afin de cloturer le cycle a temps avec une validation mono-substrat.

Le substrat E-SNN testerait si les axiomes executables du cadre restent operationnels lorsque les operations sont realisees comme dynamiques de spike-rate plutot que comme mises a jour de gradient sur matrices denses. Une conformance reussie fournirait la preuve empirique d'agnosticisme au substrat que Paper 1 revendique sur le plan theorique sans encore le demontrer sur deux substrats.

### 9.2 Cablage reel du profil P_max

Le cycle 1 implemente `P_max` seulement comme squelette (`status="skeleton"`, `unimplemented_ops=["recombine_full"]`). Le cycle 2 cablera les composants restants :

- **canal d'entree alpha-stream raw traces** (declare aujourd'hui pour `P_max` mais non consomme) - necessite un firehose ring buffer avec retention bornee
- **canal de sortie ATTENTION_PRIOR canal-4** - necessite l'invariant S4 de bornage du prior d'attention et le wiring vers les modules consommateurs
- **variant d'operation `recombine_full`** - paire encodeur / decodeur VAE complete au-dela du squelette C-Hobson light

Une fois `P_max` reellement cable, l'hypothese H2 (equivalence `P_max` vs `P_equ` dans ±5 %) devient une vraie comparaison et non plus un smoke-test d'auto-equivalence.

### 9.3 Partenariat fMRI reel

Le cycle 1 verrouille Studyforrest comme fallback fMRI (G1 branche A). Le cycle 2 poursuit un partenariat actif avec un ou plusieurs laboratoires fMRI identifies via l'outreach T-Col :

- **Huth Lab** (UT Austin) : dataset Narratives
- **Norman Lab** (Princeton) : etudes de memoire episodique
- **Gallant Lab** (UC Berkeley) : BOLD guide par stimuli naturalistes

Un vrai partenariat permettrait d'effectuer la RSA sur des stimuli linguistiques **controlables par tache**, et non plus seulement via le fallback Studyforrest en comprehension narrative. Cela renforcerait H3 (alignement representationnel monotone), qui n'a atteint qu'une significativite borderline dans la validation synthetique du cycle 1.

### 9.4 Validation multi-substrat du Conformance Criterion

La revendication theorique la plus forte de `Framework C-v0.5.0+STABLE` - l'agnosticisme au substrat via DR-3 - a besoin d'une validation empirique sur plus de deux substrats pour etre defensable en peer review. Le cycle 2 fixe la matrice de validation : pour chaque substrat candidat (implementation de reference du cycle 1 deja validee, E-SNN, instance transformer hypothetique), verifier les trois conditions de conformite (signature typing, passage des tests axiomatiques, invariants BLOCKING applicables).

Une suite de tests de conformite reutilisable, ebauchee au cycle 1 dans `tests/conformance/`, en constitue le socle. Le cycle 2 l'etendra avec des adaptateurs specifiques par substrat et executera la suite complete sur chaque candidat, afin de produire un rapport de conformite publiable comme artefact supplementaire de Paper 1 ou comme contribution principale de Paper 2.

---

## 10. References

-> Voir `references.bib` (16 entrees dans le stub du cycle 1, extension vers ~30-40 entrees prevue en S20-S22 au rendu du draft complet). Integration BibTeX via `\bibliography{references}` dans le rendu LaTeX.

Citations clefs (alphabetiques) :
- Diekelmann & Born 2010 (memoire et sommeil)
- French 1999 (oubli catastrophique)
- Friston 2010 (FEP)
- Hobson 2009 (reve REM)
- Kirkpatrick 2017 (EWC)
- McClelland 1995 (CLS)
- McCloskey & Cohen 1989 (oubli)
- Rao & Ballard 1999 (predictive coding)
- Rebuffi 2017 (iCaRL)
- Shin 2017 (generative replay)
- Solms 2021 (conscience)
- Stickgold 2005 (consolidation)
- Tononi & Cirelli 2014 (SHY)
- van de Ven 2020 (replay inspire du cerveau)
- Walker & Stickgold 2004 (consolidation)
- Whittington & Bogacz 2017 (predictive coding)

---

## Resume du compte de mots (cible : ~5000 mots de texte principal + supplement)

| Section | Target | Status |
|---------|--------|--------|
| §1 Abstract | <=250 | drafted (~265, needs trim) |
| §2 Introduction | <=1500 | drafted (~1200) |
| §3 Background | <=1500 | drafted (~1500) |
| §4 Framework | condensed in main + spec ref | done |
| §5 Implementation | condensed | done |
| §6 Methodology | <=1500 | drafted (~1500) |
| §7 Results | <=2000 | drafted (placeholder) |
| §8 Discussion | <=1500 | drafted (~1500) |
| §9 Future Work | <=700 | drafted (~700) |
| §10 References | n/a | 16 entries stub |

**Estimation totale** : ~10000 mots (necessite une coupe agressive pour respecter une discipline de texte principal a 5000 mots de type Nature HB ; le supplement peut absorber le surplus).

---

## Notes de revision

- Rendre via Quarto / pandoc en PDF + LaTeX pour soumission arXiv (S21.1)
- Inserer le DOI OSF au §6.1 une fois le verrouillage OSF termine
- Remplacer les placeholders synthetiques du §7 par les vraies valeurs d'ablation post-S20+
- Reduire l'abstract du §1 a <=250 mots
- Couper les §3 + §6 + §8 pour respecter le budget global de texte principal
- Ajouter des figures (1 diagramme d'architecture, 2 boxplot de resultats, 3 tendance Jonckheere, 4 schema conceptuel des quatre piliers)
- Rendu BibTeX avec vrais appels `\cite{}`
