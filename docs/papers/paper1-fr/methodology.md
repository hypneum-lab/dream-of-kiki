# Méthodologie (Paper 1, brouillon S20.1)

**Longueur cible** : ~1,5 page markdown (≈ 1500 mots)

---

## 6.1 Hypothèses pré-enregistrées (OSF)

Quatre hypothèses ont été pré-enregistrées sur l'Open Science
Framework (OSF) avant toute exécution empirique, selon le gabarit
Standard Pre-Data Collection. Le pré-enregistrement a été
verrouillé à S3 du cycle (référence calendaire) ; le DOI OSF est
cité dans les pages liminaires de l'article et se résout en un
enregistrement horodaté immuable. Cette discipline s'aligne sur
la pratique standard désormais explicitement recommandée pour les
méta-analyses de consolidation mnésique liée au sommeil [eLife
2025 bayesian] et reflète le protocole suivi par le projet sœur
`bouba_sens` (DOI 10.17605/OSF.IO/Q6JYN).

- **H1 — Réduction de l'oubli** : `mean(forgetting_P_equ) <
  mean(forgetting_baseline)`. Test : t de Welch, unilatéral.
  *Plancher de taille d'effet* : la méta-analyse de la
  réactivation mnésique ciblée [Hu et al. 2020 TMR] rapporte un
  Hedges' g global de 0,29 (NREM2 g = 0,32 ; SWS g = 0,27), pris
  ici comme magnitude de référence pour les gains de
  consolidation de type `P_equ`. Ce n'est pas un plancher : les
  estimations corrigées du biais du même article sont plus
  faibles (trim-and-fill g = 0,18 ; modèle de sélection à trois
  paramètres g = 0,13 ; g = −0,05 sous sélection unilatérale
  sévère).
- **H2 — Équivalence P_max** : `|mean(acc_P_max) -
  mean(acc_P_equ)| < 0.05`. Test : deux tests unilatéraux (TOST).
  *Statut cycle 1* : test de fumée d'auto-équivalence uniquement
  (P_max en squelette).
- **H3 — Alignement monotone** : `mean(acc_P_min) <
  mean(acc_P_equ) < mean(acc_P_max)`. Test :
  Jonckheere-Terpstra. *Statut cycle 1* : deux groupes
  (P_min ↔ P_equ) uniquement. *Plancher de taille d'effet pour
  le bras P_min* : la méta-analyse sur la restriction de sommeil
  [Crowley et al. 2024 sleep restriction] rapporte g = 0,29 [0,13 ;
  0,44] comme magnitude attendue du décrément de substrat
  dégradé ; son graphique en entonnoir est significativement
  asymétrique (Egger z = 3,90), bien qu'un modèle de sélection à
  trois paramètres ne détecte pas de biais de publication, et
  son intervalle de prédiction à 95 % va de −0,73 à 1,30.
- **H4 — Budget énergétique** : `mean(energy_dream / energy_awake)
  < 2.0`. Test : t à un échantillon contre seuil.

## 6.2 Tests statistiques + correction de Bonferroni

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

## 6.3 Banc de test mega-v2

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

## 6.4 Alignement RSA IRMf (Studyforrest)

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

## 6.5 Contrat de reproductibilité R1 + R3

La reproductibilité est appliquée par deux contrats :

- **R1 (run_id déterministe)** : chaque exécution est clé par un
  préfixe SHA-256 de **32 caractères hex** de `(c_version,
  profile, seed, commit_sha)` (largeur post-`df731b0`, élargie
  depuis 16 caractères suite à un signalement de collision
  64-bits à grande échelle). Réexécuter avec la même clé produit
  un `run_id` identique, vérifié par
  `harness.storage.run_registry` (cf. cross-link
  `harness/storage/run_registry.py`).
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

## 6.6 Ancrage biologique des profils et de DR-2'

Les trois profils sont calibrés sur des biomarqueurs de niveau
substrat rapportés dans la littérature du sommeil humain.
`P_min` (substrat dégradé) est ancré sur l'amplitude du creux de
l'oscillation lente et sur le gradient de synchronisation
fronto-central documentés à travers les groupes de sujets âgés
sains / aMCI / MA (N = 55 hd-EEG) [Sharon et al. 2025 Alzheimer
Dementia], où la performance cognitive décroît de façon monotone
avec la perte de cohérence de l'onde lente. La calibration est
implémentée comme un champ `so_trough_amplitude_factor` sur la
dataclass de profil (`kiki_oniric/profiles/p_min.py`), avec la
valeur ratio sans-unité `0,45` pour `P_min` (placeholder médian
aMCI) et `1,0` pour les ancres saines `P_equ` / `P_max`. Sharon
2025 rapporte bien des amplitudes absolues du creux de l'onde
lente en µV (médianes de groupe −60,18 sains, −56,95 aMCI,
−52,61 MA) ; le ratio 0,45 n'en est pas dérivé et sera affiné
empiriquement lors du pilote G2 P_min
(`scripts/pilot_g2.py`). `P_equ` (substrat
canonique) est calibré pour reproduire la bande de tailles
d'effet NREM2/SWS rapportée dans [Hu et al. 2020 TMR]. L'axiome
falsifié-puis-amendé DR-2' (réactivation entrelacée
nouveau-hippocampique + familier-cortical, voir
`docs/specs/amendments/`) hérite son ancrage biologique du
résultat d'entrelacement up-state SWS de [bioRxiv 2025
thalamocorticale].

Les ancres numériques de [@hu2020tmr] (global, NREM2, SWS) et
[@javadi2024sleeprestriction] sont encodées comme constantes
typées et figées dans `harness.benchmarks.effect_size_targets`
(`HU_2020_OVERALL`, `HU_2020_NREM2`, `HU_2020_SWS`,
`JAVADI_2024_OVERALL`) afin qu'un futur pilote G4 compare les
tailles d'effet observées aux IC 95 % publiés de façon
déterministe (`is_within_ci`, `distance_from_target`). Aucun
bump empirique (EC) n'est impliqué : ces constantes encodent
des nombres externes publiés, pas des sorties de runs
enregistrés.

---

## Notes pour révision

- Insérer le DOI OSF au §6.1 une fois l'action de verrouillage OSF
  terminée (en cours — voir `docs/osf-upload-checklist.md`)
- Remplacer les précautions liées aux substitutions synthétiques
  au §6.3 une fois les exécutions d'ablation mega-v2 réelles S20+
- Ajouter un tableau supplémentaire Méthodes : paramètres complets
  des tests statistiques (tailles d'échantillon par cellule,
  critères d'exclusion du pré-enregistrement OSF)
- Ajouter une figure supplémentaire Méthodes : schéma du pipeline
  (benchmark → prédicteur → evaluate_retained → swap → métriques)
