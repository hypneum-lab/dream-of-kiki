# Résultats (Paper 1, brouillon S18.2)

**Longueur cible** : ~1,5-2 pages markdown (≈ 1500-2000 mots)

⚠️ **Précaution (substitution synthétique, pilote G2/G4).** Toute
assertion quantitative de cette section provient de prédicteurs
mock aux niveaux de précision scriptés (50 %/70 %/85 %) enregistrés
sous le run_id `syn_s15_3_g4_synthetic_pipeline_v1` (dump
`docs/milestones/ablation-results.json`). Les chiffres valident le
*pipeline*, pas l'efficacité de P_equ sur des données linguistiques
réelles ; la section est préservée ici pour permettre aux relecteurs
d'auditer le gabarit de rapport, mais aucune assertion empirique
principale ne doit en être tirée. Les prédicteurs mega-v2 réels +
inférés par MLX interviennent en clôture du cycle 1 (S20+) et
remplaceront ces substitutions.

---

## 7.1 Viabilité de P_min (G2)

Nous avons d'abord vérifié que le profil P_min (replay + downscale
uniquement) fonctionne dans les contraintes architecturales (DR-0
redevabilité, gardes de basculement S1+S2). Sur un banc de test
retenu synthétique de 50 items, le protocole de basculement a été
commité dans 100 % des cycles tentés lorsque le prédicteur
correspondait aux sorties attendues, et a avorté avec
`S1 guard failed` dans 100 % des cycles lorsque la précision se
dégradait — établissant opérationnellement le contrat de contrôle
du basculement.

**Table 7.1 — Pilote P_min (G2, substitution synthétique)**

run_id : `syn_g2_pmin_pipeline_v1`
dump : `docs/milestones/g2-pmin-report.md`

| Seed | Précision baseline | Précision P_min | Δ |
|------|--------------|-----------|---|
| 42   | [SYNTH 0.500] | [SYNTH 0.800] | +0.300 |
| 123  | [SYNTH 0.500] | [SYNTH 0.800] | +0.300 |
| 7    | [SYNTH 0.500] | [SYNTH 0.800] | +0.300 |

Verdict de porte (validation du pipeline synthétique uniquement ;
critère Δ ≥ −0,02) : **PASS**. Voir
`docs/milestones/g2-pmin-report.md` pour les résultats bruts.

---

## 7.2 Ablation fonctionnelle de P_equ (G4)

P_equ ajoute l'opération `restructure` (source Friston FEP) et
l'opération `recombine` (source Hobson REM) aux côtés de `replay`
+ `downscale`, avec les canaux β+δ → 1+3+4 câblés. Nous avons
exécuté le runner d'ablation sur 3 profils (baseline, P_min, P_equ)
× 3 graines sur un banc de test synthétique de style mega-v2 de
500 items stratifié sur 25 domaines.

**Table 7.2 — Précision d'ablation G4 (substitution synthétique,
pilote G4)**

run_id : `syn_s15_3_g4_synthetic_pipeline_v1`
dump : `docs/milestones/ablation-results.json`

| Profil   | Précision moy. | Écart-type | Plage |
|----------|----------|-----|-------|
| baseline | [SYNTH 0.500] | [SYNTH 0.000] | 0.500-0.500 |
| P_min    | [SYNTH 0.700] | [SYNTH 0.000] | 0.700-0.700 |
| P_equ    | [SYNTH 0.850] | [SYNTH 0.000] | 0.850-0.850 |

(Remplacer par les valeurs d'ablation réelles post-S20+ ; un
nouveau run_id sera enregistré lorsque les prédicteurs réels
seront câblés.)

---

## 7.3 H1 — Réduction de l'oubli (substitution synthétique)

Test t de Welch (unilatéral) sur l'oubli (1 − précision) de P_equ
versus baseline (run_id `syn_s15_3_g4_synthetic_pipeline_v1`,
dump `docs/milestones/ablation-results.json`) :

- **Statistique** : t = [SYNTH −47.43]
- **p-value** : p < 0,001 (synthétique, sera resserré avec données
  réelles)
- **α de Bonferroni** : 0,0125
- **Issue du pipeline synthétique** : H0 rejetée sur les
  prédicteurs mock. **Aucune décision d'hypothèse empirique**
  n'est annoncée ici ; le verdict H1 authentique est différé à
  S20+ lorsque les prédicteurs mega-v2 réels seront câblés et
  qu'un run_id frais sera enregistré.

---

## 7.4 H3 — Alignement représentationnel monotone
       (substitution synthétique)

Test de tendance Jonckheere-Terpstra sur la précision à travers la
chaîne ordonnée des profils (P_min < P_equ) (run_id
`syn_s15_3_g4_synthetic_pipeline_v1`, dump
`docs/milestones/ablation-results.json`) :

- **Statistique J** : [SYNTH 9.0]
- **p-value** : [SYNTH 0.0248]
- **α de Bonferroni** : 0,0125
- **Issue du pipeline synthétique** : échoue à rejeter H0 au seuil
  corrigé par Bonferroni (rejetterait au α conventionnel = 0,05).
  **Aucune décision d'hypothèse empirique** n'est annoncée ici ;
  le cycle 2 avec P_max intégré devrait fournir le troisième
  groupe nécessaire pour renforcer le signal de tendance sur
  données réelles.

---

## 7.5 H4 — Conformité au budget énergétique (substitution synthétique)

Test t à un échantillon sur le ratio énergétique
energy(dream) / energy(awake) contre le seuil 2,0 (critère de
viabilité du master spec §7.2) (run_id
`syn_s15_3_g4_synthetic_pipeline_v1`, dump
`docs/milestones/ablation-results.json`) :

- **Moyenne d'échantillon** : [SYNTH 1.6]
- **Statistique t** : [SYNTH −5.66]
- **p-value** : [SYNTH 0.0101]
- **α de Bonferroni** : 0,0125
- **Issue du pipeline synthétique** : H0 rejetée sur l'échantillon
  mock de ratio énergétique ; le verdict **empirique** H4 est
  différé à S20+ lorsque les traces énergétiques réelles sur
  horloge murale seront enregistrées sous un run_id fraîchement
  enregistré.

---

## 7.6 H2 — Équivalence P_max (différé au cycle 2)

Conformément à la décision de SCOPE-DOWN du cycle 1 (master spec
§7.3), le profil P_max reste uniquement en squelette. Nous avons
exécuté un test de fumée d'équivalence TOST de P_equ contre
lui-même (avec une toute petite perturbation déterministe) pour
valider le pipeline statistique ; le test a correctement accepté
l'équivalence (p ≈ 5e-08). Le test réel d'équivalence H2 P_max est
différé au cycle 2 aux côtés du câblage du flux α +
ATTENTION_PRIOR canal-4.

---

## 7.7 Résumé de la porte

Parmi les 4 hypothèses pré-enregistrées :
- **H1 oubli** : significatif (PASS)
- **H2 équivalence** : test de fumée uniquement (cycle 2)
- **H3 monotone** : limite (PASS à α=0,05, échec à Bonferroni
  0,0125)
- **H4 énergie** : significatif (PASS)

**Résultat de porte G4 (validation du pipeline synthétique
uniquement)** : **PASS** — voir PRÉCAUTION ci-dessous (≥2
hypothèses significatives au α corrigé par Bonferroni). Voir
`docs/milestones/ablation-results.md` pour les données complètes +
dump JSON.

> **⚠️ PRÉCAUTION — données synthétiques uniquement.** Le verdict
> PASS ci-dessus valide le *pipeline de mesure et statistique*,
> non l'efficacité de P_equ sur des données linguistiques réelles.
> Tous les chiffres de cette section dérivent de prédicteurs mock
> aux niveaux de précision scriptés (50 % baseline, 70 % P_min,
> 85 % P_equ). Les résultats d'inférence mega-v2 + MLX réels sont
> en attente de la clôture du cycle 1 (S20+) selon les décisions
> GO-CONDITIONAL G2/G4/G5.

---

## Notes pour révision

- Remplacer toutes les valeurs [SYNTH ...] par les chiffres
  d'ablation réels post S20+ (mega-v2 réel + inférence MLX réelle)
- Ajouter des barres d'erreur / tableaux IC 95 % pour chaque
  métrique sur 3 graines
- Ajouter la Figure 1 : boxplot de distribution de précision par
  profil
- Ajouter la Figure 2 : statistique J de Jonckheere avec bande de
  confiance
- Renvoyer au §6 Méthodologie pour les détails de protocole
- Discuter le résultat limite H3 au §8 Discussion
