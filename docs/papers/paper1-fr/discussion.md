# Discussion (Paper 1, brouillon S19.1)

**Longueur cible** : ~1,5 page markdown (≈ 1500 mots)

---

## 8.1 Contribution théorique

Notre framework C-v0.5.0+STABLE est, à notre connaissance, le
premier framework formel exécutable pour la consolidation mnésique
basée sur le rêve dans les systèmes cognitifs artificiels. En
axiomatisant les quatre piliers (replay (DR-1), downscaling
(DR-2), restructuring (DR-3), recombination (DR-4)) comme
opérations composables sur un semi-groupe libre à budget additif
(voir DR-2 dans `docs/proofs/dr2-compositionality.md`), nous
explicitons ce que les travaux antérieurs laissaient implicite :
l'**ordre et la composition** des opérations de consolidation
importent, et raisonner sur leurs interactions exige davantage que
des choix d'ingénierie ad hoc.

Le Critère de Conformité (DR-3) opérationnalise l'indépendance
du substrat : tout substrat qui satisfait le typage des signatures
+ les tests de propriété axiomatiques + l'applicabilité des
invariants BLOCKING hérite des garanties du framework. Ceci diffère
qualitativement des frameworks antérieurs qui lient la théorie à
une implémentation spécifique [Kirkpatrick 2017, van de Ven 2020]
— les détails d'implémentation sont discutés dans le Paper 2.
L'inclusion en chaîne des profils DR-4 (P_min ⊆ P_equ ⊆ P_max)
structure en outre l'espace d'ablation de telle sorte que les
assertions expérimentales sur des profils plus riches ne
reposent pas par inadvertance sur des invariants de profils plus
faibles.

## 8.2 Contribution empirique

Le pipeline d'ablation synthétique (S15.3, run_id
`syn_s15_3_g4_synthetic_pipeline_v1`, dump
`docs/milestones/ablation-results.json`) démontre que la chaîne
d'évaluation statistique (Welch / TOST / Jonckheere / test t à un
échantillon sous correction de Bonferroni) est opérationnelle de
bout en bout sur un banc de test stratifié de 500 items. Trois des
quatre hypothèses pré-enregistrées passent à α = 0,0125 (H1
réduction de l'oubli, H4 conformité au budget énergétique, test
de fumée d'auto-équivalence H2), H3 tendance monotone atteignant
le seuil conventionnel 0,05 mais limite au niveau corrigé.

Bien que les valeurs rapportées soient des substitutions
synthétiques en attente de l'intégration des prédicteurs réels
mega-v2 + inférés par MLX (S20+), l'**infrastructure de mesure**
est elle-même validée : le chargeur RetainedBenchmark avec
intégrité SHA-256, le pont prédicteur `evaluate_retained`, le
harness AblationRunner et les quatre enveloppes statistiques
interopèrent proprement. Le lot synthétique ci-dessus est
enregistré sous le profil `G4_ablation` dans le registre projet
afin que le dump JSON reste traçable. Le contrat de
reproductibilité R1 (`run_id` déterministe depuis (c_version,
profile, seed, commit_sha)) est appliqué par le run registry.

## 8.3 Limites

Trois limites bornent la contribution du cycle 1 :

**(i) Précautions liées aux données synthétiques.** Tous les
résultats quantitatifs au §7 sont produits par des prédicteurs
mock aux niveaux de précision scriptés (50 % baseline, 70 % P_min,
85 % P_equ ; run_id `syn_s15_3_g4_synthetic_pipeline_v1`). Ils
valident le *pipeline*, non l'*efficacité de la consolidation*.
Les prédicteurs réels mega-v2 + inférés par MLX interviennent en
clôture du cycle 1 (S20+) ou au cycle 2 ; d'ici là, tous les
chiffres doivent être lus comme preuves de validation
d'infrastructure uniquement.

**(ii) Validation sur substrat unique.** Un substrat unique est
exercé au cycle 1. Bien que le Critère de Conformité DR-3 soit
formulé pour être indépendant du substrat, seule une instance a
passé les trois conditions de conformité. Le cycle 2 introduit un
substrat supplémentaire afin de tester empiriquement l'assertion
d'indépendance du substrat selon le Critère de Conformité DR-3.

**(iii) P_max en squelette uniquement.** Le profil P_max est
déclaré via des métadonnées (opérations cibles, canaux cibles)
mais ses handlers ne sont pas câblés. L'hypothèse H2 (équivalence
P_max vs P_equ dans ±5 %) n'est donc testée que comme test de
fumée d'auto-équivalence au cycle 1. Une évaluation H2 réelle
requiert le câblage réel de P_max (cycle 2).

## 8.4 Comparaison avec l'état de l'art

| Travail antérieur | Contribution | Apport dreamOfkiki |
|-----------|--------------|----------------------|
| van de Ven 2020 | Replay génératif | Composabilité + axiome DR-2 + Conformité |
| Kirkpatrick 2017 (EWC) | Régulariseur de consolidation synaptique | EWC subsumée sous l'opération B-Tononi SHY dans le framework |
| Tononi & Cirelli 2014 (SHY) | Thèse théorique de l'homéostasie synaptique | Opérationnalisée comme opération `downscale` à propriété non idempotente |
| Friston 2010 (FEP) | Principe d'énergie libre | Opérationnalisé comme opération `restructure` avec garde topologique S3 |
| Hobson 2009 (REM) | Théorie du rêve créatif | Opérationnalisée comme opération `recombine` avec squelette VAE allégé |
| McClelland 1995 (CLS) | Système dual hippocampe + néocortex | Intégré dans l'inclusion de profils DR-4 (P_min minimal vs P_equ plus riche) |

Nos traits distinctifs : **(a)** framework formel unifié couvrant
les quatre piliers, **(b)** Critère de Conformité exécutable
permettant la validation multi-substrat, **(c)** méthodologie
d'ablation pré-enregistrée avec bancs de test figés +
identifiants de runs déterministes, **(d)** artefacts de science
ouverte (code MIT, pré-enregistrement OSF, artefacts DOI Zenodo).

## 8.5 Réplication cross-substrat préliminaire (cycle 2)

Le cycle 2 opérationnalise la limite (ii) ci-dessus en câblant un
second substrat — `esnn_thalamocortical`, un squelette numpy LIF
à taux de décharge — aux côtés du substrat canonique
`mlx_kiki_oniric`. Le Critère de Conformité DR-3 est ré-évalué
sur les deux substrats (voir
`docs/milestones/conformance-matrix.md` et
`docs/proofs/dr3-substrate-evidence.md`), et la chaîne statistique
H1-H4 du cycle 1 est ré-exécutée par substrat
(`docs/milestones/cross-substrate-results.md`, runner
`scripts/ablation_cycle2.py`).

**Substitution synthétique — pas une revendication empirique.** Les
deux lignes de substrats partagent le même prédicteur Python mock
au cycle 2 : l'inférence spécifique au substrat est reportée au
cycle 3. Par conséquent, le verdict cross-substrat est
trivialement concordant par construction, et le pipeline émet des
p-values H1-H4 identiques sur les deux substrats (3 / 4
significatifs à Bonferroni α = 0,0125, H3 monotonie échouant sur
les deux en raison de la dispersion constante du mock). Ceci
**renforce mais ne remplace pas** les résultats H1-H4 du cycle 1
rapportés aux §7 et §8.2. Ce que cela *démontre*, c'est que les
artefacts de conformité du framework (Protocoles typés, tests de
propriété axiomatiques, gardes S2/S3) et la chaîne d'évaluation
statistique s'exécutent de bout en bout sur un second
enregistrement de substrat structurellement indépendant, ce qui
est la revendication architecturale de DR-3. Une réplication à
prédicteurs divergents sur données biologiques ou neuromorphiques
réelles constitue la cible du cycle 3.

---

## Notes pour révision

- Remplacer les précautions "synthétique" par les résultats sur
  données réelles post S20+
- Resserrer à ≤1500 mots pour la discipline du corps principal
  Nature HB
- Insérer les citations bibtex appropriées une fois references.bib
  configuré (S19.3)
