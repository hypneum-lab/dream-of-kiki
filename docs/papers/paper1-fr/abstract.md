# Résumé (Paper 1, brouillon)

**Objectif de mots** : 250 mots

---

## Brouillon v0.2 (cycle-1 closeout, 2026-04-20, miroir EN d6866f3)

**Note de portée.** Cet article rapporte un *framework formel
exécutable* pour la consolidation onirique dans les systèmes
cognitifs artificiels ; les contributions principales sont les
axiomes, le Critère de Conformité et un parcours de conformité
inter-substrats. L'évaluation empirique au niveau des hypothèses
sur des bancs de test d'apprentissage continu et d'alignement
IRMf est différée au papier compagnon (Paper 2).

L'oubli catastrophique demeure un obstacle central pour les systèmes
cognitifs artificiels apprenant séquentiellement une succession de
tâches. La consolidation mnésique inspirée du sommeil a été proposée
comme remède, les travaux antérieurs ayant exploré la réactivation
(Walker, van de Ven), l'homéostasie synaptique (Tononi), la
recombinaison créative (Hobson) et le codage prédictif (Friston) —
mais aucun framework formel unifié n'intègre ces quatre piliers en
opérations composables et indépendantes du substrat.

Nous introduisons **dreamOfkiki**, un framework formel à axiomes
exécutables. **La compositionnalité DR-2** est ici prouvée comme
théorème de semi-groupe engendré sur quatre opérations canoniques
(fermeture + additivité du budget + composition fonctionnelle +
associativité ; preuve dans
`docs/proofs/dr2-compositionality.md`). Les axiomes restants sont
DR-0 redevabilité, DR-1 conservation épisodique, DR-3 indépendance
du substrat via un Critère de Conformité exécutable (typage des
signatures ∧ tests de propriété axiomatiques ∧ applicabilité des
invariants BLOCKING), et DR-4 inclusion en chaîne des profils
(preuve dans `docs/proofs/dr4-profile-inclusion.md`). Le framework
définit 8 primitives typées (entrées α, β, γ, δ ; 4 canaux de
sortie), 4 opérations canoniques (replay, downscale, restructure,
recombine) et une ontologie de Dream Episode en quintuplet.

**Conformité inter-substrats (§5.6).** Deux substrats indépendants
— une implémentation *kiki-oniric* à base de gradients MLX et un
*E-SNN thalamocortical* numpy-LIF — satisfont tous deux les trois
conditions du Critère de Conformité, avec 9 tests de propriété
axiomatique et d'application d'invariants passants sur chacun
(porte G7 LOCKED, voir `docs/milestones/g7-esnn-conformance.md`).
Cela étaye l'assertion d'indépendance du substrat à la portée du
Paper 1, plutôt que de la différer au cycle 2.

**Validation du pipeline (§7).** Nous validons la chaîne de mesure
et statistique (quatre tests pré-enregistrés sous correction de
Bonferroni), l'injection de fautes sur les gardes des invariants
BLOCKING S1–S3, et le déterminisme de reproductibilité R1 sur des
tuples appariés du registre. Nous rapportons en complément des
mesures de portabilité inter-substrats issues du projet sœur
Nerve-WML (§7.4) qui confirment indépendamment l'indépendance du
substrat sur les tâches linéairement séparables (écart < 5 %) et
divulguent l'écart en régime non linéaire (12,1 %) comme résultat
négatif honnête. **Aucune décision d'hypothèse d'apprentissage
continu sur H1–H4 n'est annoncée dans cet article** ; l'évaluation
empirique de H1–H4 est rapportée dans le Paper 2.

Le pré-enregistrement est verrouillé sur l'Open Science Framework
(DOI 10.17605/OSF.IO/Q6JYN). L'ensemble du code, des spécifications
et du pré-enregistrement est ouvert sous MIT / CC-BY-4.0 ; les
implémentations de référence et les suites de tests sont disponibles
sur `github.com/c-geni-al/dream-of-kiki`.

---

## Notes pour révision

- v0.2 retargeté PLOS Computational Biology (pas de limite stricte
  de mots ; corps autodiscipliné à 8 000–10 000 mots)
- DOI OSF inséré (10.17605/OSF.IO/Q6JYN) ; DOI Zenodo pour les
  artefacts code+modèle pendant (post-G5)
- Resserrer le résumé à ≤300 mots si la revue le demande
  (actuellement ~410, mais PLOS CB tolère plus large que Nature HB)
