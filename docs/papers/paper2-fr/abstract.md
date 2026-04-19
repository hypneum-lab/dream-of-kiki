<!--
SPDX-License-Identifier: CC-BY-4.0
Authorship byline : dreamOfkiki project contributors
License : Creative Commons Attribution 4.0 International (CC-BY-4.0)
-->

# Résumé (Article 2, brouillon C2.13)

**Signataires** : *contributeurs du projet dreamOfkiki*
**Licence** : CC-BY-4.0

**Cible en mots** : ≤ 200 mots

---

## Brouillon v0.1 (C2.13, 2026-04-19)

> **⚠️ Substitution synthétique — article de méthodologie /
> réplication.** Chaque table empirique, figure et verdict
> rapporté ci-dessous est produit par un prédicteur mock Python
> partagé entre deux enregistrements de substrats (MLX + squelette
> numpy LIF E-SNN). Aucun matériel Loihi-2 ni cohorte IRMf ne
> participe à aucun résultat. L'Article 2 documente le **pipeline
> de réplication inter-substrats**, pas une nouvelle revendication
> empirique ; les revendications biologiques ou neuromorphiques
> de tête sont repoussées au cycle 3.

L'Article 1 introduisait le Framework C-v0.6.0+PARTIAL (axiomes
DR-0..DR-4, 8 primitives typées, 4 opérations canoniques, 3
profils) et son Critère de Conformité exécutable (DR-3).
L'Article 2 pose la question d'ingénierie que l'Article 1 ne
pouvait résoudre : *un seul substrat conforme constitue-t-il
une preuve d'indépendance du substrat ?*

Nous répondons en répliquant la chaîne pré-enregistrée H1-H4 du
cycle 1 sur deux substrats enregistrés — MLX kiki-oniric sur
Apple Silicon et un squelette E-SNN thalamocortical numpy LIF
de taux de spikes — et en exhibant une matrice de Conformité
DR-3 à deux substrats (3 conditions × 2 substrats, tous PASS ;
une troisième ligne `hypothetical_cycle3` reste N/A). Sous
Bonferroni α = 0,0125, les deux substrats s'accordent sur
4 / 4 hypothèses (H1 rejeté, H2 rejeté, H3 échec à rejeter, H4
rejeté) — *substitution synthétique ; l'accord est trivial par
construction à prédicteur partagé et valide le pipeline, pas
l'efficacité empirique du framework sur des données réelles*.
Tout le code, le pré-enregistrement et les run_ids sont
MIT / CC-BY-4.0.

---

## Notes pour révision

- Nombre de mots actuel ≈ 200 ; resserrer si le format NeurIPS
  contraint à 150.
- Insérer le DOI OSF (hérité de l'Article 1) une fois verrouillé.
- Insérer le DOI Zenodo pour le bundle d'artefacts cycle-2 au
  tag de soumission.
- Chaque passage sur ce résumé doit préserver au moins un
  drapeau `substitution synthétique` dans le bloc de précaution.
