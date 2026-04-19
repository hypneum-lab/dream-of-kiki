<!--
SPDX-License-Identifier: CC-BY-4.0
Authorship byline : dreamOfkiki project contributors
License : Creative Commons Attribution 4.0 International (CC-BY-4.0)
-->

# §9 Travaux futurs — Cycle 3 (Article 2, brouillon C2.16)

**Signataires** : *contributeurs du projet dreamOfkiki*
**Licence** : CC-BY-4.0

**Cible de longueur** : ~0,5 page markdown (≈ 500 mots)

---

## 9.1 Terminer la Phase 3 — réplication à prédicteur divergent

Le livrable cycle-3 de plus haute priorité unique est de
remplacer le prédicteur mock partagé (§6.4) par une
**inférence spécifique au substrat** :

- Forward pass MLX lisant l'état MLX de `mlx_kiki_oniric`,
  produisant de vraies prédictions par item.
- Read-out de taux de spikes LIFState lisant l'état E-SNN de
  `esnn_thalamocortical`, produisant de vraies prédictions
  par item.

Avec un prédicteur divergent, les verdicts inter-substrats
H1-H4 (§7.2) deviennent **informatifs** : l'accord n'est plus
trivial par construction ; le désaccord indiquerait un
véritable signal dépendant du substrat. L'une ou l'autre
issue est publiable — l'accord renforce empiriquement DR-3,
le désaccord contraint la revendication d'indépendance du
substrat du framework.

Atterrir cet item ferme le qualifieur `+PARTIAL` du tag
DualVer et permet un bump `C-v0.7.0+STABLE`
(`docs/milestones/g9-cycle2-publication.md` § propositions
DualVer).

## 9.2 Mapping matériel Loihi-2 réel

Si le partenariat Intel NRC se matérialise (action externe,
suivi dans `docs/milestones/g9-cycle2-publication.md` §
actions externes utilisateur), le squelette numpy LIF peut
être porté vers Loihi-2. Le port devrait préserver les
signatures des factories d'op (de sorte que C1 typage de
signature reste PASS sans churn de code) tout en échangeant
le simulateur LIFState contre l'exécution matérielle
neuromorphique.

Chemins de fallback, classés par probabilité :

- **SpiNNaker** via Norse / PyNN — substrat spiking compatible
  logiciel, déployable sans partenariat Intel.
- **Simulation Lava SDK** de Loihi-2 — toujours synthétique
  mais bit-compatible avec le modèle d'exécution réel de la
  puce.
- **Statu quo** — garder le squelette numpy LIF et repousser
  les revendications neuromorphiques jusqu'à ce que l'accès
  matériel soit confirmé.

## 9.3 Cohorte IRMf réelle (pivot T-Col)

L'outreach T-Col (§8.4) cible un partenariat de laboratoire
produisant des données IRMf linguistiques contrôlées-par-
tâche. Une cohorte même de 20 participants sur un paradigme
pré-enregistré serait suffisante pour exécuter un alignement
de type RSA du snapshot γ-sémantique du pipeline de rêve
contre les réponses BOLD, renforçant H3 (alignement
représentationnel monotone) qui échoue actuellement à
Bonferroni en raison de la dégénérescence du prédicteur mock
(§7.2).

Les délais cycle-3 pour l'acquisition de données IRMf sont
longs (IRB, scanning, prétraitement, QC) — cet item peut
glisser au cycle 4 si la formalisation du partenariat est
retardée au-delà du T2 2026.

## 9.4 Émergence de l'Article 3

Un Article 3 — comparaison de performance empirique
inter-substrats sur des données réelles — devient plausible
**seulement si** 9.1 (prédicteur divergent) et au moins l'un
de 9.2 (Loihi-2) ou 9.3 (IRMf) atterrissent avec des données
solides. En attendant, l'Article 3 est un placeholder
provisoire (`docs/papers/paper3/outline.md`) et non un
engagement.

Revues possibles pour l'Article 3, si et quand il émerge :

- **Nature Communications** ou **PNAS** (si un signal IRMf
  réel porte une revendication spécifique à la
  consolidation).
- **Neuromorphic Computing and Engineering** (IOP) ou
  **Frontiers in Neuroscience** (si l'exécution Loihi-2 porte
  une revendication d'énergie matérielle).
- Suivi **NeurIPS** (si la réplication à prédicteur divergent
  montre des deltas de performance spécifiques au substrat
  sur les benchmarks ML).

## 9.5 Séquencement Phase 4 avec acceptation Article 1

La soumission arXiv de l'Article 2 n'est **pas bloquée** sur
l'acceptation de l'Article 1, mais le séquencement
NeurIPS / TMLR l'est. Deux stratégies de séquencement
viables :

- **Préprint-d'abord** : soumettre arXiv Article 2 en même
  temps que le préprint Article 1, citer l'ID arXiv en
  cross-document. La soumission NeurIPS / TMLR suit une fois
  que l'Article 1 est au moins en révision.
- **Pivot B** : si l'acceptation de l'Article 1 est retardée
  > 6 mois, re-auto-contenir le récapitulatif de framework de
  l'Article 2 (§3 contexte + §4 préliminaire de conformité),
  ajouter un disclaimer explicite "Article 1 en révision", et
  soumettre NeurIPS / TMLR en autonome.

La contingence Pivot B est porteuse pour la revendication de
clôture cycle-2 que l'Article 2 *est* prêt à soumettre même si
l'Article 1 est retardé. L'assemblage du brouillon complet (ce
commit C2.16) est le dernier livrable d'ingénierie nécessaire
pour exécuter Pivot B.

---

## Notes pour révision

- Ré-ordonner 9.1..9.5 par priorité une fois le statut du
  partenariat Intel NRC confirmé.
- Insérer les références Gantt cycle-3 concrètes une fois
  `docs/superpowers/plans/2026-04-XX-dreamofkiki-cycle3.md`
  livré.
- Resserrer à ≤ 400 mots à la passe de pré-soumission NeurIPS.
