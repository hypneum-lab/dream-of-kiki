# Amendement : falsification empirique de DR-2

- **Date** : 2026-04-21
- **Statut** : Adopté — Option B (précondition affaiblie), 2026-04-21
- **Cible** : `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`
  §6.2 (DR-2 compositionnalité)
- **Preuve** : `tests/conformance/axioms/test_dr2_compositionality_empirical.py`

## Contexte

DR-2, tel qu'énoncé dans la spec de conception du framework C §6.2, affirme
la compositionnalité pour tout ordonnancement des opérations :

```
∀ op_1, op_2 ∈ Op,
  op_2 ∘ op_1 ∈ Op
  ∧ budget(op_2 ∘ op_1) = budget(op_1) + budget(op_2)
  ∧ effect(op_2 ∘ op_1, s) = effect(op_2, effect(op_1, s))
```

avec la réserve *« La commutativité n'est PAS affirmée »*. DR-2 avait été
signalé comme un **axiome de travail non prouvé**, avec DR-2' (compositionnalité
en ordre canonique) adopté comme repli opérationnel (voir §6.2 addendum et
`test_dr2_prime_canonical_order.py`).

## Résultat empirique

Les tests de propriétés par Hypothesis sur toutes les 24 permutations des
4 opérations canoniques `{REPLAY, DOWNSCALE, RESTRUCTURE, RECOMBINE}`
identifient une classe falsifiante sur le substrat à poids réels courant :

> Toute permutation où **RESTRUCTURE précède REPLAY** (à n'importe quelle
> distance) lève `ValueError` depuis MLX `addmm` lors de l'étape REPLAY.
> La mutation de topologie effectuée par RESTRUCTURE (échange de couche)
> laisse le MLP non appelable avec la forme d'entrée canonique `(2, 4)`
> consommée par REPLAY.

12 des 24 permutations appartiennent à cette classe ; les 12 autres (REPLAY
avant RESTRUCTURE, ordre canonique inclus) sont sûres du point de vue de la
clôture. Voir `tests/conformance/axioms/test_dr2_compositionality_empirical.py`
pour l'énumération complète (Hypothesis `@parametrize`-matérialisé plus une
sonde générique).

## Interprétation

Ceci ne **contredit pas** la spec — la spec signale explicitement DR-2 comme
non prouvé. Le résultat empirique fait passer le signalement de *« axiome de
travail non prouvé »* à *« empiriquement falsifié sur le substrat à poids
réels courant sous le schéma d'entrée canonique »*.

Périmètre de la falsification :

1. **Dépendant du substrat.** Seul le substrat MLX à poids réels a été testé.
   Un substrat où RESTRUCTURE préserve la compatibilité de signature avec les
   entrées de REPLAY peut ne pas présenter cette défaillance. DR-3
   (indépendance du substrat) garde contre la sur-généralisation du résultat.
2. **Dépendant du schéma d'entrée.** La forme d'entrée canonique `(2, 4)` est
   ce qui déclenche la discordance de forme. Un schéma d'entrée différent
   pourrait en principe éviter le problème pour des mutations de topologie
   spécifiques.
3. **La clôture est le mode de défaillance**, pas l'additivité de budget ni
   l'enchaînement d'effets. La clôture est le premier conjoint de la
   conjonction de DR-2 ; sa défaillance réfute l'ensemble de l'affirmation
   telle qu'énoncée.

## Amendement proposé au §6.2

Trois options, ordonnées par conservatisme :

### Option A — Annotation uniquement

Ajouter une note de bas de page au §6.2 citant le résultat empirique :

> **Note** (2026-04-21). La clause de clôture de DR-2 a été montrée
> empiriquement comme défaillante pour le substrat à poids réels dans la
> classe des permutations où RESTRUCTURE précède REPLAY (voir
> `tests/conformance/axioms/test_dr2_compositionality_empirical.py`).
> DR-2' reste le contrat opérationnel.

### Option B — Affaiblir DR-2 à la classe sûre observée [ADOPTÉE 2026-04-21]

Redéfinir DR-2 comme compositionnalité pour le sous-ensemble de `Op × Op`
où REPLAY ne suit pas RESTRUCTURE. Cela introduit une précondition mais
préserve une affirmation post-canonique non triviale.

### Option C — Rétrograder DR-2, promouvoir DR-2'

Retirer DR-2 de la liste des axiomes ; promouvoir DR-2' en DR-2 (l'ordre
canonique est *le* contrat de compositionnalité). Nécessite un commit errata
et un bump DualVer sur l'axe formel.

## Recommandation

**Option B adoptée le 2026-04-21** (bump FC-PATCH C-v0.7.0 → C-v0.7.1). Le
DR-2 borné par précondition est empiriquement validé par les 12 permutations
sûres du point de vue de la clôture : les 11 non canoniques sondées par la
stratégie Hypothesis dans `test_dr2_compositionality_empirical.py`, plus
l'ordre canonique couvert par `test_dr2_prime_canonical_order.py` (voir
*Correction (2026-10-25)* ci-dessous). L'Option C (rétrograder entièrement
DR-2) reste disponible pour le Papier 2 si les résultats du survey de
substrats le justifient.

## Implication DualVer

- **Option A** : pas de bump sur l'axe formel (annotation uniquement). Axe
  empirique inchangé (aucune porte franchie).
- **Option B** : bump de l'axe formel `+UNSTABLE` jusqu'à ce que la forme
  affaiblie soit re-testée à l'échelle des substrats.
- **Option C** : bump de l'axe formel `+UNSTABLE` avec un commit errata.

## Correction (2026-10-25)

Les révisions antérieures de cet amendement donnaient deux comptes différents
pour les classes de permutations : « 12 des 24 » (*Résultat empirique*,
*Références*) et « 11 permutations réussies » (*Recommandation*). Les chiffres
sont réconciliés ci-dessous à partir des propres artéfacts du dépôt, sans
rien ré-exécuter :

- **12 / 24 permutations lèvent `ValueError`** (RESTRUCTURE strictement avant
  REPLAY). Pour chaque permutation des quatre opérations, exactement l'une
  de RESTRUCTURE-avant-REPLAY ou REPLAY-avant-RESTRUCTURE s'applique, donc
  les deux classes ont la même taille (12 et 12). Le module de test énumère
  la classe falsifiante dans `_enumerate_closure_falsifiers()` (complément
  de `DR2.predicate`), et les enregistrements de run commités portent le même
  compte : `STATUS.md` → *Test suite* (« 693 passing, 12 xfailed », audité
  2026-05-10) et
  `docs/milestones/r1-cross-machine-m5-vs-m1-2026-05-20.json`
  (`functional_verdict.pytest_xfailed` = 12 sur M5, M3 Ultra et M1 Max).
- **12 / 24 permutations sont sûres du point de vue de la clôture.** Le « 11 »
  venait du plan d'implémentation
  (`docs/superpowers/plans/2026-04-21-dr2-option-b-weakening.md`,
  « ≥ 11 réussies + 12 xfailed ») : il ne compte que les permutations sûres
  non canoniques exercées par la stratégie Hypothesis, qui exclut l'ordre
  canonique car DR-2' le couvre déjà
  (`test_dr2_prime_canonical_order.py`). 11 + 1 = 12.

Chiffres retenus : **12 falsifiées / 12 sûres du point de vue de la clôture
sur 24**, cohérents avec `docs/proofs/dr2-compositionality.md`,
`docs/axioms/AXIOMS.md`, la spec framework C §6.2 et le docstring du module
de test. Le Papier 1 §4.5 nomme la classe exclue sans un compte, donc aucun
errata n'est nécessaire là. Aucun artéfact JSON dédié n'enregistre ce compte ;
en produire un nécessite un run de test et est laissé à une phase ultérieure.
Issue de suivi : hypneum-lab/dream-of-kiki#54.

## Références

- `tests/conformance/axioms/test_dr2_compositionality_empirical.py`
  — les 12 cas xfailing et les 12 cas réussis sûres du point de vue de la
  clôture (11 non canoniques + l'ordre canonique).
- `tests/conformance/axioms/test_dr2_prime_canonical_order.py`
  — DR-2' (repli, encore applicable).
- `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §6.2
  — énoncé canonique de DR-2.
- `docs/axioms/AXIOMS.md` — référence compilée des axiomes (2026-04-21).
