<!--
SPDX-License-Identifier: CC-BY-4.0
Signataires : Saillant, Clément
Licence : Creative Commons Attribution 4.0 International (CC-BY-4.0)
-->

# Introduction (Paper 1, brouillon)

**Signataires** : *Saillant, Clément*
**Licence** : CC-BY-4.0

**Longueur cible** : ~1,5 page markdown (≈ 1200 mots)

---

## 1. L'oubli catastrophique et la lacune de consolidation

Les systèmes cognitifs artificiels modernes excellent dans
l'apprentissage mono-tâche, mais se dégradent rapidement lorsqu'ils
sont entraînés séquentiellement sur plusieurs tâches — un phénomène
connu sous le nom d'**oubli catastrophique** [McCloskey & Cohen
1989, French 1999]. Malgré deux décennies de stratégies
d'atténuation (elastic weight consolidation [Kirkpatrick et al.
2017], réactivation générative [Shin et al. 2017], mémoire par
répétition [Rebuffi et al. 2017]), le champ manque toujours d'une
*théorie unifiée* expliquant pourquoi ces mécanismes fonctionnent
et quand ils doivent se composer. Les recensions récentes de la
littérature de l'apprentissage continuel identifient la
réactivation latente comme le mécanisme de consensus émergent à
travers cinq années de méthodes [van de Ven et al. 2024], sans
toutefois en dériver le consensus à partir de premiers principes.

La cognition biologique résout ce problème pendant le **sommeil**.
La réactivation hippocampique durant le NREM, la régulation à la
baisse synaptique, la restructuration prédictive des représentations
corticales et la recombinaison créative pendant le REM forment
ensemble un pipeline de consolidation multi-étapes [Diekelmann &
Born 2010, Tononi & Cirelli 2014, Robertson 2025 NRN]. Pourtant, les travaux existants
en IA n'ont intégré que des fragments de cette biologie, en se
concentrant généralement sur un mécanisme unique (p. ex. la
réactivation seule) sans théorie raisonnée de la manière dont les
mécanismes interagissent.

## 2. Quatre piliers de la consolidation mnésique basée sur le rêve

Nous identifions quatre piliers théoriques que tout framework
complet de consolidation en IA inspirée du rêve doit adresser :

- **A — consolidation Walker/Stickgold** : transfert épisodique-
  vers-sémantique via la réactivation [Walker & Stickgold 2004,
  Stickgold 2005].
- **B — SHY de Tononi** : homéostasie synaptique renormalisant les
  poids pendant le sommeil [Tononi & Cirelli 2014].
- **C — rêve créatif Hobson/Solms** : recombinaison et abstraction
  pendant le REM [Hobson 2009, Solms 2021].
- **D — FEP de Friston** : minimisation de l'énergie libre comme
  théorie unificatrice de l'inférence et de la consolidation
  [Friston 2010].

Les travaux antérieurs en IA ont implémenté A (van de Ven et al.
2020), B (Kirkpatrick et al. 2017 comme régularisation adjacente à
SHY) et des éléments de D (Rao & Ballard 1999, Whittington & Bogacz
2017), mais **aucun travail n'a formalisé la manière dont les
quatre piliers se composent** de façon indépendante du substrat,
propice à l'ablation et à la preuve.

## 3. La lacune compositionnelle

Pourquoi la composition importe-t-elle ? Empiriquement, l'ordre
dans lequel s'appliquent les opérations de consolidation modifie
l'état cognitif résultant — la réactivation avant la régulation à
la baisse préserve la spécificité épisodique, tandis que la
régulation à la baisse avant la restructuration peut effacer les
représentations mêmes que la restructuration est censée affiner.
Notre analyse (`docs/proofs/op-pair-analysis.md`) énumère les 16
paires d'opérations et constate que 12 paires croisées sont
non-commutatives, confirmant que *l'ordre fait partie du framework*
et non d'un détail d'implémentation.

Un framework formel digne de ce nom doit donc (i) spécifier les
opérations comme primitives composables à types bien définis, (ii)
expliciter quelles compositions sont valides, (iii) fournir une
théorie **exécutable** que tout substrat conforme peut implémenter
et (iv) supporter l'ablation empirique comparant différents
profils d'opérations. Aucun des travaux antérieurs ne satisfait
les quatre critères.

Trois communautés de recherche ont convergé sur le même patron
architectural depuis des directions indépendantes. L'**apprentissage
continuel** s'est arrêté sur la réactivation latente comme
atténuation dominante de l'oubli catastrophique [van de Ven et al.
2024]. La **recherche industrielle sur les LLM** réinvente la
construction sous forme de métaphore : le *sleep-time compute* de
Berkeley rapporte +13–18 % d'exactitude et une amortisation du
calcul d'un facteur 5× via un pré-traitement hors ligne du
contexte, mais sans référence explicite au sommeil biologique
[Berkeley 2025 sleep-time compute] ; les *Titans* de Google
Research introduisent un module de mémoire neuronale appris au
moment de l'inférence jusqu'à 2 M tokens, sans ancrage rigoureux
dans la théorie de la consolidation [Behrouz et al. 2024]. Les
**propositions académiques concurrentes** commencent à combler
cet écart — *Wake-Sleep Consolidated Learning* [Alfarano et al.
2024] est, à notre connaissance, l'analogue dual-phase NREM/REM
publié le plus proche, et la proposition contemporaine *Language
Models Need Sleep* couple une distillation ascendante par RL avec
un oubli intentionnel [Behrouz et al. 2026]. En affinage continu
de LLM, rejouer les propres sorties du modèle courant sur des
invites passées réduit l'oubli par rapport à un replay classique
réglé [Chen et al. 2026], et l'auto-distillation a été proposée
comme substitut de replay [Shenfeld et al. 2026], bien que
l'auto-distillation on-policy puisse aussi amplifier la dérive et
s'effondrer en post-entraînement continu [Wang et al. 2026]. Les
substrats
neuromorphiques montrent la même convergence : CLP-SNN sur Loihi 2
rapporte un facteur 70× de vitesse et 5 600× d'efficacité
énergétique sur des baselines GPU pour la réactivation hors ligne
[Hajizada et al. 2025]. Trois communautés, une même forme
architecturale, et **aucun travail publié 2024–2026 ne projette
formellement la triade SO–fuseau–ripple sur un invariant
computationnel indépendant du substrat** : la neuroscience tient
le mécanisme, l'IA tient la fonction (réactivation → non-oubli),
aucune ne tient encore le pont axiomatique. Le Framework C se
positionne sur ce pont, avec les quatre piliers ci-dessus comme
ancrages conceptuels, l'analogue empirique le plus proche
[Alfarano et al. 2024] retenu comme comparateur d'ablation
primaire du Paper 2, et les propositions concurrentes [Behrouz
et al. 2026 ; Berkeley 2025 sleep-time compute] lues comme
corroboration indépendante du besoin sous-jacent plutôt que
comme antériorité sur la contribution formelle.

## 4. Feuille de route des contributions

Dans cet article, nous présentons **dreamOfkiki**, le premier
framework formel pour la consolidation mnésique basée sur le rêve
dans les systèmes cognitifs artificiels, avec les contributions
suivantes :

1. **Framework C-v0.5.0+STABLE** : 8 primitives typées, 4 opérations
   canoniques formant un semi-groupe non commutatif (générateurs sémantiquement indépendants), 4 OutputChannels,
   ontologie de Dream Episode en quintuplet, axiomes DR-0..DR-4 avec
   Critère de Conformité exécutable (§4). Les éléments 2–4 ci-
   dessous sont rapportés dans le Paper 2 (compagnon empirique) ;
   le Paper 1 se limite aux contributions formelles et à la feuille
   de route de conformité.
2. **Feuille de route** vers la généralisation multi-substrats
   (substrats supplémentaires au-delà de l'implémentation de
   référence du cycle 1) et vers l'alignement représentationnel
   IRMf réel (partenariat de laboratoire poursuivi via la campagne
   T-Col).

Le reste de l'article est organisé comme suit : §3 passe en revue
les quatre piliers en profondeur ; §4 développe le Framework
C-v0.5.0+STABLE avec axiomes et preuves ; §5 esquisse l'approche
de validation du Critère de Conformité (les résultats empiriques
spécifiques au substrat résident dans le Paper 2) ; §6 détaille la
méthodologie ; §7 rapporte les résultats de validation du pipeline
synthétique ; §8 discute les implications et limites ; §9 esquisse
les travaux futurs du cycle 2.

---

## Notes pour révision

- Insérer les citations bibtex appropriées une fois le gestionnaire
  de références configuré
- Renvoyer aux numéros de ligne §3-§9 une fois l'article complet
  mis en page dans le gabarit de la revue cible
- Resserrer à ≤1500 mots pour la discipline Nature Human
  Behaviour du corps principal de l'introduction (cible d'envoi
  primaire ; voir aussi seuils par section dans `outline.md`)
