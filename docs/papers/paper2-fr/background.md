<!--
SPDX-License-Identifier: CC-BY-4.0
Authorship byline : dreamOfkiki project contributors
License : Creative Commons Attribution 4.0 International (CC-BY-4.0)
-->

# §3 Contexte (Article 2, brouillon C2.16)

**Signataires** : *contributeurs du projet dreamOfkiki*
**Licence** : CC-BY-4.0

**Cible de longueur** : ~0,5 page markdown (≈ 500 mots)

---

## 3.1 L'Article 1 en un paragraphe

L'Article 1 a introduit le Framework C : 8 primitives typées
(entrées α, β, γ, δ + 4 canaux de sortie), 4 opérations
oniriques canoniques (replay, downscale, restructure,
recombine) formant un semi-groupe libre sous la
compositionnalité DR-2, une ontologie de Dream Episode en
quintuplet, et les axiomes DR-0 (redevabilité), DR-1
(conservation épisodique), DR-2 (compositionnalité), DR-3
(indépendance du substrat via le Critère de Conformité),
DR-4 (inclusion de chaîne de profils). Les lecteurs non
familiers avec le framework devraient lire l'Article 1 §4
(axiomes + preuves) et §5 (implémentation de référence)
avant le §4 de cet article.

## 3.2 Quatre piliers, brièvement

L'Article 2 hérite du mapping quatre-piliers de l'Article 1
§3 :

- **Pilier A — Walker / Stickgold** : transfert
  épisodique-vers-sémantique via replay
  [@walker2004sleep; @stickgold2005sleep ; opérationnalisé
  comme `replay` dans notre framework].
- **Pilier B — Tononi SHY** : homéostasie synaptique /
  downscaling des poids [@tononi2014sleep ;
  opérationnalisé comme `downscale`, commutatif mais
  non-idempotent].
- **Pilier C — Hobson / Solms** : recombinaison créative en
  rêve REM [@hobson2009rem; @solms2021revising ;
  opérationnalisé comme `recombine`, VAE-light en cycle 1,
  VAE complet en cycle 2].
- **Pilier D — Friston FEP** : minimisation de l'énergie libre
  [@friston2010free ; opérationnalisé comme `restructure` sous
  la garde topologie S3].

L'Article 2 ne ré-argumente pas ces mappings. Il les consomme.

## 3.3 Critère de Conformité DR-3, brièvement

DR-3 spécifie les trois conditions (C1 typage de signature via
Protocols typés, C2 tests de propriétés d'axiomes passent, C3
invariants BLOCKING applicables) qu'un substrat candidat doit
satisfaire pour hériter des garanties du framework. Le
critère est exécutable : il est expédié comme batterie pytest
réutilisable dans `tests/conformance/` paramétrée par le type
d'état du substrat. Le §4 de cet article exerce DR-3 sur deux
substrats + un placeholder.

## 3.4 Travaux antérieurs de réplication multi-substrats en apprentissage continu

La réplication inter-substrats des mécanismes d'apprentissage
continu est sous-représentée dans l'art antérieur.
[@vandeven2020brain] démontrent le replay inspiré du cerveau
sur des réseaux neuronaux artificiels mais sur une seule
architecture. [@kirkpatrick2017overcoming] introduisent EWC
comme régularisateur adjacent SHY mais ne poursuivent pas
l'indépendance du substrat. [@rebuffi2017icarl] et
[@shin2017continual] sont spécifiques à l'architecture. Aucun
travail antérieur, à notre connaissance, n'expédie une
**suite de tests de conformité réutilisable** liant un
framework formel à des substrats opérationnels à travers des
modèles de matériel qualitativement distincts (tenseur dense
MLX vs LIF spiking).

C'est la contribution d'ingénierie distinctive de l'Article 2.

## 3.5 Cadrage par substitution synthétique, brièvement

L'Article 2 est un **article de méthodologie / réplication**,
pas un article de nouvelle revendication empirique. Ce
cadrage est non négociable : les deux lignes de substrats
partagent le même prédicteur mock Python (§6.4). Une
réplication à prédicteur divergent sur données réelles est
une cible cycle-3. Les lecteurs qui élident l'étiquette de
substitution synthétique lisent la portée de l'article de
travers. L'introduction §2 et la méthodologie §6.4
ré-énoncent ce cadrage avec force ; le §3 est le dernier
endroit où un lecteur peut sauter la précaution avant les
tables de résultats du §7.

---

## Notes pour révision

- Insérer les citations bibtex propres une fois
  `references.bib` étendu depuis le stub de l'Article 1.
- Resserrer à ≤ 400 mots à la passe de pré-soumission
  NeurIPS.
- Référencer §4 et §6 une fois le brouillon complet mis en
  page dans le style-file NeurIPS.
