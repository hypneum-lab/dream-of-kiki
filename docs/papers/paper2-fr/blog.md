# dreamOfkiki, acte II : quand la théorie rencontre le silicium

*De l'axiome à l'implémentation — validation inter-substrats d'un
framework de consolidation par le rêve*

**Public visé** : chercheurs et techniciens curieux.
L'article suppose que le Paper 1 (framework théorique) a été lu,
ou au moins que son existence est connue. On suppose aussi une
familiarité minimale avec la notion de substrat matériel (CPU,
GPU, neuromorphique).

---

## Un article, deux missions

Le programme dreamOfkiki produit deux articles complémentaires.
L'**Article 1**, déjà présenté ici, fait le travail théorique :
définir un framework formel de consolidation mnésique inspiré du
sommeil humain, prouver les axiomes DR-0 à DR-4, construire le
Critère de Conformité comme pont entre théorie et pratique. Il
vise *Nature Human Behaviour*, *PLoS Computational Biology* ou
*Cognitive Science* — des revues où cogniticiens et théoriciens
dialoguent.

L'**Article 2**, dont nous parlons ici, fait autre chose. Il prend
la théorie et lui demande : *est-ce que ça marche, concrètement,
sur du matériel réel, de manière reproductible, vérifiable par un
lecteur qui ne veut pas re-dériver la théorie ?* C'est un article
d'ingénierie au sens noble du terme. Il cible NeurIPS, ICML ou
TMLR — les arènes où les ingénieurs ML et les chercheurs systèmes
débattent de ce qui tient la route en production.

Les deux articles partagent un framework commun, un vocabulaire
commun, des hypothèses pré-enregistrées communes. Mais leur
angle d'attaque est diamétralement opposé : le premier prouve, le
second mesure.

## L'enjeu : sortir du laboratoire d'un seul modèle

La plupart des papers en apprentissage continu reportent des
résultats sur *une* architecture, *un* jeu de données, *un* type
de matériel. Quand on lit qu'une méthode réduit l'oubli
catastrophique de 15 %, on sait rarement si ce chiffre tient en
dehors du setup exact du papier.

dreamOfkiki prend le problème par l'autre bout. Le **framework C**
(le framework formel défini dans l'Article 1 ;
`docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`) est
pensé indépendant du substrat — c'est littéralement inscrit dans
l'axiome DR-3, *indépendance du substrat*. Mais une
revendication d'indépendance du substrat n'a de valeur que si
elle est vérifiée sur au moins deux substrats qualitativement
différents. Un seul substrat, c'est une hypothèse. Deux
substrats, c'est un début de preuve.

L'Article 2 valide donc le framework sur deux substrats :

**Substrat 1 — kiki-oniric sur MLX (Apple Silicon).** C'est
l'implémentation de référence, développée pendant le cycle 1.
Elle tourne sur des Mac Studio M3 Ultra, avec le backend MLX
d'Apple optimisé pour les puces unified memory. Déterministe,
rapide, open-source. C'est le substrat « logiciel » — des
opérations numpy ou MLX qui approximent les opérations
théoriques du framework.

**Substrat 2 — E-SNN sur Loihi-2 (Intel neuromorphique).** C'est
l'implémentation exotique, qui fait le vrai test d'indépendance.
Les Spiking Neural Networks évolutifs (E-SNN) tournent sur
Loihi-2, la puce neuromorphique d'Intel. Ici, il n'y a plus de
floating-point, plus de matrices denses : on est dans un régime
de spikes discrets et de synapses qui se modulent localement. Si
le framework tient à travers ce gap, c'est que l'indépendance
du substrat n'est pas qu'une clause de style.

## Le Critère de Conformité, mode d'emploi

La clé opérationnelle, c'est le **Critère de Conformité** défini
dans l'Article 1 et re-détaillé ici. Pour qu'un substrat soit
qualifié d'instanciation valide du framework, il doit passer
trois tests :

1. **Typage de signature** — le substrat implémente les huit
   primitives du framework (α, β, γ, δ en entrée, 1, 2, 3, 4 en
   sortie) avec les bonnes signatures typées (Python Protocol,
   TypeScript interfaces, ou équivalent).
2. **Tests de propriétés d'axiomes** — la suite de tests de
   propriétés de DR-0, DR-1 et DR-2 passe sur ce substrat, avec
   100 % de couverture sur les cas BLOCKING.
3. **Invariants BLOCKING applicables** — les invariants S1
   (non-régression), S2 (pas de NaN/Inf), S3 (topologie valide)
   et I1 (conservation épisodique) sont implémentés comme des
   checks runtime avec abort-on-violation.

C'est un critère **exécutable**. Pas une checklist narrative,
pas une prose de revue par les pairs. On lance la commande
`dream-harness conformance --substrate <S>` et on obtient un
verdict binaire. Les substrats MLX et E-SNN passent chacun ce
criblage dans l'article. D'autres substrats (transformer standard,
RWKV, state-space) sont laissés aux cycles 3+.

## L'ingénierie qui ne se voit pas dans les papiers

L'Article 2 est aussi l'occasion d'exposer le travail
d'ingénierie qui sous-tend la recherche reproductible et que les
papers théoriques minimisent habituellement :

- **Un protocole de swap atomique** (le remplacement d'un
  merge actif par un swap de worktree) qui garantit que les
  mises à jour issues du rêve ne corrompent jamais l'état de
  l'éveil. Ce protocole inclut trois gardes (S1, S2, S3) et un
  chemin de rollback trivial.
- **Un runtime de rêve concurrent** qui tourne en parallèle du
  processus éveil, avec swap de worktree pour les mises à jour
  paramétriques. L'API Future squelette en cycle 1 devient un
  vrai asyncio en cycle 2.
- **Un registre de runs déterministe** qui hache toutes les
  entrées (version du framework, profil, seed, commit_sha,
  version du benchmark) en un `run_id` SHA-256 bit-stable.
  C'est la contrainte R1 : deux exécutions avec les mêmes
  entrées produisent exactement les mêmes sorties, byte par
  byte, indéfiniment.
- **Un pipeline statistique pré-enregistré** (Welch pour
  l'hypothèse 1, TOST pour l'hypothèse 2, Jonckheere pour
  l'hypothèse 3, t une-échantillon pour l'hypothèse 4), avec
  correction de Bonferroni. Tout est pré-enregistré sur OSF
  avant collecte de données, ce qui élimine le p-hacking par
  construction.

Ces détails sont ennuyeux à lire pour un cogniticien, mais
vitaux pour un ingénieur ML qui voudrait reproduire ou étendre
le travail. L'Article 1 les passe rapidement. L'Article 2 les
met au centre.

## La matrice de validation inter-substrats

La pièce maîtresse de l'Article 2 est une **matrice de
validation inter-substrats**. En ligne, les trois profils du
framework (P_min, P_equ, P_max — ordonnés par capacité
croissante). En colonne, les huit métriques du protocole
d'évaluation (taux d'oubli, exactitude moyenne, alignement RSA
IRMf, ratio FLOPs, gain hors-ligne, énergie par épisode, qualité
de recombinaison, découverte de structure). Et dans chaque
cellule, les résultats mesurés sur MLX et sur E-SNN.

Ce qu'on cherche à voir dans cette matrice : que les effets
monotones (par exemple, P_equ améliore M1.a vs baseline sans-rêve)
se retrouvent sur les deux substrats, même si les valeurs
absolues diffèrent. Le framework ne prédit pas que MLX et E-SNN
donnent les mêmes chiffres — MLX sera probablement plus rapide,
E-SNN plus efficace en énergie — mais il prédit que les
tendances qualitatives sont conservées. Si la chaîne de profils
se comporte différemment entre les deux substrats, le framework
est en difficulté.

## Trade-offs : vitesse vs énergie

Un des résultats attendus de la comparaison MLX / E-SNN est la
caractérisation explicite des arbitrages d'ingénierie. MLX
excelle en vitesse de développement, en outillage, en densité
écosystémique ; E-SNN excelle (théoriquement) en efficacité
énergétique par opération. Les deux substrats ne sont pas
interchangeables — ils occupent des points différents sur le
Pareto front vitesse/énergie.

L'Article 2 rend cet arbitrage quantitatif. Il mesure
`FLOPs-équivalent wall-clock` sur MLX et `énergie par épisode`
(proxy sur MLX, natif sur Loihi-2). Il montre (ou réfute) que
E-SNN produit la même qualité de consolidation qu'MLX à une
fraction de l'énergie. Si ce résultat tient, la conclusion
engineering est forte : le framework n'est pas juste
compatible avec le neuromorphique, il *bénéficie* du
neuromorphique pour les déploiements edge où le budget
énergétique compte.

## Et la suite ?

L'Article 2 ne clôture pas le sujet — il l'ouvre. La section
Travaux futurs liste trois directions :

1. **Substrats additionnels** : tester le framework sur des
   transformers standard, sur RWKV, sur des state-space
   models. Chaque nouveau substrat est une validation
   supplémentaire du Critère de Conformité.
2. **Sélection dynamique de profil** : aujourd'hui, le profil
   (P_min, P_equ, P_max) est fixé à l'initialisation. Demain,
   un runtime pourrait basculer dynamiquement selon la charge
   mémoire, le budget énergétique disponible, ou des signaux
   internes (surcharge de buffer β, détection de drift).
3. **Patterns de déploiement production** : documenter comment
   un praticien déploierait un système avec consolidation par
   le rêve en production. Quelle cadence pour les épisodes de
   rêve ? Quels guardrails opérationnels ? Quels monitoring
   dashboards ?

## Pour les lecteurs francophones

L'Article 2, comme l'Article 1, sera publié en anglais dans les
actes de conférence. La version française est maintenue en
parallèle, déposée sur HAL (Hyper Articles en Ligne) pour
l'archive académique française, et synthétisée en billet de
blog (ce texte) sur le site L'Electron Rare. La politique de
maintenance parallèle est documentée dans le `CONTRIBUTING.md`
du dépôt : toute modification de la version anglaise déclenche
une mise à jour française correspondante.

Le code, lui, reste en anglais — c'est la langue de travail
standard du développement logiciel, et le framework est pensé
pour être réutilisé par la communauté internationale. Ce
bilinguisme asymétrique (code anglais, narration scientifique
bilingue) est assumé. Il reflète la réalité d'un projet porté
depuis la France mais inséré dans l'écosystème de recherche
mondial.

---

*dreamOfkiki est un programme de recherche open-source mené
par L'Electron Rare. Le code, les spécifications, les preuves
et les brouillons d'articles sont publics sur
github.com/electron-rare/dream-of-kiki. Le tableau de bord
public tourne sur dream.saillant.cc. Les hypothèses sont
pré-enregistrées sur OSF, les artefacts archivés sur Zenodo.*
