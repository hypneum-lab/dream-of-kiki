# Déclaration datée — contrôle de même norme pour H14-B (E1-small), 2026-11-02

**Rédigée le :** 2026-11-02 à 20:45 CET, à la demande du PI (2026-11-02, 20:38 CET).
**Préregistration visée :** E1-small, `docs/osf-prereg-e1-small-2026-10-25.md`
(gelée, SHA-256 `6d47d8b71829d884d3a18dda84cd21aa080f1090101d62a4bafe6bc493208463`,
commit de verrouillage `963bb24` ; `docs/osf-amendment-e1-small-2026-10-30.md:3-6`).
**Type :** déclaration d'une analyse exploratoire post hoc. Ce fichier ne modifie ni la
préregistration, ni l'amendement du 2026-10-30, ni l'amendement de réglage
(`docs/milestones/e1-small/e1-tuning-amendment-2026-10-29.json`), ni le code, ni l'analyse
enregistrée (`scripts/analyze_e1_cuda.py`). Aucune hypothèse, aucun estimand, aucun seuil,
aucun N n'est changé.
**Dépôt au moment de la rédaction :** `main` = `8a7e846`.

---

## 1. Ce qu'est le contrôle de même norme

### 1.1 Ce que fait DOWNSCALE dans le code gelé

- H14-B compare `p_equ` à `p_ctrl_vanilla_replay_tuned` ; les deux bras partagent la même
  configuration de rejeu réglée et la même suite de mises à jour, et ne diffèrent que par
  l'étape DOWNSCALE après le rejeu (`docs/osf-prereg-e1-small-2026-10-25.md:219-223`,
  `:779-782`).
- Le facteur est `SHRINK = 0.99`, non réglé (`scripts/e1_profiles.py:46`) ; il s'applique aux
  profils `p_equ` et `p_max` (`scripts/e1_profiles.py:139`).
- `apply_dream_ops` appelle `b.downscale(pr.SHRINK)` après le rejeu
  (`scripts/e1_harness.py:214-223`, appel en `:259`, rejeu en `:205`).
- `downscale` multiplie en place chaque paramètre flottant du modèle par le facteur
  (`scripts/pilot_cycle3_cuda.py:462-466`). Sur CUDA, les poids sont en bfloat16
  (`scripts/pilot_cycle3_cuda.py:228`).
- DOWNSCALE n'est pas compté dans le budget de calcul : opération élément par élément, sans
  passe avant (`docs/osf-prereg-e1-small-2026-10-25.md:704-706`).

### 1.2 Formule

Notations, pour une graine `s` :

- `θ₀` : poids de base restaurés au début de chaque cellule (`scripts/e1_harness.py:231`,
  `scripts/pilot_cycle3_cuda.py:392-397`) ;
- `θ_r(s)` : poids après le rejeu de la sélection (i), identique pour `p_equ` et
  `p_ctrl_vanilla_replay_tuned` (sélection figée dans
  `docs/milestones/e1-small/e1-tuning-amendment-2026-10-29.json:23`) ;
- `f = SHRINK = 0.99` ; `P` = l'ensemble des paramètres flottants parcourus par `downscale`
  (`model.parameters()`, `scripts/pilot_cycle3_cuda.py:464-465`) ; `N` = nombre d'éléments
  de `P`.

Pas DOWNSCALE et norme cible :

```
ΔW_DS(s) = bf16(f · θ_r(s)) − θ_r(s)          (≈ (f − 1) · θ_r(s))
ρ(s)     = ‖ΔW_DS(s)‖_F                        (≈ (1 − f) · ‖θ_r(s)‖_F = 0,01 · ‖θ_r(s)‖_F)
```

Le calcul de `ρ(s)` se fait en float32 sur la différence réellement appliquée, comme le fait
déjà `drift` (`scripts/pilot_cycle3_cuda.py:490-499`), parce que l'arrondi bfloat16 rend
l'approximation `0,01 · ‖θ_r‖` inexacte.

Bras de contrôle proposé, `p_ctrl_samenorm` (exploratoire) : même corpus, même LR, même `m`,
même suite de mises à jour, même budget de jetons `T_s` que `p_equ`, même graine `s`, mêmes
items d'évaluation (`_cell_split(s)`) ; puis, **à la place** de DOWNSCALE, on ajoute à
`θ_r(s)` une perturbation `ξ(s)` de même norme `‖ξ(s)‖_F = ρ(s)`, sur le même ensemble `P`,
sous deux variantes :

- **Variante R (direction aléatoire isotrope)** : `ξ(s) = ρ(s) · g / ‖g‖_F`, avec
  `g ~ N(0, I_N)` tiré par un générateur `torch.Generator` dédié, initialisé avec la graine
  `s` de la cellule, distinct du générateur global (le rejeu n'est pas touché).
- **Variante U (même total réparti uniformément, au sens de Kubo et al., §4)** :
  chaque élément non nul reçoit la même amplitude, vers zéro :
  `ξ_i(s) = − (ρ(s) / √N′) · signe(θ_r,i(s))`, où `N′` est le nombre d'éléments non nuls ;
  les éléments nuls restent nuls.

La variante R sépare « rétrécir » de « bouger d'autant dans une direction quelconque ». La
variante U sépare « rétrécir proportionnellement à chaque poids » (DOWNSCALE) de « rétrécir
d'autant, réparti uniformément ». Comme DOWNSCALE, la perturbation n'est pas comptée dans le
budget (aucune passe avant).

Contrastes, appariés par graine, avec les procédures appariées déjà enregistrées
(`docs/osf-prereg-e1-small-2026-10-25.md:189-202`) et la marge `δ = 0,01` déjà enregistrée
(`:181`), au niveau nominal α = 0,05, hors famille, comme `p_ctrl_lr_matched`
(`:815-818`) :

```
D_R(s) = Δ_s(p_equ) − Δ_s(p_ctrl_samenorm, variante R)
D_U(s) = Δ_s(p_equ) − Δ_s(p_ctrl_samenorm, variante U)
```

et, à titre descriptif, `Δ_s(p_ctrl_samenorm) − Δ_s(p_ctrl_vanilla_replay_tuned)`.

### 1.3 Ce que le code permet, et ce qu'il ne permet pas

- **Le code gelé ne permet pas de calculer ce contrôle sans modification.** Aucun profil
  n'applique une perturbation de même norme (`scripts/e1_profiles.py:54-57`), le backend
  n'a pas de méthode pour cela (seuls `downscale`, `restructure`, `swap_layers`,
  `scripts/pilot_cycle3_cuda.py:462-483`), et aucune cellule n'enregistre `ρ(s)`.
- Ce que les cellules enregistrent : `norm_change = ‖θ_post − θ₀‖` et la dérive relative,
  mesurées **après** DOWNSCALE et après l'annulation de RESTRUCTURE
  (`scripts/e1_harness.py:267-274` ; `scripts/e1_diagnostics.py:39-43`). Pour `p_equ`,
  `θ_post = bf16(f · θ_r)`, donc `norm_change` ne donne pas `ρ(s)`.
- Encadrement possible après lecture, sans nouveau calcul machine lourd : `‖θ₀‖` se déduit
  de `norm_change / drift` ; avec `a = ‖θ_r − θ₀‖` = `norm_change` de
  `p_ctrl_vanilla_replay_tuned` à la même graine,
  `0,01 · |‖θ₀‖ − a| ≤ (1 − f)‖θ_r‖ ≤ 0,01 · (‖θ₀‖ + a)`. Réserves : `drift` parcourt
  `state_dict()` (`scripts/pilot_cycle3_cuda.py:493-498`) et `downscale` parcourt
  `parameters()` ; les deux ensembles peuvent différer (tampons, poids liés), et l'égalité
  des `θ_r` entre les deux bras suppose un rejeu déterministe (`deterministic()`,
  `scripts/pilot_cycle3_cuda.py:393-394`) : **non vérifié ici**. Cet encadrement est une
  analyse exploratoire à faire après la lecture enregistrée, pas avant.
- Le bras lui-même demande du code nouveau (une méthode de perturbation et un profil) et de
  nouvelles cellules GPU. Un bras qui tourne sur un code différent de l'arbre gelé relève
  d'un autre protocole : il sera daté par son propre commit, `p_equ` et
  `p_ctrl_vanilla_replay_tuned` seront relancés aux mêmes graines sous ce commit pour que
  l'appariement se fasse dans le même code, et les cellules d'E1-small ne servent qu'à un
  contrôle de cohérence, jamais à un regroupement. Coût non estimé ici. Ce bras ne part
  qu'après le verrou de lecture (brainstorm, idée 1 : bras sous verrou).
- Lien avec le bras exploratoire déjà enregistré `p_ctrl_lr_matched`
  (`docs/osf-prereg-e1-small-2026-10-25.md:804-821`) : celui-ci égalise la norme totale
  `‖θ_post − θ₀‖` en augmentant le LR du rejeu (`scripts/e1_profiles.py:133`, calibration
  `:234`), donc dans la direction de la SGD ; le contrôle déclaré ici garde le rejeu
  identique et ne remplace que l'étape DOWNSCALE. Fait : la valeur figée est `c = 5.0`
  (`docs/milestones/e1-small/e1-tuning-amendment-2026-10-29.json:3`), borne haute de la
  grille `C_GRID` (`scripts/e1_profiles.py:45`). Cette déclaration n'en tire aucune
  conclusion.

## 2. Ce qu'il sépare

H14-B ne peut, par construction, que dire si l'ajout de DOWNSCALE au rejeu réglé change
`acc` (`docs/osf-prereg-e1-small-2026-10-25.md:874-876`). La préregistration avertit déjà
qu'une lecture SUPERIOR serait compatible avec une baisse implicite du LR ou une
régularisation, et pas une preuve d'un mécanisme propre au sommeil (`:882-894`).
L'amendement du 2026-10-30 note que DOWNSCALE change les poids environ 35 fois plus que la
mise à jour SGD du rejeu dans ce harnais
(`docs/osf-amendment-e1-small-2026-10-30.md:151-155`).

Le contrôle de même norme sépare « le sommeil agit » (le contenu du pas DOWNSCALE,
rétrécissement proportionnel aux poids) de « on a secoué les poids » (n'importe quel
changement de même taille). Règle de lecture, qui ne peut que restreindre une
interprétation : si la borne inférieure appariée `L` de `D_R` ou de `D_U` n'est pas
strictement positive, aucune lecture de H14-B n'est attribuée au contenu de DOWNSCALE plutôt
qu'à une perturbation de même norme.

## 3. Statut

- **Analyse exploratoire, déclarée après le gel** (commit `963bb24`) **et après le début du
  run confirmatoire** (2026-10-30, 00:05:29 CET ;
  `docs/osf-amendment-e1-small-2026-10-30.md:29-33`), **avant toute lecture des cellules**
  par l'auteur de cette déclaration et avant l'analyse enregistrée.
- **Aucune cellule d'E1-small n'a été lue par l'assistant qui rédige cette déclaration**
  (session du 2026-11-02) : ni le cache, ni le journal du run, ni un agrégat. Les seules
  sources lues sont la préregistration, l'amendement, l'amendement de réglage (fichier
  figé avant la phase confirmatoire) et le code.
- **Expositions antérieures déjà déclarées, rappelées sans les reprendre :** le 2026-10-30,
  une autre session de l'assistant a calculé et transmis au PI des moyennes par profil sur
  le cache partiel (`docs/osf-amendment-e1-small-2026-10-30.md:58-112`, §2bis), et le PI a
  lu une fois le journal du run (`:19-21`, `:36-40`). Ces expositions ont fait passer
  H14-A, H14-B et H14-C en exploratoires (`:98-108`). Cette déclaration ne peut donc pas
  revendiquer une ignorance totale des données ; elle date le contrôle avant la lecture
  enregistrée et avant toute lecture des cellules depuis ces expositions.
- **État du run :** dernier relevé en lecture seule, `inventory/compute-status.md`
  (2026-11-01, 18:10 CET, relevé de l'espace de travail du laboratoire, hors dépôt) :
  1 227 lignes sur 1 410 cellules. L'état au moment de la signature n'est pas connu de
  l'assistant.
- Ce contrôle ne fait passer aucun verrou, ne change le statut d'aucune hypothèse (toutes
  restent exploratoires), ne dépense aucun α et ne modifie aucun état décidé par
  `scripts/analyze_e1_cuda.py`.

## 4. Antériorités

- **Kubo, Delanois, Bazhenov (2026)**, *Firing Without Wiring: How REM-like Sleep
  Reactivation Rebalances Excitation and Inhibition to Orthogonalize Memories*, bioRxiv,
  DOI 10.64898/2026.10.11.757684 (v1 2026-10-18, v2 2026-10-21 ; résolu par Crossref et par
  l'API bioRxiv ; DataCite ne renvoie pas d'attributs). Le résumé indique que répartir le
  même changement synaptique total uniformément sur toutes les populations cibles abolit la
  récupération obtenue par le sommeil REM-like dans des réseaux récurrents entraînés par
  Equilibrium Propagation. C'est l'antériorité directe de la variante U. **[résumé moteur ;
  texte complet non lu ; une seule prépublication non revue : à recouper.]** Elle ne teste
  pas un LLM ni H14-B.
- **Deperrois, Petrovici, Senn, Jordan (2022)**, *Learning cortical representations through
  perturbed and adversarial dreaming*, eLife, DOI 10.7554/elife.76384 (Crossref).
- **Singh, Norman, Schapiro (2022)**, *A model of autonomous interactions between
  hippocampus and neocortex driving sleep-dependent memory consolidation*, PNAS,
  DOI 10.1073/pnas.2123432119 (Crossref).
  Ces deux références sont citées par le catalogue v2 à côté de l'amendement, pour H14 comme
  réplication de l'ordre des phases (`.github`,
  `docs/research/2026-11-01-analyse-profonde/catalogue-v2.md:33,97`) ; elles ne figurent
  pas dans le fichier d'amendement lui-même. Cette déclaration ne leur attribue pas le
  contrôle de même norme ; non relues ici.
- **Marek et al. (2026)**, arXiv:2605.26097, déjà cité par la préregistration pour la mise
  en garde DOWNSCALE (`docs/osf-prereg-e1-small-2026-10-25.md:121-129`, `:882-886`).

## 5. Critère de lecture

Aucun seuil nouveau, aucune hypothèse nouvelle. Les contrastes de §1.2 sont lus avec les
procédures appariées, la marge δ et les libellés d'état déjà enregistrés
(`docs/osf-prereg-e1-small-2026-10-25.md:181-202`, `:230-238`), au niveau nominal, comme
lectures exploratoires rapportées à côté de H14-B et de `p_ctrl_lr_matched`. Ils ne font
passer aucun verrou et ne changent ni le statut exploratoire d'E1-small ni aucun état
enregistré. La seule conséquence admise est la restriction d'interprétation de §2.

## 6. Signature

Le PI confirme, en signant, n'avoir lu aucune cellule, aucun score et aucun agrégat
d'E1-small depuis les expositions déclarées dans
`docs/osf-amendment-e1-small-2026-10-30.md` (§2, §2bis), et que cette déclaration précède
la lecture enregistrée.

- [X] Signature PI : SAILLANT Clément
- Date et heure (CET) : 06/11/2026 à 13h17
