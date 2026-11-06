# E1-small — analyse selon la préregistration gelée (brouillon, 2026-11-06)

**Statut :** brouillon, à relire par le PI. **Toutes les hypothèses H14 sont
exploratoires** (amendement du 2026-10-30, §2bis,
`docs/osf-amendment-e1-small-2026-10-30.md`, décision PI du 2026-10-31) ; ce
document ne revient pas sur ce statut et **ne donne aucun titre conjoint**
(même amendement, §2bis : « with no joint headline »). Aucune lecture
confirmatoire n'est faite d'E1-small.
**Préregistration :** `docs/osf-filing/e1-small/osf-prereg-e1-small-2026-10-25.frozen.md`,
SHA-256 `6d47d8b7…208463`, commit de verrouillage `963bb24`
(`docs/osf-filing/e1-small/FILING-NOTE.md`). Pas d'enregistrement OSF propre
à E1-small parmi les fichiers lus : les enregistrements OSF `q6jyn`
(2026-04-19) et `tpm5s` (2026-04-21) de la bibliothèque sont le projet parent
et son amendement Bonferroni, antérieurs à E1-small.
**Analyse enregistrée :** `scripts/analyze_e1_cuda.py` (prereg en-tête l.39,
§9.5 tâche 6), lancée sans modification. Elle seule décide des états de §2.
**Complément :** `scripts/e1_small_null_supplement.py` (nouveau, exploratoire) :
nuls appariés, porte d'apprentissage, puissance, encadrement de la déclaration
H14-B. Il ne change aucun état.
**Calcul :** macM1, clone frais `~/scratch/e1-analysis-0611/dream-of-kiki`
(branche `docs/e1-small-analysis-2026-11-06`, `uv sync`, `uv run`), le
2026-11-06. Aucun calcul GPU, rien n'a été lancé sur KXKM-AI.

---

## 1. Faits — artefacts et complétude

### 1.1 Où sont les données

- Le cache confirmatoire n'existe que sur **KXKM-AI**,
  `/ai/saisail/e1-small/out/run-2026-10-29/` (lu en lecture seule, copié
  sur macM1 `~/scratch/e1-analysis-0611/data/`, SHA-256 identiques).
- Les trois emplacements indiqués par la commande de l'analyse ne contiennent
  **pas** de cellule E1-small : macM1 `~/scratch/e1-small/` ne contient que la
  fumée MPS du 2026-10-25 (`out/probe-1cell`, `driver.log`) ;
  `/Volumes/T7/gb300-20261101/` (recherches `*e1*`, `*confirmatory*`) ne
  contient que d'autres campagnes (rescience, nh01, f4l) ; electron-server
  `~/scratch/gb300-20261101/` n'existe pas. E1-small n'a pas tourné sur le
  GB300 : il a tourné sur l'instrument enregistré (RTX 4090 de KXKM-AI),
  comme le prévoit la préreg (§9.2).

### 1.2 Complétude

- `driver.log` : démarrage confirmatoire 2026-10-30 00:05:30 CET (code
  `e93ea81`, amendement `3b76adc6…`) ; **fin 2026-11-01 20:20:33 CET,
  `rc=1`, 1 271 cellules** (et non 1 410).
- Cause (fin de `confirmatory.log`) : `ValueError: math domain error` dans
  `choice_kl` (`scripts/e1_diagnostics.py:74`, appel
  `scripts/e1_harness.py:276`) pendant la cellule `p_ctrl_lr_matched|2`, qui
  n'a pas été écrite. Inférence : une probabilité de choix post-rejeu
  arrondie à 0 donne `log(0)` ; non vérifié.
- Contenu : les **9 profils enregistrés × 141 graines = 1 269 cellules sont
  complètes** ; le bras exploratoire `p_ctrl_lr_matched` n'a que 2 cellules
  sur 141 (graines 0 et 1). L'ordre d'exécution est par profil, ce bras étant
  le dernier.
- Contrôles d'intégrité (`2026-11-06-e1-small-null-supplement.json`,
  `integrity`) : 0 violation d'invariant, 0 score non fini, budgets
  `matched` partout (achevé = planifié sur les 1 271 cellules ; `p_noise`
  sans mise à jour par construction), un seul `code_commit` (`e93ea81`), une
  seule `run_identity`, `backend = torch`, `linear_attn = torch`, mode
  déterministe actif, `n_eval = 162`, `pre` identique entre profils pour
  chaque graine.
- Le code E1 (`scripts/e1*`, `pilot_cycle3_cuda.py`, `analyze_e1_cuda.py`,
  `kiki_oniric/eval/statistics.py`) est identique entre le verrou `963bb24`
  et le code du run `e93ea81` (`git diff --stat`, seuls des fichiers G4/DR-3
  diffèrent) et entre `e93ea81` et `main` `b66d658`.
- Mémoire de travail périmée corrigée : la note « E1-small 1052/1410, fin
  ~2/11 03:30 » ne correspond pas au journal (arrêt le 01/11 à 20:20, 1 271).

### 1.3 Table des artefacts

| artefact | emplacement | SHA-256 |
|---|---|---|
| cache confirmatoire (1 271 lignes) | KXKM `…/run-2026-10-29/confirmatory/e1-confirmatory-torch-Qwen_Qwen3.5-4B.cells.jsonl` | `adff2ec70a45ec788253f29836d5096149f71b156d36ca8293a45ac3cb373a19` |
| amendement de réglage (run) | KXKM `…/amend/e1-tuning-amendment.json` | `3b76adc6ddf9ff7266a0351b612ef0d718951d79e00b50dfd06b0bc09fdaac49` |
| idem, dépôt (identique) | `docs/milestones/e1-small/e1-tuning-amendment-2026-10-29.json` | `3b76adc6ddf9ff7266a0351b612ef0d718951d79e00b50dfd06b0bc09fdaac49` |
| cellules de réglage (120) | KXKM `…/tuning/…cells.jsonl` | `4291d214071ed63f45236cc871223f1283d9375cf9300984f941e401585835b9` |
| cellules de calibration (60) | KXKM `…/calibrate/…cells.jsonl` | `8a1d40f6a8e7cca7228d0583ed6567beb7adcf40bc76cc7cef658de5269dc234` |
| journal confirmatoire | KXKM `…/confirmatory.log` | `9017fdfe41f545625a862ac2f8acca9b7bbe345721ae8e681390e7d80b2a7fc9` |
| journal du pilote | KXKM `…/driver.log` | `fb856160c1b6a76e02d1179e67d4bcf729c61115d6e20ef1c11c0b7fc22b2874` |
| sortie de l'analyse enregistrée | `docs/results/2026-11-06-e1-small-decision.json` | `c2da231cf899cc3bbb9c7b115b54497157143c202cd401eab33269f5bdcb01dc` |
| sortie du complément | `docs/results/2026-11-06-e1-small-null-supplement.json` | `5d78ac79370bf6ee957f17024654db63deebb5a316c052156ea0604a26a72d0a` |
| script du complément | `scripts/e1_small_null_supplement.py` | `99826304c905073cbc52e832c0e0844eda5d796d9dd81e0c6bd4cd3f38c651a4` |

Les cellules (sans texte d'item) ne sont pas versionnées ici (8,5 Mo) ;
leurs SHA-256 suffisent à les identifier.

Commandes (macM1) :

```
uv run python scripts/analyze_e1_cuda.py ../data/e1-confirmatory-torch-Qwen_Qwen3.5-4B.cells.jsonl \
  --amendment docs/milestones/e1-small/e1-tuning-amendment-2026-10-29.json --out ../decision.json
/usr/bin/lockf -k ~/scratch/.heavy.lock uv run python scripts/e1_small_null_supplement.py \
  ../data/e1-confirmatory-torch-Qwen_Qwen3.5-4B.cells.jsonl --out ../supplement.json
```

(`--draws 10000`, graine `20261106` par défaut.)

## 2. Chemins bifurquants déclarés

1. **Run avant dépôt OSF** (`docs/milestones/e1-small/deviation-run-before-osf-2026-10-29.md`,
   PR #76) ; à ce jour aucun dépôt OSF propre à E1-small n'est connu de cette
   analyse.
2. **Expositions** (amendement §2, §2bis) : le 2026-10-30 à 13:12 CET, des
   moyennes intermédiaires ont été transmises au PI, dont `p_min` +0,057 sur
   ses **141 cellules déjà complètes** ; la moyenne finale de `p_min` est
   +0,0566. Le bras `p_min`, terme X de H14-A, était donc entièrement connu du
   PI avant la fin du run. Le PI a lu une fois le journal (14:22 CET). D'où le
   passage de H14-A/B/C en exploratoires.
3. **Pilotes et réglage :** fumées MPS (abandonnée) et KXKM (non
   déterministe puis déterministe, jamais analysées) ; réglage 10 graines
   (1000-1009) et calibration 60 cellules, figés dans l'amendement de réglage
   avant la première graine confirmatoire (PR #77, fusion
   2026-10-29T23:05:10Z ; début confirmatoire 19 s plus tard). Valeurs
   figées : sélection (i) vanilla = sélection (ii) OPR = (LR 3e-4, m 2),
   `c = 5.0` (borne haute de la grille), `p̂_disc = 0.1269`, `n_eval = 162`,
   proxy `SD_D = 0.0286`.
4. **Run incomplet pour le bras exploratoire.** La préreg compte 1 410
   cellules confirmatoires (10 profils, en-tête l.46) ; la règle « run
   incomplet » (§4, §8) est formulée en « 141 graines complètes » et le code
   enregistré la lit sur les 9 profils enregistrés (`n_complete_seeds = 141`,
   `confirmatory: true`). Les deux lectures divergent ; c'est sans effet ici
   puisque tout est déjà exploratoire.
5. **Sortie brute du code enregistré.** Le script imprime « joint headline:
   J2 » et `confirmatory: true` : il ignore l'amendement. Ces deux champs
   **ne sont pas repris** (amendement §2bis).
6. **Analyses post hoc annoncées** (amendement §4, déclaration H14-B du
   2026-11-02, signée le 2026-11-06 13:17) : traitées en §5.
7. **Complément de ce document** (nuls par inversion de signe, bootstrap,
   puissance à SD observée) : non préenregistré ; il ne fait que calibrer
   les seuils déjà enregistrés.

## 3. Méthode du nul apparié (complément)

Pour chaque contraste (X, Y), statistique par graine `D_s = Δ_s(X) − Δ_s(Y)`
(mêmes cellules, même appariement, mêmes exclusions — aucune ici — que le
code enregistré). Nul : **inversion de signe appariée** de `D_s`
(échange des étiquettes des deux bras au sein de chaque graine), N = 10 000
tirages ; pour la non-infériorité, inversion de `D_s + δ` (H0 : μ = −δ).
Rapporté : médiane et P99 de `D̄` sous H0, p = (k+1)/(N+1), et la part des
tirages nuls qui passent le seuil enregistré (`L > 0` ou `L > −δ`). Hypothèse
du nul : distribution de `D_s` symétrique sous H0 (échangeabilité des bras
au sein d'une graine). La plus petite p atteignable est 1/10 001 ≈ 1,0·10⁻⁴.

## 4. Résultats par hypothèse (lectures exploratoires)

Constantes enregistrées : δ = 0,01 ; α = 1/60 (bornes unilatérales à
1 − α) ; n = 141 paires retenues pour chaque contraste (0 exclusion, 0 graine
OPR infaisable, taux d'effondrement ≤ 0,7 %, porte de 20 % non atteinte).

### Porte d'apprentissage (préalable)

- Fait : Δ moyen par profil (post − pre, `acc`) : `p_min` +0,057,
  `p_equ` +0,063, `p_ctrl_vanilla_replay_tuned` +0,063, `p_ctrl_onpolicy_replay`
  +0,045, `p_max` +0,037, `p_ctrl_template` +0,013, `p_ctrl_shuffle` +0,013,
  `p_ctrl_sft` −0,0004, `p_noise` +0,004 (SD 0,049, bruit d'échantillonnage
  d'items). Précision de base moyenne 0,336 (`pre`).
- Fait : pour chaque profil de rejeu sauf `p_ctrl_sft`, Δ > 0 avec p
  d'inversion de signe 1,0·10⁻⁴ ; `p_ctrl_sft` p = 0,62 ; `p_noise` p = 0,18.
- Inférence : la porte d'apprentissage est franchie pour les bras des trois
  contrastes H14 (le modèle a bougé au-delà du bruit d'items) ; elle ne l'est
  pas pour `p_ctrl_sft`.

### H14-A — `p_min` vs `p_ctrl_template` (supériorité)

- Règle (prereg §2, l.216-218, l.232-238) : SUPERIOR ssi `L > 0` et
  `D̄ ≥ δ` ; BOUNDED-NO-GAIN ssi `U < δ`.
- Observé : `D̄ = +0,0432`, SD_D 0,0270, `[L, U] = [+0,0383 ; +0,0481]`,
  g_z 1,59 ; `D_s > 0` sur 131 graines, `= 0` sur 5, `< 0` sur 5 ; état du code
  enregistré : SUPERIOR.
- Nul apparié : médiane +0,0000, P99 +0,0097, p = 1,0·10⁻⁴ (k = 0/10 000) ;
  `L > 0` passé par 1,6 % des tirages nuls (attendu α ≈ 1,7 % : seuil
  calibré). Bootstrap 95 % de `D̄` : [+0,0388 ; +0,0476].
- Contrôle pouvant échouer : `p_ctrl_template` a un Δ propre (+0,013) et un
  état non forcé ; non tautologique. Discordance post entre bras 0,141.
- Puissance à SD observée (0,027, n = 141) : SUPERIOR à μ = δ 0,50, à
  μ = 1,5 δ 0,99 ; BOUNDED à μ = 0 0,99.
- Par source (descriptif) : `realtimeqa` +0,062, `daily_oracle` +0,047,
  `mmlu_cf` +0,021, bornes inférieures toutes > 0.
- **Formulation :** dépasse le seuil préenregistré (`L > 0`, `D̄ ≥ δ`) ; nul
  apparié p = 1,0·10⁻⁴ → **va dans le sens de** « le rejeu avec réponse
  améliore `acc` par rapport au rejeu sans réponse, à jetons égaux »,
  lecture exploratoire. Réserves : format et contenu de réponse ne sont pas
  séparés (prereg §1, l.150-156) ; le bras `p_min` était connu du PI avant la
  fin du run (§2.2 ci-dessus) ; le contrôle « options seules » n'est pas
  calculé (§5).

### H14-B — `p_equ` vs `p_ctrl_vanilla_replay_tuned` (supériorité)

- Règle : comme H14-A (l.219-223).
- Observé : `D̄ = −2·10⁻¹⁸` (somme des écarts en items exactement 0),
  SD_D 0,0101, `[L, U] = [−0,0018 ; +0,0018]` ; `D_s > 0` sur 48 graines,
  `= 0` sur 41, `< 0` sur 52 ; état du code enregistré : BOUNDED-NO-GAIN (TOST
  secondaire équivalent à ±δ).
- Nul apparié : médiane 0,0000, P99 +0,0019, p = 0,52.
- Sensibilité du contraste (fait) : discordance post entre les deux bras
  0,016 en moyenne (max 0,043), **13 graines sur 141 ont des réponses post
  identiques** ; DOWNSCALE change pourtant les poids environ 25 fois plus
  que le rejeu (encadrement §5.3 : ρ ≈ 8,32 contre ‖θ_r − θ₀‖ ≈ 0,325).
- Inférence (non testée) : un rétrécissement uniforme de 1 % des poids
  d'un transformeur à normalisation RMS modifie surtout l'échelle des
  activations que les normalisations compensent ; DOWNSCALE change donc peu
  de réponses. Le contraste est peu sensible : `U < δ` était presque acquis
  à cette discordance. Le contrôle peut échouer (13/141 graines identiques
  seulement), mais faiblement ; il est classé **contrôle à faible pouvoir de
  falsification**, pas « non informatif ».
- Puissance à SD observée (0,0101) : BOUNDED à μ = 0 ≈ 1,00 ; SUPERIOR à
  μ = δ 0,50.
- Par source : moyennes −0,0002 / +0,0004 / −0,0002, toutes dans ±0,004.
- **Formulation :** ne dépasse pas le seuil de supériorité (`L = −0,0018`) ;
  atteint la borne préenregistrée `U < δ` ; nul apparié p = 0,52 → **va dans
  le sens de** « DOWNSCALE ajoute moins de δ au rejeu réglé dans ce
  réglage », lecture exploratoire, sous la réserve de sensibilité ci-dessus.
  Aucune lecture sur un mécanisme de sommeil (prereg §7, l.882-894).

### H14-C — `p_equ` vs `p_ctrl_onpolicy_replay` (non-infériorité à δ)

- Règle (l.224-225, l.242-247) : NON-INFERIOR ssi `L > −δ`.
- Observé : `D̄ = +0,0187`, SD_D 0,0323, `[L, U] = [+0,0129 ; +0,0245]`,
  g_z 0,58 ; `D_s > 0` sur 96, `= 0` sur 13, `< 0` sur 32 ; état du code
  enregistré : NON-INFERIOR. Corpus OPR : 7 à 21 items par graine (aucune
  graine vide) ; charge de notation 1 802-2 423 jetons, budget OPR donc
  inférieur à T_s (règle §5.4) ; 1 cellule OPR sur 141 marquée effondrement.
- Nul apparié (H0 : μ = −δ) : médiane −0,0100, P99 −0,0016, p = 1,0·10⁻⁴ ;
  `L > −δ` passé par 1,65 % des tirages nuls (calibré). Bootstrap 95 % :
  [+0,0133 ; +0,0241].
- Contrôle pouvant échouer : oui (32 graines avec `D_s < 0` ; discordance 0,124).
- Puissance à SD observée : NI à μ = 0 0,94.
- Par source : `daily_oracle` +0,022, `mmlu_cf` +0,022, `realtimeqa` +0,012.
- **Formulation :** dépasse le seuil préenregistré de non-infériorité ; nul
  apparié p = 1,0·10⁻⁴ → **va dans le sens de** « `p_equ` n'est pas pire que
  le rejeu MC-OPR restreint de δ ou plus », lecture exploratoire. Réserve :
  OPR reçoit moins de jetons de mise à jour (charge de notation) et un corpus
  plus court ; la comparaison est celle, à calcul total égal, que fixe la
  préreg, pas une comparaison à jetons de mise à jour égaux.

### Titre conjoint

Non donné (amendement §2bis). Le code enregistré imprime J2 ; ce champ est
écarté.

## 5. Contrôles, secondaires et analyses annoncées

### 5.1 Signature de défendabilité (secondaire, prereg l.267-271)

- Fait : la signature `dream > p_ctrl_sft > p_ctrl_shuffle ≈ p_noise`
  **n'est pas observée** : `p_ctrl_sft − p_ctrl_shuffle = −0,0134`
  (`[L, U]` à 95 % unilatéral `[−0,0174 ; −0,0094]`, nul p = 1,00) ;
  `p_ctrl_sft` ne bouge pas (Δ −0,0004 ; perte de rejeu moyenne 0,19 contre
  1,5-1,9 pour les autres bras, 792 mises à jour médianes contre 200). Les
  trois profils de rêve dépassent tous les contrôles sauf
  `p_ctrl_vanilla_replay_tuned` (et OPR pour `p_max`).
- Inférence : ce contrôle pouvait échouer et échoue sur le maillon
  `sft > shuffle` ; cela ne touche aucun état H14 mais limite la lecture
  « le contenu du rejeu compte » aux contrastes enregistrés.
- Autres secondaires notables (α nominal 0,05) : `p_min − p_ctrl_vanilla_replay_tuned`
  = −0,0069 (la configuration réglée fait mieux que le défaut) ;
  `p_max − p_ctrl_vanilla_replay_tuned` = −0,0263 (RESTRUCTURE dégrade) ;
  `p_max − OPR` = −0,0076. Table complète : `2026-11-06-e1-small-decision.json`,
  champ `secondary`.

### 5.2 Bras exploratoire `p_ctrl_lr_matched` (prereg §5.4, l.804-821)

- Fait : 2 cellules sur 141 (graine 0 : Δ +0,086 ; graine 1 : Δ −0,049,
  marquée effondrement) ; LR = 5 × 3e-4 = 1,5e-3. Le code enregistré donne
  UNINTERPRETABLE (`differential_exclusion`, `collapse`).
- **Formulation : non calculable** à n = 2 ; la cellule de la graine 2 a
  arrêté le run (§1.2).

### 5.3 Contrôle de même norme H14-B (déclaration du 2026-11-02)

- Le bras `p_ctrl_samenorm` n'a pas été exécuté (code nouveau et cellules GPU
  requis, déclaration §1.3) : **non calculé**.
- Encadrement sans calcul lourd (déclaration §1.3), exploratoire, après la
  lecture enregistrée : ‖θ₀‖ médian 832,4 (identique depuis les deux bras à
  1,4·10⁻¹⁶ près en relatif) ; a = ‖θ_r − θ₀‖ médian 0,325 ;
  `ρ ∈ [8,321 ; 8,328]` (médianes) ; rapport ρ / a ≈ 25,6 (l'amendement
  §4.3 annonçait « environ 35 fois »). Réserve de la déclaration : `drift`
  parcourt `state_dict()` et `downscale` `parameters()` ; non vérifié ici.
- Conséquence de la règle de lecture de la déclaration (§2) : aucune lecture
  de H14-B n'est attribuée au contenu de DOWNSCALE (la borne `L` de `D_R` /
  `D_U` n'existe pas).

### 5.4 Analyses annoncées par l'amendement §4

1. Contrôle « options seules » : **non calculé** (notation du modèle de base
   sur les items confirmatoires sur l'instrument enregistré, donc GPU de
   KXKM-AI ; aucun lancement sans accord du PI ; un passage MPS serait un autre
   instrument, jamais regroupé, prereg §8).
2. Lecture par source : faite (ci-dessus, et `per_source` du JSON).
3. Daily Oracle sous le hasard : à la base, `daily_oracle` reste la source la
   plus basse en réglage (0,185, manifeste gelé) ; les gains H14 par source
   sont du même signe pour les trois sources.

## 6. Ce qui n'a pas pu être calculé

- Bras `p_ctrl_lr_matched` (139 cellules manquantes, arrêt du run).
- Bras de même norme (non exécuté) et contrôle « options seules » (GPU).
- Le cas de la cellule `p_ctrl_lr_matched|2` (cause exacte de `log(0)`) :
  non reproduit, ce qui demanderait le GPU.

## 7. Faits et inférences — résumé

Faits : 1 269/1 269 cellules enregistrées complètes et valides, 2/141
cellules du bras exploratoire ; états du code enregistré SUPERIOR (A),
BOUNDED-NO-GAIN (B), NON-INFERIOR (C) ; nuls appariés p = 1,0·10⁻⁴ (A, C) et
0,52 (B) ; seuils calibrés (taux de passage sous H0 ≈ α) ; signature
`sft > shuffle` non observée.

Inférences : A et C vont dans le sens des lectures préenregistrées, en
exploratoire ; B va dans le sens d'un apport de DOWNSCALE inférieur à δ, mais
le contraste est peu sensible (1,6 % de réponses discordantes) ; rien
n'est dit sur un mécanisme propre au sommeil ni sur le modèle 35B.
