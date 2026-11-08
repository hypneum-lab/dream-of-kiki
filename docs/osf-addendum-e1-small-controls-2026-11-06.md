# Addendum de contrôle — E1-small : bras de même norme (H14-B) et « options seules », 2026-11-06

**Rédigé le :** 2026-11-06 (après-midi, CET), sur décision du PI du 2026-11-06 à 13:58 CET
(« lancer les deux contrôles GPU maintenant »), condition : GPU de KXKM-AI libre.
**Préregistration visée :** E1-small,
`docs/osf-filing/e1-small/osf-prereg-e1-small-2026-10-25.frozen.md` (gelée, SHA-256
`6d47d8b7…208463`, commit de verrouillage `963bb24`), non modifiée.
**Type :** addendum de contrôle d'une campagne déjà gelée (principe 3 de la roadmap v2 du
laboratoire, dépôt `.github` ; ligne non relue ici). Ce n'est pas un nouveau gel. Aucune
hypothèse, aucun estimand, aucun seuil, aucun N de la préregistration n'est changé ; aucun état
décidé par `scripts/analyze_e1_cuda.py` n'est modifié.
**Documents sources (non modifiés) :** amendement du 2026-10-30
(`docs/osf-amendment-e1-small-2026-10-30.md`), déclaration H14-B du 2026-11-02 signée le
2026-11-06 à 13:17 (`docs/osf-declaration-h14b-same-norm-control-2026-11-02.md`), rapport
d'analyse `docs/results/2026-11-06-e1-small-analysis.md` (PR #91, fusionnée sur `main`,
`365bf14`).
**Base de la branche :** `main` = `365bf14`.

---

## 1. Statut épistémique : exploratoire et informé

- Les hypothèses H14-A, H14-B et H14-C sont déjà exploratoires (amendement, §2bis,
  `docs/osf-amendment-e1-small-2026-10-30.md:98-108`).
- **L'analyse des bras enregistrés a déjà été lue** (rapport #91). En particulier, avant la
  rédaction de cet addendum, étaient connus : l'état BOUNDED-NO-GAIN de H14-B avec
  `D̄ ≈ 0` et une discordance post entre bras de 0,016 (13 graines sur 141 à réponses
  identiques) ; l'encadrement `ρ ∈ [8,321 ; 8,328]` contre `‖θ_r − θ₀‖ ≈ 0,325` ; la
  précision de base moyenne 0,336 ; la position basse de `daily_oracle`
  (`docs/results/2026-11-06-e1-small-analysis.md:157`, `:188-214`, `:270-293`).
- Tout ce que produisent les deux bras ci-dessous est donc **exploratoire et informé** par
  ces lectures. Les règles de lecture de §4 sont écrites avant le lancement, mais après la
  lecture de #91 ; leurs seuils nouveaux (§4.2) sont choisis en connaissant ces résultats.
- Ces contrôles ne dépensent aucun α, ne font passer aucun verrou et ne changent ni le statut
  exploratoire d'E1-small ni aucun état enregistré.

## 2. Bras 1 — contrôle de même norme pour H14-B

### 2.1 Définition (celle de la déclaration du 2026-11-02, §1.2)

Pour chaque graine `s` des **141 graines enregistrées 0-140** (aucune autre), quatre cellules,
dans le mode `controls` du harnais, ordre graine par graine :

| profil | corpus | LR, m | après le rejeu |
|---|---|---|---|
| `p_equ` (relancé) | faits | sélection (i) : 3e-4, 2 | DOWNSCALE `f = 0,99` (inchangé) |
| `p_ctrl_vanilla_replay_tuned` (relancé) | faits | sélection (i) | rien (inchangé) |
| `p_ctrl_samenorm_r` | faits | sélection (i) | perturbation de même norme, variante R |
| `p_ctrl_samenorm_u` | faits | sélection (i) | perturbation de même norme, variante U |

Tout le reste est identique aux cellules enregistrées : corpus FROZEN
(`f8379b81…c0a2`, manifeste `093fdf61…02ab`), amendement de réglage `3b76adc6…ac49`
(sélection (i), `n_eval = 162`), `_cell_split(s)`, budget `T_s`, suite de mises à jour,
instrument enregistré (CUDA, `torch` fallback, tokenizer brut, instantané RAM,
`torch.optim.SGD`, mode déterministe, `expandable_segments:True`), poids `851bf6e8…`.

Notations de la déclaration : `θ_r(s)` = poids après le rejeu ; `P` = `model.parameters()`
flottants (ceux que parcourt `downscale`, `scripts/pilot_cycle3_cuda.py`) ; `N` = nombre
d'éléments de `P`.

- **Norme cible.** `ρ(s) = ‖bf16(f · θ_r(s)) − θ_r(s)‖_F`, calculée en float32 dans la cellule,
  sur `θ_r(s)`, avec la même opération que `downscale` mais sans modifier les poids
  (`TorchBackend.downscale_norm`).
- **Variante R.** `ξ(s) = ρ(s) · g / ‖g‖_F`, `g ~ N(0, I_N)` tiré en float32 par un
  `torch.Generator` dédié sur le GPU, initialisé avec `s` ; deux passes régénèrent le même `g`
  (norme, puis application), par blocs de 2²⁵ éléments dans l'ordre de `parameters()`. Le
  générateur global et le rejeu ne sont pas touchés.
- **Variante U.** `ξ_i(s) = −(ρ(s)/√N′) · signe(θ_r,i(s))` sur les `N′` éléments non nuls ;
  les éléments nuls restent nuls.
- **Application.** `θ ← bf16(θ_r + ξ)`, calcul en float32 puis écriture en bfloat16, comme
  les poids du run. La cellule enregistre `ρ(s)`, `‖ξ(s)‖` (= `ρ(s)` en float32) et la
  **norme réellement appliquée** `‖bf16(θ_r + ξ) − θ_r‖_F` (champ `ops.samenorm`).
  L'arrondi bfloat16 rend la norme appliquée différente de `ρ(s)` ; c'est le pendant exact de
  `ρ(s)`, lui-même mesuré après arrondi. Aucune remise à l'échelle itérative n'est faite :
  la norme appliquée est mesurée, puis contrôlée par la porte de conformité de §4.2.
- **Relance de `p_equ` et `p_ctrl_vanilla_replay_tuned`.** Comme le prévoit la déclaration
  (§1.3, `docs/osf-declaration-h14b-same-norm-control-2026-11-02.md:109-114`), ces deux bras
  sont relancés aux mêmes graines sous le même commit, et l'appariement se fait **dans ce
  commit uniquement**. Les cellules d'E1-small ne servent qu'à un contrôle de cohérence
  (§4.2), jamais à un regroupement.

### 2.2 Ce qu'il contrôle

H14-B ne dit que si l'ajout de DOWNSCALE au rejeu réglé change `acc`
(prereg `:219-223`, `:874-876`). La préregistration avertit qu'une lecture favorable serait
compatible avec une baisse implicite du LR ou une régularisation (`:882-894`). Le bras de même
norme sépare le contenu du pas DOWNSCALE (rétrécissement proportionnel aux poids) d'un
changement quelconque de même taille (R : direction aléatoire ; U : même total réparti
uniformément, vers zéro) (déclaration §2, `:124-139`).

## 3. Bras 2 — contrôle « options seules »

### 3.1 Définition (amendement du 2026-10-30, §4.1, `:132-138`)

- Modèle de base `Qwen/Qwen3.5-4B` @ `851bf6e8…`, restauré, sans rejeu.
- Les **1 098 items de la partition CONFIRMATORY** du corpus FROZEN, notés une fois chacun
  (mode `options-only`) : la base est la même pour toutes les graines, donc la valeur de la
  graine `s` s'obtient sur ses 162 items d'évaluation `_cell_split(s)`, sans recalcul.
- **Question retirée.** L'invite enregistrée est `"Date: AAAA-MM-JJ\n" + question +
  "\nAnswer:"` (préfixe de date pour les sources datées ; `scripts/e1_corpus_core.py:61`,
  `scripts/e1_corpus_io.py:51`). L'invite « options seules » est le seul enrobage
  `"Answer:"` : préfixe de date et question retirés. Les choix gardent leur espace initial et
  leur ordre permuté gelé.
- **Règle de notation enregistrée** : `_score_choice` (log-probabilité moyenne des jetons du
  choix), argmax, égalités vers le plus petit indice (prereg `:734-735`).
- Sortie : `e1-options-only.json` (scores par item, argmax, réponse, source, `item_sha256`).

**Restriction déclarée.** L'amendement annonce aussi la part « du changement de chaque
profil » atteignable par les options seules. Cette part demanderait de noter en « options
seules » les poids post-rejeu de chaque profil, qui n'ont pas été conservés : il faudrait
relancer les 1 269 cellules. **Elle n'est pas lancée** ; seule la partie « modèle de base » est
exécutée. Le PI peut demander la partie par profil séparément.

### 3.2 Ce qu'il contrôle

Combien de la précision de base est atteignable sans la question, avec la règle de notation
normalisée par la longueur. Cela calibre un artefact de format au niveau de la base :
réponse correcte plus longue dans environ 34 % des items `daily_oracle`
(amendement `:149-150`), précision de base `daily_oracle` sous le hasard (0,185 en réglage).
Ce contrôle ne porte pas directement sur H14-A (un changement post − pre) ; il borne la part de
la précision de base qui ne dépend pas de la question. La préregistration reconnaît déjà que
format et contenu de réponse ne sont pas séparés (`:150-156`).

## 4. Règles de lecture (écrites avant le lancement)

### 4.1 Communes

- Procédures appariées enregistrées (prereg `:189-202`), marge `δ = 0,01` (`:181`), libellés
  d'état de §2 (`:230-238`), au niveau nominal α = 0,05 (bornes unilatérales à 0,95), hors
  famille, comme `p_ctrl_lr_matched` (`:815-818`).
- **Nul apparié identique au rapport #91** : inversion de signe appariée de `D_s`,
  10 000 tirages, graine `20261106`, `p = (k+1)/(N+1)`, fonction `signflip` de
  `scripts/e1_small_null_supplement.py` (rapport #91, §3) ; plus petite p atteignable
  ≈ 1,0·10⁻⁴.
- **Un contrôle qui ne peut pas échouer est « non informatif »** ; il est rapporté, sans
  lecture.
- Validité reprise de la préreg : violation d'invariant ou score non fini → cellule exclue
  (`:331-333`) ; plus de 7 graines perdues pour un contraste → UNINTERPRETABLE (`:344-349`) ;
  taux d'effondrement > 20 % d'un bras → UNINTERPRETABLE (`:857-862`).
- Formulation : « va dans le sens de », jamais « confirme / infirme » ; aucun titre conjoint ;
  rien sur un mécanisme propre au sommeil ni sur le 35B.

### 4.2 Bras de même norme

- **Contrastes** (déclaration §1.2) : `D_R(s) = Δ_s(p_equ) − Δ_s(p_ctrl_samenorm_r)`,
  `D_U(s) = Δ_s(p_equ) − Δ_s(p_ctrl_samenorm_u)`, tous dans ce commit ; descriptif :
  `Δ_s(variante) − Δ_s(p_ctrl_vanilla_replay_tuned)` ; et H14-B recalculé dans ce commit
  (`p_equ` relancé − `p_ctrl_vanilla_replay_tuned` relancé), descriptif.
- **Règle de restriction** (déclaration §2, `:136-139`) : si la borne inférieure `L` de `D_R`
  ou de `D_U` n'est pas strictement positive, aucune lecture de H14-B n'est attribuée au
  contenu de DOWNSCALE plutôt qu'à une perturbation de même norme. Vu #91 (`D̄ ≈ 0` pour
  H14-B), cette issue est quasi certaine ; elle est dite telle quelle.
- **Porte d'apprentissage** (comme #91, §4) : pour chacun des quatre bras, Δ > 0 avec
  p d'inversion de signe < 0,05. Un bras qui ne la passe pas rend « non informatifs » les
  contrastes qui l'utilisent.
- **Porte de conformité de la perturbation** (nouvelle) : rapport
  `norme appliquée / ρ(s)` hors de [0,5 ; 2] sur plus de 7 graines (5 %, le seuil de §4 de la
  préreg) → la variante est « non conforme » et ses contrastes « non informatifs ».
- **Sensibilité** (nouveaux seuils, choisis après #91) : pour chaque contraste, réponses post
  identiques entre les deux bras sur plus de 70 des 141 graines → « non informatif » (le
  contraste ne peut presque pas différer de 0) ; sinon discordance post moyenne < 0,05 →
  « contrôle à faible pouvoir de falsification » (classement de #91 pour H14-B) ; sinon
  « informatif ».
- **Cohérence** : sous le mode déterministe, les cellules relancées de `p_equ` et
  `p_ctrl_vanilla_replay_tuned` doivent être identiques aux cellules d'E1-small
  (`pre_correct`, `post_correct`, scores, `norm_change`), comparées **par empreinte**. Un écart
  est rapporté comme incident d'instrument ; l'appariement reste celui du nouveau commit.

### 4.3 « Options seules »

- Par graine : `O_s` = précision « options seules » de la base sur les 162 items de
  `_cell_split(s)` ; `Q_s` = `pre` des cellules enregistrées (même items, même base, déjà lu
  dans #91).
- **Contrastes** : `Q_s − O_s` (la question apporte-t-elle quelque chose au-delà des
  options ?) et `O_s − 0,25` (les options seules dépassent-elles le hasard ?), bornes
  appariées et nul par inversion de signe comme en §4.1.
- **Peut échouer** dans les deux sens : `O` peut être au hasard ou proche de `Q`. Pas de
  porte d'apprentissage (aucun entraînement) : dit tel quel.
- **Lecture** : `L(Q − O) > 0` → va dans le sens de « la question apporte de la précision
  au-delà des options seules » ; `L(O − 0,25) > 0` → va dans le sens de « une partie de la
  précision de base est atteignable sans la question » ; `U(O − 0,25) < δ` → options seules
  au plus δ au-dessus du hasard. Descriptifs : part `(O − 0,25)/(Q − 0,25)`, valeurs par
  source (`daily_oracle` en premier), précision item par item sur les 1 098 items avec un
  intervalle binomial.
- **Réserve** : les graines partagent des items (162 items tirés parmi 1 058 par graine), les
  `O_s` ne sont pas indépendants ; l'inférence est conditionnelle au corpus gelé (prereg
  §5.2, `:652-655`).

## 5. Le plantage du 2026-11-01 et le garde-fou numérique

- **Fait** (rapport #91, `docs/results/2026-11-06-e1-small-analysis.md:47-50`) : le run s'est
  arrêté le 2026-11-01 à 20:20:33 CET sur `ValueError: math domain error` dans `choice_kl`
  (`scripts/e1_diagnostics.py:74`, appel `scripts/e1_harness.py:276`) pendant la cellule
  `p_ctrl_lr_matched|2` (LR 1,5e-3), non écrite.
- **Déduction** (lecture du code, non reproduite sur GPU) : `math.log` ne lève cette erreur que
  pour un argument ≤ 0 ; les termes `a = 0` sont filtrés et un softmax est ≥ 0 ; c'est donc
  `math.log(b)` avec `b = 0` exactement, c'est-à-dire un `exp` qui sous-déborde en float64 :
  un score de choix post plus de ~745 en dessous du meilleur.
- **Correctif minimal** (`scripts/e1_diagnostics.py`, `choice_kl`) : quand `b = 0`, `log b` est
  pris en espace logarithmique (log-sum-exp) ; tout terme avec `b > 0` est calculé comme
  avant, donc toute valeur déjà calculée est reproduite bit pour bit (test
  `test_choice_kl_underflow_guard`). C'est un garde-fou numérique sur une mesure descriptive
  (`kl_ref`, aucune porte, aucun état) ; **pas un changement de méthode**.
- **Nécessaire ?** Pour `p_equ` et `p_ctrl_vanilla_replay_tuned`, non : 282 cellules sans
  incident. Pour les deux variantes, le comportement post-perturbation n'est pas connu : le
  garde-fou évite qu'un seul sous-débordement arrête la campagne. Il ne relance pas le bras
  `p_ctrl_lr_matched` (hors de cet addendum).

## 6. Code ajouté (même commit que cet addendum)

- `scripts/e1_profiles.py` : profils `p_ctrl_samenorm_r` / `_u` (`SAMENORM`,
  `CONTROL_PROFILES`), hors de `ALL_PROFILES` : le mode `confirmatory` est inchangé.
- `scripts/pilot_cycle3_cuda.py` : `downscale_norm`, `perturb_same_norm` (par blocs,
  mémoire temporaire bornée à quelques centaines de Mo).
- `scripts/e1_harness.py` : modes `controls` (graines 0-140, ordre graine par graine, lignes
  de progression **sans score**) et `options-only` (base, partition CONFIRMATORY, aucun score
  imprimé) ; fichier `HL_STOP_FILE` vérifié entre deux cellules ; instrument enregistré
  étendu à ces deux modes, toutes ses autres vérifications inchangées.
- `scripts/e1_corpus_io.py` : les deux modes lisent la partition CONFIRMATORY d'un corpus
  FROZEN.
- `scripts/e1_diagnostics.py` : garde-fou de §5.
- `scripts/run_e1_small_controls.sh` : pilote (vérifie les empreintes d'amendement, de
  manifeste et de corpus, puis `options-only`, puis `controls`, journal `out/driver.log`,
  fichier `STOP`).
- Tests : `tests/unit/scripts/test_e1_controls.py`, `test_e1_diagnostics.py`.

## 7. Enregistrement du lancement

*Complété après le lancement. Aucune valeur de résultat (`pre`, `post`, `Δ`, score, précision)
n'a été lue ; seuls des champs de progression et d'intégrité l'ont été.*

**État de KXKM-AI avant le lancement** (lecture seule) :

- 14:03 (relevé de l'orchestrateur) puis 14:07 et 14:17 CET : aucune session tmux ; GPU
  1 369 Mio utilisés sur 24 564 (seul processus GPU : ComfyUI, PID 712018, 652 Mio ; NVML lit
  ~43 % d'utilisation à vide) ; 22 681 Mio libres ; RAM disponible 24 Gio sur 62.
- Job en cours : P1bis/P8c (`/ai/saisail/p1bis-p8c-20261104`, relancé le 2026-11-05,
  `experiments.hopfield_dream_conf.p8c_nocap --confirmatory`), **sur CPU uniquement** (aucune
  mémoire GPU), 2 à 4 processus à 100 % d'un cœur. La condition d'arrêt du brief (VRAM des deux
  jobs > 22 Go) n'est pas atteinte : P8c 0 Go + ce run ≈ 19,4 Go alloués.
- Tests unitaires E1 (CPU, venv de la campagne, clone frais, `CUDA_VISIBLE_DEVICES` vide) :
  121 réussis (`test_e1_controls`, `test_e1_diagnostics`, `test_e1_profiles`,
  `test_pilot_cycle3_cuda`, `test_e1_study`, `test_e1_small_torch`, `test_e1_corpus_io`,
  `test_analyze_e1_cuda`, `test_e1_budget`) ; `test_e1_schedule.py` non lancé (`hypothesis`
  absent du venv de la campagne ; module non modifié).

**Lancement :**

- Clone frais `/ai/saisail/e1-small-controls-20261106/dream-of-kiki`, branche
  `docs/e1-small-controls-addendum-2026-11-06`, commit `8c1d57a` (propre).
- Environnement de la campagne : venv `/ai/saisail/e1-small/venv`, poids
  `/ai/saisail/e1-small/models` (10 fichiers vérifiés par le runner), corpus
  `/ai/saisail/e1-small/corpus/e1-corpus.jsonl` ; `CUDA_HOME` non requis (aucun noyau
  compilé, fallback torch).
- Contrôle mémoire : `HL_MIN_FREE_MIB=21000` (valeur du run enregistré ; pic d'une cellule
  ≈ 20,1 Go). Le seuil générique « 4 Go libres » ne suffit pas à ce harnais.
- Commande :
  `cd /ai/saisail/e1-small-controls-20261106 && tmux new-session -d -s e1s-controls "bash
  /ai/saisail/e1-small-controls-20261106/dream-of-kiki/scripts/run_e1_small_controls.sh"`
- Départ : 2026-11-06 14:17:45 CET (`out/driver.log`, empreintes amendement `3b76adc6…ac49`,
  manifeste `093fdf61…02ab`, corpus `f8379b81…c0a2` vérifiées) ; tmux `e1s-controls` ;
  PID pilote 3394033, harnais 3404927 (étape `controls`).
- Arrêt propre : `touch /ai/saisail/e1-small-controls-20261106/STOP` (vérifié entre deux
  étapes et entre deux cellules ; la cellule en cours se termine).
- Sortie : `/ai/saisail/e1-small-controls-20261106/out/{options-only,controls}/`, journaux
  `options-only.log`, `controls.log` (sans score), `driver.log`.

**Premiers relevés (intégrité seulement) :**

- `options-only` : terminé 14:25:54 CET, `rc=0`, 1 098 items CONFIRMATORY notés, instrument
  enregistré (`backend torch`, `linear_attn torch`, déterministe), code `8c1d57a` ;
  `e1-options-only.json` SHA-256 `061054d425b4a9a5504bc7dc451f3485cabff4abad129fa9aaf62b7fc4f6de4a`.
- `controls`, graine 0 (4 cellules, 14:47 CET) : budgets atteints (7 605 / 7 605 jetons),
  aucune violation d'invariant, `kl_ref` défini, pic `max_memory_allocated` 19,43 Go ; porte de
  conformité : norme appliquée / ρ = 1,012 (R) et 1,022 (U), `N′` = 4 205 751 296 éléments non
  nuls (tous les paramètres).
- Durées graine 0 : 312, 248, 404 et 299 s (`p_equ`, vanilla, R, U), contre 177 s en
  moyenne pour `p_equ` et vanilla dans E1-small ; la contention CPU de P8c (4 cœurs pleins) est
  la cause probable, non isolée.

**ETA (564 cellules) :** au rythme de la graine 0 (≈ 1 262 s par graine), ≈ 49 h, fin vers
le 2026-11-08 ~16:00 CET ; au rythme moyen d'E1-small (177 s par cellule, 708 à 908 s par
graine selon le surcoût des perturbations), ≈ 28 à 36 h, fin entre le 2026-11-07 ~18:30 et le
2026-11-08 ~02:30 CET. Départ de l'étape `controls` : 14:26 CET.

## 8. À coller dans le champ d'amendement OSF

---

**Title:** E1-small: control addendum — H14-B same-norm arm and options-only control

**Summary:** Exploratory control addendum of an already frozen campaign, PI decision of
2026-11-06. Written after the registered analysis of E1-small had been read (all H14
hypotheses already exploratory; amendment of 2026-10-30, §2bis); everything it produces is
exploratory and informed by that reading, spends no α and changes no registered state.
(1) H14-B same-norm arm declared on 2026-11-02: after the same tuned replay as `p_equ`, a
perturbation of the same norm as the DOWNSCALE step replaces it (R: isotropic random
direction; U: uniform magnitude toward zero), on seeds 0-140, with `p_equ` and
`p_ctrl_vanilla_replay_tuned` re-run under the same commit for pairing. (2) Options-only
control announced on 2026-10-30: the base model scores the 1,098 confirmatory items with the
question removed (prompt "Answer:"); the per-profile part is not run. Reading rules, written
before launch: registered paired procedures at nominal α = 0.05, the paired sign-flip null
of the analysis report, a learning gate, a perturbation-conformity gate and sensitivity
labels; a control that cannot fail is reported as non-informative. A numerical guard was
added to a descriptive diagnostic (`choice_kl`) that stopped the run on 2026-11-01; it
reproduces every earlier value bit for bit and changes no method.

---

## 9. End record (2026-11-08)

Data record only. No analysis, no verdict and no interpretation is made in this section; the
reading rules of section 8 apply to the analysis, which is the PI's.

- Host: kxkm-ai (RTX 4090).
- Start: 2026-11-06 14:17:45 CET. End: 2026-11-08 05:34:03 CET (`out/driver.log`, final line
  `done`; `options-only rc=0`, `controls rc=0 cells=564`).
- Code commit `8c1d57a43248a2e6ebe722cd21f2021123881d49`; amendment sha256 `3b76adc6…`;
  manifest sha256 `093fdf61…`; corpus sha256 `f8379b81…` (full hashes in the first line of
  `driver.log`).
- Cells: 564 = 4 profiles x 141 seeds (0-140). Profiles: `p_equ`,
  `p_ctrl_vanilla_replay_tuned`, `p_ctrl_samenorm_r`, `p_ctrl_samenorm_u` (141 cells each).
  All 564 cells carry `budget.status = matched` and `invariant_violation = null`.
- Options-only control: `options-only/e1-options-only.json` (written 2026-11-06 14:25 CET).
- Files in this repository: `docs/milestones/e1-small/controls-2026-11-06/` (cells, options-only
  result, three logs, `SHA256SUMS`). sha256 of the cells file:
  `7b3ec589b358f8676f5c18177f1751ded806cf0b28257d0ee352cfb4b8a06247`
  (`e1-controls-torch-Qwen_Qwen3.5-4B.cells.jsonl`, 564 lines); of the options-only result:
  `061054d425b4a9a5504bc7dc451f3485cabff4abad129fa9aaf62b7fc4f6de4a`.
- Archive on electron-server (verified by sha256 against kxkm-ai on 2026-11-08):
  `~/hypneum-archives/e1-small-controls-20261106-20261108/out/`.

---

## 10. Results (2026-11-08)

Written on 2026-11-08 on the PI's decision to record the numbers, the paired null, the
section 4.2 consistency check and the pre-registered reading rules applied to them. Every
number below is read from `controls_analysis.json` and `sec42.json` (section 10.8). Rules are
cited by line of this file (`l.N`) and translated from the French text of sections 4.1-4.3;
labels in French are the addendum's own. Nothing here goes beyond what those rules assign.
All quantities are exploratory and informed (section 1), at nominal alpha = 0.05
(one-sided bounds at 0.95), outside any family (`l.128-130`).

**Procedure notes.** The repository holds no analysis script for the controls cells. The
script `e1/analyse_controls.py` was written on 2026-11-08 on macM1 for this purpose; it
implements sections 4.1-4.3 and reuses `bounds`, `signflip` and the bootstrap of
`scripts/e1_small_null_supplement.py`. It was not pre-registered and is not part of the
commit `8c1d57a` that produced the cells. Paired sign-flip null as in `l.131-134`: 10,000
draws, seed `20261106`, `p = (k+1)/(N+1)`, smallest attainable `p` = 9.999e-05 (reported
below as 1.0e-04). Bootstrap intervals are descriptive. `L` and `U` are the paired one-sided
95% lower and upper bounds. `g_z` is the mean divided by the standard deviation of the paired
differences. "Discordance" is the mean over seeds of the share of the 162 evaluation items
whose post-replay correctness differs between the two arms.

### 10.1 Integrity and validity

Rule (`l.137-139`): "invariant violation or non-finite score -> cell excluded; more than 7
seeds lost for a contrast -> UNINTERPRETABLE; collapse rate > 20% of an arm ->
UNINTERPRETABLE."

- Cells: 564 = 4 profiles x 141 seeds (`p_ctrl_samenorm_r` 141, `p_ctrl_samenorm_u` 141, `p_ctrl_vanilla_replay_tuned` 141, `p_equ` 141); missing: 0.
- Invariant violations: 0; non-finite scores: 0; `budget.status`: matched for all cells.
- Collapse rate: `p_ctrl_samenorm_r` 0%, `p_ctrl_samenorm_u` 0%, `p_ctrl_vanilla_replay_tuned` 0%, `p_equ` 0%.
- `n_eval` = 162 items for every cell; `pre_correct` identical across the four arms within each seed: true. Single code commit `8c1d57a`, single amendment sha256 `3b76adc6...`.
- No cell is excluded, no seed is lost, no arm exceeds the collapse threshold: the validity rule excludes nothing and does not trigger UNINTERPRETABLE.

### 10.2 Consistency check (section 4.2)

Rule (`l.164-167`): "under the deterministic mode, the re-run cells of `p_equ` and
`p_ctrl_vanilla_replay_tuned` must be identical to the E1-small cells (`pre_correct`,
`post_correct`, scores, `norm_change`), compared by fingerprint. A discrepancy is reported as
an instrument incident; the pairing remains that of the new commit."

- `p_equ`: 141/141 seeds identical to the main-run cell on all non-metadata fields (`pre_correct`, `post_correct`, scores, `norm_change`, `delta`, `mean_replay_loss`, `drift`, `kl_ref`, `lost_correct`; max absolute difference 0 on each); items whose post-replay correctness differs: 0.
- `p_ctrl_vanilla_replay_tuned`: 141/141, same fields, same result; items differing: 0.
- `eval_sources` fingerprint `a7da36fa60af` on both sides; learning rate, `m` and replay count identical (3e-4, 2, 40); DOWNSCALE `f` = 0.99 for `p_equ`, none for vanilla, on both sides.
- Only `code_commit` differs: main run `e93ea81`, controls run `8c1d57a`. Library versions, corpus and model are equal (`versions_diff` empty).
- No discrepancy is found; no instrument incident is to be reported under the rule.
- The H14-B difference recomputed from the main-run cells gives mean -1.97e-18, sd 0.010053, 100 seeds with a non-zero paired difference and 128 seeds with differing `post_correct`; the same quantity from the controls cells is identical in mean (-1.97e-18) and its sd is 0.010089 in section 10.4 (ratio 1.00357 = sqrt(141/140): the two sd differ by the degrees-of-freedom convention of the two scripts, as checked numerically on these values, not in code).
- Context, not part of the rule: the main run stopped with `rc=1` at 1271 of 1410 cells (`p_ctrl_lr_matched` 2 of 141 seeds; section 5). The cells compared here are the 282 complete cells of the two re-run arms. The main-run output is archived on electron-server at `~/hypneum-archives/e1-small-main-run-20261029-20261101/`.

### 10.3 Learning gate

Rule (`l.153-155`): "for each of the four arms, delta > 0 with sign-flip p < 0.05. An arm that
does not pass the gate makes the contrasts that use it non-informative."

| arm | pre | post | mean Δ | L (95%) | sign-flip p | seeds with Δ > 0 | gate |
|---|---|---|---|---|---|---|---|
| `p_equ` | 0.3357 | 0.3991 | +0.0634 | +0.0586 | 1.0e-04 | 136/141 | passed |
| `p_ctrl_vanilla_replay_tuned` | 0.3357 | 0.3991 | +0.0634 | +0.0585 | 1.0e-04 | 136/141 | passed |
| `p_ctrl_samenorm_r` | 0.3357 | 0.3988 | +0.0631 | +0.0583 | 1.0e-04 | 136/141 | passed |
| `p_ctrl_samenorm_u` | 0.3357 | 0.4000 | +0.0643 | +0.0596 | 1.0e-04 | 137/141 | passed |

All four arms: Δ > 0 with sign-flip p = 1.0e-04 (the smallest attainable). Learning gate: passed
for the four arms; no contrast is made non-informative by this rule. `pre` is the same for
the four arms (same base, same 162 items per seed).

### 10.4 Contrasts D_R, D_U and descriptives

Contrasts (`l.145-148`): `D_R(s) = Δ_s(p_equ) − Δ_s(p_ctrl_samenorm_r)`, `D_U(s) = Δ_s(p_equ) − Δ_s(p_ctrl_samenorm_u)`;
descriptives: each variant minus vanilla, and H14-B recomputed in this commit. n = 141 seeds each.

| contrast | mean | sd | L | U | bootstrap 95% CI | sign-flip p | g_z | discordance | seeds with identical post | null P99 of mean |
|---|---|---|---|---|---|---|---|---|---|---|
| D_R = Δ(p_equ) − Δ(samenorm_r) | +0.00031 | 0.01127 | −0.00127 | +0.00188 | [−0.00153; +0.00210] | 0.391 | +0.027 | 0.0201 | 6 | 0.00223 |
| D_U = Δ(p_equ) − Δ(samenorm_u) | −0.00088 | 0.01202 | −0.00255 | +0.00080 | [−0.00285; +0.00101] | 0.815 | −0.072 | 0.0194 | 4 | 0.00228 |
| desc: Δ(samenorm_r) − Δ(vanilla) | −0.00031 | 0.00974 | −0.00166 | +0.00105 | [−0.00188; +0.00131] | 0.631 | −0.031 | 0.0164 | 9 | 0.00197 |
| desc: Δ(samenorm_u) − Δ(vanilla) | +0.00088 | 0.01147 | −0.00072 | +0.00247 | [−0.00105; +0.00271] | 0.185 | +0.076 | 0.0197 | 8 | 0.00228 |
| desc: H14-B recomputed, Δ(p_equ) − Δ(vanilla) | 0.00000 | 0.01009 | −0.00141 | +0.00141 | [−0.00166; +0.00162] | 0.523 | 0.000 | 0.0158 | 13 | 0.00201 |

Seeds with paired difference below / equal to / above 0: D_R 53 / 34 / 54; D_U 54 / 39 / 48. The observed means of D_R (+0.00031) and D_U (−0.00088) are below the 99th percentile of the sign-flip null of the mean (0.00223 and 0.00228). The H14-B recomputation reproduces, in this commit, the mean difference of the main run (-1.97e-18) with 13 seeds of identical post answers.

### 10.5 Reading under the pre-registered rules

**Restriction rule** (`l.149-152`): "if the lower bound `L` of `D_R` or `D_U` is not strictly
positive, no reading of H14-B is attributed to the content of DOWNSCALE rather than to a
perturbation of the same norm." Observed: L(D_R) = −0.00127 and L(D_U) = −0.00255; neither is strictly greater than 0.
The restriction rule applies (to both contrasts).

**Compliance gate** (`l.156-158`): "ratio applied norm / rho(s) outside [0.5; 2] on more than 7
seeds -> variant non-compliant, contrasts non-informative." Observed: median ratio 1.0121 for R (min and max equal to the median at 4 decimals) and 1.0220 for U (idem); seeds outside [0.5; 2]:
0 (R) and 0 (U). Both variants are within the gate. The target norm rho(s) is 7.677 (median; min and max equal at 3 decimals). A ratio near 1 follows from the construction of the same-norm controls (the perturbation is built to have the DOWNSCALE norm; the residual 1.2% and 2.2% come from the bfloat16 rounding of section 2.1), so the gate is informative about the implementation only, not about the contrasts.

**Sensitivity rule** (`l.159-163`): "post answers identical between the two arms on more than 70
of 141 seeds -> non-informative; otherwise mean post discordance < 0.05 -> `contrôle à faible
pouvoir de falsification` (low falsification power control); otherwise informative."
Observed: identical-post seeds 6 (D_R) and 4 (D_U), both at most 70; mean discordance 0.0201 (D_R) and 0.0194 (D_U), both below 0.05 (maximum per-seed discordance 0.0494 and 0.0494).
The first branch is not met and the second is: the label assigned by the rule to D_R and D_U is
`contrôle à faible pouvoir de falsification`. (The label "non informatif" belongs to the first branch and is not assigned.)

Combined with the restriction rule, the addendum assigns no attribution of the H14-B
difference to the content of DOWNSCALE, from controls whose power to differ is rated low by
its own sensitivity rule. The descriptive H14-B recomputation (mean -1.97e-18, L −0.00141, U +0.00141, discordance 0.0158, 13 seeds with identical post answers) is reported without reading.

### 10.6 Options-only control

Procedure and rule: `l.171-187`. Base model, no replay, question removed, 1,098 CONFIRMATORY
items scored once each (corpus `f8379b81...`, partition sha256 `546f44c6...`, status FROZEN); the
`eval_sources` of the cells match: 141/141 seeds. Overall: 300/1098 = 0.2732, Clopper-Pearson 95% [0.2470; 0.3006].
Per-seed mean over the 162 evaluation items: O = 0.2728, Q = 0.3357.

| source | k/n (items) | options-only accuracy | Clopper-Pearson 95% | O, seed-pooled | Q, seed-pooled |
|---|---|---|---|---|---|
| `daily_oracle` | 62/338 | 0.1834 | [0.1436; 0.2289] | 0.1809 | 0.1789 |
| `mmlu_cf` | 132/380 | 0.3474 | [0.2995; 0.3976] | 0.3461 | 0.4938 |
| `realtimeqa` | 106/380 | 0.2789 | [0.2344; 0.3270] | 0.2808 | 0.3157 |

Paired contrasts over 141 seeds (`l.174-176`):

| contrast | mean | sd | L | U | bootstrap 95% CI | sign-flip p | g_z | null P99 of mean |
|---|---|---|---|---|---|---|---|---|
| Q − O | +0.06282 | 0.03479 | +0.05797 | +0.06767 | [+0.05713; +0.06843] | 1.0e-04 | +1.796 | 0.01458 |
| O − 0.25 | +0.02283 | 0.03018 | +0.01862 | +0.02704 | [+0.01780; +0.02782] | 1.0e-04 | +0.752 | 0.00729 |

Reading rule (`l.179-184`), applied term by term:

- `L(Q − O) > 0`: L = +0.05797; the condition is met. The addendum's phrasing for this branch is "goes in the direction of 'the question brings accuracy beyond the options alone'".
- `L(O − 0.25) > 0`: L = +0.01862; the condition is met. Phrasing: "goes in the direction of 'part of the base accuracy is reachable without the question'".
- `U(O − 0.25) < δ` with δ = 0.01: U = +0.02704, which is not below 0.01; the condition is not met, so the "options alone at most δ above chance" phrasing is not assigned.
- Descriptive share `(O − 0.25)/(Q − 0.25)` of the means: 0.267.
- By source (descriptive): `daily_oracle` options-only accuracy 0.1834 with Clopper-Pearson upper bound 0.2289, below 0.25.

Caveat (`l.185-187`): the seeds share items (162 items drawn among 1,058 per seed), so the `O_s` are not independent; inference is conditional on the frozen corpus (prereg section 5.2). The sign-flip p and the bounds above should be read with this reserve.

### 10.7 Not done / limits

- The per-profile part of the options-only control (declared not launched, section 3.1) was not run.
- `p_ctrl_lr_matched` remains at 2 of 141 seeds in the main run (section 5 and 10.2); this addendum does not re-run it.
- The analysis script was written on 2026-11-08, after the data existed; it was not pre-registered. Its statistical routines are those of `scripts/e1_small_null_supplement.py`.
- Environment of the analysis (macM1): Python 3.14.4, numpy 2.5.4, scipy 1.18.1, torch 2.11.0 (CPU; used only for `randperm` in `_cell_split`), run with `nice -n 10` and `python -I`.
- The thresholds of the compliance and sensitivity rules were chosen after reading #91 (section 1); the options-only and same-norm results are exploratory.
- Within-seed `pre` is identical across arms by construction, so the contrasts measure post-replay differences only.

### 10.8 Reproducibility pointers

- Analysis host macM1, directory `~/hypneum-analysis-20261108/e1/` (script `analyse_controls.py`, output `out/controls_analysis.json`; §4.2 check in `sec42/`). Command: `nice -n 10 .venv/bin/python -I analyse_controls.py $PWD` with 10,000 draws and seed `20261106`; code under analysis `8c1d57a` (PR head `9333898` not used).
- Input sha256: cells `7b3ec589b358f8676f5c18177f1751ded806cf0b28257d0ee352cfb4b8a06247`; options-only result `061054d425b4a9a5504bc7dc451f3485cabff4abad129fa9aaf62b7fc4f6de4a`; corpus `f8379b810353d25e3c85e4e94b9a8d02553ea8c18abdef631a0289339497c0a2`.
- Controls output archive: electron-server `~/hypneum-archives/e1-small-controls-20261106-20261108/out/`; T7 on macM1 `/Volumes/T7/hypneum-archives/e1-small-controls-20261106-20261108/`.
- Main-run archive (for section 10.2): electron-server `~/hypneum-archives/e1-small-main-run-20261029-20261101/`; T7 `/Volumes/T7/hypneum-archives/e1-small-main-run-20261029-20261101/`.
- In this repository: `docs/milestones/e1-small/controls-2026-11-06/` (cells, options-only result, logs, `SHA256SUMS`; section 9).
