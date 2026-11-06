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
