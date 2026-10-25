"""E1 profiles, tuning selection and LR matching (task 4 of section 9).

**Gate** : ``docs/osf-prereg-e1-replay-controls-2026-10-23.md`` sections
5.2 (``n_eval`` rule) and 5.4 (profiles, tuning, ``p_ctrl_lr_matched``).
**Mode** : stdlib only. Builds each profile's replay corpus from the
seed's replay items, gives each profile its (LR, m, DOWNSCALE,
RESTRUCTURE) configuration, selects the tuned configurations on the
tuning seeds and calibrates the LR multiplier c, then writes the frozen
tuning amendment that confirmatory cells require.

Profiles (9 registered + 1 exploratory):

=============================  ==========  =================  =========
profile                        corpus      LR, m              ops
=============================  ==========  =================  =========
p_min                          facts       default            -
p_equ                          facts       selection (i)      DOWNSCALE
p_max                          facts       default            DOWNSCALE, RESTRUCTURE
p_ctrl_sft                     neutral     default            -
p_ctrl_shuffle                 shuffled    default            -
p_noise                        none        -                  -
p_ctrl_template                prompts     default            -
p_ctrl_vanilla_replay_tuned    facts       selection (i)      -
p_ctrl_onpolicy_replay         OPR         selection (ii)     -
p_ctrl_lr_matched (explor.)    facts       c x LR (i), m (i)  -
=============================  ==========  =================  =========

Default: LR 1e-4, m = 5. DOWNSCALE factor 0.99 (untuned).
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

Item = dict[str, Any]
Config = tuple[float, int]  # (lr, m)

DEFAULT: Config = (1e-4, 5)
GRID_LR = (3e-5, 1e-4, 3e-4)
GRID_M = (2, 5)
GRID: tuple[Config, ...] = tuple((lr, m) for lr in GRID_LR for m in GRID_M)
C_GRID = (1.0, 1.5, 2.0, 3.0, 5.0)
SHRINK = 0.99
TIE = 1e-9
SD_STAR = 0.039661  # section 5.2, as printed by scripts/prereg_power_2026_10.py
N_EVAL_FLOOR = 130
N_REPLAY = 40
MAX_OPR_DROPPED_TUNING = 2

REGISTERED = ("p_min", "p_equ", "p_max", "p_ctrl_sft", "p_ctrl_shuffle", "p_noise",
              "p_ctrl_template", "p_ctrl_vanilla_replay_tuned", "p_ctrl_onpolicy_replay")
EXPLORATORY = ("p_ctrl_lr_matched",)
ALL_PROFILES = REGISTERED + EXPLORATORY
TUNED = ("p_equ", "p_ctrl_vanilla_replay_tuned", "p_ctrl_onpolicy_replay", "p_ctrl_lr_matched")

# Neutral filler of the matched-compute SFT control (unchanged from the
# first CUDA slice): content unrelated to the corpus.
NEUTRAL_TEXTS = (
    "The weather changed slowly from one day to the next.",
    "A long road wound quietly through the empty valley.",
    "She closed the wooden door and walked down the street.",
    "Clouds drifted above the distant grey hills at dusk.",
    "The old clock kept its steady rhythm through the night.",
)


@dataclass(frozen=True)
class ProfileConfig:
    profile: str
    corpus: str  # facts | neutral | shuffled | prompts | opr | none
    lr: float | None
    m: int | None
    downscale: bool
    restructure: bool
    scoring_charged: bool  # OPR's 160 scoring passes are charged (section 5.4)


@dataclass(frozen=True)
class Amendment:
    """The frozen tuning amendment (section 5.4); required by confirmatory cells."""

    selection_vanilla: Config
    selection_opr: Config
    c: float
    n_eval: int

    def as_dict(self) -> dict[str, Any]:
        return {"selection_vanilla": list(self.selection_vanilla),
                "selection_opr": list(self.selection_opr), "c": self.c, "n_eval": self.n_eval}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Amendment:
        sv, so = d["selection_vanilla"], d["selection_opr"]
        return cls((float(sv[0]), int(sv[1])), (float(so[0]), int(so[1])),
                   float(d["c"]), int(d["n_eval"]))


def profile_config(profile: str, amendment: Amendment | None) -> ProfileConfig:
    if profile not in ALL_PROFILES:
        raise ValueError(f"unknown profile {profile!r}")
    if profile in TUNED and amendment is None:
        raise ValueError(f"{profile} needs the frozen tuning amendment")
    lr, m = DEFAULT
    if profile in ("p_equ", "p_ctrl_vanilla_replay_tuned"):
        assert amendment is not None
        lr, m = amendment.selection_vanilla
    elif profile == "p_ctrl_onpolicy_replay":
        assert amendment is not None
        lr, m = amendment.selection_opr
    elif profile == "p_ctrl_lr_matched":
        assert amendment is not None
        lr, m = amendment.c * amendment.selection_vanilla[0], amendment.selection_vanilla[1]
    corpus = {"p_ctrl_sft": "neutral", "p_ctrl_shuffle": "shuffled", "p_noise": "none",
              "p_ctrl_template": "prompts", "p_ctrl_onpolicy_replay": "opr"}.get(profile, "facts")
    if profile == "p_noise":
        return ProfileConfig(profile, "none", None, None, False, False, False)
    return ProfileConfig(profile, corpus, lr, m,
                         downscale=profile in ("p_equ", "p_max"),
                         restructure=profile == "p_max",
                         scoring_charged=profile == "p_ctrl_onpolicy_replay")


def fact_text(item: Item) -> str:
    return str(item["prompt"]) + str(item["choices"][int(item["answer"])])


def profile_texts(
    corpus: str,
    replay_items: Sequence[Item],
    seed: int,
    shuffle: Callable[[str, int], str] | None = None,
    opr_choice: Sequence[int] | None = None,
) -> list[str]:
    """Replay corpus in ``_cell_split`` order for a corpus kind."""
    facts = [fact_text(it) for it in replay_items]
    if corpus == "none":
        return []
    if corpus == "facts":
        return facts
    if corpus == "neutral":
        reps = len(facts) // len(NEUTRAL_TEXTS) + 1
        return list(NEUTRAL_TEXTS * reps)[: max(1, len(facts))]
    if corpus == "shuffled":
        if shuffle is None:
            raise ValueError("shuffled corpus needs a shuffle function")
        return [shuffle(f, seed) for f in facts]
    if corpus == "prompts":
        return [str(it["question_prompt"]) for it in replay_items]
    if corpus == "opr":
        if opr_choice is None or len(opr_choice) != len(replay_items):
            raise ValueError("OPR corpus needs one pre-replay argmax per replay item")
        return opr_texts(replay_items, opr_choice)
    raise ValueError(f"unknown corpus kind {corpus!r}")


def opr_texts(replay_items: Sequence[Item], argmax: Sequence[int]) -> list[str]:
    """Restricted MC-OPR: prompt + the checkpoint's argmax choice, reward-1 items only."""
    return [str(it["prompt"]) + str(it["choices"][a])
            for it, a in zip(replay_items, argmax, strict=True) if a == int(it["answer"])]


# --- tuning ------------------------------------------------------------------


def select_config(mean_delta: Mapping[Config, float]) -> Config:
    """Highest mean delta; ties within 1e-9 go to the lower LR, then smaller m."""
    if set(mean_delta) != set(GRID):
        raise ValueError(f"tuning needs exactly the 6 grid configurations (got {sorted(mean_delta)})")
    best = sorted(GRID)[0]
    for cfg in sorted(GRID):
        if mean_delta[cfg] > mean_delta[best] + TIE:
            best = cfg
    return best


def calibrate_c(norms: Mapping[float, Sequence[float]], ref: Sequence[float]) -> float:
    """c with the smallest |log(median norm(c) / median norm(p_equ))|; ties to smaller c."""
    if set(norms) != set(C_GRID):
        raise ValueError(f"calibration needs c in {C_GRID} (got {sorted(norms)})")
    r = statistics.median(ref)
    best, best_v = None, math.inf
    for c in sorted(C_GRID):
        v = abs(math.log(statistics.median(norms[c]) / r))
        if v < best_v - 1e-12:
            best, best_v = c, v
    assert best is not None
    return best


def n_eval_rule(p_disc: float, confirmatory_size: int) -> dict[str, Any]:
    """Section 5.2: ``max(130, ceil(p_disc / (SD*^2 / 2)))``, capped at size - 40."""
    raw = max(N_EVAL_FLOOR, math.ceil(p_disc / (SD_STAR ** 2 / 2) - 1e-12))
    cap = confirmatory_size - N_REPLAY
    return {"n_eval_rule": raw, "n_eval": min(raw, cap), "capped": raw > cap, "cap": cap}


def tuning_amendment(cells: Sequence[Mapping[str, Any]], confirmatory_size: int) -> dict[str, Any]:
    """Freeze selections (i) and (ii), c, the SD_D proxy, p_disc and n_eval.

    ``cells`` are the tuning and calibration cell records of
    ``pilot_cycle3_cuda.py``: ``kind`` in {tune_vanilla, tune_opr,
    calib_lr, calib_ref}, ``seed``, ``lr``, ``m``, ``c`` (calib_lr),
    ``delta``, ``post_correct`` (per eval item), ``opr_corpus_size``
    (tune_opr) and ``norm_change`` (calibration).
    """
    seeds = sorted({int(c["seed"]) for c in cells})
    tune = {(c["kind"], int(c["seed"]), float(c["lr"]), int(c["m"])): c for c in cells
            if c["kind"] in ("tune_vanilla", "tune_opr")}
    calib = {(int(c["seed"]), float(c["c"])): c for c in cells if c["kind"] == "calib_lr"}
    calib_ref = {int(c["seed"]): c for c in cells if c["kind"] == "calib_ref"}

    def mean_over(kind: str, keep: Sequence[int]) -> dict[Config, float]:
        return {(lr, m): statistics.fmean(float(tune[(kind, s, lr, m)]["delta"]) for s in keep)
                for lr, m in GRID}

    sel_v = select_config(mean_over("tune_vanilla", seeds))
    empty = sorted({int(c["seed"]) for c in cells
                    if c["kind"] == "tune_opr" and int(c["opr_corpus_size"]) == 0})
    opr_seeds = [s for s in seeds if s not in empty]
    opr_default = len(empty) > MAX_OPR_DROPPED_TUNING
    sel_o = DEFAULT if opr_default else select_config(mean_over("tune_opr", opr_seeds))
    diffs, disc = [], []
    for s in opr_seeds:
        v = tune[("tune_vanilla", s, sel_v[0], sel_v[1])]
        o = tune[("tune_opr", s, sel_o[0], sel_o[1])]
        diffs.append(float(v["delta"]) - float(o["delta"]))
        pv, po = list(v["post_correct"]), list(o["post_correct"])
        if len(pv) != len(po) or not pv:
            raise ValueError(f"seed {s}: post_correct lengths differ or are empty")
        disc.append(sum(a != b for a, b in zip(pv, po, strict=True)) / len(pv))
    for (s, c), cell in calib.items():  # calibration must use selection (i)
        if not (math.isclose(float(cell["lr"]), c * sel_v[0]) and int(cell["m"]) == sel_v[1]):
            raise ValueError(f"calib_lr seed {s} c={c} not run at c x LR(i), m(i)")
    for s, cell in calib_ref.items():
        if not (math.isclose(float(cell["lr"]), sel_v[0]) and int(cell["m"]) == sel_v[1]):
            raise ValueError(f"calib_ref seed {s} not run at selection (i)")
    norms = {c: [float(calib[(s, c)]["norm_change"]) for s in seeds] for c in C_GRID}
    ref = [float(calib_ref[s]["norm_change"]) for s in seeds]
    c_sel = calibrate_c(norms, ref)
    p_disc = statistics.fmean(disc) if disc else float("nan")
    rule = n_eval_rule(p_disc, confirmatory_size)
    return {
        "selection_vanilla": list(sel_v), "selection_opr": list(sel_o),
        "opr_default_kept": opr_default, "opr_empty_tuning_seeds": empty,
        "c": c_sel, "sd_d_proxy": statistics.stdev(diffs) if len(diffs) > 1 else None,
        "sd_d_proxy_exceeds_sd_star": (statistics.stdev(diffs) > SD_STAR
                                       if len(diffs) > 1 else None),
        "p_disc": p_disc, **rule, "tuning_seeds": seeds,
    }
