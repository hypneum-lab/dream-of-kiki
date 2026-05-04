"""G6-Studio Path A pilot — real-LoRA SpikingKiki-V4 35B-A3B-V4 × MMLU CL stream.

First real-LLM-scale validation of framework C 4-channel coupling
(REPLAY + DOWNSCALE + RESTRUCTURE + RECOMBINE) on the real
``SpikingKiki-35B-A3B-V4`` spiking LIF Qwen substrate (58 GB,
31 070 ``.npz`` modules, 34 LoRA adapters). Per subdomain : real
``mlx_lm.tuner.lora`` fine-tune → optional profile-dependent
``dream_episode_real()`` whose four handlers operate on the **live**
LoRA delta tensors → ``mlx_lm.generate`` letter-argmax MMLU eval.

Load-bearing change vs G6 Path B (``experiments/g6_mmlu_stream``) :
the four handlers mutate the live LoRA delta dict by reference
rather than synthetic payloads disjoint from the eval surface,
fixing the spectator caveat documented in
``experiments/g6_mmlu_stream/run_g6.py:539-578``.

Pre-registration : ``docs/osf-prereg-g6-studio-path-a.md``
(LOCKED at commit ``fae8c32``, 2026-05-04).
Plan : ``docs/superpowers/plans/2026-05-04-g6-studio-path-a-real-lora.md``.
"""
from __future__ import annotations
