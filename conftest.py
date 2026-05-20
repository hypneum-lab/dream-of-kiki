"""Root conftest.

Skip MLX-dependent test modules on non-Darwin platforms. MLX ships
wheels only for arm64-darwin, so importing these modules on Linux CI
runners fails with `ImportError: libmlx.so: cannot open shared object
file`. Use collect_ignore_glob to drop them before collection so the
ImportError never fires.
"""
from __future__ import annotations

import sys

collect_ignore_glob: list[str] = []

if sys.platform != "darwin":
    collect_ignore_glob = [
        "tests/conformance/axioms/test_dr3_*.py",
        "tests/conformance/test_baseline_registration.py",
        "tests/unit/experiments/test_g4_*.py",
        "tests/unit/experiments/test_g5_*.py",
        "tests/unit/experiments/test_g6_*.py",
        "tests/unit/test_esnn_norse.py",
        "tests/unit/test_g4_quinto_*.py",
        "tests/unit/test_micro_kiki_*.py",
        "tests/unit/test_qwen_mlx_*.py",
        "tests/unit/test_real_eval.py",
        "tests/unit/test_substrate_*.py",
        "tests/unit/test_wake_sleep_baseline_adapter.py",
    ]
