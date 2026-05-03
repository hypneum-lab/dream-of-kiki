"""Smoke test for the G5 cross-substrate pilot driver — package import only."""
from __future__ import annotations


def test_g5_package_importable() -> None:
    """The `experiments.g5_cross_substrate` package and its `run_g5`
    module must import without side effects (matches G4-bis pattern)."""
    from experiments.g5_cross_substrate import run_g5  # noqa: F401

    assert hasattr(run_g5, "main")
    assert callable(run_g5.main)


def test_g5_main_help_returns_zero() -> None:
    """`main(['--help'])` exits cleanly via SystemExit(0) (argparse)."""
    import pytest

    from experiments.g5_cross_substrate import run_g5

    with pytest.raises(SystemExit) as exc_info:
        run_g5.main(["--help"])
    assert exc_info.value.code == 0
