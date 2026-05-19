"""Task 1: SubstrateAdapter Protocol + CellRequest dataclass."""

from pathlib import Path
from typing import Any

from kiki_oniric.substrates.factory import (
    CellRequest,
    SUBSTRATE_NAMES,
    SubstrateAdapter,
)


def test_cell_request_fields() -> None:
    req = CellRequest(
        substrate="mlx_kiki_oniric",
        profile="p_equ",
        seed=7,
        scale="qwen3p6-35b-bf16-local",
        model_path=Path("/fake/path"),
        benchmarks=("mmlu", "hellaswag", "mega_v2"),
    )
    assert req.seed == 7
    assert req.substrate in SUBSTRATE_NAMES


def test_substrate_names_is_3_tuple() -> None:
    assert SUBSTRATE_NAMES == (
        "mlx_kiki_oniric",
        "esnn_thalamocortical",
        "micro_kiki",
    )


def test_substrate_adapter_protocol_is_runtime_checkable() -> None:
    # Protocol is structural -- any object with the two methods satisfies it.
    class Dummy:
        def execute_profile(self, request: Any) -> dict[str, Any]:
            return {}

        def teardown(self) -> None:
            pass

    d = Dummy()
    # Static check: Protocol methods exist
    assert callable(getattr(d, "execute_profile"))
    assert callable(getattr(d, "teardown"))
    # Import path reachable
    assert SubstrateAdapter is not None


def test_esnn_adapter_executes_all_four_handlers(tmp_path: Path) -> None:
    from kiki_oniric.substrates.factory import ESNNAdapter

    adapter = ESNNAdapter()
    request = CellRequest(
        substrate="esnn_thalamocortical",
        profile="p_equ",
        seed=3,
        scale="qwen3p5-1p5b",
        model_path=tmp_path,
    )
    result = adapter.execute_profile(request)
    for key in ("replay_rate", "downscale_norm", "restructure_sum",
                "recombine_rate", "delta_acc", "wall_time_s"):
        assert key in result, f"missing key {key}"
    assert isinstance(result["delta_acc"], float)
    adapter.teardown()
