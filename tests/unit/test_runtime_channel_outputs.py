"""Unit tests for the channel_outputs log field and collection."""
from __future__ import annotations

import numpy as np

from kiki_oniric.dream.channels import WeightUpdate
from kiki_oniric.dream.episode import Operation
from kiki_oniric.dream.runtime import EpisodeLogEntry


def test_log_entry_channel_outputs_defaults_empty() -> None:
    entry = EpisodeLogEntry(
        episode_id="de-x",
        operations_executed=(Operation.REPLAY,),
        completed=True,
    )
    assert entry.channel_outputs == ()


def test_log_entry_accepts_channel_outputs() -> None:
    wu = WeightUpdate(lora_delta={"l0": np.zeros(2, dtype=np.float32)})
    entry = EpisodeLogEntry(
        episode_id="de-y",
        operations_executed=(Operation.REPLAY,),
        completed=True,
        channel_outputs=(wu,),
    )
    assert entry.channel_outputs == (wu,)
