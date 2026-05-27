"""Smoke tests for MoveOracle training, saving, and loading."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

from ai.cage.refine.move_oracle import MoveOracle, load_move_oracle
from ai.cage.refine.train import train


def _smoke_save_dir(tmp_root: str) -> Path:
    path = Path(tmp_root) / "move_oracle_smoke"
    if path.exists():
        shutil.rmtree(path)
    return path


class TestMoveOracleSmoke:
    def test_train_and_save(self, tmp_path: Path) -> None:
        save_dir = _smoke_save_dir(str(tmp_path))
        _ = train(
            samples=80,
            epochs=2,
            batch_size=8,
            hidden_dim=16,
            num_layers=2,
            cycle_lengths=[3, 4, 5],
            rwpe_dim=4,
            seed=0,
            save_dir=save_dir, workers=1,
        )
        assert (save_dir / "weights.pt").exists()
        assert (save_dir / "info.json").exists()

    def test_info_json_records_feature_config(self, tmp_path: Path) -> None:
        save_dir = _smoke_save_dir(str(tmp_path))
        _ = train(
            samples=80,
            epochs=2,
            batch_size=8,
            hidden_dim=16,
            num_layers=2,
            cycle_lengths=[3, 4, 5],
            rwpe_dim=4,
            seed=0,
            save_dir=save_dir, workers=1,
        )
        with open(save_dir / "info.json") as f:
            info = json.load(f)
        fc = info["training"]["feature_config"]
        assert fc["cycle_lengths"] == [3, 4, 5]
        assert fc["rwpe_dim"] == 4

    def test_weights_reload(self, tmp_path: Path) -> None:
        save_dir = _smoke_save_dir(str(tmp_path))
        _ = train(
            samples=80,
            epochs=2,
            batch_size=8,
            hidden_dim=16,
            num_layers=2,
            cycle_lengths=[3, 4, 5],
            rwpe_dim=4,
            seed=0,
            save_dir=save_dir, workers=1,
        )
        loaded, cycle_lengths, rwpe_dim = load_move_oracle(save_dir)
        assert isinstance(loaded, MoveOracle)
        assert cycle_lengths == [3, 4, 5]
        assert rwpe_dim == 4
        # Sanity: forward pass doesn't raise
        _ = loaded.eval()
        for p in loaded.parameters():
            assert not torch.isnan(p).any()
