"""Entrypoint shim for the excision-repair PPO trainer.

The implementation lives in :mod:`ai.cage.excision.rl.train`; this module keeps
``python -m ai.cage.excision.train`` working as the public training entrypoint.
"""

from __future__ import annotations

from ai.cage.excision.rl.train import main, train_repair_ppo

__all__ = ["main", "train_repair_ppo"]


if __name__ == "__main__":
    main()
