"""Module entrypoint for the voltage girth predictor trainer.

The implementation lives in ``ai.cage.voltage.supervised.train``; this thin
wrapper exposes it as ``ai.cage.voltage.train`` so the supercomputer submit
scripts (``python -m ai.cage.voltage.train``) and the public training command
have a stable path. The predictor is always a single (k, g)-independent model:
pass the full target grid via ``--targets``.
"""

from __future__ import annotations

from typing import cast

from ai.cage.utils import parse_targets
from ai.cage.voltage.supervised.train import parse_args, train

__all__ = ["parse_args", "train"]


if __name__ == "__main__":
    args = parse_args()
    raw_cycle_lengths = cast(str, args.cycle_lengths).strip()
    parsed_cycle_lengths: list[int] | None = (
        [int(x) for x in raw_cycle_lengths.split(",") if x.strip()]
        if raw_cycle_lengths
        else None
    )
    _ = train(
        targets=parse_targets(cast("str | None", args.targets)),
        num_samples=cast(int, args.samples),
        max_group_order=cast(int, args.max_group_order),
        epochs=cast(int, args.epochs),
        batch_size=cast(int, args.batch_size),
        hidden_dim=cast(int, args.hidden_dim),
        num_layers=cast(int, args.num_layers),
        lr=cast(float, args.lr),
        print_every=cast(int, args.print_every),
        seed=cast(int, args.seed),
        weight_decay=cast(float, args.weight_decay),
        model_id_override=cast("str | None", args.model_id),
        cycle_lengths=parsed_cycle_lengths,
        rwpe_dim=cast(int, args.rwpe_dim),
        workers=cast("int | None", args.workers),
        pos_weight=cast(float, args.pos_weight),
    )
