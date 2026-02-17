## Completed Refactoring
- [x] Refactor `ai/cage/rl/train.py` to use `ai.registry` for model management
- [x] Update `ai/registry.py` to support `cage` task
- [x] Implement `get_config()` and `get_name()` in `ActorCritic` (`ai/cage/rl/model.py`)
- [x] Standardize CLI arguments in `ai/cage/rl/train.py` (`--name`, `--hidden-dim`, etc.)

## Pending Features
- [ ] Implement r-neighborhood preprocessing in dataset creation pipeline
