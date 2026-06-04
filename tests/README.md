# tests/

Pytest suite covering graph algorithms, search/generation, voltage/excision/refine logic, model loading/parsing, route validation, and the results framework.

---

## How to run

**Full fast suite** (default — slow-marked tests are skipped):

```bash
uv run pytest -q
```

**Single file:**

```bash
uv run pytest tests/test_voltage.py -q
```

**Single test:**

```bash
uv run pytest tests/test_voltage.py::test_cyclic_group_operations -q
```

**Useful flags:**

- `-k <expr>` — run tests whose name matches an expression, e.g. `-k "girth or regularity"`
- `-x` — stop on first failure
- `--all` — include `@pytest.mark.slow` tests (skipped by default)

**Type-check touched files** (required before reporting completion):

```bash
uv run basedpyright path/to/file.py
```

---

## How to add a test

1. **Location:** place the file in `tests/`. Name it `test_<subject>.py`.
2. **Function naming:** every test function must be named `test_*`.
3. **Shared fixtures:** `conftest.py` provides a `--all` flag, a `@pytest.mark.slow` skip mechanism, and a 60-second hard timeout (SIGALRM) on non-slow tests. Import nothing extra — fixtures are auto-discovered.
4. **Mark slow tests** with `@pytest.mark.slow` if they take more than a few seconds; they are skipped in the default run.

### Philosophy

- Prefer **deterministic unit tests** for graph invariants: girth/regularity checks, voltage/Cayley construction, search transition rules, label-generation logic.
- Keep tests **fast**: tiny smoke runs only — a few steps, small groups (order < 20), small graphs (< 30 nodes).
- Never add tests that depend on long training jobs, large model downloads, or expensive catalogue-wide searches.
- Graph-algorithm coverage has priority over frontend/API tests.

### Minimal example

```python
# tests/test_my_feature.py
import networkx as nx
from backend.utils.graph_utils import compute_girth


def test_triangle_has_girth_3() -> None:
    g: nx.Graph[int] = nx.cycle_graph(3)
    assert compute_girth(g) == 3
```
