"""Benchmark module registry — each import self-registers via register().

This wave: only the example module exists.
Next wave: add real benchmark modules here in the same pattern:
    from . import node_tasks   # noqa: F401
    from . import cage         # noqa: F401
    from . import refine       # noqa: F401
    from . import excision     # noqa: F401
"""

from . import _example  # noqa: F401
from . import node_tasks  # noqa: F401  — registers "degree" and "min_cycle"
from . import cage  # noqa: F401
from . import refine  # noqa: F401
from . import excision  # noqa: F401
from . import throughput  # noqa: F401
