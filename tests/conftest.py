"""Pytest support: ``--all`` flag, slow-test skipping, and a hard timeout.

By default, tests marked ``@pytest.mark.slow`` are skipped so the suite stays
fast. Pass ``--all`` to run every test, including the slow ones.

A hard per-test timeout (``PER_TEST_TIMEOUT`` seconds, SIGALRM-based) prevents
any non-slow test from hanging the suite; slow-marked tests are exempt so they
can run to completion under ``--all``.
"""

from __future__ import annotations

import signal
from collections.abc import Generator

import pytest

PER_TEST_TIMEOUT = 60


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--all",
        action="store_true",
        default=False,
        help="Run all tests, including those marked @pytest.mark.slow.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--all"):
        return
    skip_slow = pytest.mark.skip(reason="slow test; pass --all to run it")
    for item in items:
        if item.get_closest_marker("slow") is not None:
            item.add_marker(skip_slow)


@pytest.fixture(autouse=True)
def _enforce_per_test_timeout(
    request: pytest.FixtureRequest,
) -> Generator[None, None, None]:
    if request.node.get_closest_marker("slow") is not None:
        yield
        return

    def _handler(signum: int, frame: object) -> None:
        raise TimeoutError(f"test exceeded {PER_TEST_TIMEOUT}s hard timeout")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(PER_TEST_TIMEOUT)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
