"""Shared utilities for the cage module."""

from __future__ import annotations

import urllib.request
from http.client import HTTPResponse
from pathlib import Path
from typing import TypedDict, cast

import networkx as nx

_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "cages"
_HOG_BASE = "https://houseofgraphs.org/data/cages"
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# fmt: off
# All (k,g) → filename entries from the House of Graphs cage survey table.
# Sourced from the cage survey page at houseofgraphs.org/cages.
_HOG_TABLE: dict[tuple[int, int], str] = {
    (3,  5): "cagesk3g05.g6",
    (3,  6): "cagesk3g06.g6",
    (3,  7): "cagesk3g07.g6",
    (3,  8): "cagesk3g08.g6",
    (3,  9): "cagesk3g09.g6",
    (3, 10): "cagesk3g10.g6",
    (3, 11): "cagesk3g11.g6",
    (3, 12): "cagesk3g12.g6",
    (3, 13): "k3_g13_n272.g6",
    (3, 14): "k3_g14_n384.g6",
    (3, 15): "k3_g15_n620.g6",
    (3, 16): "k3_g16_n936.g6",
    (3, 17): "k3_g17_n2048.g6",
    (3, 18): "k3_g18_n2560.g6",
    (3, 19): "k3_g19_n4324.g6",
    (3, 20): "k3_g20_n5376.g6",
    (4,  5): "cagesk4g05.g6",
    (4,  6): "cagesk4g06.g6",
    (4,  7): "cagesk4g07.g6",
    (4,  8): "cagesk4g08.g6",
    (4,  9): "k4_g9_n270.g6",
    (4, 10): "k4_g10_n320.g6",
    (4, 11): "k4_g11_n713.g6",
    (4, 12): "k4_g12_n728.g6",
    (5,  5): "cagesk5g05.g6",
    (5,  6): "cagesk5g06.g6",
    (5,  7): "k5_g7_n152.g6",
    (5,  8): "k5_g8_n170.g6",
    (5,  9): "k5_g9_n1116.g6",
    (5, 10): "k5_g10_n1296.g6",
    (5, 11): "k5_g11_n2688.g6",
    (5, 12): "k5_g12_n2730.g6",
    (6,  5): "cagesk6g05.g6",
    (6,  6): "cagesk6g06.g6",
    (6,  7): "k6_g7_n294.g6",
    (6,  8): "k6_g8_n312.g6",
    (6, 11): "k6_g11_n7783.g6",
    (7,  5): "cagesk7g05.g6",
    (7,  6): "cagesk7g06.g6",
    (7,  7): "k7_g7_n632.g6",
    (7,  8): "k7_g8_n658.g6",
    (8,  5): "k8_g5_n80.g6",
    (8,  6): "k8_g6_n114.g6",
    (8,  7): "k8_g7_n774.g6",
    (8,  8): "k8_g8_n800.g6",
    (9,  5): "k9_g5_n96.g6",
    (9,  7): "k9_g7_n1104.g6",
    (9,  8): "k9_g8_n1170.g6",
    (10, 5): "k10_g5_n124.g6",
    (10, 7): "k10_g7_n1608.g6",
    (10, 8): "k10_g8_n1640.g6",
    (11, 5): "k11_g5_n154.g6",
    (11, 6): "k11_g6_n240.g6",
    (11, 7): "k11_g7_n2576.g6",
    (11, 8): "k11_g8_n2618.g6",
    (12, 5): "k12_g5_n203.g6",
    (12, 6): "k12_g6_n266.g6",
    (12, 7): "k12_g7_n2890.g6",
    (12, 8): "k12_g8_n2928.g6",
    (13, 5): "k13_g5_n226.g6",
    (13, 6): "k13_g6_n336.g6",
    (13, 7): "k13_g7_n4292.g6",
    (13, 8): "k13_g8_n4342.g6",
    (14, 5): "k14_g5_n280.g6",
    (14, 6): "k14_g6_n366.g6",
    (14, 7): "k14_g7_n4716.g6",
    (14, 8): "k14_g8_n4760.g6",
    (15, 5): "k15_g5_n310.g6",
    (16, 5): "k16_g5_n336.g6",
    (17, 5): "k17_g5_n436.g6",
    (18, 5): "k18_g5_n468.g6",
    (18, 6): "k18_g6_n614.g6",
    (19, 5): "k19_g5_n500.g6",
    (19, 6): "k19_g6_n720.g6",
    (20, 6): "k20_g6_n762.g6",
}
# fmt: on


class HogCageEntry(TypedDict):
    k: int
    g: int
    filename: str
    cached: bool


def hog_list_available() -> list[tuple[int, int]]:
    """Return all (k,g) pairs in the HoG cage survey, sorted."""
    return sorted(_HOG_TABLE.keys())


def hog_is_available(k: int, g: int) -> bool:
    """Return True if this (k,g) pair is in the HoG cage survey."""
    return (k, g) in _HOG_TABLE


def hog_download(k: int, g: int, *, force: bool = False) -> Path:
    """Download the (k,g) cage file from HoG and cache it in data/cages/.

    Returns the local path. Skips download if already cached unless *force* is True.
    Raises KeyError if the pair is not in the HoG survey.
    """
    if (k, g) not in _HOG_TABLE:
        raise KeyError(f"No cage data for (k={k}, g={g}) in the HoG survey")

    filename = _HOG_TABLE[(k, g)]
    local = _DATA_DIR / filename
    if local.exists() and not force:
        return local

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{_HOG_BASE}/{filename}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with cast(HTTPResponse, urllib.request.urlopen(req, timeout=30)) as resp:
        data: bytes = resp.read()
    _ = local.write_bytes(data)
    return local


def hog_download_all(*, force: bool = False, verbose: bool = True) -> None:
    """Download every cage file in the HoG survey."""
    total = len(_HOG_TABLE)
    for i, (k, g) in enumerate(sorted(_HOG_TABLE.keys()), 1):
        try:
            local = _DATA_DIR / _HOG_TABLE[(k, g)]
            if local.exists() and not force:
                if verbose:
                    print(f"[{i}/{total}] (k={k}, g={g}) already cached")
                continue
            _ = hog_download(k, g, force=force)
            if verbose:
                filename = _HOG_TABLE[(k, g)]
                print(f"[{i}/{total}] (k={k}, g={g}) downloaded → {filename}")
        except Exception as e:
            print(f"[{i}/{total}] (k={k}, g={g}) FAILED: {e}")


def get_cages(k: int, g: int) -> list[nx.Graph[int]]:
    """Return all (k,g)-regular record graphs from the HoG survey.

    Downloads and caches the file on first call. The file may contain multiple
    non-isomorphic graphs (e.g. (3,9) has 18). Raises KeyError if the pair is
    not in the HoG survey.
    """
    local = hog_download(k, g)
    with open(local, "rb") as f:
        lines = [line.strip() for line in f if line.strip()]
    return cast("list[nx.Graph[int]]", [nx.from_graph6_bytes(line) for line in lines])
