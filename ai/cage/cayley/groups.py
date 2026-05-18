"""Group helpers for Cayley graph search.

Reuses ai.cage.voltage.groups.FiniteGroup as the element-level representation.
Adds utilities that the Cayley search needs but the voltage code doesn't:
element order, involution detection, and a catalogue of candidate groups
that includes the non-abelian families relevant to (k,g)-cage records
(PGL(2,p), SL(2,p)) on top of the abelian pool.
"""

from __future__ import annotations

import random

from ai.cage.voltage.groups import (
    FiniteGroup,
    cyclic_group,
    dihedral_group,
    direct_product,
    pgl2,
    sl2,
)


def element_order(group: FiniteGroup, a: int) -> int:
    """Order of element a in group (smallest n >= 1 with a^n = e)."""
    if a == 0:
        return 1
    cur = a
    n = 1
    while cur != 0:
        cur = group.mult(cur, a)
        n += 1
        if n > group.order:
            raise RuntimeError(
                f"element_order: did not terminate (group={group.name}, a={a})"
            )
    return n


def non_identity_elements(group: FiniteGroup) -> list[int]:
    """All elements of G except the identity."""
    return list(range(1, group.order))


def involutions(group: FiniteGroup) -> list[int]:
    """Elements of order 2 (a != e, a * a = e)."""
    return [a for a in range(1, group.order) if group.mult(a, a) == 0]


def conjugacy_class_size(group: FiniteGroup, a: int) -> int:
    """Size of the conjugacy class of a (orbit under g a g^{-1})."""
    seen: set[int] = set()
    for g in range(group.order):
        conj = group.mult(group.mult(g, a), group.inv(g))
        seen.add(conj)
    return len(seen)


def available_groups(
    max_order: int,
    *,
    include_matrix: bool = True,
    include_abelian: bool = True,
    include_dihedral: bool = True,
) -> list[FiniteGroup]:
    """Build a deduplicated catalogue of candidate finite groups.

    PSL/PGL/SL families are the historically productive Cayley-graph choices
    for high-girth (k,g)-records. Abelian/dihedral are kept for sanity checks
    and small cases.
    """
    groups: list[FiniteGroup] = []
    seen: set[str] = set()

    def _add(g: FiniteGroup) -> None:
        if g.order > max_order:
            return
        if g.name in seen:
            return
        seen.add(g.name)
        groups.append(g)

    if include_matrix:
        # PGL(2, p) has order p(p^2 - 1).
        # SL(2, p) has order p(p^2 - 1).
        # PSL(2, p) = SL(2, p) / {+/- I} has order p(p^2 - 1)/2 for p > 2;
        # we approximate the PSL via PGL/SL here because we don't have a
        # quotient implementation; PGL and SL together cover the useful range.
        for p in (3, 5, 7, 11, 13):
            order_est = p * (p * p - 1)
            if order_est <= max_order:
                _add(pgl2(p))
                _add(sl2(p))

    if include_abelian:
        for n in range(2, max_order + 1):
            _add(cyclic_group(n))

        for a in range(2, min(20, max_order)):
            for b in range(a, min(20, max_order)):
                if a * b <= max_order:
                    _add(direct_product(cyclic_group(a), cyclic_group(b)))

    if include_dihedral:
        for n in range(3, max_order // 2 + 1):
            if 2 * n <= max_order:
                _add(dihedral_group(n))

    return groups


def symmetric_closure(group: FiniteGroup, seeds: list[int]) -> list[int]:
    """Close a set of group elements under inversion; deduplicate."""
    out: list[int] = []
    seen: set[int] = set()
    for s in seeds:
        if s == 0 or s in seen:
            continue
        _ = seen.add(s)
        out.append(s)
        s_inv = group.inv(s)
        if s_inv != s and s_inv not in seen:
            _ = seen.add(s_inv)
            out.append(s_inv)
    return out


def random_generating_set(group: FiniteGroup, k: int) -> list[int] | None:
    """Sample a random symmetric set of size k.

    Each involution contributes degree 1; each non-involution pair {a, a^{-1}}
    contributes degree 2. Slots are sampled without replacement.

    Returns None if no valid (involutions, pairs) composition fits the group
    (e.g. odd k but no involutions).
    """
    invols = involutions(group)
    non_invols = [a for a in non_identity_elements(group) if group.mult(a, a) != 0]

    seen_pair: set[int] = set()
    non_invol_reps: list[int] = []
    for a in non_invols:
        if a in seen_pair:
            continue
        _ = seen_pair.add(a)
        a_inv = group.inv(a)
        _ = seen_pair.add(a_inv)
        non_invol_reps.append(a)

    compositions: list[tuple[int, int]] = []
    for p in range(0, k // 2 + 1):
        i = k - 2 * p
        if i < 0:
            continue
        if i > len(invols):
            continue
        if p > len(non_invol_reps):
            continue
        compositions.append((i, p))

    if not compositions:
        return None

    i_count, p_count = random.choice(compositions)
    chosen_invols = random.sample(invols, i_count)
    chosen_pairs = random.sample(non_invol_reps, p_count)
    return symmetric_closure(group, chosen_invols + chosen_pairs)


__all__ = [
    "FiniteGroup",
    "available_groups",
    "conjugacy_class_size",
    "cyclic_group",
    "dihedral_group",
    "direct_product",
    "element_order",
    "involutions",
    "non_identity_elements",
    "pgl2",
    "random_generating_set",
    "sl2",
    "symmetric_closure",
]
