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
    semidirect_product_cyclic,
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
    include_semidirect: bool = True,
    abelian_cap: int = 300,
    semidirect_cap: int = 600,
) -> list[FiniteGroup]:
    """Build a deduplicated catalogue of candidate finite groups.

    PGL/SL families are the historically productive Cayley-graph choices for
    high-girth (k,g)-records. Abelian/dihedral are kept for sanity checks and
    small cases; semidirect products of cyclic groups add non-abelian variety
    at moderate order.

    Each FiniteGroup carries a full O(order^2) multiplication table, so the
    catalogue cannot materialise every cyclic group up to a large max_order
    without exhausting memory. abelian_cap / semidirect_cap bound the order
    of those families independently of max_order, while the matrix groups
    (few in number, but individually large) are allowed up to max_order.
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
        # PGL(2, p) and SL(2, p) both have order p(p^2 - 1) for prime p.
        # PSL(2, p) = SL(2, p) / {+/- I} has order p(p^2 - 1)/2 for p > 2;
        # we approximate the PSL via PGL/SL here because we don't have a
        # quotient implementation; PGL and SL together cover the useful range.
        for p in (3, 5, 7, 11, 13, 17):
            order_est = p * (p * p - 1)
            if order_est <= max_order:
                _add(pgl2(p))
                _add(sl2(p))

    if include_abelian:
        ab_cap = min(max_order, abelian_cap)
        for n in range(2, ab_cap + 1):
            _add(cyclic_group(n))

        for a in range(2, min(20, ab_cap)):
            for b in range(a, min(20, ab_cap)):
                if a * b <= ab_cap:
                    _add(direct_product(cyclic_group(a), cyclic_group(b)))

    if include_dihedral:
        dih_cap = min(max_order, abelian_cap)
        for n in range(3, dih_cap // 2 + 1):
            if 2 * n <= dih_cap:
                _add(dihedral_group(n))

    if include_semidirect:
        # Z_n ⋊ Z_m with a non-trivial action phi (phi^m ≡ 1 mod n). One
        # non-trivial action per (n, m) is enough for catalogue variety.
        sd_cap = min(max_order, semidirect_cap)
        for n in range(3, sd_cap + 1):
            for m in (2, 3, 4, 5, 6):
                if n * m > sd_cap:
                    continue
                for phi in range(2, n):
                    if pow(phi, m, n) == 1:
                        _add(semidirect_product_cyclic(n, m, phi))
                        break

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
    "semidirect_product_cyclic",
    "sl2",
    "symmetric_closure",
]
