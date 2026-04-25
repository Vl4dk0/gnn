"""Finite group representations for voltage graph constructions.

Groups are stored as lookup tables for O(1) multiplication and inversion.
Every element is an integer in 0..order-1, with 0 always being the identity.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import override

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class FiniteGroup:
    """A finite group represented by its multiplication and inverse tables."""

    name: str
    order: int
    mult_table: NDArray[np.intp]  # shape (order, order)
    inv_table: NDArray[np.intp]  # shape (order,)

    def mult(self, a: int, b: int) -> int:
        result: int = self.mult_table[a, b].item()
        return result

    def inv(self, a: int) -> int:
        result: int = self.inv_table[a].item()
        return result

    def identity(self) -> int:
        return 0

    def elements(self) -> range:
        return range(self.order)

    @override
    def __repr__(self) -> str:
        return f"FiniteGroup({self.name!r}, order={self.order})"


def _build_tables(
    order: int, mult_fn: Callable[[int, int], int]
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Build multiplication and inverse tables from a callable (a,b)->c."""
    mt = np.zeros((order, order), dtype=np.intp)
    for a in range(order):
        for b in range(order):
            mt[a, b] = mult_fn(a, b)

    inv = np.zeros(order, dtype=np.intp)
    for a in range(order):
        for b in range(order):
            if mt[a, b] == 0:
                inv[a] = b
                break
    return mt, inv


# ── Cyclic group Z_n ─────────────────────────────────────────────────────


def cyclic_group(n: int) -> FiniteGroup:
    """Cyclic group Z_n with addition mod n."""
    mt, inv = _build_tables(n, lambda a, b: (a + b) % n)
    return FiniteGroup(name=f"Z_{n}", order=n, mult_table=mt, inv_table=inv)


# ── Dihedral group D_n (order 2n) ────────────────────────────────────────


def dihedral_group(n: int) -> FiniteGroup:
    """Dihedral group D_n of order 2n.

    Elements encoded as:
        rotation r^k  -> k          (0 <= k < n)
        reflection s*r^k -> n + k   (0 <= k < n)

    Multiplication rules:
        r^a * r^b       = r^{(a+b) mod n}
        r^a * s*r^b     = s*r^{(b-a) mod n}
        s*r^a * r^b     = s*r^{(a+b) mod n}
        s*r^a * s*r^b   = r^{(b-a) mod n}
    """
    order = 2 * n

    def _mult(a: int, b: int) -> int:
        a_is_s = a >= n
        b_is_s = b >= n
        a_r = a % n
        b_r = b % n
        if not a_is_s and not b_is_s:
            return (a_r + b_r) % n
        if not a_is_s and b_is_s:
            return n + (b_r - a_r) % n
        if a_is_s and not b_is_s:
            return n + (a_r + b_r) % n
        # both reflections
        return (b_r - a_r) % n

    mt, inv = _build_tables(order, _mult)
    return FiniteGroup(name=f"D_{n}", order=order, mult_table=mt, inv_table=inv)


# ── Direct product G1 x G2 ──────────────────────────────────────────────


def direct_product(g1: FiniteGroup, g2: FiniteGroup) -> FiniteGroup:
    """Direct product G1 x G2.

    Element (a, b) is encoded as a * |G2| + b.
    """
    n1 = g1.order
    n2 = g2.order
    order = n1 * n2

    def _mult(x: int, y: int) -> int:
        a1, a2 = divmod(x, n2)
        b1, b2 = divmod(y, n2)
        r1: int = g1.mult_table[a1, b1].item()
        r2: int = g2.mult_table[a2, b2].item()
        return r1 * n2 + r2

    mt, inv = _build_tables(order, _mult)
    return FiniteGroup(
        name=f"{g1.name}x{g2.name}", order=order, mult_table=mt, inv_table=inv
    )


# ── Semidirect product Z_n ⋊_φ Z_m ──────────────────────────────────────


def semidirect_product_cyclic(n: int, m: int, phi_1: int) -> FiniteGroup:
    """Semidirect product Z_n ⋊ Z_m where the generator of Z_m acts on Z_n
    by multiplication by phi_1.

    Requires phi_1^m ≡ 1 (mod n).

    Element (a, b) with a in Z_n, b in Z_m is encoded as a * m + b.
    Multiplication: (a1, b1) * (a2, b2) = (a1 + phi_1^b1 * a2 mod n, b1 + b2 mod m)
    """
    order = n * m

    # Precompute powers of phi_1 mod n
    phi_powers = [1] * m
    for i in range(1, m):
        phi_powers[i] = (phi_powers[i - 1] * phi_1) % n

    def _mult(x: int, y: int) -> int:
        a1, b1 = divmod(x, m)
        a2, b2 = divmod(y, m)
        new_a = (a1 + phi_powers[b1] * a2) % n
        new_b = (b1 + b2) % m
        return new_a * m + new_b

    mt, inv = _build_tables(order, _mult)
    return FiniteGroup(
        name=f"Z_{n}_rtimes_Z_{m}", order=order, mult_table=mt, inv_table=inv
    )


# ── General linear groups over finite fields ─────────────────────────────


def _gf_mult(a: int, b: int, p: int) -> int:
    return (a * b) % p


def _gf_add(a: int, b: int, p: int) -> int:
    return (a + b) % p


def _gf_inv_mult(a: int, p: int) -> int:
    """Multiplicative inverse of a in GF(p), a != 0."""
    return pow(a, p - 2, p)


def pgl2(p: int) -> FiniteGroup:
    """PGL(2, p) — projective general linear group over GF(p).

    Elements are Möbius transformations z -> (az+b)/(cz+d) on GF(p) ∪ {∞}.
    Represented by equivalence classes of 2x2 matrices [[a,b],[c,d]] with
    ad-bc != 0, up to scalar multiples.

    We normalize so that the first nonzero entry in (a, b, c, d) reading
    left-to-right is 1. Elements are stored as integer indices.

    Order: p(p^2-1) for prime p.
    """
    # Enumerate normalized representatives
    reps: list[tuple[int, int, int, int]] = []
    for a in range(p):
        for b in range(p):
            for c in range(p):
                for d in range(p):
                    det = (a * d - b * c) % p
                    if det == 0:
                        continue
                    # Normalize: divide by first nonzero entry
                    first_nonzero = 0
                    for v in (a, b, c, d):
                        if v != 0:
                            first_nonzero = v
                            break
                    inv_fn = _gf_inv_mult(first_nonzero, p)
                    na = (a * inv_fn) % p
                    nb = (b * inv_fn) % p
                    nc = (c * inv_fn) % p
                    nd = (d * inv_fn) % p
                    rep = (na, nb, nc, nd)
                    if rep not in reps:
                        reps.append(rep)

    order = len(reps)
    rep_to_idx = {r: i for i, r in enumerate(reps)}

    # Ensure identity (1,0,0,1) is element 0
    identity_rep = (1, 0, 0, 1)
    if identity_rep in rep_to_idx:
        id_idx = rep_to_idx[identity_rep]
        if id_idx != 0:
            # Swap
            reps[0], reps[id_idx] = reps[id_idx], reps[0]
            rep_to_idx = {r: i for i, r in enumerate(reps)}

    def _mat_mult(idx_a: int, idx_b: int) -> int:
        a1, b1, c1, d1 = reps[idx_a]
        a2, b2, c2, d2 = reps[idx_b]
        ra = _gf_add(_gf_mult(a1, a2, p), _gf_mult(b1, c2, p), p)
        rb = _gf_add(_gf_mult(a1, b2, p), _gf_mult(b1, d2, p), p)
        rc = _gf_add(_gf_mult(c1, a2, p), _gf_mult(d1, c2, p), p)
        rd = _gf_add(_gf_mult(c1, b2, p), _gf_mult(d1, d2, p), p)
        # Normalize
        first_nonzero = 0
        for v in (ra, rb, rc, rd):
            if v != 0:
                first_nonzero = v
                break
        inv_fn = _gf_inv_mult(first_nonzero, p)
        nr = (
            (ra * inv_fn) % p,
            (rb * inv_fn) % p,
            (rc * inv_fn) % p,
            (rd * inv_fn) % p,
        )
        return rep_to_idx[nr]

    mt, inv = _build_tables(order, _mat_mult)
    return FiniteGroup(name=f"PGL(2,{p})", order=order, mult_table=mt, inv_table=inv)


def sl2(p: int) -> FiniteGroup:
    """SL(2, p) — special linear group over GF(p).

    Elements are 2x2 matrices [[a,b],[c,d]] with ad-bc = 1 over GF(p).
    Order: p(p^2-1) for prime p.
    """
    reps: list[tuple[int, int, int, int]] = []
    for a in range(p):
        for b in range(p):
            for c in range(p):
                for d in range(p):
                    if (a * d - b * c) % p == 1:
                        reps.append((a, b, c, d))

    order = len(reps)
    rep_to_idx = {r: i for i, r in enumerate(reps)}

    # Ensure identity is element 0
    identity_rep = (1, 0, 0, 1)
    id_idx = rep_to_idx[identity_rep]
    if id_idx != 0:
        reps[0], reps[id_idx] = reps[id_idx], reps[0]
        rep_to_idx = {r: i for i, r in enumerate(reps)}

    def _mat_mult(idx_a: int, idx_b: int) -> int:
        a1, b1, c1, d1 = reps[idx_a]
        a2, b2, c2, d2 = reps[idx_b]
        ra = (a1 * a2 + b1 * c2) % p
        rb = (a1 * b2 + b1 * d2) % p
        rc = (c1 * a2 + d1 * c2) % p
        rd = (c1 * b2 + d1 * d2) % p
        return rep_to_idx[(ra, rb, rc, rd)]

    mt, inv = _build_tables(order, _mat_mult)
    return FiniteGroup(name=f"SL(2,{p})", order=order, mult_table=mt, inv_table=inv)
