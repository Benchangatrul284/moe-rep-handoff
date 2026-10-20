"""The two-card split of the hardware router, copied from our repo
(rtl_verification/moe/moe_route_model.py, the replicas v2 path) as the reference for the tests.

Card A (rank 0) and card B (rank 1) each run the picks only they hold. The f = r + U shared picks
(replicated experts and shared units) go t to card 0 and f - t to card 1:
t = clamp(ceil((f + b - a) / 2), 0, f), a / b = picks only card 0 / card 1 holds.
"""


def rep_split(top, lo, nloc, rep_n, rank, shared_u=0):
    """t, the replicated ids card A takes and the shared units card A takes."""
    ua_hi = lo if rank else lo + nloc
    rep = sorted(i for i in top if i < rep_n)
    a = sum(1 for i in top if rep_n <= i < ua_hi)
    b = len(top) - len(rep) - a
    f = len(rep) + shared_u
    t = max(0, min(f, (f + b - a + 1) // 2))
    return t, set(rep[:max(0, t - shared_u)]), set(range(min(t, shared_u)))


def assign(top, lo, nloc, rep_n, rank, shared_u=0):
    """This card's entries: [(position, slot)], shared units first (position -1 - j), then the
    top-k entries in rank order. lo = rep_n + rank * nloc."""
    assert rank in (0, 1) and 0 <= shared_u <= min(2, len(top)) and lo == rep_n + rank * nloc
    _, take_a, take_sh = rep_split(top, lo, nloc, rep_n, rank, shared_u)
    out = [(-1 - j, rep_n + nloc + j) for j in range(shared_u) if (j in take_sh) != bool(rank)]
    for p, i in enumerate(top):
        if i < rep_n:
            if (i in take_a) != bool(rank):
                out.append((p, i))
        elif lo <= i < lo + nloc:
            out.append((p, i - (lo - rep_n)))
    return out
