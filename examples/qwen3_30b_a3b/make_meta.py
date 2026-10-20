#!/usr/bin/env python3
"""An example way to choose replicas and the two-card split from routing traces, written as a
moe_rep_v2 file. It is one method, not the only one: replace it with your own experiment.

Per MoE layer (a layer whose trace rows are all -1 is dense: R 0, identity):
  1. replicas = the R experts picked most often on the calibration traces;
  2. the other E - R experts split (E - R) / 2 per card by local search: swap one expert of card 0
     with one of card 1 while the mean cost max(a, b, ceil((k + U) / 2)) drops (a / b = picks
     that only card 0 / card 1 holds; replicated picks fill in, README section 2);
  3. a few rounds of replica <-> non-replica swaps, the split searched again after each.
perm[l] = replicas, then card 0's experts, then card 1's (each sorted).

    python make_meta.py --calib traces/trace_calib_chat.npy --model qwen3-30b-a3b \
        --hf-repo Qwen/Qwen3-30B-A3B-Instruct-2507 --rep 8 --out q30_r8.json

Traces: .npy [layers, tokens, top_k] original expert ids (collect_traces.py writes them).
Needs numpy and torch (CPU is fine for ~50k positions; --device cuda for more).
"""
import argparse
import json
import time

import numpy as np
import torch


def onehot(ids, E, dev):
    """[T, k] ids -> [T, E] float."""
    t = torch.as_tensor(ids.astype(np.int64), device=dev)
    m = torch.zeros(t.shape[0], E, device=dev)
    m.scatter_(1, t, 1.0)
    return m


def cost(M, A, B, floor):
    """Mean over tokens of max(a, b, floor)."""
    return float(torch.clamp(torch.maximum(M[:, A].sum(1), M[:, B].sum(1)), min=floor).mean())


def split_search(M, pool, A0, floor, iters=400):
    """Equal split of `pool` from start A0: the best single swap each step (exact deltas)."""
    A = list(A0)
    B = [e for e in pool if e not in set(A)]
    for _ in range(iters):
        MA, MB = M[:, A], M[:, B]
        a, b = MA.sum(1), MB.sum(1)
        f0 = torch.clamp(torch.maximum(a, b), min=floor)
        gm = torch.clamp(torch.maximum(a - 1, b + 1), min=floor) - f0     # an expert leaves card 0
        gp = torch.clamp(torch.maximum(a + 1, b - 1), min=floor) - f0     # an expert joins card 0
        D = (MA.T @ gm)[:, None] + (MB.T @ gp)[None, :] - MA.T @ ((gm + gp)[:, None] * MB)
        v, idx = D.flatten().min(0)
        if v >= -1e-6 * M.shape[0]:
            break
        i, j = divmod(int(idx), len(B))
        A[i], B[j] = B[j], A[i]
    return A, B


def balanced_start(pool, freq):
    """Most frequent first onto the lighter card, equal sizes."""
    A, B, fa, fb = [], [], 0.0, 0.0
    for e in sorted(pool, key=lambda e: -freq[e]):
        if (fa <= fb and len(A) < len(pool) // 2) or len(B) >= len(pool) // 2:
            A.append(e)
            fa += freq[e]
        else:
            B.append(e)
            fb += freq[e]
    return A


def best_split(M, pool, freq, floor, rng, starts):
    best = None
    for A0 in [balanced_start(pool, freq)] + [list(rng.permutation(pool)[:len(pool) // 2]) for _ in range(starts)]:
        A, B = split_search(M, pool, A0, floor)
        c = cost(M, A, B, floor)
        if best is None or c < best[0]:
            best = (c, A, B)
    return best[1], best[2]


def replica_swaps(M, rep, A, B, floor, rounds, E):
    """Swap a replica with an expert of one card (it takes the replica's place there), best swap
    first, then search the split again."""
    rep, A, B = list(rep), list(A), list(B)
    for _ in range(rounds):
        a, b = M[:, A].sum(1), M[:, B].sum(1)
        best = (cost(M, A, B, floor), None)
        for side, S, own, other in ((0, A, a, b), (1, B, b, a)):
            for ri, r in enumerate(rep):
                new = own[:, None] - M[:, S] + M[:, r][:, None]           # [T, |S|]: S[si] -> replica, r -> card
                c = torch.clamp(torch.maximum(new, other[:, None]), min=floor).mean(0)
                v, si = c.min(0)
                if float(v) < best[0] - 1e-9:
                    best = (float(v), (side, int(si), ri))
        if best[1] is None:
            break
        side, si, ri = best[1]
        S = A if side == 0 else B
        S[si], rep[ri] = rep[ri], S[si]
        pool = [e for e in range(E) if e not in set(rep)]
        A, B = split_search(M, pool, A, floor)
    return rep, A, B


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", nargs="+", required=True, help="calibration traces (.npy [L, T, k]), concatenated")
    ap.add_argument("--model", required=True, help="our model key (README table), e.g. qwen3-30b-a3b")
    ap.add_argument("--hf-repo", default="", help="the HF repo id, for the record")
    ap.add_argument("--num-experts", type=int, default=128)
    ap.add_argument("--rep", type=int, default=8, help="R of every MoE layer (even)")
    ap.add_argument("--rep-layers", help="JSON list, one R per layer (overrides --rep)")
    ap.add_argument("--shared-units", type=int, default=0, help="U (0, or 2 for a two-unit shared expert)")
    ap.add_argument("--starts", type=int, default=2, help="random split starts besides the balanced one")
    ap.add_argument("--rounds", type=int, default=4, help="replica swap rounds")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.time()
    tr = np.concatenate([np.load(f) for f in a.calib], axis=1)
    L, T, k = tr.shape
    E, U = a.num_experts, a.shared_units
    reps = json.loads(a.rep_layers) if a.rep_layers else [a.rep] * L
    floor = (k + U + 1) // 2
    rng = np.random.default_rng(a.seed)
    rep_layers, perm = [], []
    for l in range(L):
        if (tr[l] == -1).all():                     # dense layer
            rep_layers.append(0)
            perm.append(list(range(E)))
            continue
        R = reps[l]
        M = onehot(tr[l], E, a.device)
        freq = M.sum(0).cpu().numpy()
        hot = [int(e) for e in np.argsort(-freq, kind="stable")[:R]]
        pool = [e for e in range(E) if e not in set(hot)]
        A, B = best_split(M, pool, freq, floor, rng, a.starts)
        if R:
            hot, A, B = replica_swaps(M, hot, A, B, floor, a.rounds, E)
        rep_layers.append(R)
        perm.append(sorted(int(e) for e in hot) + sorted(int(e) for e in A) + sorted(int(e) for e in B))
        ident = cost(M, list(range(E // 2)), list(range(E // 2, E)), floor)
        print(f"layer {l:3d}  R {R:3d}  calib cost {cost(M, A, B, floor):.3f}  (identity R 0: {ident:.3f})  "
              f"{time.time() - t0:.0f} s", flush=True)
    meta = {"schema": "moe_rep_v2", "model": a.model, "hf_repo": a.hf_repo, "num_experts": E, "top_k": k,
            "num_layers": L, "num_ranks": 2, "shared_units": U, "rep_layers": rep_layers, "perm": perm,
            "note": f"make_meta.py: hot replicas + local-search split on {T} calibration positions "
                    f"({', '.join(a.calib)}), seed {a.seed}"}
    with open(a.out, "w") as f:
        json.dump(meta, f)
    print(f"{a.out}: {L} layers, R {sorted(set(rep_layers))} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
