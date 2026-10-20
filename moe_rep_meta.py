#!/usr/bin/env python3
"""MoE expert replicas: the moe_rep_v2 metadata file and what it does to a HF checkpoint.

The same file runs on both sides of the hand-off (docs/moe_rep_handoff.md):

  init    --hf DIR --model KEY --out META   an identity file to start from (R = 0, no reorder)
  check   META [--hf DIR]                   the file is valid (for that checkpoint, and for our
                                            hardware when run inside the repo)
  layout  META [--json OUT]                 which original expert each card holds, slot by slot
  balance META --traces F.npy               the two-card cost the hardware sees, from routing traces
  apply   META --hf DIR --out DIR           the reordered HF checkpoint: expert j of layer l is
                                            original expert perm[l][j] (the same function)

Id space (docs/design/moe_rep2/spec.md): in MoE layer l, hardware id j runs original expert
perm[l][j]. Ids [0, R) sit on both cards, [R, R+u) on card 0 only, [R+u, E) on card 1 only,
u = (E - R) / 2. U > 0 cuts the shared expert into U units that both cards hold.

Needs numpy; init, check --hf and apply also torch + safetensors.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections import defaultdict

import numpy as np

SCHEMA = "moe_rep_v2"
REP_MAX = 128                # moe_router.sv REP_MAX
E_MAX, K_MAX = 256, 8        # moe_router.sv E_MAX / K_MAX
SHARED_UNITS = (0, 2)        # spec "User decisions": U = 2 or the TP-split shared expert
W4_BYTES = 4.25 / 8          # MXINT4: 4-bit elements + one 8-bit scale per 32
ROUTER_SUFFIXES = ("gate.weight", "gate.e_score_correction_bias",      # Qwen3, DeepSeek, Mixtral, ...
                   "router.proj.weight", "router.per_expert_scale")    # Gemma-4
TOP_K_KEYS = ("num_experts_per_tok", "num_experts_per_token", "top_k_experts")   # Kimi: _token
EXPERT = re.compile(r"^(.*\.layers\.(\d+)\.(?:.*\.)?)experts\.(?:(\d+)\.)?(.+)$")
SHARED = re.compile(r"^shared_experts?\.")


def fail(msg):
    sys.exit(f"error: {msg}")


# ---------------------------------------------------------------- the file

def load(path):
    raw = open(path, "rb").read()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def moe_layers(meta):
    """Layers whose row is not the identity or whose R is not 0; with --hf the checkpoint decides."""
    return [l for l in range(meta["num_layers"])
            if meta["rep_layers"][l] or meta["perm"][l] != list(range(meta["num_experts"]))]


def is_int(v):
    return type(v) is int


def validate(meta, moe=None):
    """Errors in the file itself; moe = the checkpoint's MoE layers when known."""
    err = []
    if meta.get("schema") != SCHEMA:
        return [f"schema {meta.get('schema')!r}, expected {SCHEMA!r}"]
    for key in ("model", "num_experts", "top_k", "num_layers", "rep_layers", "perm"):
        if key not in meta:
            err.append(f"missing {key!r}")
    if err:
        return err
    E, k, L, U = meta["num_experts"], meta["top_k"], meta["num_layers"], meta.get("shared_units", 0)
    if not all(is_int(v) for v in (E, k, L, U)):
        return ["num_experts, top_k, num_layers and shared_units must be JSON integers"]
    if meta.get("num_ranks", 2) != 2:
        err.append("num_ranks must be 2 (replicas are a two-card layout)")
    if not 2 <= E <= E_MAX or E % 2:
        err.append(f"num_experts {E}: even, 2..{E_MAX}")
    if not 1 <= k <= min(K_MAX, E):
        err.append(f"top_k {k}: 1..{min(K_MAX, E)}")
    if U not in SHARED_UNITS:
        err.append(f"shared_units {U}: 0, or 2 (a shared expert of two routed-expert sizes); "
                   f"a one-unit shared expert stays TP-split (moe_rep2 spec)")
    if len(meta["rep_layers"]) != L or len(meta["perm"]) != L:
        err.append(f"rep_layers and perm need {L} rows (one per layer, dense layers included)")
        return err
    for l, (r, p) in enumerate(zip(meta["rep_layers"], meta["perm"])):
        if not is_int(r) or r % 2 or not 0 <= r <= min(REP_MAX, E):
            err.append(f"layer {l}: R {r!r} must be an even integer 0..{min(REP_MAX, E)}")
        if not isinstance(p, list) or not all(is_int(v) for v in p) or sorted(p) != list(range(E)):
            err.append(f"layer {l}: perm is not a permutation of the integers 0..{E - 1}")
        elif moe is not None and l not in moe and (r or p != list(range(E))):
            err.append(f"layer {l} is dense: R 0 and the identity perm")
    return err


def card_lists(meta, l):
    """Original experts of layer l: (replicated, card 0 only, card 1 only)."""
    E, R, p = meta["num_experts"], meta["rep_layers"][l], meta["perm"][l]
    u = (E - R) // 2
    return p[:R], p[R:R + u], p[R + u:]


# ---------------------------------------------------------------- the checkpoint

def cfg_get(cfg, *keys):
    for c in (cfg, cfg.get("text_config", {})):
        for k in keys:
            if k in c:
                return c[k]
    return None


def shard_files(hf):
    idx = os.path.join(hf, "model.safetensors.index.json")
    if os.path.exists(idx):
        return sorted(set(json.load(open(idx))["weight_map"].values()))
    if os.path.exists(os.path.join(hf, "model.safetensors")):
        return ["model.safetensors"]
    fail(f"{hf}: no model.safetensors(.index.json)")


def scan(hf):
    """Names and shapes of every tensor, and the MoE structure they imply. Experts outside the
    decoder stack stay as they are: layers past num_hidden_layers (GLM-4.7-Flash's
    multi-token-prediction layer) and other stacks (Qwen3.6's mtp.layers)."""
    from safetensors import safe_open
    shapes = {}
    for fn in shard_files(hf):
        with safe_open(os.path.join(hf, fn), "pt") as f:
            for n in f.keys():
                shapes[n] = tuple(f.get_slice(n).get_shape())
    cfg = json.load(open(os.path.join(hf, "config.json")))
    L = cfg_get(cfg, "num_hidden_layers")
    stacks = defaultdict(set)                       # the name before ".layers." -> its layers
    for n in shapes:
        m = EXPERT.match(n)
        if m and int(m.group(2)) < L:
            stacks[m.group(1).rsplit(".layers.", 1)[0]].add(int(m.group(2)))
    decoder = max(stacks, key=lambda k: len(stacks[k]), default=None)
    prefix, ids, fused = {}, defaultdict(set), {}
    for n, s in shapes.items():
        m = EXPERT.match(n)
        if not m or int(m.group(2)) >= L or m.group(1).rsplit(".layers.", 1)[0] != decoder:
            continue
        l = int(m.group(2))
        if prefix.setdefault(l, m.group(1)) != m.group(1):
            fail(f"layer {l}: experts under two prefixes ({prefix[l]}, {m.group(1)})")
        if m.group(3) is None:
            fused[n] = s[0]
        else:
            ids[(l, m.group(4))].add(int(m.group(3)))
    return {"shapes": shapes, "prefix": prefix, "ids": ids, "fused": fused, "cfg": cfg}


def check_ckpt(meta, ck):
    """Errors between the file and the checkpoint; also the per-expert and shared sizes."""
    E, err = meta["num_experts"], []
    L, k = cfg_get(ck["cfg"], "num_hidden_layers"), cfg_get(ck["cfg"], *TOP_K_KEYS)
    if L != meta["num_layers"]:
        err.append(f"num_layers {meta['num_layers']}, the checkpoint has {L}")
    if k != meta["top_k"]:
        err.append(f"top_k {meta['top_k']}, the checkpoint has {k}")
    for (l, rest), got in ck["ids"].items():
        if got != set(range(E)):
            err.append(f"layer {l} {rest}: experts {min(got)}..{max(got)} ({len(got)}), expected 0..{E - 1}")
    for n, e in ck["fused"].items():
        if e != E:
            err.append(f"{n}: {e} experts, expected {E}")
    expert = defaultdict(int)          # parameters of one expert, per layer
    for (l, rest), got in ck["ids"].items():
        expert[l] += int(np.prod(ck["shapes"][f"{ck['prefix'][l]}experts.0.{rest}"]))
    for n in ck["fused"]:
        l = int(EXPERT.match(n).group(2))
        expert[l] += int(np.prod(ck["shapes"][n])) // E
    for l, pre in ck["prefix"].items():
        routers = [n for n in ck["shapes"] if n.startswith(pre) and n[len(pre):] in ROUTER_SUFFIXES]
        if not any(n.endswith(".weight") for n in routers):
            err.append(f"layer {l}: no router under {pre} ({', '.join(ROUTER_SUFFIXES)})")
        for n in routers:
            if ck["shapes"][n][0] != E:
                err.append(f"{n}: {ck['shapes'][n][0]} rows, expected {E}")
    shared = {}
    for l, pre in ck["prefix"].items():
        for proj in ("gate_proj.weight", "w1.weight", "up_proj.weight"):    # up: relu^2, no gate
            g = [s for n, s in ck["shapes"].items()
                 if n.startswith(pre) and SHARED.match(n[len(pre):]) and n.endswith(proj)]
            if g:
                shared[l] = g[0][0]
                break
    return err, dict(expert), shared


def moe_inter(ck, l):
    """The routed experts' intermediate size in layer l: gate projection rows (up for a
    relu^2 expert without one; half the rows of a fused [E, 2I, d] gate_up)."""
    pre, shapes = ck["prefix"][l], ck["shapes"]
    for rest in ("gate_proj.weight", "w1.weight", "up_proj.weight"):
        if f"{pre}experts.0.{rest}" in shapes:
            return shapes[f"{pre}experts.0.{rest}"][0]
    if f"{pre}experts.gate_up_proj" in shapes:
        return shapes[f"{pre}experts.gate_up_proj"][1] // 2
    return None


def hw_check(meta):
    """Our generators' checks (regen.sh runs the same), on a 32-port card: (errors, the
    model's MoE layers, weights + tables in MiB per channel). None outside the repo."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if not os.path.exists(os.path.join(root, "common", "model_configs.py")):
        return None
    os.environ["VVD_NUM_HBM"] = "32"
    sys.path[:0] = [root, os.path.join(root, "data_prep")]
    from common.model_configs import (MODEL_CONFIGS, hf_layer_block, hf_num_layers, moe_rep_meta_blocks,
                                      replicate_experts_v2, shard)
    import address_map
    mc = MODEL_CONFIGS.get(meta["model"])
    if mc is None:
        return [f"model {meta['model']!r} is not in common/model_configs.py"], None, None
    # rows are HF decoder layers; a hybrid stack may hold one in two blocks (mixer, FFN)
    L = hf_num_layers(mc)
    moe = {l for l in range(L) if mc.layer_is_moe(hf_layer_block(mc, l))}
    err = [f"{key} {meta[key]}, our {meta['model']} has {want}"
           for key, want in (("num_experts", mc.num_experts), ("top_k", mc.top_k), ("num_layers", L))
           if meta[key] != want]
    if err:
        return err, moe, None
    try:
        sh = shard(mc, 2)
        reps, perm = moe_rep_meta_blocks(sh, meta["rep_layers"], meta["perm"])
        rmc = replicate_experts_v2(sh, reps, meta.get("shared_units", 0), perm)
    except ValueError as e:
        return [str(e)], moe, None
    fixed = address_map.weight_layout(rmc, 4, rmc.num_layers).hbm0_15_end * 4 / 2**20
    cap = address_map.DRAM_CH_BYTE_SIZE / 2**20
    if fixed >= cap:
        return [f"does not fit: weights and tables take {fixed:.0f} MiB of each {cap:.0f} MiB channel "
                f"before any K/V (lower the largest R)"], moe, fixed
    return [], moe, fixed


# ---------------------------------------------------------------- commands

def cmd_init(a):
    ck = scan(a.hf)
    E = {e for e in ck["fused"].values()} | {len(v) for v in ck["ids"].values()}
    if len(E) != 1:
        fail(f"expert counts {sorted(E)}: not one MoE geometry")
    E, L = E.pop(), cfg_get(ck["cfg"], "num_hidden_layers")
    meta = {"schema": SCHEMA, "model": a.model, "hf_repo": a.repo or "", "num_experts": E,
            "top_k": cfg_get(ck["cfg"], *TOP_K_KEYS), "num_layers": L,
            "num_ranks": 2, "shared_units": 0, "rep_layers": [0] * L,
            "perm": [list(range(E)) for _ in range(L)], "note": "identity (moe_rep_meta.py init)"}
    with open(a.out, "w") as f:
        json.dump(meta, f)
    print(f"{a.out}: E {E}, top-k {meta['top_k']}, {L} layers ({len(ck['prefix'])} MoE), R 0, identity")


def cmd_check(a):
    meta, sha = load(a.meta)
    err = validate(meta)
    if err:
        print("\n".join(f"  {e}" for e in err[:40]))
        fail(f"{a.meta}: {len(err)} problem(s)")
    ck = scan(a.hf) if a.hf else None
    hw = hw_check(meta)
    moe = set(ck["prefix"]) if ck else hw[1] if hw else None
    err = validate(meta, moe) + (hw[0] if hw else [])
    if ck and hw and hw[1] is not None and hw[1] != set(ck["prefix"]):
        err.append(f"MoE layers: the checkpoint has {sorted(ck['prefix'])}, our table {sorted(hw[1])}")
    expert = {}
    if ck:
        e2, expert, shared = check_ckpt(meta, ck)
        err += e2
        U = meta.get("shared_units", 0)
        for l in ck["prefix"]:
            if U and shared.get(l) != U * moe_inter(ck, l):
                err.append(f"layer {l}: shared_units {U} needs a shared expert of {U} x "
                           f"{moe_inter(ck, l)} (it has {shared.get(l, 0)})")
    if err:
        print("\n".join(f"  {e}" for e in err[:40]))
        fail(f"{a.meta}: {len(err)} problem(s)")
    E, U = meta["num_experts"], meta.get("shared_units", 0)
    moe = sorted(moe) if moe is not None else moe_layers(meta)
    reps = [meta["rep_layers"][l] for l in moe]
    rmax = max(reps, default=0)
    print(f"{a.meta}: OK  (sha256 {sha})")
    print(f"  model {meta['model']}, E {E}, top-k {meta['top_k']}, {meta['num_layers']} layers "
          f"({len(moe)} MoE), shared units {U}")
    print(f"  R per MoE layer: mean {np.mean(reps) if reps else 0:.1f}, max {rmax}; "
          f"{sum(meta['perm'][l] != list(range(E)) for l in moe)} layers reordered"
          + ("" if ck or hw else " (MoE layers = the layers the file touches; --hf gives the real set)"))
    print(f"  expert slots per card per layer: {rmax + (E - rmax) // 2 + U} (R 0: {E // 2}"
          f"{' + the TP-split shared expert' if U else ''}); the largest R sizes every layer")
    if expert:
        extra = (rmax + U) / 2 * sum(expert[l] for l in moe) * W4_BYTES / 2**30
        print(f"  extra W4 expert storage per card vs R 0: {extra:.2f} GiB (less room for the K/V pool)")
    if hw:
        print(f"  our hardware check: OK; weights and tables {hw[2]:.0f} MiB of each 512 MiB channel, "
              f"the rest is K/V")
    else:
        print("  our hardware check: skipped (outside the repo; regen.sh runs it)")


def cmd_layout(a):
    meta, sha = load(a.meta)
    err = validate(meta)
    if err:
        fail("; ".join(err))
    out = {"meta_sha256": sha, "shared_units": meta.get("shared_units", 0), "layers": {}}
    for l in moe_layers(meta):
        rep, c0, c1 = card_lists(meta, l)
        out["layers"][str(l)] = {"R": len(rep), "both_cards": rep, "card0_only": c0, "card1_only": c1}
        if not a.json:
            print(f"layer {l:3d}  R {len(rep):3d}  both {rep}\n           card 0 {c0}\n           card 1 {c1}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"{a.json}: {len(out['layers'])} layers")


def two_card_cost(hw, R, E, U):
    """hw [T, k]: hardware ids picked. Each card runs its own picks; the f = r + U shared ones
    (picks below R and the U shared units) split t to card 0, f - t to card 1 with
    t = clamp(ceil((f + b - a) / 2), 0, f) (moe_router.sv; moe_route_model.rep_split).
    Returns max(n0, n1) per token."""
    u = (E - R) // 2
    k = hw.shape[1]
    r = (hw < R).sum(1)
    a = ((hw >= R) & (hw < R + u)).sum(1)
    b = k - r - a
    f = r + U
    t = np.clip((f + b - a + 1) // 2, 0, f)
    return np.maximum(a + t, b + f - t)


def cmd_balance(a):
    meta, _ = load(a.meta)
    err = validate(meta)
    if err:
        fail("; ".join(err))
    tr = np.load(a.traces)
    E, L, k, U = meta["num_experts"], meta["num_layers"], meta["top_k"], meta.get("shared_units", 0)
    if tr.ndim != 3 or tr.shape[0] != L or tr.shape[2] != k:
        fail(f"traces {tr.shape}: expected [{L}, tokens, {k}] original expert ids (-1 on dense layers)")
    if not np.issubdtype(tr.dtype, np.integer):
        fail(f"traces are {tr.dtype}: integer expert ids")
    for l in range(L):
        row = tr[l]
        if (row == -1).all():
            continue
        if row.min() < 0 or row.max() >= E:
            fail(f"layer {l}: ids outside 0..{E - 1} (a dense layer's row is all -1)")
        if (np.sort(row, axis=1)[:, 1:] == np.sort(row, axis=1)[:, :-1]).any():
            fail(f"layer {l}: a token picks the same expert twice")
    floor = (k + U + 1) // 2
    now, base = [], []
    print(f"layer    R   cost   R0-identity   (ideal {floor}, in experts per card)")
    for l in range(L):
        orig = tr[l].astype(np.int64)
        if (orig == -1).all():
            continue
        inv = np.argsort(meta["perm"][l])
        c = two_card_cost(inv[orig], meta["rep_layers"][l], E, U).mean()
        c0 = two_card_cost(orig, 0, E, U).mean()
        now.append(c)
        base.append(c0)
        print(f"{l:5d} {meta['rep_layers'][l]:4d} {c:6.3f} {c0:13.3f}")
    print(f"mean   {np.mean(now):.3f}  (R 0 identity {np.mean(base):.3f}, ideal {floor})")


def remap(name, t, perm, inv, prefix):
    """The tensor's name and value in the reordered checkpoint, and what was done. perm[l]:
    index tensor (row j = original row perm[l][j]); inv[l][orig] = new expert number."""
    m = EXPERT.match(name)
    if m and prefix.get(int(m.group(2))) == m.group(1):
        l = int(m.group(2))
        if m.group(3) is not None:
            return f"{m.group(1)}experts.{inv[l][int(m.group(3))]}.{m.group(4)}", t, "renamed"
        return name, t[perm[l]], "permuted"
    for l, pre in prefix.items():
        if name.startswith(pre) and name[len(pre):] in ROUTER_SUFFIXES:
            return name, t[perm[l]], "permuted"
    return name, t, "kept"


def cmd_apply(a):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    meta, sha = load(a.meta)
    src, dst = os.path.realpath(a.hf), os.path.realpath(a.out)
    if os.path.commonpath([src, dst]) in (src, dst):
        fail("--out and --hf must not contain each other")
    if os.path.exists(a.out) and os.listdir(a.out):
        fail(f"{a.out} is not empty")
    ck = scan(a.hf)
    err = validate(meta, set(ck["prefix"])) or check_ckpt(meta, ck)[0]
    if err:
        fail("; ".join(err[:10]))
    os.makedirs(a.out, exist_ok=True)
    inv = {l: np.argsort(meta["perm"][l]).tolist() for l in ck["prefix"]}
    perm = {l: torch.tensor(meta["perm"][l]) for l in ck["prefix"]}
    weight_map, count, digest = {}, defaultdict(int), {}
    for fn in shard_files(a.hf):
        out = {}
        with safe_open(os.path.join(a.hf, fn), "pt") as f:
            st_meta = f.metadata() or {"format": "pt"}
            for n in f.keys():
                new, t, how = remap(n, f.get_tensor(n), perm, inv, ck["prefix"])
                out[new] = t.contiguous()
                count[how] += 1
        save_file(out, os.path.join(a.out, fn), metadata=st_meta)
        for n, t in out.items():
            weight_map[n] = fn
            h = hashlib.sha256(f"{n}|{t.dtype}|{tuple(t.shape)}|".encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy())
            digest[n] = h.hexdigest()
        print(f"  {fn}: {len(out)} tensors", flush=True)
    idx = os.path.join(a.hf, "model.safetensors.index.json")
    if os.path.exists(idx):
        j = json.load(open(idx))
        j["weight_map"] = dict(sorted(weight_map.items()))
        with open(os.path.join(a.out, "model.safetensors.index.json"), "w") as f:
            json.dump(j, f, indent=2)
    for n in os.listdir(a.hf):
        src = os.path.join(a.hf, n)
        if n.startswith(".") or n.endswith(".safetensors") or n == "model.safetensors.index.json":
            continue
        (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, os.path.join(a.out, n))
    shutil.copy2(a.meta, os.path.join(a.out, "moe_rep_meta.json"))
    content = hashlib.sha256("".join(digest[n] for n in sorted(digest)).encode()).hexdigest()
    with open(os.path.join(a.out, "moe_rep_apply.json"), "w") as f:
        json.dump({"meta_sha256": sha, "source": os.path.abspath(a.hf), "content_sha256": content,
                   "tensors": dict(count)}, f, indent=1)
    print(f"{a.out}: {dict(count)}; content sha256 {content}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init", help="an identity file for a checkpoint")
    p.add_argument("--hf", required=True)
    p.add_argument("--model", required=True, help="our model key (docs/moe_rep_handoff.md table)")
    p.add_argument("--repo", help="the HF repo id, for the record")
    p.add_argument("--out", required=True)
    p = sub.add_parser("check", help="validate a file")
    p.add_argument("meta")
    p.add_argument("--hf", help="the original HF checkpoint directory")
    p = sub.add_parser("layout", help="per-card expert lists")
    p.add_argument("meta")
    p.add_argument("--json", help="write them here instead of printing")
    p = sub.add_parser("balance", help="two-card cost from routing traces")
    p.add_argument("meta")
    p.add_argument("--traces", required=True, help=".npy [layers, tokens, top_k] original expert ids")
    p = sub.add_parser("apply", help="write the reordered HF checkpoint")
    p.add_argument("meta")
    p.add_argument("--hf", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"init": cmd_init, "check": cmd_check, "layout": cmd_layout, "balance": cmd_balance,
     "apply": cmd_apply}[a.cmd](a)


if __name__ == "__main__":
    main()
