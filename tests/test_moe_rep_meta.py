#!/usr/bin/env python3
"""Tests of moe_rep_meta.py on tiny random HF models (no download, CPU, about a minute).

1. apply keeps the function: logits of the reordered checkpoint equal the original's, and its
   router scores are the original's in perm order (Qwen3-MoE, DeepSeek-V3 with the score bias and
   two shared experts, Mixtral). Negative: the experts reordered without the router differ.
2. apply on a fused-expert checkpoint (Gemma-4 style [E, ...] tensors) matches the per-expert path;
   a layer past num_hidden_layers (GLM-4.7-Flash's MTP layer) stays as it is.
3. check refuses broken files (perm not a permutation or not integers, odd R, R on a dense layer, U vs
   the shared size, top-k); balance refuses bad traces; apply refuses an output inside the input.
   With ACCEL_REPO=<our repo> also our hardware check: a geometry that differs from our table or does
   not fit HBM.
4. balance's split rule gives the per-card counts of the hardware router (router_rule_ref.assign).
5. The Qwen3-30B-A3B example: its meta is valid and beats the identity layout on held-out traces.

usage: python tests/test_moe_rep_meta.py [work dir]   (torch, transformers >= 5, safetensors)
"""
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
import transformers
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOL = os.path.join(ROOT, "moe_rep_meta.py")      # main() runs a copy in the work dir
EXAMPLE = os.path.join(ROOT, "examples", "qwen3_30b_a3b")
ACCEL = os.environ.get("ACCEL_REPO")              # our repo, for the hardware check (optional)
REPO_TOOL = ACCEL and os.path.join(ACCEL, "data_prep", "tools", "moe_rep_meta.py")
BOARD = ACCEL and os.path.join(ACCEL, "docs", "design", "moe_rep2", "board")
sys.path[:0] = [HERE, ROOT]
import moe_rep_meta as mrm             # noqa: E402
import router_rule_ref                 # noqa: E402

transformers.utils.logging.disable_progress_bar()
E, K = 8, 2
CONFIGS = {
    "qwen3_moe": (0, lambda: transformers.Qwen3MoeConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, moe_intermediate_size=16, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, num_experts=E, num_experts_per_tok=K,
        norm_topk_prob=True)),
    # transformers 5.0 sizes DeepSeek-V3's routed experts by intermediate_size: equal to
    # moe_intermediate_size here, so the shared expert is two units of one routed expert
    "deepseek_v3": (2, lambda: transformers.DeepseekV3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, moe_intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=4, n_routed_experts=E, n_shared_experts=2,
        num_experts_per_tok=K, first_k_dense_replace=1, n_group=1, topk_group=1, q_lora_rank=None,
        kv_lora_rank=16, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8)),
    "mixtral": (0, lambda: transformers.MixtralConfig(
        vocab_size=64, hidden_size=32, intermediate_size=16, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, num_local_experts=E, num_experts_per_tok=K)),
}
fails = []


def check(cond, what):
    print(("PASS " if cond else "FAIL ") + what, flush=True)
    if not cond:
        fails.append(what)


def tool(*args, path=None):
    r = subprocess.run([sys.executable, path or TOOL, *args], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def random_meta(ck, U, seed):
    rng = random.Random(seed)
    L = ck["cfg"]["num_hidden_layers"]
    perm, reps = [], []
    for l in range(L):
        p = list(range(E))
        if l in ck["prefix"]:
            rng.shuffle(p)
        perm.append(p)
        reps.append(rng.choice([0, 2, 4, 6, 8]) if l in ck["prefix"] else 0)
    return {"schema": mrm.SCHEMA, "model": "tiny", "num_experts": E, "top_k": K, "num_layers": L,
            "num_ranks": 2, "shared_units": U, "rep_layers": reps, "perm": perm}


def logits(path, ids):
    m = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    with torch.no_grad():
        return m(ids).logits


def router_logits(path, ids):
    m = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    with torch.no_grad():
        out = m(ids, output_router_logits=True)
    return out.router_logits


def model_tests(work):
    ids = torch.randint(0, 64, (2, 12), generator=torch.Generator().manual_seed(1))
    for name, (U, cfg) in CONFIGS.items():
        torch.manual_seed(0)
        m = transformers.AutoModelForCausalLM.from_config(cfg())
        if name == "deepseek_v3":       # a nonzero score bias, so that its reorder is visible
            for layer in m.model.layers[1:]:
                layer.mlp.gate.e_score_correction_bias.copy_(torch.randn(E))
        src, out = os.path.join(work, name), os.path.join(work, name + "_rep")
        m.save_pretrained(src)
        ck = mrm.scan(src)
        meta = random_meta(ck, U, seed=7)
        mp = os.path.join(work, name + ".json")
        json.dump(meta, open(mp, "w"))
        rc, log = tool("check", mp, "--hf", src)
        check(rc == 0, f"{name}: check accepts a random file (U {U})\n{log if rc else ''}")
        rc, log = tool("apply", mp, "--hf", src, "--out", out)
        check(rc == 0, f"{name}: apply{'' if rc == 0 else chr(10) + log}")
        if rc:
            continue
        a, b = logits(src, ids), logits(out, ids)
        err = (a - b).abs().max().item()
        check(err < 1e-5, f"{name}: reordered logits equal the original (max |diff| {err:.2e})")
        try:
            ra, rb = router_logits(src, ids), router_logits(out, ids)
            moe = sorted(ck["prefix"])
            ok = all(torch.allclose(rb[i], ra[i][:, meta["perm"][l]], atol=1e-6)
                     for i, l in enumerate(moe[-len(ra):]))
            check(ok, f"{name}: router score j of the reordered model = original score perm[l][j]")
        except (TypeError, ValueError, AttributeError) as e:
            print(f"     ({name}: no router logits output: {e})")
        # negative: experts reordered, router left as it was
        neg = os.path.join(work, name + "_neg")
        shutil.copytree(out, neg)
        st = load_file(os.path.join(out, "model.safetensors"))
        orig = load_file(os.path.join(src, "model.safetensors"))
        for n in st:
            if any(n.endswith(s) for s in mrm.ROUTER_SUFFIXES) and ".layers." in n:
                st[n] = orig[n]
        save_file(st, os.path.join(neg, "model.safetensors"), metadata={"format": "pt"})
        err = (a - logits(neg, ids)).abs().max().item()
        check(err > 1e-3, f"{name}: negative, router not reordered -> logits differ ({err:.2e})")
        if name == "deepseek_v3":       # negative: everything reordered but the score bias
            st = load_file(os.path.join(out, "model.safetensors"))
            for n in st:
                if n.endswith("gate.e_score_correction_bias"):
                    st[n] = orig[n]
            save_file(st, os.path.join(neg, "model.safetensors"), metadata={"format": "pt"})
            err = (a - logits(neg, ids)).abs().max().item()
            check(err > 1e-3, f"{name}: negative, score bias not reordered -> logits differ ({err:.2e})")


def fused_test(work):
    """A Gemma-4 style checkpoint: experts stacked [E, ...], router.proj / per_expert_scale."""
    src = os.path.join(work, "qwen3_moe")
    st = load_file(os.path.join(src, "model.safetensors"))
    fused = {}
    for n, t in st.items():
        m = mrm.EXPERT.match(n)
        if m and m.group(3) is not None:
            key = f"{m.group(1)}experts.{m.group(4).replace('.weight', '')}"
            fused.setdefault(key, [None] * E)[int(m.group(3))] = t
        elif n.endswith("mlp.gate.weight"):
            fused[n.replace("mlp.gate.weight", "mlp.router.proj.weight")] = t
            fused[n.replace("mlp.gate.weight", "mlp.router.per_expert_scale")] = torch.arange(E, dtype=t.dtype)
        else:
            fused[n] = t
    fused = {n: torch.stack(v) if isinstance(v, list) else v for n, v in fused.items()}
    fdir, fout = os.path.join(work, "fused"), os.path.join(work, "fused_rep")
    os.makedirs(fdir, exist_ok=True)
    save_file(fused, os.path.join(fdir, "model.safetensors"), metadata={"format": "pt"})
    cfg = json.load(open(os.path.join(src, "config.json")))
    json.dump(cfg, open(os.path.join(fdir, "config.json"), "w"))
    rc, log = tool("apply", os.path.join(work, "qwen3_moe.json"), "--hf", fdir, "--out", fout)
    check(rc == 0, "fused: apply" + ("" if rc == 0 else "\n" + log))
    if rc:
        return
    got = load_file(os.path.join(fout, "model.safetensors"))
    ref = load_file(os.path.join(work, "qwen3_moe_rep", "model.safetensors"))
    meta = json.load(open(os.path.join(work, "qwen3_moe.json")))
    ok = True
    for n, t in got.items():
        m = mrm.EXPERT.match(n)
        if m:
            per = [ref[f"{m.group(1)}experts.{j}.{m.group(4)}.weight"] for j in range(E)]
            ok &= torch.equal(t, torch.stack(per))
        elif n.endswith("router.proj.weight"):
            ok &= torch.equal(t, ref[n.replace("router.proj.weight", "gate.weight")])
        elif n.endswith("router.per_expert_scale"):
            l = int(n.split(".layers.")[1].split(".")[0])
            ok &= torch.equal(t, torch.tensor(meta["perm"][l], dtype=t.dtype))
    check(ok, "fused: stacked experts, router.proj and per_expert_scale match the per-expert path")


def mtp_test(work):
    """Layer 2's MoE copied as layer 3 = num_hidden_layers: apply leaves it alone."""
    src = os.path.join(work, "qwen3_moe")
    st = load_file(os.path.join(src, "model.safetensors"))
    st.update({n.replace("layers.2.", "layers.3."): t.clone() for n, t in st.items() if ".layers.2.mlp." in n})
    mdir, mout = os.path.join(work, "mtp"), os.path.join(work, "mtp_rep")
    os.makedirs(mdir, exist_ok=True)
    save_file(st, os.path.join(mdir, "model.safetensors"), metadata={"format": "pt"})
    shutil.copy(os.path.join(src, "config.json"), mdir)
    rc, log = tool("apply", os.path.join(work, "qwen3_moe.json"), "--hf", mdir, "--out", mout)
    check(rc == 0, "mtp: apply" + ("" if rc == 0 else "\n" + log))
    if rc:
        return
    got = load_file(os.path.join(mout, "model.safetensors"))
    ref = load_file(os.path.join(work, "qwen3_moe_rep", "model.safetensors"))
    ok = all(torch.equal(got[n], st[n]) for n in st if ".layers.3." in n)
    ok &= all(torch.equal(got[n], t) for n, t in ref.items())
    check(ok, "mtp: the layer past num_hidden_layers is unchanged, the others as without it")


def check_refusals(work):
    src = os.path.join(work, "deepseek_v3")
    good = json.load(open(os.path.join(work, "deepseek_v3.json")))
    moe = [l for l in range(good["num_layers"]) if good["perm"][l] != list(range(E)) or good["rep_layers"][l]]
    cases = {
        "perm not a permutation": lambda m: m["perm"][moe[0]].__setitem__(0, m["perm"][moe[0]][1]),
        "odd R": lambda m: m["rep_layers"].__setitem__(moe[0], 3),
        "R on the dense layer 0": lambda m: m["rep_layers"].__setitem__(0, 2),
        "U 1 beside two shared experts": lambda m: m.__setitem__("shared_units", 1),
        "wrong top_k": lambda m: m.__setitem__("top_k", 3),
        "top_k above the router's 8": lambda m: m.__setitem__("top_k", 9),
        "perm entries as floats": lambda m: m["perm"].__setitem__(moe[0], [float(v) for v in m["perm"][moe[0]]]),
    }
    for what, edit in cases.items():
        m = json.loads(json.dumps(good))
        edit(m)
        p = os.path.join(work, "bad.json")
        json.dump(m, open(p, "w"))
        rc, _ = tool("check", p, "--hf", src)
        check(rc != 0, f"check refuses: {what}")


def repo_checks(work):
    """Our hardware check: our model table and the 32-port HBM budget."""
    if not ACCEL:
        print("     (repo checks skipped: set ACCEL_REPO to our repo to run them)")
        return
    q = json.load(open(os.path.join(BOARD, "q30b_r2b.json")))
    rc, log = tool("check", os.path.join(BOARD, "q30b_r2b.json"), path=REPO_TOOL)
    check(rc == 0 and "MiB of each 512 MiB channel" in log, "repo check: the demo's Qwen3-30B meta fits")
    cases = {
        "top_k 6 against our Qwen3-30B (8)": ("qwen3-30b-a3b", dict(q, top_k=6)),
        "Mixtral with all 8 experts on both cards (735 MiB > 512)":
            ("mixtral-8x7b", {"schema": mrm.SCHEMA, "model": "mixtral-8x7b", "num_experts": 8, "top_k": 2,
                              "num_layers": 32, "num_ranks": 2, "shared_units": 0, "rep_layers": [8] * 32,
                              "perm": [list(range(8))] * 32}),
    }
    for what, (_, m) in cases.items():
        p = os.path.join(work, "repo_bad.json")
        json.dump(m, open(p, "w"))
        rc, log = tool("check", p, path=REPO_TOOL)
        check(rc != 0, f"repo check refuses: {what}")
    m = dict(cases["Mixtral with all 8 experts on both cards (735 MiB > 512)"][1], rep_layers=[0] * 32)
    json.dump(m, open(os.path.join(work, "repo_ok.json"), "w"))
    rc, _ = tool("check", os.path.join(work, "repo_ok.json"), path=REPO_TOOL)
    check(rc == 0, "repo check: Mixtral without replicas fits")


def io_refusals(work):
    mp = os.path.join(work, "qwen3_moe.json")
    meta = json.load(open(mp))
    L = meta["num_layers"]
    rng = np.random.default_rng(0)
    good = np.stack([np.stack([rng.choice(E, K, replace=False) for _ in range(50)]) for _ in range(L)])
    bad = {"ok": good, "float ids": good.astype(np.float32), "an id out of range": np.where(good == 0, E, good),
           "a token picking one expert twice": np.concatenate([good[:, :, :1], good[:, :, :1]], axis=2)}
    for what, tr in bad.items():
        p = os.path.join(work, "trace.npy")
        np.save(p, tr)
        rc, _ = tool("balance", mp, "--traces", p)
        check((rc == 0) == (what == "ok"), f"balance {'accepts' if what == 'ok' else 'refuses'} traces: {what}")
    rc, _ = tool("apply", mp, "--hf", os.path.join(work, "qwen3_moe"),
                 "--out", os.path.join(work, "qwen3_moe", "rep"))
    check(rc != 0, "apply refuses an output directory inside the input")


def balance_test():
    rng = np.random.default_rng(3)
    for Ex, k, U in ((128, 8, 0), (64, 6, 2), (8, 2, 0)):
        for R in (0, 2, Ex // 4, Ex - 2, Ex):
            if R > 128:
                continue
            u = (Ex - R) // 2
            tops = np.stack([rng.choice(Ex, k, replace=False) for _ in range(300)])
            cost = mrm.two_card_cost(tops, R, Ex, U)
            ref = []
            for top in tops.tolist():
                n = [len(router_rule_ref.assign(top, R + rank * u, u, R, rank, U)) for rank in (0, 1)]
                ref.append(max(n))
            check(np.array_equal(cost, ref), f"balance rule = router_rule_ref.assign (E {Ex}, k {k}, R {R}, U {U})")


def collect_test():
    """collect_traces.Recorder on a tiny Qwen3-MoE (one dense layer): the experts the model routes to."""
    sys.path.insert(0, EXAMPLE)
    import collect_traces
    torch.manual_seed(0)
    cfg = CONFIGS["qwen3_moe"][1]()
    cfg.mlp_only_layers = [0]
    m = transformers.AutoModelForCausalLM.from_config(cfg).eval()
    rec = collect_traces.Recorder(m, K)
    ids = torch.randint(0, 64, (1, 20), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        out = m(ids, output_router_logits=True)
    tr = rec.trace(cfg.num_hidden_layers, 20)
    ref = [r.topk(K, dim=-1).indices for r in out.router_logits]
    ok = (tr[0] == -1).all() and sorted(rec.layers) == [1, 2]
    ok &= all(np.array_equal(np.sort(tr[l], 1), np.sort(ref[-2 + j].numpy(), 1)) for j, l in enumerate((1, 2)))
    check(ok, f"collect_traces: recorded picks = the model's routing, dense layer -1 ({sorted(set(rec.how.values()))})")


def example_test():
    meta = os.path.join(EXAMPLE, "q30_r8.json")
    rc, log = tool("check", meta)
    check(rc == 0, "example: q30_r8.json is valid" + ("" if rc == 0 else "\n" + log))
    rc, log = tool("balance", meta, "--traces", os.path.join(EXAMPLE, "traces", "trace_heldout_chatbig.npy"))
    m = re.search(r"mean\s+([\d.]+)\s+\(R 0 identity ([\d.]+)", log)
    ok = rc == 0 and m is not None and float(m.group(1)) < float(m.group(2))
    check(ok, f"example: held-out cost below the identity layout ({log.strip().splitlines()[-1] if rc == 0 else log})")


def main():
    global TOOL
    work = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="moe_rep_meta_")
    os.makedirs(work, exist_ok=True)
    TOOL = shutil.copy(TOOL, os.path.join(work, "moe_rep_meta.py"))
    model_tests(work)
    fused_test(work)
    mtp_test(work)
    check_refusals(work)
    repo_checks(work)
    io_refusals(work)
    balance_test()
    collect_test()
    example_test()
    print(f"{'ALL PASS' if not fails else f'{len(fails)} FAILED'} ({work})")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
