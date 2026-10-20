#!/usr/bin/env python3
"""Routing traces of a HF MoE model: per MoE layer, the top-k original expert ids of every position.

Hooks each layer's router (`mlp.gate`, or `block_sparse_moe.gate` for Mixtral) and records the
experts it picks: the module's own indices when it returns them (newer transformers), otherwise the
top-k of its logits (softmax routers, e.g. Qwen3-MoE: same order as the softmax). A router with a
score bias or group-limited routing (DeepSeek-V3, GLM, Moonlight) must return its indices, or the
picks come out wrong: check the printed source line on such a model.

Sources: wikitext-2 windows (teacher-forced) and chat (greedy answers to the prompts below, then the
answer positions teacher-forced, the way the board decodes). Writes int16 [layers, tokens, top_k]
(-1 on dense layers): trace_{calib,heldout}_{wt2,chat}.npy, chat_samples.txt, trace_meta.json.

    python collect_traces.py --hf Qwen/Qwen3-30B-A3B-Instruct-2507 --out traces_new

Qwen3-30B-A3B in bf16 needs ~61 GB of GPU memory (device_map auto spreads it over the GPUs).
The shipped traces/ came from our W4 export instead (W4 weights dequantized, bf16 activations), close
to what the board routes; HF bf16 weights pick slightly different experts on near ties.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "Explain how a transformer language model generates text, step by step.",
    "Write a Python function that checks whether a string is a palindrome, with tests.",
    "What are the main differences between TCP and UDP? Give examples of when to use each.",
    "請用繁體中文介紹台灣的夜市文化。",
    "Summarize the causes of the French Revolution in five bullet points.",
    "Write a short story about a robot who learns to paint.",
    "How does a mixture-of-experts model decide which experts to use?",
    "Solve: a train leaves at 3 pm at 80 km/h, another at 4 pm at 100 km/h. When does the second catch up?",
    "請解釋什麼是量化 (quantization)，以及它在大型語言模型上的用途。",
    "Give me a weekly workout plan for a beginner who has 30 minutes a day.",
    "What is the difference between a process and a thread in an operating system?",
    "Translate into English: 今天天氣很好，我們去公園散步吧。",
    "Write a haiku about autumn rain, then explain its imagery.",
    "Describe how HBM memory differs from DDR memory for accelerators.",
    "List five tips for writing clear technical documentation.",
    "請幫我寫一封email，向教授請假一天。",
    "Explain the Fourier transform to a high-school student.",
    "What are the pros and cons of electric cars compared to hybrids?",
    "Write a SQL query that returns the top 3 customers by total order value.",
    "Why is the sky blue? Answer in detail.",
    "請比較 FPGA 與 ASIC 在設計流程上的差異。",
    "Give a recipe for a simple vegetable curry.",
    "What is a hash table and how does it handle collisions?",
    "Write a limerick about a cat who loves coffee.",
]


def routers(model):
    """{decoder layer index: router module}; layers without one are dense."""
    out = {}
    for i, ly in enumerate(model.model.layers):
        for name in ("mlp", "block_sparse_moe"):
            g = getattr(getattr(ly, name, None), "gate", None)     # a dense MLP has gate_proj, not gate
            if isinstance(g, torch.nn.Module):
                out[i] = g
                break
    return out


class Recorder:
    def __init__(self, model, k):
        self.k, self.rec, self.how = k, {}, {}
        self.layers = routers(model)
        for i, g in self.layers.items():
            g.register_forward_hook(self.hook(i))

    def hook(self, i):
        def fn(_mod, _inp, out):
            idx = None
            if isinstance(out, tuple):
                ints = [t for t in out if torch.is_tensor(t) and not t.is_floating_point() and t.shape[-1] == self.k]
                idx = ints[0] if ints else None
                logits = out[0]
            else:
                logits = out
            if idx is None:
                idx = logits.float().topk(self.k, dim=-1).indices
                self.how[i] = "top-k of the router logits"
            else:
                self.how[i] = "the router's own indices"
            self.rec[i] = idx.reshape(-1, self.k).to(torch.int16).cpu()
        return fn

    def trace(self, L, T):
        """[L, T, k] of the last forward, -1 on dense layers."""
        tr = np.full((L, T, self.k), -1, np.int16)
        for i, t in self.rec.items():
            tr[i] = t.numpy()[-T:]
        return tr


def forward(model, rec, ids, keep_from, L):
    """One teacher-forced sequence: traces of positions keep_from.., NLL sum and count."""
    rec.rec.clear()
    x = torch.tensor([ids], device=model.device)
    with torch.no_grad():
        logits = model(x).logits[0].float()
    nll = torch.nn.functional.cross_entropy(logits[keep_from:-1], x[0, keep_from + 1:].to(logits.device), reduction="sum")
    return rec.trace(L, len(ids))[:, keep_from:], float(nll), len(ids) - keep_from - 1


def chat(model, tok, prompts, max_new):
    """Greedy answers: [(ids, prompt length)]."""
    out = []
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
        out.append((g[0].tolist(), enc["input_ids"].shape[1]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf", required=True, help="HF repo id or local checkpoint directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--win", type=int, default=2048, help="wikitext window length")
    ap.add_argument("--skip", type=int, default=32, help="first positions of a window not recorded")
    ap.add_argument("--calib-windows", type=int, default=32)
    ap.add_argument("--heldout-windows", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=320, help="chat answer length")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.hf)
    model = AutoModelForCausalLM.from_pretrained(a.hf, dtype=torch.bfloat16, device_map="auto").eval()
    cfg = model.config.get_text_config()
    k = next(getattr(cfg, n) for n in ("num_experts_per_tok", "num_experts_per_token", "top_k_experts") if hasattr(cfg, n))
    L = cfg.num_hidden_layers
    rec = Recorder(model, k)
    print(f"loaded in {time.time() - t0:.0f} s: {L} layers, {len(rec.layers)} MoE, top-{k}", flush=True)
    meta = {"hf": a.hf, "win": a.win, "skip": a.skip, "dtype": "bf16", "sources": {}}

    from datasets import load_dataset
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    for split, part, n in (("calib", "train", a.calib_windows), ("heldout", "test", a.heldout_windows)):
        ids = tok("\n\n".join(wt[part]["text"]))["input_ids"]
        trs, nll, cnt = [], 0.0, 0
        for s in range(0, min(len(ids) - a.win, n * a.win), a.win):
            tr, x, y = forward(model, rec, ids[s:s + a.win], a.skip, L)
            trs.append(tr)
            nll, cnt = nll + x, cnt + y
        arr = np.concatenate(trs, axis=1)
        np.save(os.path.join(a.out, f"trace_{split}_wt2.npy"), arr)
        meta["sources"][f"{split}/wt2"] = {"windows": len(trs), "positions": arr.shape[1], "ppl": float(np.exp(nll / cnt))}
        print(f"{split}/wt2: {arr.shape[1]} positions, ppl {np.exp(nll / cnt):.3f} ({time.time() - t0:.0f} s)", flush=True)

    seqs = chat(model, tok, PROMPTS, a.max_new)
    with open(os.path.join(a.out, "chat_samples.txt"), "w") as f:
        for ids, plen in seqs:
            f.write(tok.decode(ids[plen:]) + "\n=====\n")
    cut = len(seqs) * 2 // 3
    for split, part in (("calib", seqs[:cut]), ("heldout", seqs[cut:])):
        arr = np.concatenate([forward(model, rec, ids, plen, L)[0] for ids, plen in part], axis=1)
        np.save(os.path.join(a.out, f"trace_{split}_chat.npy"), arr)
        meta["sources"][f"{split}/chat"] = {"sequences": len(part), "positions": arr.shape[1]}
        print(f"{split}/chat: {arr.shape[1]} positions ({time.time() - t0:.0f} s)", flush=True)
    meta["router"] = sorted(set(rec.how.values()))
    json.dump(meta, open(os.path.join(a.out, "trace_meta.json"), "w"), indent=1)
    print(f"router picks from: {meta['router']}; done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
