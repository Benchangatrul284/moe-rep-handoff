# 範例：Qwen3-30B-A3B

從 routing trace 到一個可以交付的 meta 檔，完整走一遍。下面的指令都在這個目錄下執行。

| 檔案 | 內容 |
|---|---|
| `traces/trace_{calib,heldout}_{chat,chatbig}.npy` | 我們取好的 routing trace，uint8 `[48, tokens, 8]` |
| `collect_traces.py` | 自己取 trace 的範例（HF 模型 + router hook） |
| `make_meta.py` | 產生 meta 的**一種**方法（熱門 expert 複製 + 分卡局部搜尋）。換成你自己的實驗就是這一步 |
| `q30_r8.json` | 用 `make_meta.py` 產出的 meta：每層 R = 8 |
| `make_meta.log`、`balance_heldout.txt` | 產生過程和 held-out 的負載 |

## 1. routing trace

`traces/` 裡的是我們用板子上的 W4 權重（反量化回 bf16，activation bf16）取的，跟板子上 route 的結果最接近：

| 檔案 | 來源 | token 數 |
|---|---|---|
| `trace_calib_chat.npy` | 16 個聊天 prompt 的 greedy 回答 | 4750 |
| `trace_calib_chatbig.npy` | 約 100 個模板 prompt（中英文、程式）的回答 | 33529 |
| `trace_heldout_chat.npy` | 另外 8 個聊天 prompt | 2247 |
| `trace_heldout_chatbig.npy` | 另外約 50 個模板 prompt | 18144 |

只記回答的位置（teacher-forced），因為 demo 是聊天：decode 時每個 token 一個一個過 MoE，負載平衡最有影響的就是這段。calib 拿來設計，heldout 只拿來量，避免 overfit。

自己取（要 GPU，bf16 約 61 GB；`device_map="auto"` 會分到多張卡）：

```sh
python collect_traces.py --hf Qwen/Qwen3-30B-A3B-Instruct-2507 --out traces_new
```

會寫出 wikitext-2 和聊天的 trace（int16，dense 層 −1）。用 HF 的 bf16 權重和我們的 W4 權重，分數接近平手的 token 會選到不同的 expert，差異不大。換別的模型時注意：router 帶 score bias 或 group-limited routing 的模型（DeepSeek-V3、GLM、Moonlight）要讓 router 自己回傳 index，結尾印的 `router picks from` 要是 `the router's own indices`。

## 2. 產生 meta

```sh
python make_meta.py --calib traces/trace_calib_chat.npy traces/trace_calib_chatbig.npy \
    --model qwen3-30b-a3b --hf-repo Qwen/Qwen3-30B-A3B-Instruct-2507 --rep 8 --out q30_r8.json
```

CPU 約 25 秒。每個 MoE 層：

1. 在 calib trace 上被選最多次的 R 個 expert 拿去複製；
2. 其他 E − R 個平分兩卡：從「熱門的輪流放到比較輕的卡」開始，每一步做一次讓平均成本 max(a, b, ⌈k/2⌉) 下降最多的交換（卡 0 一個和卡 1 一個互換），直到沒有改善；
3. 再試著把一個複製的 expert 和某張卡上的 expert 互換，有改善就換，然後重做第 2 步。

`perm[l]` = 複製的（排序）+ 卡 0 的（排序）+ 卡 1 的（排序）。同樣的輸入、同樣的 `--seed`（CPU）產出的檔案完全一樣（`q30_r8.json` 的 sha256 `4b22183e…88b4a3`）。

這只是一個 baseline。可以改的方向：每層不同的 R（`--rep-layers`）、考慮 expert 之間的共現、用更多或不同的 trace、把 prefill 也算進來。

## 3. 檢查和算負載

```sh
python ../../moe_rep_meta.py check q30_r8.json            # 有 HF 權重就加 --hf $HF
python ../../moe_rep_meta.py balance q30_r8.json --traces traces/trace_heldout_chatbig.npy
python ../../moe_rep_meta.py layout q30_r8.json | head    # 每層哪些 expert 放哪張卡
```

held-out 的平均成本（每張卡每層要算的 expert 數，下限 4）。R = 8 是交付的 `q30_r8.json`，其他列是同一支程式換 `--rep`：

| 配置 | 每張卡每層 expert slot | heldout_chat | heldout_chatbig |
|---|---|---|---|
| R = 0、identity（不做任何事） | 64 | 5.076 | 5.080 |
| R = 0、只重排 | 64 | 4.709 | 4.635 |
| **R = 8** | 68 | **4.298** | **4.134** |
| R = 16 | 72 | 4.130 | 4.051 |
| R = 32 | 80 | 4.041 | 4.015 |

- 只重排就拿掉 35–40 % 的多餘成本（高於下限 4 的部分），不花 HBM；複製 8 個再拿掉剩下的大半。
- R 每多 2，每張卡多放約 120 MB（Qwen3-30B 一個 W4 expert 48 層），從 K/V 空間扣。R = 8 到 32 只再少 0.1–0.3，但多用 1.4 GB。
- 這個成本降低不會等比例變成 tok/s：每層 MoE 還有一段固定成本，實測 trace 成本降 16–18 % 時 tok/s 多 3–8 %。最後以板子為準。

我們這邊跑硬體檢查的結果（`q30_r8.json`）：`our hardware check: OK; weights and tables 256 MiB of each 512 MiB channel`。

## 4.（建議）重排 HF 模型確認功能

```sh
python ../../moe_rep_meta.py apply q30_r8.json --hf $HF --out $HF_rep
```

用 `$HF_rep` 跑 PPL 或聊天，應該和原模型一樣（上層 README 第 4 節第 6 步）。

## 5. 交付

```sh
sha256sum q30_r8.json
python ../../moe_rep_meta.py balance q30_r8.json --traces traces/trace_heldout_chatbig.npy > balance.txt
```

交 `q30_r8.json`、sha256、`balance.txt`。
