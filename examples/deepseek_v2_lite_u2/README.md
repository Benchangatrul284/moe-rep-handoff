# 範例：DeepSeek-V2-Lite，shared expert 切成 2 個 unit（U = 2）

| 檔案 | 內容 |
|---|---|
| `v2lite_u2.json` | U = 2、每層 R = 0、perm identity：只切 shared expert，不複製也不重排 |

## 1. meta 怎麼寫

```json
{
  "schema": "moe_rep_v2",
  "model": "deepseek-v2-lite",
  "hf_repo": "deepseek-ai/DeepSeek-V2-Lite-Chat",
  "num_experts": 64,
  "top_k": 6,
  "num_layers": 27,
  "num_ranks": 2,
  "shared_units": 2,
  "rep_layers": [0, 0, 0, ...],
  "perm": [[0, 1, 2, ..., 63], ...],
  "note": "..."
}
```

- shared expert 只用 **`shared_units`** 一個整數表示，所有 MoE 層共用，不能逐層設。
- shared expert **不在 `perm` 裡**。`perm`、`rep_layers` 只管 64 個 routed expert，寫法和 U = 0 時一樣：27 格 R、27 列 perm，第 0 層是 dense，填 R = 0、identity。
- 怎麼切成 unit 不用寫：我們這邊沿 intermediate 維度切成 U 段，每段一個 routed expert 大小。V2-Lite 的 shared expert 是 2 × 1408，剛好 2 段；`check --hf` 會逐層確認這件事。
- U 和 R、perm 互相獨立：要複製、重排，就照主 README 第 3、4 節設 `rep_layers`、`perm`，`shared_units` 留 2。

## 2. 硬體怎麼用這兩個 unit

| | U = 0（TP 切法） | U = 2 |
|---|---|---|
| shared expert 放哪 | 兩張卡各放一半 | 兩個 unit 兩張卡都有 |
| 每個 token | 每張卡固定算一半（一個 expert 大小） | 兩個 unit 當成「每次都被選到的副本」，由 t 規則分給比較閒的卡 |
| 成本 | max(a, b) + 1 | max(a, b, ⌈(k + 2) / 2⌉) |
| 每張卡每層 expert slot（R = 0） | 32 + 1 | 32 + 2 = 34 |

例如 k = 6，選到的 a = 4、b = 2：U = 0 成本 5；U = 2 時兩個 unit 都給卡 1，兩邊各 4。

## 3. 板子上量過

2026-10-16，r2s170 @ 170 MHz，兩卡，真實文字 prompt（256 prefill → 512），kernel decode tok/s，4 次平均：

| 配置 | decode | 變化 |
|---|---|---|
| U = 0、R = 0（原本的 TP 切法） | 314.7 | — |
| **U = 2、R = 0、identity（`v2lite_u2.json`）** | **340.4** | **+8.2 %** |
| U = 2 加上隨機 R（平均 17）、隨機 perm | 343.6 | +9.2 % |

V2-Lite 的收益幾乎都來自 shared unit；再加複製只多 1 %。

## 4. 檢查

```sh
python ../../moe_rep_meta.py check v2lite_u2.json      # 有 HF 權重就加 --hf $HF
```

我們這邊跑硬體檢查的結果：`our hardware check: OK; weights and tables 133 MiB of each 512 MiB channel, the rest is K/V`。
