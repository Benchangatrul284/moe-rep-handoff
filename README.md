# MoE expert 複製／重排：交接用工具

雙卡跑 MoE 時，每一層的 expert 一半放卡 0、一半放卡 1，比較慢的那張卡決定這一層花多久。這個 repo 讓做實驗的人（學弟）決定：

- 哪些 expert 要兩張卡都放（**複製**）；
- 其他 expert 怎麼分到兩張卡（**重排**）。

決定的結果寫成一個 **meta 檔**（`moe_rep_v2` 格式）交給我們。我們用它產生上卡的 bundle，在 FPGA 上跑。**meta 檔是兩邊唯一的約定。**

```
學弟（GPU）                                                  我們（FPGA）
原始 HF 模型 → routing trace
   │ 決定每層的 R 和 perm（實驗本身，方法自由）
   ▼
meta.json ── check / balance / apply 自己驗證 ──(只交這個檔)──▶ check（多一道硬體檢查）
                                                              ▶ 兩張卡各產一包 bundle
                                                              ▶ 上卡：聊天要通順，量 tok/s
```

| 檔案 | 用途 |
|---|---|
| `moe_rep_meta.py` | 工具本體，單一檔案。`check`（不帶 `--hf`）、`layout`、`balance` 只要 numpy；`init`、`check --hf`、`apply` 還要 torch + safetensors |
| `tests/test_moe_rep_meta.py` | 工具的測試（tiny 隨機模型，CPU 約一分鐘） |
| `tests/router_rule_ref.py` | 硬體分配規則的 Python 參考（從我們 repo 的 `moe_route_model.py` 抄來） |
| `examples/qwen3_30b_a3b/` | **完整範例**：Qwen3-30B-A3B 的 trace、產生 meta 的範例方法、產出的 meta 和負載數字 |

安裝：`pip install -r requirements.txt`（測試和 `collect_traces.py` 要 transformers >= 5）。

`moe_rep_meta.py` 和我們 repo 的 `data_prep/tools/moe_rep_meta.py` 是同一份；裡面提到的 `docs/...`、`moe_router.sv` 是我們 repo 的路徑。

---

## 1. 為什麼要複製或重排

每個 token 在每層選 top-k 個 expert，兩張卡同時算各自那幾個：

> **成本 = max(卡 0 要算幾個 expert, 卡 1 要算幾個 expert)**

以 Qwen3-30B（k = 8）為例：8 個都落在卡 0，成本就是 8，卡 1 整段閒著；剛好 4 / 4，成本是 4，這是下限。

| 手段 | 意思 | 代價 |
|---|---|---|
| **重排**（`perm`） | 決定哪個 expert 放哪張卡，例如把常一起被選的 expert 拆到不同卡 | 無 |
| **複製**（每層的 `R`） | R 個 expert 兩張卡各放一份。被選到時由硬體當場決定給哪張卡算，把兩邊補平 | 每張卡多放 R/2 個 expert，佔 HBM，K/V 的空間（context）變小 |
| **shared expert 切 unit**（`U`） | DeepSeek 一類模型的 shared expert 每個 token 都跑。把它切成 U 份，當成「每次都被選到的副本」，一樣交給硬體分配 | 每張卡多放 U/2 個 expert 大小 |

板子上量過（kernel decode tok/s，真實文字 prompt）：

| 模型 | 配置 | tok/s 變化 |
|---|---|---|
| DeepSeek-V2-Lite | U = 2 | +8.2 % |
| DeepSeek-V2-Lite | U = 2 加上隨機 R | +9.2 % |
| Qwen3-30B | 熱門 expert 複製 8 個 | +3.2 % |

---

## 2. 硬體怎麼看 expert：hw id

每個 MoE 層的 expert 在硬體裡編號 0 .. E−1，叫 **hw id**。meta 檔裡 `perm[l][j]` 的意思是：

> **第 l 層的 hw id j，用的是原始 HF 模型第 `perm[l][j]` 號 expert。**

| hw id | 放在哪 |
|---|---|
| `0 .. R-1` | 兩張卡都有（複製） |
| `R .. R+u-1` | 只在卡 0，u = (E − R) / 2 |
| `R+u .. E-1` | 只在卡 1 |
| U 個 shared unit | 兩張卡都有 |

所以每一層的 `perm` 從前往後讀：前 R 個要複製，接著 u 個放卡 0，最後 u 個放卡 1。

**例子**：E = 8、R = 2、`perm = [5, 1, 0, 2, 3, 4, 6, 7]`。

- 原始 expert 5、1：兩張卡都有。
- 卡 0 自己的：0、2、3。
- 卡 1 自己的：4、6、7。

`python moe_rep_meta.py layout meta.json` 會把每層的這三組列出來。

### 被複製的 expert 給哪張卡算

token 選到的 k 個 expert 裡，記：

| 符號 | 意思 |
|---|---|
| a | 只在卡 0 的有幾個 |
| b | 只在卡 1 的有幾個 |
| r | 被複製的有幾個 |
| f | r + U（shared unit 也算進來） |

- 卡 0 拿 t 個：先拿 shared unit，再拿 hw id 最小的複製 expert。卡 1 拿剩下的 f − t 個。
- t = clamp(⌈(f + b − a) / 2⌉, 0, f)。

兩張卡看到的 top-k 相同，各自算出同一個 t，不用互相溝通。結果是：

> **成本 = max(a, b, ⌈(k + U) / 2⌉)**

只要「只在某一張卡」的 expert 沒有一邊多到壓倒另一邊，被複製的就能把兩邊補平。所以設計 meta 檔要追求：

- 常被選的 expert 拿去複製；
- 不複製的 expert 在兩張卡上被選到的機會差不多。

`balance` 指令就是用這條規則算的（`tests/` 裡和硬體規則的參考實作逐筆比對過）。

---

## 3. meta 檔格式（`moe_rep_v2`）

```json
{
  "schema": "moe_rep_v2",
  "model": "qwen3-30b-a3b",
  "hf_repo": "Qwen/Qwen3-30B-A3B-Instruct-2507",
  "num_experts": 128,
  "top_k": 8,
  "num_layers": 48,
  "num_ranks": 2,
  "shared_units": 0,
  "rep_layers": [8, 8, 8, ...],
  "perm": [[5, 10, 28, ...], ...],
  "note": "怎麼產生的：資料、方法、seed"
}
```

| 欄位 | 規則 |
|---|---|
| `schema` | 固定 `"moe_rep_v2"` |
| `model` | 我們的 model key，見第 6 節的表。我們靠它做硬體檢查，打錯會被擋下 |
| `hf_repo` | 原始模型的 HF repo，只是紀錄 |
| `num_experts`、`top_k`、`num_layers` | 必須和原始模型的 config 一致（`check --hf` 會比對） |
| `num_ranks` | 固定 2 |
| `shared_units` | U，0 或 2。2 只給 shared expert 剛好等於兩個 routed expert 大小的模型（第 6 節）。shared expert 只有一個 expert 大的模型維持 TP 切法，填 0，工具會擋 1 |
| `rep_layers` | 每一層一個 R，**dense 層也要有一格**（填 0）。R 是偶數，0 ≤ R ≤ min(128, E)。不同層可以不同 |
| `perm` | 每一層一個長度 E 的排列（0 .. E−1 各出現一次）。dense 層填 identity `[0, 1, ..., E-1]` |
| `note` | 寫清楚怎麼來的，之後才查得到 |

- **只重排、不複製**：R = 0，只改 perm。
- **只複製、不重排**：perm 用 identity，被複製的就是原始 0..R−1 號。
- **交出去就不要改**：要改就換新檔名。檔案的 sha256 是兩邊對帳用的。

---

## 4. 要做的事

下面的 `$HF` 是原始模型的目錄（含 `config.json` 和 `*.safetensors`）。完整跑過一次的例子在 [`examples/qwen3_30b_a3b/`](examples/qwen3_30b_a3b/README.md)。

**1. 拿原始模型。** 用第 6 節表裡的 repo 和 revision：

```sh
huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --revision 0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe --local-dir $HF
```

**2. 取 routing trace。** `.npy` 整數陣列，shape `[num_layers, tokens, top_k]`，內容是每個 token 在每層選到的**原始** expert 編號（router 的 top-k），dense 層整列填 −1。範例：`examples/qwen3_30b_a3b/collect_traces.py`。

**3. 決定每層的 R 和 perm。** 這是實驗本身，方法自由。範例的方法在 `examples/qwen3_30b_a3b/make_meta.py`。或從一個空白檔開始改：

```sh
python moe_rep_meta.py init --hf $HF --model qwen3-30b-a3b \
    --repo Qwen/Qwen3-30B-A3B-Instruct-2507 --out q30_x.json      # R 全 0、identity
```

**4. 檢查**，必須印出 `OK`：

```sh
python moe_rep_meta.py check q30_x.json --hf $HF
```

它檢查欄位、R、排列，dense 層必須 R = 0 且是 identity，以及和 checkpoint 的 expert 數、top-k、層數、router、shared expert 大小一致。也印出每張卡每層放幾個 expert、比 R = 0 多用多少 HBM。我們這邊會再跑一次，多檢查我們的 model 表和 HBM 放不放得下。

**5. 算負載**，**用沒拿來設計的 trace（held-out）**：

```sh
python moe_rep_meta.py balance q30_x.json --traces trace_heldout.npy
```

印出每層的平均成本（一張卡要算的 expert 數）、同一份 trace 在 R = 0、identity 下的成本、下限 ⌈(k + U)/2⌉。

**6.（建議）產生重排後的模型，確認功能沒變：**

```sh
python moe_rep_meta.py apply q30_x.json --hf $HF --out $HF_rep    # 新目錄，不能在 $HF 裡面
```

- `$HF_rep` 是一般的 HF 模型目錄，transformers 可以直接載入。第 j 號 expert 就是原始的 `perm[l][j]`，router 的列（和 score bias）一起重排。
- 數學上和原模型是同一個函數，logits、PPL 應該一樣，只差加法順序（fp32 約 1e−7）和分數剛好平手時的 top-k。
- 結束時印出 `content sha256`（也寫在 `$HF_rep/moe_rep_apply.json`）。

> 重排後的 HF 模型裡沒有「複製」：複製是硬體的擺放方式（同一個 expert 在兩張卡的 HBM 各放一份），模型的數學不變，所以 HF 格式裡一個 expert 只有一份。哪些 expert 被複製，看 `layout`。

**7. 交付**：meta 檔、它的 `sha256sum`、`balance` 的輸出（held-out trace），有跑 `apply` 的話再附 `content sha256`。

### 我們收到之後

1. 在我們的 repo 裡 `check`（多一道硬體檢查：和產 bundle 用的是同一個函式）。
2. 直接從我們量化好的 checkpoint 產兩張卡的 bundle（`regen.sh --moe-rep-meta`），不需要 HF 權重：router 的列、score bias、每個 expert slot 都按同一個 perm 取。量化是每個 expert、每一列各自做，所以「先重排再量化」和「先量化再重排」完全相同。bundle 裡記了 perm 的 sha256，對得回 meta 檔。
3. 上卡：聊天要通順（通過標準），在同一顆 image 上和現在的配置比 kernel tok/s。

---

## 5. 測試

```sh
python tests/test_moe_rep_meta.py
```

- `apply` 不改變模型：tiny Qwen3-MoE、DeepSeek-V3（score bias + 兩個 shared expert）、Mixtral 重排後的 logits 和原模型相同（≤ 1e−5，實測 ~1e−7），router 分數正好是原分數按 perm 重排。反向對照：漏重排 router 或 score bias，logits 就差 ~1e−2。
- fused expert（Gemma-4 的 `[E, ...]` tensor）和逐個 expert 的結果相同；`num_hidden_layers` 之後的 MTP 層不動。
- `check` 擋得下壞檔，`balance` 擋得下壞 trace。
- `balance` 的分配規則和硬體規則的參考實作（`tests/router_rule_ref.py`）逐 token 相同。
- `collect_traces.py` 記到的就是模型實際 route 到的 expert。
- 範例的 meta 合法，held-out 成本比 identity 低。

我們這邊設 `ACCEL_REPO=<我們的 repo>` 再跑，會多跑硬體檢查的測試。

---

## 6. 能用的模型和限制

| `model` | HF repo（revision） | E | k | MoE 層 | U 可用 | 板子上跑過複製 |
|---|---|---|---|---|---|---|
| `qwen3-30b-a3b` | Qwen/Qwen3-30B-A3B-Instruct-2507（0d7cf239） | 128 | 8 | 48/48 | 0 | ✓ |
| `deepseek-v2-lite` | deepseek-ai/DeepSeek-V2-Lite-Chat（85864749） | 64 | 6 | 26/27 | 0、2 | ✓ |
| `moonlight-16b-a3b` | moonshotai/Moonlight-16B-A3B-Instruct（4e735b07） | 64 | 6 | 26/27 | 0、2 | ✓ |
| `deepseek-moe-16b` | deepseek-ai/deepseek-moe-16b-chat（eefd8ac7） | 64 | 6 | 27/28 | 0、2 | 還沒 |
| `glm-4.7-flash-h32` | zai-org/GLM-4.7-Flash（7dd20894） | 64 | 4 | 46/47 | 0 | 還沒 |
| `mellum2-12b-a2.5b` | JetBrains/Mellum2-12B-A2.5B-Instruct（273d9c1c） | 64 | 8 | 28/28 | 0 | 還沒 |
| `gemma4-26b-a4b-kv4` | google/gemma-4-26B-A4B-it（4d7ae498） | 128 | 8 | 30/30 | 0 | 還沒 |
| `mixtral-8x7b` | mistralai/Mixtral-8x7B-Instruct-v0.1（eba92302） | 8 | 2 | 32/32 | 0 | 還沒 |
| `nemotron-3-nano-kv4` | nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16（bf77c317） | 128 | 6 | 23/52 | 0、2 | ✓ |
| `qwen3.6-35b-a3b-kv4` | Qwen/Qwen3.6-35B-A3B（995ad96e） | 256 | 8 | 40/40 | 0 | ✓ |
| `kimi-linear-48b-a3b` | moonshotai/Kimi-Linear-48B-A3B-Instruct（e1df551a） | 256 | 8 | 26/27 | 0 | ✓ |

表上的模型都用 HF 上的 header（tensor 名字和 shape）跑過工具的檢查。

- **OLMoE**：只能單卡，不適用。

hybrid 模型（MoE 層之間夾著 Mamba / linear attention）：

- `num_layers` 和 meta 的列一律是 **HF 的 decoder 層**（Nemotron 52、Qwen3.6 40、Kimi-Linear 27）。不是 MoE 的層（dense、Mamba、attention）填 R = 0、identity。我們的硬體把 Qwen3.6、Kimi 的一層拆成兩個 block，由我們這邊換算。
- Qwen3.6 的 MTP 層（`mtp.layers.0`）不給格子，工具不動它。
- U：Nemotron 的 shared expert 是兩個 routed expert 大，可以 U = 2；Qwen3.6（shared 帶 sigmoid gate）、Kimi（shared 只有一個 expert 大）維持 TP，U = 0。
- 2026-10-20 板測：隨機 meta，Nemotron（U = 0、2）、Qwen3.6、Kimi 兩卡 selftest 都 bit-exact、聊天通順；tok/s 還沒量。

其他限制：

- **HBM 由最大的 R 決定。** 每張卡每層的 expert slot 數是 `max R + (E − max R)/2 + U`，**每一層都按全部層裡最大的 R 配置**：只要有一層 R = 64，其他層也得留一樣多的空間。max R 每多 2，每張卡大約多放：Qwen3-30B、V2-Lite、Moonlight 120 MB，GLM 230 MB，Mixtral 3 GB。多出來的從 K/V pool 扣，context 變短（`check --hf` 印 GiB 數；Mixtral 8 個全複製要 735 MiB > 每個 channel 512 MiB，放不下）。
- **U = 2** 只有 shared expert 剛好等於兩個 routed expert 大小時能用（V2-Lite、Moonlight、DeepSeek-MoE-16B、Nemotron-3-Nano）。
- **GLM 的第 47 層**是 MTP 層，不在 `num_hidden_layers` 裡。工具不動它，meta 也不用給它格子。
- **`balance` 的數字不等於板子上的 tok/s。** 實測 trace 成本降 16–18 %，tok/s 只多 3–8 %，因為每層 MoE 還有一段固定成本。最後以板子上的 tok/s 為準。
- **trace 的精度。** 我們的 trace 用 W4 權重（板子上的量化）；用 bf16 的 HF 模型取也可以，路由只有小差異，不影響這個約定。
