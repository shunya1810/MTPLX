# Apple M1 Max での MTPLX 長 context ベンチマーク（2K〜256K）

[English](README.md) | 日本語

`m1max-longctx` ブランチと upstream の MTPLX を、MacBook Pro M1 Max（64 GB）で比べた計測です。
モデルは **Qwen3.8-27B-MTPLX-Optimized-Speed（FP16）**、MTP depth 3 です。
計測は、MTPLX 本体の OpenAI 互換 HTTP サーバー（`mtplx serve`）にリクエストを送る形で行いました。
どちらの commit も既定の設定のまま動かしています。
ブランチの変更は M1 系の GPU（`applegpu_g13*`）でだけ自動で有効になります。
2026年9月25日に、31 セルを 8.6 時間かけて計測しました。

**主な結果**（baseline → optimized）：

- **128K の decode**：fp16 KV で 10.4 → **16.7 tok/s**（+61%）、q8 KV で 4.6 → **18.9 tok/s**（4.1倍）になりました。
  32K 以上では、q8 KV の decode が 1.9〜4.1倍になっています。
- **128K のコールド TTFT**：MMA prefill attention により 30.6 → **24.6 分**（-20%）に縮みました。
  32K までの prefill はほぼ変わりません（8K の fp16 KV の +7% を除き ±1% 以内）。
- **128K のピークメモリ**：fp16 KV で 46.9 → **40.7 GB**、q8 KV で **34.1 GB** です。
- **256K が既定の設定で動くようになりました**：q8 KV で、75 分の prefill のあと **14.7 tok/s** で decode し、ピークは **41.7 GB** です。
  baseline は既定の設定のままだと、256K の prefill 中にメモリ不足で止まります。
- **短いプロンプト**：16K までの fp16 KV では、decode は baseline の -3%〜+5% の範囲に収まっています。

| | commit | |
|---|---|---|
| **baseline** | [`1de2b1c`](https://github.com/youssofal/MTPLX/commit/1de2b1c049136ed117af0c6712baaadd81820b51) | upstream の `main`（2.12.0 と README の更新）、既定設定 |
| **optimized** | [`66910a1`](https://github.com/shunya1810/MTPLX/commit/66910a1efd3c6ccc2e37414a8f8b3db608f7a04f) | `shunya1810/MTPLX` の `m1max-longctx` ブランチ、既定設定 |

グラフの表記は英語版と共通です。

## decode の速度

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/decode-vs-context-dark.svg">
  <img alt="プロンプト長ごとの decode tok/s。baseline と optimized、fp16 KV と q8 KV" src="charts/decode-vs-context-light.svg">
</picture>

## 最初の token までの時間と prefill

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/ttft-vs-context-dark.svg">
  <img alt="プロンプト長ごとのコールド TTFT（両対数）" src="charts/ttft-vs-context-light.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/prefill-vs-context-dark.svg">
  <img alt="プロンプト長ごとの prefill tok/s" src="charts/prefill-vs-context-light.svg">
</picture>

## リクエスト全体の所要時間

コールドのリクエスト1回分を、prefill と生成（最大 256 token）に分けて示します。
128K 以上では prefill が全体の 97〜99.6% を占めるため、コールドのリクエスト1回だけを見ると decode の改善はほとんど表に出ません。
改善が効くのは会話の2ターン目以降です。
プレフィックスキャッシュで prefill が省かれ、生成する token がすべて decode の速度で出てきます。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/e2e-off-dark.svg">
  <img alt="リクエスト全体の所要時間を prefill と生成に分けたもの（fp16 KV）" src="charts/e2e-off-light.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/e2e-q8-dark.svg">
  <img alt="リクエスト全体の所要時間を prefill と生成に分けたもの（q8 KV）" src="charts/e2e-q8-light.svg">
</picture>

## ピークメモリ

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/peak-memory-vs-context-dark.svg">
  <img alt="プロンプト長ごとの MLX ピークメモリ。物理 RAM と GPU working set の上限線つき" src="charts/peak-memory-vs-context-light.svg">
</picture>

## 全セルの数値

各欄は「baseline → **optimized**（変化率）」の形で示します。
CSV は [`results/summary.csv`](results/summary.csv)、リクエストごとのサーバー指標は [`results/raw-cells.jsonl`](results/raw-cells.jsonl) にあります。

### fp16 KV

| context | プロンプト token 数 | prefill tok/s | TTFT（コールド） | decode tok/s | E2E（コールド） | ピークメモリ | 出力一致 |
|---|---:|---:|---:|---:|---:|---:|:-:|
| 2K | 2,048 | 149.3 → **149.7** (+0%) | 13.9 s → **13.8 s** (-0%) | 26.26 → **26.12** (-1%) | 22.5 s → **22.5 s** (+0%) | 23.5 → **23.5 GB** | ✅ |
| 4K | 4,097 | 148.5 → **149.4** (+1%) | 27.7 s → **27.6 s** (-1%) | 26.66 → **25.78** (-3%) | 36.4 s → **36.5 s** (+0%) | 24.0 → **24.0 GB** | ✅ |
| 8K | 8,202 | 137.3 → **146.4** (+7%) | 59.9 s → **56.2 s** (-6%) | 25.26 → **26.30** (+4%) | 68.1 s → **64.1 s** (-6%) | 25.1 → **25.1 GB** | ✅ |
| 16K | 16,378 | 139.4 → **141.1** (+1%) | 118 s → **116 s** (-1%) | 22.49 → **23.71** (+5%) | 128 s → **126 s** (-1%) | 27.0 → **27.0 GB** | ✅ |
| 32K | 32,778 | 131.7 → **131.5** (-0%) | 249 s → **249 s** (+0%) | 19.77 → **23.16** (+17%) | 262 s → **260 s** (-1%) | 29.7 → **29.7 GB** | ✅ |
| 64K | 65,530 | 107.2 → **112.1** (+5%) | 10.2 min → **585 s** (-4%) | 16.42 → **21.20** (+29%) | 10.5 min → **598 s** (-5%) | 35.0 → **32.7 GB** | ✅ |
| 128K | 131,082 | 71.3 → **88.8** (+24%) | 30.6 min → **24.6 min** (-20%) | 10.36 → **16.72** (+61%) | 31.0 min → **24.9 min** (-20%) | 46.9 → **40.7 GB** | ❌ |
| 256K | 259,003 | <sub>ref.</sub> 51.8 → **57.1** | <sub>ref.</sub> 83.3 min → **75.6 min** | <sub>ref.</sub> 3.54 → **13.86** | **76.0 min** | <sub>ref.</sub> 57.3 → **56.7 GB** | — |

### q8 KV

| context | プロンプト token 数 | prefill tok/s | TTFT（コールド） | decode tok/s | E2E（コールド） | ピークメモリ | 出力一致 |
|---|---:|---:|---:|---:|---:|---:|:-:|
| 2K | 2,048 | 149.4 → **148.3** (-1%) | 13.9 s → **14.0 s** (+1%) | 23.84 → **26.90** (+13%) | 23.9 s → **22.9 s** (-4%) | 23.5 → **23.5 GB** | ✅ |
| 4K | 4,097 | 146.8 → **147.0** (+0%) | 28.1 s → **28.0 s** (-0%) | 22.13 → **26.25** (+19%) | 36.9 s → **36.9 s** (+0%) | 24.0 → **24.0 GB** | ❌ |
| 8K | 8,202 | 144.8 → **144.9** (+0%) | 56.8 s → **56.8 s** (-0%) | 23.26 → **26.92** (+16%) | 65.8 s → **64.6 s** (-2%) | 25.1 → **25.1 GB** | ✅ |
| 16K | 16,378 | 140.8 → **140.8** (+0%) | 117 s → **117 s** (-0%) | 19.87 → **23.17** (+17%) | 128 s → **127 s** (-1%) | 27.0 → **27.0 GB** | ❌ |
| 32K | 32,778 | 131.3 → **131.3** (+0%) | 250 s → **250 s** (-0%) | 12.13 → **22.83** (+88%) | 270 s → **261 s** (-4%) | 29.7 → **29.7 GB** | ✅ |
| 64K | 65,530 | 106.7 → **112.0** (+5%) | 10.2 min → **585 s** (-5%) | 8.47 → **21.56** (+155%) | 10.7 min → **598 s** (-7%) | 35.0 → **32.7 GB** | ✅ |
| 128K | 131,082 | 71.4 → **88.6** (+24%) | 30.6 min → **24.7 min** (-19%) | 4.57 → **18.92** (+314%) | 31.5 min → **24.9 min** (-21%) | 46.9 → **34.1 GB** | ❌ |
| 256K | 259,003 | <sub>ref.</sub> 48.8 → **57.2** | <sub>ref.</sub> 88.4 min → **75.4 min** | <sub>ref.</sub> 4.86 → **14.66** | **75.8 min** | <sub>ref.</sub> 52.3 → **41.7 GB** | — |

### q4 KV（optimized のみ、256K）

| context | プロンプト token 数 | prefill tok/s | TTFT（コールド） | decode tok/s | E2E（コールド） | ピークメモリ |
|---|---:|---:|---:|---:|---:|---:|
| 256K | 259,003 | 57.2 | 75.5 min | 12.76 | 75.8 min | 41.7 GB |

<sub>ref.：256K の baseline は今回測り直していません。
upstream の既定の prefill chunk（2048）では、256K の prefill が約48分でメモリ不足により止まります。
参考値は、以前のセッションで chunk を手動で 512 に下げて完走させたときのものです。
commit も計測スクリプトも今回と異なるため、同じ条件での比較ではなく、桁の目安として載せています。</sub>

## 出力の一致

temperature 0 での生成テキストは、プロンプトと KV モードの組 16 のうち 12 で両 commit がバイト単位で一致しました。
残りの4組は、途中まで同じで、ある位置から分かれます。

| 組 | 最初に異なる文字の位置 |
|---|---:|
| 4K q8 | 5 |
| 16K q8 | 969（全体は約 1,000 文字） |
| 128K fp16 | 528 |
| 128K q8 | 127 |

M1 向けのカーネルは attention の足し合わせの順序が異なります。
そのため、上位2候補がほぼ同点の位置では、greedy decode の選ぶ token が入れ替わることがあります。
同じ入れ替わりは、同じ commit の中でも起きています。
128K q8 では、baseline 自身のコールドのリクエスト（全体を prefill）とウォームのリクエスト（プレフィックスキャッシュを使用）で生成テキストが異なりました。
しかも、baseline のコールドの出力は optimized のウォームの出力とバイト単位で一致しています。
この4組では出力の長さが異なるので、E2E の比較は生成 token 数が少し違う条件どうしの比較になります。
decode tok/s は token あたりの値なので、この影響を受けません。

## 計測方法

- **セルごとにサーバーを起動し直す**：一つのセルは「commit、プロンプト長、KV モード」の組です。
  セルごとに `mtplx serve` を新しく起動するので、セッションキャッシュや Metal アロケータの状態はセル間で引き継がれません。
  SSD セッションキャッシュは off にし、セルの間に 60 秒の待ち時間を置きました。
- **リクエスト**：同じストリーミング `/v1/chat/completions` リクエストを3回（256K は1回）送ります。
  temperature 0（`top_k` 1、seed 固定）、thinking は無効、`max_tokens` は 256 です。
  1回目（**コールド**）はプロンプト全体を prefill し、TTFT、prefill、E2E、ピークメモリはこの回から取ります。
  2回目以降（**ウォーム**）は RAM のプレフィックスキャッシュに当たって prefill を省きます。
  KV の長さと生成テキストは1回目と同じなので、decode の標本を増やす目的にだけ使います。
- **プロンプト**：決定的に生成した合成テレメトリの行を目標 token 数に切りそろえ、固定の質問を付けたものです（`scripts/bench_longctx.py` の `build_telemetry_prompt`）。
  「プロンプト token 数」はチャットテンプレート適用後の、サーバーが数えた値です。
  256K は 262,144 token の context window に 256 token の回答が収まるよう、259,000 token にしました。
- **指標**：
  - **TTFT（コールド）**：リクエスト送信から最初のストリーミング token を受け取るまでの、クライアント側の経過時間です。
  - **prefill tok/s**：プロンプト token 数を、サーバーの `prompt_eval_time_s` で割った値です。
  - **decode tok/s**：サーバーの `decode_tok_s` の、全リクエストでの中央値です。
    MTP で受理された draft を含む確定 token 数を decode の経過時間で割っています。
  - **E2E（コールド）**：1回目のリクエストの、送信から最後の token までのクライアント側の経過時間です。
  - **ピークメモリ**：1回目のリクエスト後にサーバーが報告する `peak_memory_bytes`（MLX のピーク確保量）です。
    グラフの「physical RAM」は物理メモリの 68.7 GB（64 GiB）です。
    「GPU working set」は Metal の `recommendedMaxWorkingSetSize`（55.7 GB）で、物理メモリとは別の上限です。
- **出力の一致**：ストリーミングで受け取ったテキストの SHA-256 を、セル内のリクエスト間と、同じプロンプトと KV モードの両 commit 間で比べました（表の「出力一致」）。

環境の詳細は [`configs/environment.json`](configs/environment.json) と [`configs/model.lock.json`](configs/model.lock.json) にあります。

### この計測の限界

- 1台のマシンで1回通しただけの結果です。
  decode は1セルあたり3標本（256K は1標本）、prefill と TTFT は1セルあたりコールド1標本です。
- 2K〜4K のコールド TTFT には、最初のリクエストでだけかかる処理（kernel のコンパイルなど）が含まれます。
  サーバーは `--warmup-tokens 0` で起動しています。
- 単一ストリームの応答時間だけを測りました。同時実行やバッチ処理は対象外です。
- 128K q8 では、両 commit とも3回目のリクエストが RAM のプレフィックスキャッシュに当たらず、prefill をやり直しました。
  decode の標本としてはそのまま使えます。
- 256K の fp16 KV では、ピーク（56.7 GB）が Metal の推奨 working set（55.7 GB）をわずかに超えました。
  リクエストは最後まで完了しています。
- 計測前から約 1.9 GB の swap が使われていましたが、計測中に増えてはいません。

## ブランチで変更した内容

`git log --oneline 1de2b1c..66910a1`：

- `66910a1` Keep the dense MMA verify route off below a 4096-key capacity
- `f359091` MMA kernel: split the QK reduction into two accumulators for q8/q4 KV
- `ad87b04` Tests: MMA verify/prefill kernel numerics across layouts and bails
- `9e77f1c` Flash-style MMA prefill attention for long prefixes on M1
- `9ce0015` Tests: pin GPU-family gates, cover M1 chunk cap and pages-layout demote
- `e5d1e77` Cap the prefill chunk at 512 above 163,840 prompt tokens on M1
- `ba22a23` Route dense verify and MTP draft attention through the MMA kernel on M1
- `54afe35` GraphBank: lift the compiled-verify context fence for paged adapters on M1
- `439a460` M1 long-context: MMA split-K attention kernel and paged/quantized adapter fixes
- `1e33cfb` Long-context diagnostics: KV cache receipts, route trace, tail-mask elision for quantized adapter

## 再現手順

```bash
git clone https://github.com/shunya1810/MTPLX && cd MTPLX
git remote add upstream https://github.com/youssofal/MTPLX && git fetch upstream
git checkout m1max-longctx
cd docs/benchmarks/m1max-longctx
MTPLX_REPO=$(git rev-parse --show-toplevel) \
MODEL_PATH=~/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed-FP16 \
MTPLX_PY="$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python" \
WORK_DIR=/tmp/m1max-longctx-bench \
./run_matrix.sh            # 試すだけなら --only 8k-off-baseline 8k-off-optimized を付ける
```

セルの一覧、実行順、タイムアウトは [`configs/plan.json`](configs/plan.json) にあります。
このマシンでは全セルの計測に 8.6 時間かかりました。
`scripts/summarize.py` は `results/raw-cells.jsonl` から `results/` と `charts/` を作り直します。
Python の標準ライブラリだけで動きます。
