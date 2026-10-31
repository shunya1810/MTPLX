# M1 Mac 向け MTPLX

[![Release](https://img.shields.io/github/v/release/shunya1810/MTPLX)](https://github.com/shunya1810/MTPLX/releases/latest)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

[English](README.md) | 日本語

[MTPLX](https://github.com/youssofal/MTPLX) を M1 系の Mac で速く動かすための非公式のフォークです。MTPLX 2.12.2 に、M1 系の GPU でだけ有効になる変更を加えています。対象は Qwen3.5・Qwen3.6・Qwen3.8 系のモデルで、長いプロンプトとコーディングエージェントでの利用を主な用途にしています。

本家と同じく、モデル自身の MTP（multi-token prediction）ヘッドで投機的デコードを行うので、出力の分布は本家と変わりません。OpenAI 互換と Anthropic 互換の API で動きます。

- **M3 以降の Mac では、[本家の MTPLX](https://github.com/youssofal/MTPLX) を使ってください。** 
- **本家の MTPLX アプリが必要です。** このフォークは、アプリの Python の実行環境と、アプリでダウンロードしたモデルを使います。
- MTPLX のプロジェクトとは関係のない、非公式のフォークです。変更は AI（Claude）の支援を受けて作り、MacBook Pro M1 Max（64 GB）1 台で計測しました。本家の README は [README.upstream.md](README.upstream.md) にあります。

## 主な計測値

このフォークと本家 MTPLX 2.12.2（このフォークの元の版）を、同じ Mac・同じ条件で測りました。太字がこのフォークです。

### prefill（tok/s）

キャッシュのない状態で、約 8K・32K・64K・128K トークンのプロンプトを読み込む速さです。

| モデル | | 8K | 32K | 64K | 128K |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | このフォーク | **1,065** | **1,028** | **771** | **499** |
|  | 本家 2.12.2 | 957 | 795 | 607 | 318 |
| Ornith-1.5-35B-A3B | このフォーク | **1,069** | **1,034** | **766** | **499** |
|  | 本家 2.12.2 | 954 | 795 | 606 | 317 |
| Qwen3.8-27B | このフォーク | **233** | **211** | **177** | **139** |
|  | 本家 2.12.2 | 150 | 139 | 121 | 87 |
| Huihui-Qwen3.8-27B-abliterated | このフォーク | **234** | **220** | **181** | **141** |
|  | 本家 2.12.2 | 150 | 139 | 123 | 87 |
| Swift-1.5-Qwen3.8-27b | このフォーク | **234** | **220** | **181** | **132** |
|  | 本家 2.12.2 | 150 | 139 | 123 | 86 |

### decode（tok/s）

「短い質問」は、思考（thinking）ありの短い質問 3 本に、1,500 トークンずつ答えさせたときの速さです。8K〜128K は、その長さのプロンプトを読んだ後に、512 トークン答えさせたときの速さです。

| モデル | | 短い質問 | 8K | 32K | 64K | 128K |
|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | このフォーク | **107.6** | **101.8** | **86.5** | **76.5** | **60.4** |
|  | 本家 2.12.2 | 58.0 | 53.4 | 46.8 | 39.1 | 24.9 |
| Ornith-1.5-35B-A3B | このフォーク | **110.5** | **106.8** | **94.6** | **81.9** | **63.8** |
|  | 本家 2.12.2 | 63.6 | 61.0 | 86.8 | 29.3 | 16.2 |
| Qwen3.8-27B | このフォーク | **32.5** | **30.3** | **30.2** | **27.4** | **22.8** |
|  | 本家 2.12.2 | 23.2 | 22.7 | 18.6 | 9.1 | 6.1 |
| Huihui-Qwen3.8-27B-abliterated | このフォーク | **32.0** | **33.9** | **30.1** | **27.8** | **23.3** |
|  | 本家 2.12.2 | 24.2 | 22.9 | 19.9 | 10.3 | 6.0 |
| Swift-1.5-Qwen3.8-27b | このフォーク | **31.5** | **32.3** | **29.0** | **27.0** | **16.9** |
|  | 本家 2.12.2 | 24.3 | 23.5 | 19.3 | 10.7 | 5.9 |

### ピークのメモリ

それぞれの長さのプロンプトの読み込みと応答の間に、MLX が割り当てたメモリの最大値です。

| モデル | | 8K | 32K | 64K | 128K |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | このフォーク | **32.6 GB** | **32.6 GB** | **32.6 GB** | **32.6 GB** |
|  | 本家 2.12.2 | 22.6 GB | 24.7 GB | 27.3 GB | 32.2 GB |
| Ornith-1.5-35B-A3B | このフォーク | **32.6 GB** | **32.6 GB** | **32.6 GB** | **32.6 GB** |
|  | 本家 2.12.2 | 23.9 GB | 26.0 GB | 28.6 GB | 33.5 GB |
| Qwen3.8-27B | このフォーク | **25.1 GB** | **27.4 GB** | **30.6 GB** | **35.5 GB** |
|  | 本家 2.12.2 | 25.5 GB | 29.7 GB | 36.1 GB | 45.0 GB |
| Huihui-Qwen3.8-27B-abliterated | このフォーク | **20.2 GB** | **22.9 GB** | **26.9 GB** | **35.0 GB** |
|  | 本家 2.12.2 | 21.0 GB | 25.2 GB | 31.6 GB | 41.0 GB |
| Swift-1.5-Qwen3.8-27b | このフォーク | **20.2 GB** | **22.9 GB** | **26.9 GB** | **35.0 GB** |
|  | 本家 2.12.2 | 21.0 GB | 25.2 GB | 31.6 GB | 41.3 GB |

### 246K のプロンプト（繰り返しのない文章）

[apple-silicon-llm-bench](https://github.com/shunya1810/apple-silicon-llm-bench) のベンチ 2 の計測です。245,637 トークンのテレメトリの記録を読ませて、thinking ありで 3 ターン答えさせました（temperature 1.0、1 ターン最大 4,096 トークン）。

| モデル | | prefill | decode（1 / 2 / 3 ターン目） | wired メモリのピーク |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | このフォーク | **253 tok/s** | **48.5 / 46.2 / 45.9** | **34.6 GB** |
|  | 本家 2.12.2 | 147 tok/s | 11.5 / 13.2 / 12.6 | 49.6 GB |
| Qwen3.8-27B | このフォーク | **80 tok/s** | **20.9 / 21.0 / 18.2** | **44.4 GB** |
|  | 本家 2.12.2 | 1 ターン目の途中でメモリ不足で停止 | — | 58.4 GB |

### 計測の条件

- Qwen3.6-35B-A3B と Qwen3.8-27B は v2.12.2-m1.10 で測り直しました（2026-10-20）。ほかの行は v2.12.2-m1.8 の計測です。
- v2.12.2-m1.11 で、長い履歴への prefill の attention を直しました。途中の和を fp16 で持っていたため、27B では 19.2 万〜25.6 万トークンの cold のプロンプトの多くが NaN で止まり、3.2 万トークン以上では両方のモデルで出力が少し変わっていました。prefill の速さは変わりません。bench 3 では 3.2 万〜12.8 万トークンの decode の差が数 % だったので、上の表は測り直していません。修正後の 19.2 万〜25.6 万トークンの値は [apple-silicon-llm-bench](https://github.com/shunya1810/apple-silicon-llm-bench) の bench 3 にあります（27B は prefill 78〜90 tok/s、decode 17〜21 tok/s）。
- v2.12.2-m1.12 では、これらのモデルの速さは変わりません。Qwen3.8-27B と Qwen3.6-35B-A3B で m1.11 と比べ、1.6K トークンでの prefill と verify 1 回の時間の差は 2% 未満だったので、表は測り直していません。
- 8K〜128K のプロンプトは、1 万 1 千トークンの HTML ファイルを繰り返したものです。約 256K（22 回の繰り返し）まで読ませると、3 回目の繰り返しから Qwen3.8-27B 自身の次のトークンの予測が当たらなくなり（本家でも同じ）、モデルの hidden state を読む draft ヘッドも当たらなくなります。そのため 256K 前後は、上の繰り返しのない文章の計測で示しています。
- MacBook Pro M1 Max 64 GB、macOS 26。プロファイルは `turbo`、文脈の上限は 256K、KV キャッシュは 8bit、temperature は 0.6（seed 固定）です。
- MTP の深さは、このフォークで一番速かった値（「[対象のモデル](#対象のモデル)」の表に記載）を、両方のエンジンで使いました。
- 長さごとにサーバーを起動し直して、cold の状態から測りました。同じ項目を複数回測ったものは、速いほうの値です（GPU の温度などで、遅い回が出ることがありました）。
- 本家の Ornith-1.5-35B-A3B の「32K」の decode（86.8 tok/s）は、答えにファイルの引用が多く、本家の context-copy がそれを写したため、ほかの長さより高くなっています。
- このフォークの Qwen3.6-35B-A3B 系のピーク（32.6 GB）は、モデルを読み込むときの値です。読み込みの際に MoE の重みを並べ替えるので、一時的にメモリが増えます。応答中に使うメモリは 21〜24 GB です（128K で 24.4 GB、本家は 26.2 GB）。
- Swift-1.5-Qwen3.8-27b は、128K の decode が、同じ形の Huihui-Qwen3.8-27B-abliterated より遅くなりました（16.9 と 23.3 tok/s）。繰り返し測っても同じで、原因は調べている途中です。

## 対象のモデル

### 計測したモデル

| モデル | 種類 | 量子化 | 重み | MTP の深さ |
|---|---|---|---|---|
| [Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16](https://huggingface.co/Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16) | MoE（全体 35B、1 トークンあたり 3B） | 4bit（64 値ごと）、一部 8bit | 21 GB | 1 |
| Ornith-1.5-35B-A3B（[ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) を MTPLX の forge で変換） | MoE（Qwen3.6-35B-A3B の派生） | 4bit（64 値ごと）、一部 8bit | 21 GB | 1 |
| [Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16](https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16) | dense（27B） | 4bit（32 値ごと）、一部 8bit | 20 GB | 3 |
| Huihui-Qwen3.8-27B-abliterated（[huihui-ai/Huihui-Qwen3.8-27B-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated) を MTPLX の forge で変換） | dense（27B） | 4bit（64 値ごと） | 16 GB | 3 |
| Swift-1.5-Qwen3.8-27b（[ukisai/Swift-1.5-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b) を MTPLX の forge で変換） | dense（27B、Qwen3.8-27B の派生） | 4bit（64 値ごと） | 16 GB | 3 |

- 4bit の重みは、32 個または 64 個の値ごとに 1 組の倍率を持ちます。32 個ごとのほうが精度は高くなりますが、重みが大きくなり、decode は遅くなります。
- [Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality-FP16](https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality-FP16)（8bit）も動作を確かめています。重みが約 9 GB 大きく、decode は Speed より 1〜2 割遅くなります。

### そのほかのモデル

- Qwen3.5・3.6・3.8 系のモデルは、M1 ではプロファイルを指定しないと `turbo` で起動します。系統は `config.json` の `model_type` で判定するので、派生モデルや自分で forge したモデルも対象です。
- Qwen3.6-35B-A3B と同じ構成のモデル（expert 256 個、1 トークンあたり 8 個、隠れ層の幅 2048、40 層、本体が 4bit・64 値ごと）は、A3B 用の最適化（GPU 上で decode の round を続けて投入する経路、ファイルの書き直しを速くする context-copy など）も有効になります。
- `turbo` の verify 用の kernel が対応する量子化は、4bit・6bit・8bit の affine です。それ以外の量子化（3bit、mxfp4、3 値など）では、標準の行列積で計算します。
- ほかの系統のモデルは、本家と同じく `sustained` で起動します。`--profile turbo`、またはアプリの設定のプロファイルで `turbo` を選ぶと、M1 向けの変更が効く場合があります。この組み合わせは計測していません。
- モデルごとに一番速い MTP の深さは、「[MTP の深さを測り直す](#mtp-の深さを測り直す)」で確かめられます。

## 必要なもの

- M1 系の Mac（計測は M1 Max 64 GB、macOS 26）
- Metal のツールチェーンが入った Xcode（`xcrun metal` が動くこと）
- 一度起動した [MTPLX アプリ](https://github.com/youssofal/MTPLX)。モデルは、アプリのモデル画面からダウンロードします。

## インストール

```bash
git clone --branch v2.12.2-m1.12 https://github.com/shunya1810/MTPLX.git MTPLX-m1
cd MTPLX-m1
scripts/build-m1-native-extensions.sh   # 1 分ほど。「mfa_attn: ok」「ane_ffn: ok」「qsa_kernels: ok」が出れば完了
```

ネイティブ拡張（Neural Engine の prefill と、M1 向けの attention）を作らなくてもサーバーは動きますが、prefill は遅くなります。

### アプリから使う

アプリのエンジンを、このフォークに切り替えます。

```bash
scripts/m1-app-use-checkout.sh install
```

アプリを終了して開き直すと、アプリはこのフォークのエンジンで動きます。Dock や Finder など、どこから開いても同じです。この設定は、ログインし直しても残ります。元に戻すときは `scripts/m1-app-use-checkout.sh uninstall` を実行して、アプリを開き直します。

### サーバーとして使う

MTPLX アプリを終了してから（エンジンを 2 つ同時に動かすとメモリに収まりません）、サーバーを起動します。`--model` には、アプリでダウンロードしたモデルのフォルダを指定します。

```bash
PYTHONPATH="$PWD" "$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python" -m mtplx.cli serve \
  --model "$HOME/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16" \
  --context-window 262144 --paged-kv-quantization q8
```

サーバーは `http://127.0.0.1:8000` で待ち受けます。次のコマンドで応答を確かめます。

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model": "mtplx", "messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 64}'
```

使い方の詳細は [docs/m1max-longctx/USAGE.ja.md](docs/m1max-longctx/USAGE.ja.md) にあります。

### 新しい版に更新する

```bash
git fetch --tags
git checkout <新しいタグ>               # 例: v2.12.2-m1.12
scripts/build-m1-native-extensions.sh   # ネイティブ拡張が変わった版だけ必要です（リリースノートに書きます）
```

アプリから使っている場合は、更新した後にアプリを開き直します。

## コーディングエージェントで使う

OpenAI または Anthropic の API を使うクライアントなら、どれでも使えます。

- OpenCode: `~/.config/opencode/opencode.json` に provider を追加します。

  ```json
  {
    "provider": {
      "mtplx": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "MTPLX (M1)",
        "options": { "baseURL": "http://127.0.0.1:8000/v1", "apiKey": "local", "timeout": false },
        "models": { "mtplx": { "name": "MTPLX", "reasoning": true, "tool_call": true,
                               "limit": { "context": 262144, "output": 32768 } } }
      }
    }
  }
  ```

- Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8000 claude`

Qwen3.6 系には reasoning effort の段階がなく、思考はオンかオフだけです。Qwen3.8 系は `low` から `xhigh` まで選べます。

## MTP の深さを測り直す

MTP の深さ（1 回の draft で先読みするトークンの数）は、モデルによって一番速い値が違います。次のスクリプトは、深さ 1・2・3 で順にサーバーを起動して decode の速さを測り、一番速い深さを示します。1 モデルで 15 分ほどかかります。

```bash
scripts/m1-depth-sweep.py --model "$HOME/.mtplx/models/<モデルのフォルダ>"
```

`--apply-app` を付けると、結果をアプリのモデルごとの設定に書き込みます。アプリは終了時に設定を書き戻すので、アプリを終了した状態で実行してください。

## このフォークの変更点

- prefill: 計算の一部を Neural Engine に任せて、GPU と並べて処理します。キャッシュした履歴への attention は、M1 向けに調整した flash-attention で計算します。長い履歴に短い追加（ツールの結果など）を読み込むときは、8bit の KV を直接読みます。
- decode: 1 round ごとの CPU と GPU のやり取りを減らしています。Qwen3.6-35B-A3B 系では、前の round の結果を CPU が読む前に、次の round を GPU に投入します。ファイルを書き直すときは、プロンプトと一致する部分をそのまま写します（context-copy、完全一致のときだけ）。
- メモリ: KV キャッシュを 8bit のページで持ち、長いプロンプトはそのページに直接 prefill します（dense のモデルで 19.2 万トークンを超えるプロンプトは、fp16 で読んでから詰め直します）。メモリの見積もりは、このフォークで測った値に合わせています。64 GB の Mac で、256K トークンの会話が既定の設定のまま動きます。
- 既定の設定: Qwen3.5・3.6・3.8 系は `turbo` で起動します。アプリから Qwen3.6-35B-A3B 系を起動したときも、A3B 用の経路を使います。
- Qwen4 への下準備（実際の Qwen4 のモデルではまだ確かめていません）: 本家の `qwen4_exp` の経路は、Qwen3.8-Flash-Next（hidden 2560、MoE、query 24 / KV 2 ヘッド）用に書かれています。Qwen4 のモデルが同じアーキテクチャを別の大きさで使った場合に備えて、dense の層、ほかの幅、ほかのヘッド構成も扱えるようにしました。M1 では、この系統の bf16 のパックを fp16 に変換して動かします。
- エージェントの会話（M1 以外の Mac でも有効）: 履歴の書き換えをまたいで同じ会話を見つけ直す、thinking の有無を切り替えてもキャッシュを使い続ける、ツールの引数の空白を保つ、macOS が長い GPU の処理を止めたときに要求をやり直す、保存した状態の一番深いところから再開する、などの修正です。

変更の一つ一つと計測の結果は、[CHANGES.ja.md](docs/m1max-longctx/CHANGES.ja.md) と[リリースノート](https://github.com/shunya1810/MTPLX/releases)にあります。

## 切り替えスイッチ

M1 向けの既定は、一つずつ環境変数で切れます。`MTPLX_M1_LONG_CONTEXT=0` で、すべての M1 向けの既定が切れます。

| 環境変数 | M1 での既定 | 切れるもの |
|---|---|---|
| `MTPLX_ANE_PREFILL=0` / `MTPLX_ANE_PREFILL_MOE=0` | オン | Neural Engine の prefill（dense / MoE） |
| `MTPLX_MFA_PREFILL=0` | オン | flash-attention の prefill |
| `MTPLX_QWEN4_SAMPLED_DRAFT_CHAIN=0` | オン | draft を GPU 上で続けて作る経路 |
| `MTPLX_M1_PAGED_PREFILL=0` | 32K 以上でオン | 8bit のページへの直接の prefill |
| `MTPLX_M1_ENGINE_RAM_FRACTION=0.75` | メモリの 13/16 | エンジンに割り当てるメモリの上限（既定より小さくするとき） |
| `MTPLX_M1_TURBO_DEFAULT=0` | Qwen3.5・3.6・3.8 系は `turbo` | `turbo` の既定 |
| `MTPLX_M1_FP16_CAST=0` | `qwen4_exp` 系でオン | bf16 のパックを fp16 に変換して動かす処理（`=1` でほかの系統でもオン。fp16 のパックでは何もしません） |
| `MTPLX_M1_MOE_LANE=0` | MoE のモデルでオン | MoE 向けの既定すべて（下の 2 行も含む） |
| `MTPLX_M1_RESIDENT_DECODE=0` | Qwen3.6-35B-A3B 系でオン | GPU 上で round を続けて投入する decode |
| `MTPLX_CONTEXT_COPY=0` | Qwen3.6-35B-A3B 系でオン | context-copy |

## 既知の制限

- 計測は M1 Max 64 GB の 1 台だけです。M1 Pro、M1 Ultra、ほかのメモリの容量でも同じコードが動きますが、確かめていません。
- Neural Engine の経路は、Apple の非公開の `AppleNeuralEngine.framework` を使っています。macOS の更新で動かなくなる可能性があり、その場合は GPU の prefill に戻ります。
- Neural Engine の prefill は int8 で計算するので、GPU の prefill とビット単位では一致しません（Qwen3.6-35B-A3B で、24,576 か所のうち 98.25% で次のトークンが一致）。`MTPLX_ANE_PREFILL=0` / `MTPLX_ANE_PREFILL_MOE=0` で GPU だけの prefill にできます。
- 思考ありのモデルは、難しいエージェントの作業で非常に長く考えることがあります。思考の長さは既定では制限しません。ツールを渡す要求では、`MTPLX_THINKING_BUDGET=<トークン数>` で上限を付けられます。
- 本家のテストの一部は本家の既定を前提にしていて、M1 向けの既定がオンだと失敗します。多くは `MTPLX_M1_LONG_CONTEXT=0` で通ります。

## ライセンスと謝辞

本家と同じ Apache-2.0 です（[LICENSE](LICENSE)、[NOTICE](NOTICE)）。Neural Engine のクライアントと FFN のプログラムの構成は [Splash](https://github.com/incoai/splash) から取り入れています（`native_extensions/ane_ffn/NOTICE`）。flash-attention の kernel は [metal-flash-attention](https://github.com/philipturner/metal-flash-attention) にならっています（`native_extensions/mfa_attn/MFA_LICENSE.txt`）。
