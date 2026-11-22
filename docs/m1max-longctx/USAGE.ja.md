# このフォーク（m1max-longctx）でサーバーを動かす

[English](USAGE.md)

`m1max-longctx` ブランチは、M1 Max で長い context を扱うための変更を upstream の MTPLX に足したものです（変更点と計測は [ベンチマーク](../benchmarks/m1max-longctx/README.ja.md)）。
ここでは、このブランチのサーバーを手元で起動して使う方法を説明します。

## 使い方は2通り

- **MTPLX アプリで使う**: アプリの開発者向けの仕組み（ソースのチェックアウトをエンジンとして使う）で、アプリの画面からこのブランチを動かします（下の「アプリで動かす」）
- **サーバーだけを起動する**: コマンドでサーバーを起動し、サーバー自身のチャット画面か OpenAI 互換の API から使います（下の「サーバーだけを起動する」）

どちらも、アプリの実行環境（`~/Library/Application Support/MTPLX/runtime-venv`）の Python を使い、このリポジトリの `mtplx` を読ませます。
`scripts/install-m1.sh --app`（新しいリリースに追従するなら `--auto-update` も）を実行すると、アプリはどこから開いてもこのリポジトリで動きます。実行しなければ、アプリ同梱のエンジンで動きます。

## 用意するもの

- Apple Silicon の Mac（計測は M1 Max 64GB、macOS 26）
- MTPLX アプリを一度起動して、実行環境（mlx などの依存関係）とモデルを入れておく。以下の手順は、アプリの実行環境の Python を使います（計測もすべてこの方法です）
- モデル: `~/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed-FP16`（アプリのモデル画面から取得したもの）

依存関係を自分の venv に入れて動かす方法は検証していません。

```bash
git clone -b m1max-longctx https://github.com/shunya1810/MTPLX.git
cd MTPLX
```

以下のコマンドは、この `MTPLX` ディレクトリで実行します。

### ネイティブ拡張のビルド（チェックアウトごとに 1 回、1 分ほど）

M1 向けの機能のうち 3 つはネイティブ拡張です。metal-flash-attention による prefill の attention、Neural Engine の prefill、draft chain に必要な MTP キャッシュのその場書き込みです。

```bash
scripts/build-m1-native-extensions.sh
```

アプリの実行環境の Python から `.build-venv` を作り、`mlx`（実行環境と同じ版）、`nanobind` 2.15.0、`cmake`、`setuptools` を入れて、2 つの拡張をその場でビルドします。最後に `mfa_attn: ok` / `ane_ffn: ok` と表示されれば完了です。Metal のツールチェーン（`xcrun metal`）が入った Xcode が必要です。チェックアウトを更新したら、もう一度実行してください。
拡張がなくてもサーバーは動きます。その場合、これらの機能は GPU だけの処理に戻ります（prefill が遅くなり、draft は 1 段ずつになります）。

## アプリで動かす

いちばん簡単なのは `scripts/install-m1.sh --app`（または `scripts/m1-app-use-checkout.sh install`）です。下の 2 つの環境変数を、ログインのたびに設定します。この節の残りは、その中で行っていることの説明です。

アプリには、環境変数 `MTPLX_APP_ALLOW_SOURCE_WRAPPER=1` と `MTPLX_APP_SOURCE_WRAPPER_PATH=<リポジトリ>/bin/mtplx` を渡すと、アプリ専用の実行環境より優先してそのラッパーでエンジンを起動する仕組みがあります（`apps/MTPLXApp/Sources/MTPLXAppCore/Services/MTPLXCommandBuilder.swift`）。
`bin/mtplx` は、`~/Library/Application Support/MTPLX/runtime.env` に書いた `MTPLX_RUNTIME_VENV_PY` の Python で、このリポジトリの `mtplx` を動かします。

[`scripts/open-mtplx-app-with-checkout.command`](../../scripts/open-mtplx-app-with-checkout.command) がこれをまとめて行います（Finder からダブルクリックでも実行できます）。

1. `runtime.env` がなければ、アプリの実行環境の Python を指す1行を書く
2. アプリが動いていれば終了し、残っている MTPLX のサーバーを止める（アプリは前回のサーバーが残っていると、それを引き継ぐため）
3. 2つの環境変数を付けてアプリを開く

アプリが開いたら、アプリの画面でエンジンを開始してください。
手で行う場合は次のとおりです。

```bash
printf 'MTPLX_RUNTIME_VENV_PY="%s"\n' "$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python" \
  > "$HOME/Library/Application Support/MTPLX/runtime.env"
open -a /Applications/MTPLX.app \
  --env MTPLX_APP_ALLOW_SOURCE_WRAPPER=1 \
  --env MTPLX_APP_SOURCE_WRAPPER_PATH="$PWD/bin/mtplx"
```

確かめ方: エンジンのプロセスの環境変数に、このリポジトリを指す `PYTHONPATH` があれば、このブランチで動いています。

```bash
ps eww -p "$(pgrep -f mtplx.server.openai | head -1)" -o command= | grep -o 'PYTHONPATH=[^ ]*'
```

注意:

- アプリの設定（context window、SSD キャッシュ、KV の量子化など）がそのまま使われます。128K / 256K を使うなら、設定で context window を 262,144 に上げ、SSD キャッシュを on にしてください
- KV の量子化: アプリの選択肢は off / q8 / q4 で、`auto` はありません。アプリは off のときにエンジンへ何も渡さないので、このブランチでは off を選ぶと M1 系では `auto`（131,072 トークン以上だけ q8）になります（アプリのコードから読み取った動作で、画面からは未確認）。q8 を選ぶと全長で q8 です
- リポジトリの作業ツリーの内容がそのまま動きます。別のブランチに切り替えると、次にエンジンを起動したときにその内容で動きます
- 元に戻すには、アプリを終了し、`runtime.env` を消して、アプリを普通に開きます

## サーバーだけを起動する

MTPLX アプリが動いていれば終了します（エンジンは GPU とメモリを約 20 GB 使うため、2つ同時には動かせません）。

```bash
bin/mtplx-m1 serve --model Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed-FP16 \
  --profile turbo --depth 3 --host 127.0.0.1 --port 8000 \
  --context-window 262144 --no-auth --yes
```

- `bin/mtplx-m1` は、アプリの実行環境の Python で、このリポジトリの `mtplx` を動かします。`--model` には `~/.mtplx/models` のフォルダ名か、パスを指定します
- `MTPLX is ready.` と表示されたら使えます（モデルの読み込みに十数秒〜1分）
- 止めるときは Ctrl-C

主なオプション:

| オプション | 既定 | 説明 |
|---|---|---|
| `--kv-quant` | 未指定（M1 系では `auto`） | `auto` は 131,072 トークン以上のプロンプトだけ q8 KV。`off` / `q8` / `q4` を指定するとそれに従う |
| `--context-window` | モデルの設定 | 256K まで使うなら `262144` |
| `--ssd-session-cache` | `on` | 会話のキャッシュを SSD にも保存し、サーバーを再起動しても続きから再開できる |
| `--max-tokens` | サーバーの既定 | 1回の応答の上限 |
| `--port` | 8000 | |

## Qwen3.8-Flash-Next（Unsloth UD-Q2_0）

Qwen3.8-Flash-Next（全体 125B、1 トークンあたり 6B を使う MoE、48 層、51B の n-gram 埋め込み表つき）は、Unsloth の UD-Q2_0 の GGUF なら 64 GB の M1 Mac に載ります。このフォークは、GGUF の expert と n-gram 表をその場で読み、残りをモデルのフォルダに変換します。アプリのモデル画面からは入手できません。

必要なもの: 64 GB の Mac、約 78 GB の空きディスク（GGUF 70.5 GB、変換後のフォルダ 6 GB、MTP ヘッド 1.7 GB）、上のネイティブ拡張。準備は次の 1 行です（途中で止まっても続きからダウンロードし、そのあと CPU で数分かけて変換します）。

```bash
bin/mtplx-m1 get flash-next
```

GGUF は `~/models/Qwen3.8-Flash-Next-UD-Q2_0-GGUF` に、モデルのフォルダは `~/.mtplx/models/Qwen3.8-Flash-Next-UD-Q2_0-MTPLX` にできるので、アプリのモデル一覧にも出ます（アプリから起動するのは試していません）。モデルのフォルダは expert と n-gram 表を GGUF から読むので、GGUF は消さないでください。GGUF の置き場所は `--models-dir` で変えられます。メモリを多く使うアプリを閉じてから起動します。

```bash
bin/mtplx-m1 serve --model Qwen3.8-Flash-Next-UD-Q2_0-MTPLX \
  --profile turbo --host 127.0.0.1 --port 8000 --no-auth --yes
```

- MTP の深さの既定は 3 です。64 GB の Mac では文脈の上限が既定で 65,536 トークンになります。それより長いプロンプトは `--context-window 262144` を付けて起動します。M1 Max 64 GB で、既定の int8 の KV（下）なら、12.8 万・19.2 万・25.6 万トークンのコードのプロンプトがそれぞれ 1 回で完走しました（cold の読み込みが 338 / 524 / 700 秒、decode が 40.5 / 38.8 / 36.8 tok/s。v2.12.3-m1.18 で各 1 回）。`--paged-kv-quantization off`（fp16 の KV）では、19.2 万トークンはプロンプトの終わり近くでメモリの番人に止められ、25.6 万トークンは受け付けません。
- KV は既定で int8 です。1 トークン・1 ヘッドごとの float16 のスケール付きで、1 トークンあたり約 24 KB が約 12 KB になります。`--paged-kv-quantization off`（または `MTPLX_QSA_KV_Q8=0`）で fp16 に戻ります。draft の頭の 1 層は fp16 のままです。perplexity と 1 round あたりの受理数は、ばらつきの範囲で変わりませんでした（対の比較は m1.18 の変更内容にあります）。
- ピークのメモリは 2K で約 50 GB、8K で約 52 GB、12.8 万トークンで約 53 GB、19.2〜25.6 万トークンで約 53〜54 GB です（v2.12.3-m1.18、int8 の KV）。n-gram 表（26.8 GB）は SSD から読み、この数字には入りません。
- M1 Max 64 GB での実測（思考あり、コード / 英語 / 日本語 / エージェントのログ）: decode は 2K のプロンプトの後で 47.8 / 42.0 / 43.7 / 45.3 tok/s、3.2 万トークンの後で 44.2 / 41.6 / 38.4 / 48.6 tok/s、12.8 万トークンの後で 37.5 / 37.2 / 34.8 / 42.8 tok/s です。cold なプロンプトの読み込みは 2K で約 5.4 秒、8K で約 20.5 秒、3.2 万トークンで約 83 秒、12.8 万トークンで約 440 秒です。
- Unsloth のファイルを勧める理由: ISTA の Q2_0 の GGUF（Splash が使うもの。こちらも使えます）と比べて、perplexity が英語の小説 / 日本語の小説 / コード / エージェントのログで 7.30 / 34.6 / 2.77 / 2.65 から 6.77 / 20.2 / 2.47 / 1.94 に下がります。その代わり decode とプロンプトの読み込みが数 % 遅くなります（v2.12.3-m1.16 での計測）。プロンプトの中の事実を答える質問は 48 問すべて正解でした（ISTA は 3.2 万トークンで 2 問を外しました）。Unsloth は routed の expert を IQ2_XXS / IQ2_S（gate、up）と Q2_0 / IQ4_NL（down）で、dense の層を Q5_K〜Q8_0 で量子化しています。ISTA の dense は Q2_0〜Q4_K です。

## Inkling-2-Mini（Thinking Machines）

Inkling-2-Mini（MoE、expert 128 個から 8 個、24 層のうち 12 層を 3 周する構造）は、公式の重み（NVFP4 / MXFP8、約 20 GB）のまま動きます。MTP の draft は使わず、自己回帰（AR）で動きます。v2.12.3-m1.19 から対応しています。

必要なもの: 32 GB 以上の Mac（計測は 64 GB だけです）、約 20 GB の空きディスク、上のネイティブ拡張。forge で `~/.mtplx/models` に置くと、アプリのモデル一覧にも出ます（変換はしないので数分で終わります。ダウンロードした重みとは hardlink でつなぐので、同じディスクならディスクは増えません）。

```bash
bin/mtplx-m1 get inkling2   # = mtplx forge build --repo thinkingmachines/Inkling-2-Mini ... into ~/.mtplx/models
```

サーバーだけで使うときは、forge したフォルダ（またはダウンロードしたフォルダ）を指定します。

```bash
bin/mtplx-m1 serve --model thinkingmachines-Inkling-2-Mini-MTPLX --host 127.0.0.1 --port 8000 --no-auth --yes
```

- M1 Max 64 GB での実測（v2.12.3-m1.19、サーバー、temperature 0）: prefill は 2K で 496、1.6 万トークンで 495〜516、3 万トークンで 469 tok/s、decode は 49 / 45 / 42 tok/s です。
- MTP の深さや `--paged-kv-quantization q8` を指定しても、AR と fp16 の KV に自動で切り替わります。`mtplx tune` には対応していません。
- 本家の Inkling のパック（`Youssofal/Inkling-2-Mini-MTPLX-*`。MTPLX 2.13 のアプリが勧めるもので、M1 では FP16 版）も、このフォークの実装で動きます。M1 系ではこの実装が既定です（`MTPLX_INKLING2_FORK_LANE=0` で本家 2.13 の実装になりますが、M1 Max では Metal の threadgroup の上限で request が失敗しました）。Optimized Speed FP16 で、2K / 16K / 32K の prefill 509 / 522 / 492 tok/s、decode 48.9 / 45.5 / 42.2 tok/s でした。
- 思考の深さ（reasoning effort）は low / medium / high / xhigh で、既定は high です。難しいエージェントの作業では長く考えるので、時間の制限がある使い方では medium が向きます。sampling の既定は temperature 1.0 です。
- 文脈は 1M トークンまでのモデルです。既定の上限は Mac のメモリから決まり、起動時のログに出ます。それより長いプロンプトは `--context-window` で広げます（実測は CHANGES の m1.19 の節にあります）。1 トークンあたりの KV は 32 KB で、長いほど prefill と decode が遅くなります（25.6 万トークンで prefill 236 tok/s、decode 19.8 tok/s、52 万トークンで 154 tok/s と 12.0 tok/s）。64 GB の Mac では 1M は fp16 の KV が収まりません。

## 使う

- ブラウザのチャット画面: <http://127.0.0.1:8000/>
- OpenAI 互換の API: `http://127.0.0.1:8000/v1`（`--no-auth` を付けているので API キーは不要。localhost にだけ公開されます）

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model": "mtplx", "messages": [{"role": "user", "content": "こんにちは"}], "max_tokens": 128}'
```

Open WebUI、OpenCode、Claude Code などの設定は `mtplx.cli connect` で表示できます（同じく `PYTHONPATH` を付けて実行します）。

## 長い context を使うときの目安（M1 Max 64GB、実測）

- 1ターン目は prompt 全体を prefill するので、128K で約 25 分、256K で約 75 分かかります
- 同じ会話の2ターン目以降は、キャッシュから復元するので数秒です（128K 約 2 s、256K 約 3.4 s）。新しく足したトークンの分だけ prefill します
- 256K の2ターン目のピークメモリは約 50 GB です。ほかのアプリでメモリを大きく使うと、スワップが発生することがあります
