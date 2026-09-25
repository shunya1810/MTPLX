# このフォーク（m1max-longctx）でサーバーを動かす

[English](USAGE.md)

`m1max-longctx` ブランチは、M1 Max で長い context を扱うための変更を upstream の MTPLX に足したものです（変更点と計測は [ベンチマーク](../benchmarks/m1max-longctx/README.ja.md)）。
ここでは、このブランチのサーバーを手元で起動して使う方法を説明します。

## 使い方は2通り

- **MTPLX アプリで使う**: アプリの開発者向けの仕組み（ソースのチェックアウトをエンジンとして使う）で、アプリの画面からこのブランチを動かします（下の「アプリで動かす」）
- **サーバーだけを起動する**: コマンドでサーバーを起動し、サーバー自身のチャット画面か OpenAI 互換の API から使います（下の「サーバーだけを起動する」）

どちらも、アプリの実行環境（`~/Library/Application Support/MTPLX/runtime-venv`）の Python を使い、このリポジトリの `mtplx` を読ませます。
アプリを普通に開いたときは、これまでどおりアプリ同梱の upstream のリリース版で動きます。

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

## アプリで動かす

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
PYTHONPATH="$PWD" "$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python" -m mtplx.cli serve \
  --model "$HOME/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed-FP16" \
  --profile turbo --depth 3 --host 127.0.0.1 --port 8000 \
  --context-window 262144 --no-auth --yes
```

- `PYTHONPATH="$PWD"` で、アプリの実行環境の Python に、このリポジトリの `mtplx` を読ませます
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
