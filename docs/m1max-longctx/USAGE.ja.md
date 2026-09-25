# このフォーク（m1max-longctx）でサーバーを動かす

[English](USAGE.md)

`m1max-longctx` ブランチは、M1 Max で長い context を扱うための変更を upstream の MTPLX に足したものです（変更点と計測は [ベンチマーク](../benchmarks/m1max-longctx/README.ja.md)）。
ここでは、このブランチのサーバーを手元で起動して使う方法を説明します。

## MTPLX アプリとの関係

- MTPLX アプリ（`/Applications/MTPLX.app`）は、アプリに同梱された upstream のリリース版を、アプリ専用の Python 環境（`~/Library/Application Support/MTPLX/runtime-venv`）で動かします。**アプリからこのブランチは使われません。**
- アプリには、外部で起動したサーバーにつなぐ機能がありません。設定のポートが使われていると、アプリは別のポートで自分のサーバーを起動します。
- アプリの実行環境をこのブランチに差し替える方法は検証していません。アプリは同梱の版を入れ直すことがあるので、差し替えはおすすめしません。

このブランチを使うときは、アプリを終了し、下の手順でサーバーを起動します。サーバー自身がブラウザのチャット画面と OpenAI 互換の API を持っているので、アプリがなくても使えます。
アプリに戻るときは、このサーバーを止めてからアプリを開いてください。

## 用意するもの

- Apple Silicon の Mac（計測は M1 Max 64GB、macOS 26）
- MTPLX アプリを一度起動して、実行環境（mlx などの依存関係）とモデルを入れておく。以下の手順は、アプリの実行環境の Python を使います（計測もすべてこの方法です）
- モデル: `~/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed-FP16`（アプリのモデル画面から取得したもの）

依存関係を自分の venv に入れて動かす方法は検証していません。

## 起動

```bash
git clone -b m1max-longctx https://github.com/shunya1810/MTPLX.git
cd MTPLX
```

MTPLX アプリが動いていれば終了します（GPU とメモリを約 20 GB 使うため、2つ同時には動かせません）。

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
