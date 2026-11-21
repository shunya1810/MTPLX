# Inkling 2 Mini — 試用メモ（2026-11-21、branch inkling-s1）

（m1max-longctx-bench-work/inkling-s1/ が外部の削除で消えたため、要点だけをここに再記録した）

## agent-mini（OpenCode 2、各 1 回、temperature 1.0 / top_p 1.0 / top_k なし）

結果は llm-bench の results/agent-mini/inkling2-s1-try1 と try2-medium にある。

| タスク | effort high | effort medium |
|---|---|---|
| A ヘプタゴン | timeout（思考 21.6K トークン） | 合格 7/7、732 s（思考 13K） |
| B バグ修正 | 合格 5/5、213 s | － |
| C 型注釈 | 合格 14/14、361 s | － |
| Den / Dja 読解レポート | 失敗（memory guard の中止。18f173b9 で修正） | timeout（prefill 220〜400 tok/s と decode 23〜26 tok/s では 15 分に収まらない） |
| E ペリカン SVG | timeout（思考 19.5K） | SVG あり（採点なし） |

## 直したこと

- 18f173b9：InklingVisitCache.mtplx_boundary_checkpoints=False。これまで prompt 途中の境界ごとに 48 visit 分の KV を複製し、5.3K トークンで 12.8 GB を bank に保存していた。memory_plan の KV は 32,768 B/token にした（以前は 98,304）。修正後は中止 0 回、bank は 1.64 GB。

## 残っていること

- 速度は下の「速度改善」を参照。DSpark drafter（参照実装は Ollama の mlxrunner/model/inkling2/draft.go）は未着手。
- agent 用の既定 effort を medium にするかどうか（high では思考が長すぎる）。
- server の committed-reasoning が ChatML 前提のため、Inkling では前ターンの思考が履歴に残らない。postcommit が retokenized_history_mismatch になる（性能だけの問題）。
- 未実施：日本語の複数ターン、needle 32K/64K の battery（t7_chat_battery.py は削除されたので作り直しが必要）。

## 会話テスト（server、forge 済み pack、sampling は既定の 1.0）

- 日本語の 3 ターン（京都の紅葉）：
  - 自然な日本語で、prefix cache も効いていた（cached 1088 → 1803）。decode は約 27 tok/s。
  - ただし行き方の説明は曖昧だった。model card のとおり、知識系は弱い。
- effort low / medium / high / xhigh：
  - すべて正解（バットとボール）。
  - 思考は 92 / 198 / 198 / 238 トークン。
- `enable_thinking: false`：思考は非表示になり、本文は「Bonjour」。template の都合で、内部では思考している。
- max_tokens=64 で思考の途中に打ち切ると、finish は length で、本文は空。
- needle：

  | 文脈 | 結果 | prefill | TTFT | decode |
  |---|---|---|---|---|
  | 32K | 正解 | 313 tok/s | 102 s | 25.6 tok/s |
  | 64K | 正解 | 274 tok/s | 233 s | 22.8 tok/s |

  prefill の中止は 0 回。

## forge とアプリ（ba755150）

- forge は MTP contract の校正を必須にしていたため、Inkling（DSpark drafter）は calibrate で失敗していた。そこで AR-only の lane を作った：
  - 変換しない
  - hardlink でミラーする
  - AR で 3 問を検証する
  - AR-only として `mtplx_runtime.json` に書き込む
- アプリで使えない理由：
  1. アプリの runtime（MTPLX-app）は、起動のたびに origin/m1max-longctx（m1.18）へ更新されるので、Inkling 対応が入っていない。
  2. アプリのモデル一覧は `~/.mtplx/models` と「設定 → フォルダを追加」で登録したフォルダしか見ない。Flash-Next（`setup_flashnext_unsloth.py` の既定は `~/models`）も、同じ理由で一覧に出なかった。
- アプリと同じ起動形（`--depth 1`、q8 KV、SSD cache on、256K）でも、このブランチなら動く。MTP は AR に、q8 は off に自動で切り替わる。
- `mtplx tune` は unsupported_tune_model を返す（tune_policy が非対応）。そのため、アプリの自動チューニングは走らない想定。

## 速度改善（2026-11-21 後半、bd114a30..7c111e02）

### いちばん効いた 2 点

- server は Inkling を bf16 で動かしていた。M1 の fp16 cast が qwen4 専用だったためである。
  - M1 の GPU には bf16 演算がない。fp16 専用の高速経路（scaled experts、prefill の MoE glue）も使われていなかった。
  - 午前の agent 試験の 220〜400 tok/s（prefill）と 23〜27 tok/s（decode）は、この bf16 経路の数字である。
  - d8017a7e で inkling2_loop も既定で fp16 にした。fp16 の PPL は bf16 と同じ（測定済み）。
- server の AR ループは、token を host で sample してから次の forward を組み、encode していた。
  - Inkling の decode は 1 token あたり CPU を約 16 ms 使う（graph 構築 3 ms と encode 13 ms）。GPU は約 21 ms で、両者が重ならず直列になっていた。
  - 7c111e02 で、device 上で token を引き、host が読む前に次の forward を投げるようにした。
  - model 側の `mtplx_ar_pipeline_default` で有効になる。`MTPLX_AR_PIPELINE=1/0` で強制できる。
  - 文法制約、penalty、全語彙の top-p では従来のループのまま。

### server での実測（fp16、temperature 0、300 トークン生成）

| 文脈 | prefill | decode（前） | decode（後） |
|---|---|---|---|
| 2K | 383〜443 tok/s | 37.0 tok/s | 47.9 tok/s |
| 16K | 422 tok/s | 34.9 tok/s | 44.7 tok/s |
| 30K | 396 tok/s | 33.0 tok/s | 41.8 tok/s |

- chat battery（日本語 3 ターン、effort 4 段階、thinking off、打ち切り、needle 32K）と並列 tool call は合格した。
- decode-path の NLL は chunked と一致する（4 コーパスで |d| 0.005〜0.03 nats、argmax 一致 98.7〜99.5%）。
- PPL（en/ja/code/agent）は 6.599/10.305/2.160/1.790。基準は 6.593/10.320/2.157/1.791。

### kernel 側の変更

- post_mid / pre_row：attention の後ろと前の行処理を、それぞれ 1 kernel にした。
- MXFP8 experts（layer 20〜23）を fused kernel にした。E4M3 はレジスタ上で decode する（`(b & 0x7F) << 7` が値 / 256 の fp16 になる）。
- prefill の MoE glue を 3 kernel にした。約 15 回あった全行パスが減り、1 visit あたり 4.6 ms 短くなった。prefill は +5%。

### 試して入れなかったもの

- split-K の decode attention（bias を kernel 内で計算する版を 3 種類）。単体では速く見えたが、decode ループの中では MLX sdpa + mask に負けた。
- prefill attention の GQA 共有版。元の kernel が 4.2〜4.4 TFLOPS 出ていて、共有版のほうが遅かった。
- prefill の chunk を 4K / 8K にする案は +2〜3% にとどまった。projection を fp16 dense にする案は約 1.5% にとどまった。

### この先の候補

- DSpark：verify の幅を広げる経路が遅く、今のままでは使えない（1 token 18 ms、2 token 54 ms、8 token 91 ms。L>1 は prefill 用の汎用経路に落ちる）。drafter を入れる前に、2〜9 token の forward を decode 並みに組む必要がある。
- prefill の expert GEMM：gather_qmm は 5.4 TFLOPS で、dense の nvfp4 qmm（7.0）より遅い。segment ごとに揃えた自前 GEMM なら、prefill は 7〜13% 速くなる見込み。
