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

- 速度：agent ループ中の decode が 23〜27 tok/s、長い文脈の suffix prefill が約 220 tok/s。実用面では 27B と同程度の速さ。
  - DSpark drafter（参照実装は Ollama の mlxrunner/model/inkling2/draft.go）
  - 長い文脈での prefill
- agent 用の既定 effort を medium にするかどうか（high では思考が長すぎる）。
- server の committed-reasoning が ChatML 前提のため、Inkling では前ターンの思考が履歴に残らない。postcommit が retokenized_history_mismatch になる（性能だけの問題）。
- 未実施：日本語の複数ターン、needle 32K/64K の battery（t7_chat_battery.py は削除されたので作り直しが必要）。
