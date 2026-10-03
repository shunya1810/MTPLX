"""MTPLX score_prompt_logprobs, executable body from 50de43bb:mtplx/generation.py.

Executed with the generation module globals by the parent-parity test.
"""
from __future__ import annotations

def score_prompt_logprobs(
    rt: MTPLXRuntime,
    prompt_ids: list[int],
    *,
    top_k: int,
    chunk_size: int = 256,
) -> dict[str, Any]:
    """The parent scoring path; executable body preserved below."""

    import numpy as np

    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    top_k = max(1, int(top_k))
    chunk_size = max(16, int(chunk_size))
    cache = _make_target_prefill_cache(rt)
    n = len(prompt_ids)
    prompt_array = mx.array([prompt_ids])
    token_logprobs: list[float | None] = []
    top_entries: list[list[tuple[int, float]]] = []
    started = time.perf_counter()
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        chunk = prompt_array[:, start:end]
        with attention_phase("prefill"):
            logits, _hidden = _forward_ar_optional_hidden(
                rt,
                chunk,
                cache=cache,
                hidden_variant=None,
                emit_logits=True,
            )
        rows_logits = logits[0]
        row_lse = _row_logsumexp_f32(rows_logits)
        k = min(top_k, int(rows_logits.shape[-1]))
        top_idx = _exact_top_k_ids(rows_logits, k)
        top_vals = _logprobs_at(rows_logits, row_lse, top_idx)
        # Positions start..end-1 predict prompt tokens start+1..end; the
        # final prompt position has no target inside the prompt.
        target_rows = min(end, n - 1) - start
        if target_rows > 0:
            targets = mx.array(
                [prompt_ids[start + 1 : start + 1 + target_rows]]
            )[0][:, None]
            target_lp = _logprobs_at(
                rows_logits[:target_rows], row_lse[:target_rows], targets
            )[:, 0]
        else:
            target_lp = None
        if target_lp is not None:
            mx.eval(top_idx, top_vals, target_lp)
        else:
            mx.eval(top_idx, top_vals)
        idx_np, vals_np = _sorted_top_k(np.array(top_idx), np.array(top_vals))
        rows = end - start
        for row in range(rows):
            # The last prompt position's distribution predicts a token
            # outside the prompt; keep its top-K out of the echoed contract.
            if start + row >= n - 1:
                break
            top_entries.append(
                [
                    (int(idx_np[row, col]), float(vals_np[row, col]))
                    for col in range(idx_np.shape[1])
                ]
            )
        if target_lp is not None:
            token_logprobs.extend(float(v) for v in np.array(target_lp))
        del logits, rows_logits, row_lse, top_idx, top_vals
    return {
        "positions": top_entries,
        "token_logprobs": token_logprobs,
        "prompt_tokens": n,
        "elapsed_s": time.perf_counter() - started,
    }
