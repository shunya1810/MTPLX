"""Greedy speculative decoding for Inkling 2 with its DSpark block drafter.

A round: the drafter proposes ``k`` tokens after the anchor (the last
emitted, not yet forwarded token), the target forwards [anchor, d1..dk]
(one compiled multi-row forward, tapping the DSpark features), the longest
prefix of drafts equal to the target's own argmax is accepted, the target
caches step back over the rejected rows (``Model.rollback``), the accepted
rows' features become drafter context, and the target's token after the
last accepted draft is the next anchor. Output equals the target's greedy
decoding up to the numerics of the multi-row forward.

Drafter approximations (they change proposals, never the output): the
recurrent head corrects only the base logits' top ``CANDIDATES`` per row.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import mlx.core as mx

from .inkling2_dspark import DSpark, _rms

CANDIDATES = 64


class DraftContext:
    """The drafter's per-request state: context keys / values per draft
    layer (local layers keep the last window), the recurrent head's
    conditioning row and the number of context positions."""

    def __init__(self, draft: DSpark):
        self.draft = draft
        self.k = [None] * draft.config.num_hidden_layers
        self.v = [None] * draft.config.num_hidden_layers
        self.length = 0
        self.last_feature = None

    def append(self, features: mx.array) -> None:
        """features [1, n, taps * D] of the next n committed positions."""
        d = self.draft
        c = d.config
        n = features.shape[1]
        for i, layer in enumerate(d.layers):
            k, v = d.context_kv(i, features, self.length)
            if self.k[i] is not None:
                k = mx.concatenate([self.k[i], k], axis=2)
                v = mx.concatenate([self.v[i], v], axis=2)
            if layer.local and k.shape[2] > c.sliding_window_size:
                k = k[:, :, -c.sliding_window_size :]
                v = v[:, :, -c.sliding_window_size :]
            self.k[i], self.v[i] = k, v
        self.length += n
        self.last_feature = d.head_feature(features[:, -1:])[:, 0]

    def propose(self, anchor_embedding: mx.array, anchor: mx.array, lm_head, vocab: int, softcap: float, k: int) -> mx.array:
        """k greedy proposals after the anchor (lazy int32 [k])."""
        d = self.draft
        h = d.block_forward(anchor_embedding, list(zip(self.k, self.v)), self.length)
        rows = h[:, 1 : 1 + k]
        base = lm_head(rows)[..., :vocab]
        return select_candidates(d, rows, base, anchor, self.last_feature, softcap)


def select_candidates(d: DSpark, block_hidden, base_logits, anchor, feature, softcap: float) -> mx.array:
    """The recurrent head over each row's top CANDIDATES base logits."""
    c = d.config
    h = d.markov_head
    r = c.markov_rank
    R = block_hidden.shape[1]
    cand = mx.argpartition(-base_logits[0], kth=CANDIDATES - 1, axis=-1)[:, :CANDIDATES]  # [R, C]
    base_c = mx.take_along_axis(base_logits[0], cand, axis=-1).astype(mx.float32)  # [R, C]
    fold_w = d._fold_w[cand]  # [R, C, r]
    fold_b = d._fold_b[cand].astype(mx.float32)  # [R, C]
    state = mx.zeros((1, r), d.dtype)
    last = anchor.reshape(1).astype(mx.int32)
    feature = feature.reshape(1, -1).astype(d.dtype)
    out = []
    for i in range(R):
        prev = h.markov_w1(last).astype(d.dtype)
        gates = h.rnn_proj(mx.concatenate([state, prev, block_hidden[:, i].astype(d.dtype), feature], axis=-1))
        keep = mx.sigmoid(gates[:, :r].astype(mx.float32))
        update = mx.tanh(gates[:, r : 2 * r].astype(mx.float32))
        latent = mx.tanh(gates[:, 2 * r :].astype(mx.float32)).astype(d.dtype)
        state = (keep * state.astype(mx.float32) + (1.0 - keep) * update).astype(d.dtype)
        corr = (fold_w[i] @ latent[0]).astype(mx.float32) + fold_b[i]
        scores = base_c[i] + corr
        if softcap:
            scores = mx.tanh(scores / softcap) * softcap
        last = cand[i][mx.argmax(scores)].reshape(1).astype(mx.int32)
        out.append(last)
    return mx.concatenate(out)


@dataclass
class SpecStats:
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    tokens: int = 0
    seconds: float = 0.0
    accept_hist: list = field(default_factory=lambda: [0] * 9)


def spec_generate(model, draft: DSpark, prompt_ids: list, *, max_tokens: int, k: int = 3,
                  stop_ids: set | None = None, prefill_chunk: int = 2048) -> tuple[list, SpecStats]:
    """Greedy speculative generation (one request, B = 1)."""
    lm = model.model.language_model
    text = model.args.text
    vocab, softcap = text.unpadded_vocab_size, text.final_logit_softcapping
    stop_ids = stop_ids or set()
    lm.set_draft_taps(draft.config.target_layer_ids)
    try:
        cache = model.make_cache()
        ctx = DraftContext(draft)
        logits = None
        for s0 in range(0, len(prompt_ids), prefill_chunk):
            chunk = prompt_ids[s0 : s0 + prefill_chunk]
            logits = model(mx.array([chunk]), cache=cache)
            ctx.append(lm.last_draft_features)
            mx.eval(logits, ctx.k, ctx.v)
        anchor = int(mx.argmax(logits[0, -1, :vocab]).item())
        out = [anchor]
        st = SpecStats()
        t0 = time.perf_counter()
        while len(out) < max_tokens and anchor not in stop_ids:
            a_arr = mx.array([anchor], dtype=mx.int32)
            drafts = ctx.propose(lm.embed(a_arr[None]), a_arr, model.lm_head, vocab, softcap, k)
            block = mx.concatenate([a_arr, drafts])[None]
            logits = model(block, cache=cache)
            feats = lm.last_draft_features
            target = mx.argmax(logits[0, :, :vocab], axis=-1)
            mx.eval(target, drafts, feats)
            d = drafts.tolist()
            t = target.tolist()
            acc = 0
            while acc < k and d[acc] == t[acc]:
                acc += 1
            model.rollback(cache, k - acc)
            ctx.append(feats[:, : acc + 1])
            new = d[:acc] + [t[acc]]
            st.rounds += 1
            st.drafted += k
            st.accepted += acc
            st.accept_hist[acc] += 1
            for tok in new:
                out.append(tok)
                if tok in stop_ids or len(out) >= max_tokens:
                    break
            anchor = out[-1]
        st.seconds = time.perf_counter() - t0
        st.tokens = len(out) - 1
        return out, st
    finally:
        lm.set_draft_taps(None)
