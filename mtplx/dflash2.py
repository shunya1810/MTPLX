"""DFlash 2 block drafter for Qwen3.8-Flash-Next on MLX.

incoai/Qwen3.8-Flash-Next-DFlash2 (Apache-2.0) is a 5-layer block drafter: from the
last committed token (the anchor) and mask tokens it produces a whole block of draft
tokens in one forward pass, reading the target's hidden states as context.

  context  taps (the mean of the four hyper-connection streams after the zero-based
           layers 5, 15, 25, 35, 45 of the target, 5 x 2560 per committed position)
           -> fc (12800 -> 2560) -> hidden_norm; every layer projects that one vector
           to its own K (k_norm, rope) and V; a position's K/V go into a sliding window
           (2,048 positions) and never change.
  block    ``block_size`` rows: embedding of the anchor, then of the mask token. Per layer
           rms -> dynamic 2-tap grouped conv (prepare) -> q / k / v (q_norm, k_norm, rope)
           -> attention over the window and the block's own keys (not causal; keys within
           2,048 positions of the query) -> o -> conv (finish) + residual
           -> rms -> conv (prepare) -> gate / up -> silu * up -> down -> conv (finish) + residual.
  select   final norm; the target's output head over the rows 1..; each row keeps its top
           ``selector_top_k`` candidates (their logits are the unary scores) and a rank-256
           bilinear score between the previous pick and each candidate is added; the path
           is walked left to right (argmax, or a sample of softmax(score / temperature),
           whose probabilities are the draft distribution q of the speculative sampling).

The block's own K/V are never stored, so a rejected draft leaves nothing to roll back.
Port of the reference `dflash/model.py` (z-lab/dflash, MIT) written from its description;
numerics are checked against a torch transcription in m1max-longctx-bench-work/fn-s8.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn


@dataclass
class DFlash2Config:
    hidden_size: int = 2560
    num_layers: int = 5
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 7680
    sliding_window: int = 2048
    block_size: int = 8
    conv_kernel_size: int = 2
    conv_group_size: int = 16
    selector_rank: int = 256
    selector_top_k: int = 16
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    mask_token_id: int = 248077
    target_layer_ids: tuple = (5, 15, 25, 35, 45)
    vocab_size: int = 248320
    input_embedding_scale: float = 1.0
    output_multiplier: float = 1.0

    @classmethod
    def from_dir(cls, path: str | Path) -> "DFlash2Config":
        raw = json.loads((Path(path) / "config.json").read_text())
        df = raw.get("dflash_config", {})

        def pick(name: str, default: Any) -> Any:
            return df.get(name, raw.get(name, default))

        rope = raw.get("rope_parameters") or {}
        return cls(
            hidden_size=int(raw["hidden_size"]),
            num_layers=int(raw["num_hidden_layers"]),
            num_heads=int(raw["num_attention_heads"]),
            num_kv_heads=int(raw["num_key_value_heads"]),
            head_dim=int(raw.get("head_dim", raw["hidden_size"] // raw["num_attention_heads"])),
            intermediate_size=int(raw["intermediate_size"]),
            sliding_window=int(raw.get("sliding_window") or 2048),
            block_size=int(pick("block_size", 16)),
            conv_kernel_size=int(pick("conv_kernel_size", 2)),
            conv_group_size=int(pick("conv_group_size", 16)),
            selector_rank=int(pick("selector_rank", 256)),
            selector_top_k=int(pick("selector_top_k", 16)),
            rope_theta=float(rope.get("rope_theta", raw.get("rope_theta", 10000.0))),
            rms_norm_eps=float(raw.get("rms_norm_eps", 1e-6)),
            mask_token_id=int(pick("mask_token_id", 0)),
            target_layer_ids=tuple(int(i) for i in pick("target_layer_ids", (5, 15, 25, 35, 45))),
            vocab_size=int(raw["vocab_size"]),
            input_embedding_scale=float(pick("input_embedding_scale", 1.0)),
            output_multiplier=float(pick("output_multiplier", 1.0)),
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class GroupedDynamicCausalConv(nn.Module):
    """Two-tap causal convolution along the block whose kernel is a learned base
    plus a per-position, per-group dynamic part. ``prepare`` runs before the
    attention / MLP, ``finish`` after it (the dynamic kernel of ``finish`` comes
    from the same projection as ``prepare``'s)."""

    def __init__(self, hidden_size: int, kernel_size: int, group_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.group_size = group_size
        self.groups = hidden_size // group_size
        self.base_kernel = mx.zeros((2, kernel_size, hidden_size))
        self.kernel_projection = nn.Linear(hidden_size, 2 * kernel_size * self.groups, bias=False)

    def _convolve(self, hidden: mx.array, dynamic: mx.array, base: mx.array) -> mx.array:
        # hidden [L, H], dynamic [L, ks, groups], base [ks, H]
        length, hidden_size = hidden.shape
        blocks = hidden.reshape(length, self.groups, self.group_size)
        out = mx.zeros_like(blocks)
        for offset in range(self.kernel_size):
            if offset == 0:
                values = blocks
            else:
                values = mx.concatenate(
                    [mx.zeros((offset, self.groups, self.group_size), blocks.dtype), blocks[: length - offset]],
                    axis=0,
                )
            kernel = base[offset].reshape(1, self.groups, self.group_size).astype(hidden.dtype)
            out = out + kernel * values
            out = out + dynamic[:, offset, :, None] * values
        return out.reshape(length, hidden_size)

    def prepare(self, hidden: mx.array) -> tuple[mx.array, mx.array]:
        length = hidden.shape[0]
        dynamic = self.kernel_projection(hidden).reshape(length, 2, self.kernel_size, self.groups)
        return self._convolve(hidden, dynamic[:, 0], self.base_kernel[0]), dynamic[:, 1]

    def finish(self, hidden: mx.array, dynamic: mx.array) -> mx.array:
        return self._convolve(hidden, dynamic, self.base_kernel[1])


class Attention(nn.Module):
    def __init__(self, cfg: DFlash2Config):
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.scale = cfg.head_dim ** -0.5

    def context_kv(self, ctx: mx.array, pos0: int) -> tuple[mx.array, mx.array]:
        """K [kv_heads, n, head_dim] (k_norm, rope at ``pos0``...) and V of ``n`` context rows."""
        cfg = self.cfg
        n = ctx.shape[0]
        k = self.k_norm(self.k_proj(ctx).reshape(n, cfg.num_kv_heads, cfg.head_dim)).transpose(1, 0, 2)
        v = self.v_proj(ctx).reshape(n, cfg.num_kv_heads, cfg.head_dim).transpose(1, 0, 2)
        k = mx.fast.rope(k, cfg.head_dim, traditional=False, base=cfg.rope_theta, scale=1.0, offset=pos0)
        return k, v

    def __call__(
        self,
        hidden: mx.array,
        ctx_k: Optional[mx.array],
        ctx_v: Optional[mx.array],
        ctx_pos0: int,
        start: int,
    ) -> mx.array:
        """Attention of the block rows (positions start...) over the window and the block."""
        cfg = self.cfg
        rows = hidden.shape[0]
        q = self.q_norm(self.q_proj(hidden).reshape(rows, cfg.num_heads, cfg.head_dim)).transpose(1, 0, 2)
        q = mx.fast.rope(q, cfg.head_dim, traditional=False, base=cfg.rope_theta, scale=1.0, offset=start)
        k_blk, v_blk = self.context_kv(hidden, start)
        if ctx_k is not None and ctx_k.shape[1] > 0:
            n_ctx = ctx_k.shape[1]
            k = mx.concatenate([ctx_k, k_blk], axis=1)
            v = mx.concatenate([ctx_v, v_blk], axis=1)
            kpos = mx.concatenate([ctx_pos0 + mx.arange(n_ctx), start + mx.arange(rows)])
        else:
            k, v = k_blk, v_blk
            kpos = start + mx.arange(rows)
        qpos = start + mx.arange(rows)
        visible = mx.abs(qpos[:, None] - kpos[None, :]) < cfg.sliding_window
        mask = mx.where(visible, 0.0, -mx.inf).astype(q.dtype)
        out = mx.fast.scaled_dot_product_attention(
            q[None], k[None], v[None], scale=self.scale, mask=mask
        )[0]
        return self.o_proj(out.transpose(1, 0, 2).reshape(rows, cfg.num_heads * cfg.head_dim))


class MLP(nn.Module):
    def __init__(self, cfg: DFlash2Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def __call__(self, x: mx.array, down_scale: float = 1.0) -> mx.array:
        """``down_scale`` multiplies the activation entering the down projection (the output
        is down_scale times the true one): the bfloat16 checkpoint's MLP outputs reach
        1e5-1e6, past float16's 65,504."""
        act = nn.silu(self.gate_proj(x)) * self.up_proj(x)
        if down_scale != 1.0:
            act = act * down_scale
        return self.down_proj(act)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: DFlash2Config):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attention_conv = GroupedDynamicCausalConv(cfg.hidden_size, cfg.conv_kernel_size, cfg.conv_group_size)
        self.mlp_conv = GroupedDynamicCausalConv(cfg.hidden_size, cfg.conv_kernel_size, cfg.conv_group_size)

    def __call__(self, h, ctx_k, ctx_v, ctx_pos0, start, dtype, down_scale: float = 1.0):
        """``h`` is the float32 residual stream [rows, hidden]; the blocks compute in ``dtype``."""
        x, kernel = self.attention_conv.prepare(self.input_layernorm(h).astype(dtype))
        x = self.attention_conv.finish(self.self_attn(x, ctx_k, ctx_v, ctx_pos0, start), kernel)
        h = h + x.astype(mx.float32)
        x, kernel = self.mlp_conv.prepare(self.post_attention_layernorm(h).astype(dtype))
        # the down projection runs on act * down_scale (see MLP); finish is linear in its input
        x = self.mlp_conv.finish(self.mlp(x, down_scale), kernel).astype(mx.float32)
        return h + (x / down_scale if down_scale != 1.0 else x)


class CandidateSelector(nn.Module):
    def __init__(self, cfg: DFlash2Config):
        super().__init__()
        self.top_k = cfg.selector_top_k
        self.predecessor_codebook = mx.zeros((cfg.vocab_size, cfg.selector_rank))
        self.successor_codebook = mx.zeros((cfg.vocab_size, cfg.selector_rank))
        self.hidden_projection = nn.Linear(cfg.hidden_size, cfg.selector_rank, bias=False)

    def select(
        self,
        hidden: mx.array,
        unary: mx.array,
        candidates: mx.array,
        anchor: mx.array,
        temperature: float,
        key: Optional[mx.array] = None,
    ):
        """Walk the block's path. ``unary`` [rows, k] logits of ``candidates`` [rows, k]
        (token ids). Returns (path [rows] int32, probs [rows, k] float32 or None);
        ``probs`` is softmax(score / temperature) of every row (the draft distribution
        of the row's pick over its candidates) when ``temperature`` > 0."""
        proj = self.hidden_projection(hidden)
        pred = anchor.reshape(())
        path, probs = [], []
        keys = mx.random.split(key, hidden.shape[0]) if (temperature > 0 and key is not None) else None
        for p in range(hidden.shape[0]):
            bilinear = self.successor_codebook[candidates[p]].astype(mx.float32) @ (
                self.predecessor_codebook[pred].astype(mx.float32) * proj[p].astype(mx.float32)
            )
            scores = unary[p].astype(mx.float32) + bilinear
            if temperature > 0:
                q = mx.softmax(scores / temperature, axis=-1)
                idx = mx.random.categorical(mx.log(q + 1e-30), key=keys[p] if keys is not None else None)
                probs.append(q)
            else:
                idx = mx.argmax(scores, axis=-1)
            pred = candidates[p][idx]
            path.append(pred)
        return mx.stack(path).astype(mx.int32), (mx.stack(probs) if probs else None)


class DFlash2Draft(nn.Module):
    """The residual stream is float32 (the checkpoint is bfloat16 and its massive
    activations pass float16's range); the blocks compute in ``dtype`` (float16 on the M1
    family) with the MLP's down projection fed act / 64."""

    DOWN_SCALE_F16 = 1.0 / 64.0

    def __init__(self, cfg: DFlash2Config, dtype: mx.Dtype = mx.float16):
        super().__init__()
        self.cfg = cfg
        self.dtype = dtype
        self.down_scale = 1.0 if dtype == mx.float32 else self.DOWN_SCALE_F16
        self.layers = [DecoderLayer(cfg) for _ in range(cfg.num_layers)]
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.fc = nn.Linear(len(cfg.target_layer_ids) * cfg.hidden_size, cfg.hidden_size, bias=False)
        self.hidden_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.candidate_selector = CandidateSelector(cfg)

    # -- context ---------------------------------------------------------

    def context_features(self, taps: mx.array) -> mx.array:
        """[n, 5 * hidden] target taps -> the context vector every layer projects."""
        return self.hidden_norm(self.fc(taps))

    def new_context(self) -> "DFlash2Context":
        return DFlash2Context(self.cfg)

    def update_context(self, ctx: "DFlash2Context", taps: mx.array, pos0: int) -> None:
        """Append the committed rows ``pos0`` ... ``pos0 + n - 1`` (taps [n, 5 * hidden])."""
        feats = self.context_features(taps)
        kvs = [layer.self_attn.context_kv(feats, pos0) for layer in self.layers]
        ctx.append(kvs, pos0, taps.shape[0])

    # -- block ---------------------------------------------------------

    def block_hidden(self, ctx: "DFlash2Context", embeddings: mx.array, start: int) -> mx.array:
        """Final-normed hidden states [rows, hidden] of a block whose first row (the
        anchor) sits at position ``start`` (embeddings [rows, hidden])."""
        h = embeddings.astype(mx.float32)
        for i, layer in enumerate(self.layers):
            k, v = (ctx.k[i], ctx.v[i]) if ctx.length else (None, None)
            h = layer(h, k, v, ctx.pos0, start, self.dtype, self.down_scale)
        return self.norm(h).astype(self.dtype)

    def propose(
        self,
        ctx: "DFlash2Context",
        anchor_token: int | mx.array,
        start: int,
        embed: Callable[[mx.array], mx.array],
        head: Callable[[mx.array], tuple[mx.array, mx.array]],
        temperature: float = 0.0,
        rows: Optional[int] = None,
        key: Optional[mx.array] = None,
    ):
        """One block: (tokens [rows - 1] int32, probs [rows - 1, k] | None, candidates [rows - 1, k]).

        ``embed(ids)`` is the target's embedding (rows [n] -> [n, hidden]); ``head(hidden)``
        returns (logits [n, V'], token_ids [V']) over the draft vocabulary (the target's
        whole head, or its pruned rows and their ids).
        """
        cfg = self.cfg
        rows = cfg.block_size if rows is None else rows
        anchor = mx.array(anchor_token, dtype=mx.int32).reshape(1)
        ids = mx.concatenate([anchor, mx.full((rows - 1,), cfg.mask_token_id, dtype=mx.int32)])
        emb = embed(ids)
        if cfg.input_embedding_scale != 1.0:
            emb = emb * cfg.input_embedding_scale
        hidden = self.block_hidden(ctx, emb, start)[1:]
        logits, token_ids = head(hidden)
        logits = logits * cfg.output_multiplier
        logits = logits.astype(mx.float32)
        # the top-k candidates of every row, unsorted (the selector scores all k of them)
        part = mx.argpartition(-logits, cfg.selector_top_k - 1, axis=-1)[:, : cfg.selector_top_k]
        unary = mx.take_along_axis(logits, part, axis=-1)
        candidates = token_ids[part]
        tokens, probs = self.candidate_selector.select(hidden, unary, candidates, anchor, temperature, key)
        return tokens, probs, candidates


class DFlash2Context:
    """The committed rows' K/V per layer, the last ``sliding_window`` positions."""

    def __init__(self, cfg: DFlash2Config):
        self.cfg = cfg
        self.k: list[Optional[mx.array]] = [None] * cfg.num_layers
        self.v: list[Optional[mx.array]] = [None] * cfg.num_layers
        self.pos0 = 0
        self.length = 0

    def reset(self) -> None:
        self.k = [None] * self.cfg.num_layers
        self.v = [None] * self.cfg.num_layers
        self.pos0 = 0
        self.length = 0

    def append(self, kvs: Sequence[tuple[mx.array, mx.array]], pos0: int, n: int) -> None:
        """Rows ``pos0`` ... ``pos0 + n - 1``. A row that does not follow the held run
        starts a new run (the old rows are dropped)."""
        window = self.cfg.sliding_window
        if self.length and pos0 != self.pos0 + self.length:
            self.reset()
        for i, (k, v) in enumerate(kvs):
            if self.k[i] is None:
                self.k[i], self.v[i] = k, v
            else:
                self.k[i] = mx.concatenate([self.k[i], k], axis=1)
                self.v[i] = mx.concatenate([self.v[i], v], axis=1)
        if not self.length:
            self.pos0 = pos0
        self.length += n
        extra = self.length - window
        if extra > 0:
            for i in range(len(self.k)):
                self.k[i] = self.k[i][:, extra:]
                self.v[i] = self.v[i][:, extra:]
            self.pos0 += extra
            self.length = window


def load_dflash2(path: str | Path, dtype: mx.Dtype = mx.float16, quantize_bits: int = 0) -> DFlash2Draft:
    """Load the safetensors checkpoint (bf16) as ``dtype``; ``quantize_bits`` 8 / 4 quantizes
    the linear layers (group size 64)."""
    cfg = DFlash2Config.from_dir(path)
    model = DFlash2Draft(cfg, dtype=dtype)
    weights = {}
    for shard in sorted(Path(path).glob("*.safetensors")):
        weights.update(mx.load(str(shard)))
    renamed = {}
    for key, value in weights.items():
        # the norms that read the float32 residual stream keep float32 weights
        stream_norm = key.endswith(("input_layernorm.weight", "post_attention_layernorm.weight")) or key == "norm.weight"
        renamed[key] = value.astype(mx.float32 if stream_norm else dtype)
    model.load_weights(list(renamed.items()), strict=True)
    if quantize_bits:
        def predicate(name: str, module: nn.Module) -> bool:
            return isinstance(module, nn.Linear) and "kernel_projection" not in name and "hidden_projection" not in name

        nn.quantize(model, group_size=64, bits=quantize_bits, class_predicate=predicate)
    mx.eval(model.parameters())
    return model
