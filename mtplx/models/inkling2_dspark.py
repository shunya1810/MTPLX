"""DSpark block drafter for Inkling 2 (``mtp.safetensors``, ``mtp_config``).

One forward proposes a block: the anchor (the committed, not yet verified
token) and ``block_size - 1`` learned mask embeddings attend bidirectionally
over context rows built from target features, then a small recurrent
("markov") head corrects each mask row's logits with the previous proposal.

Target features: the residual stream entering visits ``target_layer_ids``
(all but the last) and the final hidden (the last id equals the visit count),
concatenated per token. A row ``row`` of the softmaxed ``fuse_weight`` mixes
the taps for draft layer ``row`` (the last row feeds the recurrent head).

Layout follows Ollama's mlxrunner inkling2 draft (draft.go / dflash.go): the
context keys take RoPE at their own target positions, the block continues
after them, local layers see the last ``sliding_window_size`` context rows
plus the block, residual sums round to the activation dtype before the next
addition, and the recurrent head's latent projection is folded into the
target head once.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


@dataclass
class DSparkConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    local_layer_ids: tuple
    sliding_window_size: int
    block_size: int
    markov_rank: int
    target_layer_ids: tuple

    @classmethod
    def from_dict(cls, cfg: dict) -> "DSparkConfig":
        if cfg.get("markov_head_type", "rnn_hidden") != "rnn_hidden" or cfg.get("q_bias") or cfg.get("o_bias"):
            raise ValueError("unsupported DSpark configuration")
        return cls(
            hidden_size=int(cfg["hidden_size"]),
            num_hidden_layers=int(cfg["num_hidden_layers"]),
            num_attention_heads=int(cfg["num_attention_heads"]),
            num_key_value_heads=int(cfg["num_key_value_heads"]),
            head_dim=int(cfg["head_dim"]),
            intermediate_size=int(cfg["intermediate_size"]),
            rms_norm_eps=float(cfg["rms_norm_eps"]),
            rope_theta=float(cfg["rope_theta"]),
            local_layer_ids=tuple(int(i) for i in cfg.get("local_layer_ids", ())),
            sliding_window_size=int(cfg["sliding_window_size"]),
            block_size=int(cfg["block_size"]),
            markov_rank=int(cfg["markov_rank"]),
            target_layer_ids=tuple(int(i) for i in cfg["target_layer_ids"]),
        )


def _rms(x: mx.array, weight: mx.array, eps: float, dtype) -> mx.array:
    return mx.fast.rms_norm(x.astype(mx.float32), weight.astype(mx.float32), eps).astype(dtype)


def _residual(residual: mx.array, update: mx.array, dtype) -> mx.array:
    # the stored residual is rounded before the (float32) addition
    return residual.astype(dtype).astype(mx.float32) + update.astype(mx.float32)


class _Attention(nn.Module):
    def __init__(self, c: DSparkConfig):
        super().__init__()
        d, kv = c.hidden_size, c.num_key_value_heads * c.head_dim
        self.q_proj = nn.Linear(d, c.num_attention_heads * c.head_dim, bias=False)
        self.k_proj = nn.Linear(d, kv, bias=False)
        self.v_proj = nn.Linear(d, kv, bias=False)
        self.k_proj_ctx = nn.Linear(d, kv, bias=False)
        self.v_proj_ctx = nn.Linear(d, kv, bias=False)
        self.o_proj = nn.Linear(c.num_attention_heads * c.head_dim, d, bias=False)
        self.q_norm = nn.RMSNorm(c.head_dim, eps=c.rms_norm_eps)
        self.k_norm = nn.RMSNorm(c.head_dim, eps=c.rms_norm_eps)
        self.k_norm_ctx = nn.RMSNorm(c.head_dim, eps=c.rms_norm_eps)


class _MLP(nn.Module):
    def __init__(self, c: DSparkConfig):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)


class DraftLayer(nn.Module):
    def __init__(self, c: DSparkConfig, local: bool):
        super().__init__()
        self.local = local
        self.self_attn = _Attention(c)
        self.mlp = _MLP(c)
        self.input_layernorm = nn.RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.dflare_norm = nn.RMSNorm(c.hidden_size, eps=c.rms_norm_eps)


class MarkovHead(nn.Module):
    def __init__(self, c: DSparkConfig, vocab_size: int):
        super().__init__()
        r, d = c.markov_rank, c.hidden_size
        self.dflare_norm = nn.RMSNorm(d, eps=c.rms_norm_eps)
        self.markov_w1 = nn.Embedding(vocab_size, r)
        self.rnn_proj = nn.Linear(2 * (r + d), 3 * r, bias=True)
        self.hidden_out = nn.Linear(r, d, bias=True)


class DSpark(nn.Module):
    def __init__(self, c: DSparkConfig, vocab_size: int):
        super().__init__()
        self.config = c
        self.layers = [DraftLayer(c, i in c.local_layer_ids) for i in range(c.num_hidden_layers)]
        self.fuse_weight = mx.zeros((c.num_hidden_layers + 1, len(c.target_layer_ids)))
        self.mask_embed = mx.zeros((c.hidden_size,))
        self.norm = nn.RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.markov_head = MarkovHead(c, vocab_size)

    # ---- loading

    @classmethod
    def load(cls, model_dir: Path, *, vocab_size: int, dtype=mx.float16) -> "DSpark":
        model_dir = Path(model_dir)
        cfg = json.loads((model_dir / "config.json").read_text())["mtp_config"]
        draft = cls(DSparkConfig.from_dict(cfg), vocab_size)
        weights = mx.load(str(model_dir / "mtp.safetensors"))
        weights = {k: (v.astype(dtype) if v.dtype in (mx.bfloat16, mx.float16, mx.float32) else v) for k, v in weights.items()}
        draft.load_weights(list(weights.items()), strict=True)
        mx.eval(draft.parameters())
        draft.dtype = dtype
        return draft

    def fold_head(self, lm_head) -> None:
        """Fold markov_head.hidden_out into the target head: logits correction
        = (W_head @ W_out) latent + W_head @ b_out, computed once in float32."""
        if isinstance(lm_head, nn.QuantizedLinear):
            w = mx.dequantize(
                lm_head.weight, lm_head.scales, lm_head.get("biases"),
                group_size=lm_head.group_size, bits=lm_head.bits,
                mode=getattr(lm_head, "mode", "affine"),
            )
        else:
            w = lm_head.weight
        w = w.astype(mx.float32)
        out = self.markov_head.hidden_out
        self._fold_w = (w @ out.weight.astype(mx.float32)).astype(self.dtype)
        self._fold_b = (w @ out.bias.astype(mx.float32)).astype(self.dtype)
        mx.eval(self._fold_w, self._fold_b)

    # ---- forward pieces

    def _mix_weights(self) -> mx.array:
        w = self.__dict__.get("_fuse_softmax")
        if w is None:
            w = mx.softmax(self.fuse_weight.astype(mx.float32), axis=-1)
            self.__dict__["_fuse_softmax"] = w
        return w

    def mix(self, features: mx.array, row: int, norm: nn.RMSNorm) -> mx.array:
        c = self.config
        B, L, _ = features.shape
        taps = len(c.target_layer_ids)
        x = features.reshape(B, L, taps, c.hidden_size).astype(mx.float32)
        w = self._mix_weights()[row].reshape(1, 1, taps, 1)
        return _rms((x * w).sum(axis=2), norm.weight, c.rms_norm_eps, self.dtype)

    def _heads(self, x, proj, norm, n_heads, offset):
        c = self.config
        B, L, _ = x.shape
        y = proj(x).reshape(B, L, n_heads, c.head_dim)
        if norm is not None:
            y = _rms(y, norm.weight, c.rms_norm_eps, self.dtype)
        y = y.transpose(0, 2, 1, 3)
        if offset is not None:
            y = mx.fast.rope(y, c.head_dim, traditional=False, base=c.rope_theta, scale=1.0, offset=offset)
        return y

    def context_kv(self, index: int, features: mx.array, start: int):
        """Context keys / values of draft layer ``index`` for target feature
        rows at positions start, start + 1, ..."""
        c = self.config
        layer = self.layers[index]
        ctx = self.mix(features, index, layer.dflare_norm)
        a = layer.self_attn
        k = self._heads(ctx, a.k_proj_ctx, a.k_norm_ctx, c.num_key_value_heads, start)
        v = self._heads(ctx, a.v_proj_ctx, None, c.num_key_value_heads, None)
        return k, v

    def block_forward(self, anchor_embedding: mx.array, ctx_kv: list, block_start: int) -> mx.array:
        """anchor_embedding [B, 1, D] (the target's normalized embedding);
        ctx_kv[i] = (k, v) of every context row draft layer i sees, at
        positions ending at block_start - 1. Returns the normalized block
        hidden [B, block, D]."""
        c = self.config
        B = anchor_embedding.shape[0]
        n = c.block_size
        masks = mx.broadcast_to(self.mask_embed.astype(self.dtype), (B, n - 1, c.hidden_size))
        x = mx.concatenate([anchor_embedding.astype(self.dtype), masks], axis=1)
        scale = 1.0 / math.sqrt(c.head_dim)
        for layer, (kc, vc) in zip(self.layers, ctx_kv):
            a = layer.self_attn
            xn = _rms(x, layer.input_layernorm.weight, c.rms_norm_eps, self.dtype)
            q = self._heads(xn, a.q_proj, a.q_norm, c.num_attention_heads, block_start)
            kb = self._heads(xn, a.k_proj, a.k_norm, c.num_key_value_heads, block_start)
            vb = self._heads(xn, a.v_proj, None, c.num_key_value_heads, None)
            if kc is not None:
                if layer.local and kc.shape[2] > c.sliding_window_size:
                    kc = kc[:, :, -c.sliding_window_size :]
                    vc = vc[:, :, -c.sliding_window_size :]
                k = mx.concatenate([kc, kb], axis=2)
                v = mx.concatenate([vc, vb], axis=2)
            else:
                k, v = kb, vb
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
            out = out.transpose(0, 2, 1, 3).reshape(B, n, -1)
            res = _residual(x, a.o_proj(out), self.dtype)
            ff = _rms(res, layer.post_attention_layernorm.weight, c.rms_norm_eps, self.dtype)
            m = layer.mlp
            ffo = m.down_proj(nn.silu(m.gate_proj(ff)) * m.up_proj(ff))
            x = _residual(res, ffo, self.dtype)
        return _rms(x, self.norm.weight, c.rms_norm_eps, self.dtype)

    def head_feature(self, last_features: mx.array) -> mx.array:
        """The recurrent head's conditioning: the most recent committed row."""
        return self.mix(last_features, self.config.num_hidden_layers, self.markov_head.dflare_norm)

    def select_greedy(self, block_hidden, base_logits, anchor, feature, softcap: float) -> mx.array:
        """block_hidden [1, R, D] / base_logits [1, R, V] for the mask rows,
        anchor int token array [1], feature [1, D]. Returns R greedy tokens
        (lazy int32 [R]); each step conditions on the previous choice."""
        c = self.config
        h = self.markov_head
        r = c.markov_rank
        state = mx.zeros((1, r), self.dtype)
        last = anchor.reshape(1)
        feature = feature.reshape(1, -1).astype(self.dtype)
        out = []
        V = base_logits.shape[-1]
        for i in range(block_hidden.shape[1]):
            prev = h.markov_w1(last)
            gates = h.rnn_proj(mx.concatenate([state, prev.astype(self.dtype), block_hidden[:, i].astype(self.dtype), feature], axis=-1))
            keep = mx.sigmoid(gates[:, :r].astype(mx.float32))
            update = mx.tanh(gates[:, r : 2 * r].astype(mx.float32))
            latent = mx.tanh(gates[:, 2 * r :].astype(mx.float32)).astype(self.dtype)
            state = (keep * state.astype(mx.float32) + (1.0 - keep) * update).astype(self.dtype)
            corr = (latent @ self._fold_w[:V].T + self._fold_b[:V]).astype(mx.float32)
            scores = base_logits[:, i].astype(mx.float32) + corr
            if softcap:
                scores = mx.tanh(scores / softcap) * softcap
            last = mx.argmax(scores, axis=-1).astype(mx.int32)
            out.append(last)
        return mx.concatenate(out)
