#!/usr/bin/env python3
"""Convert the ISTA GSQ-RCO Qwen3.8-Flash-Next GGUF into an MTPLX qwen4_exp model dir.

Source: ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF ``Q2_0`` (the GGUF Splash
serves on 64 GB Macs), two splits:

* ``...-00001-of-00002.gguf``: the trunk. Routed experts are Q2_0 (GGUF type
  42: 64 weights per block, one fp16 d, 2-bit codes q worth d * (q - 1)); the
  dense projections mix Q2_0, Q3_K, Q4_0, Q4_K, Q5_0, Q5_K, Q6_K, Q8_0, IQ4_XS
  and IQ4_NL per tensor (RCO allocation); hyper-connection mixers, routers,
  GDN alpha/beta and the QSA indexer are BF16.
* ``...-00002-of-00002.gguf``: the PLE n-gram table ``per_layer_token_embd``,
  320M rows of 160 values in IQ4_NL (90 bytes a row, 28.8 GB).

What the converter writes (CPU only, no GPU work):

* dense projections: dequantized to float32 and requantized to MLX affine
  8-bit g64 (the Optimized-Speed recipe's width for every projection it keeps
  at 8 bits), so every fusion group (GDN in_proj, QSA qkv, shared-expert
  gate+up) shares one packing. A Q2_0/Q4_0/Q8_0 group is reproduced up to the
  fp16 rounding of its scale and bias; K-quant sub-blocks of 16 and the IQ4
  codebooks are requantized at an error below 1/255 of the group range.
* hyper-connection mixers, routers, PLE key/value and norms: float (fp16), as
  the Optimized-Speed pack keeps them unquantized.
* routed experts: NOT copied. config.json records each expert tensor's offset
  in the GGUF and qwen4_exp builds MLX affine 2-bit g64 tensors from the
  blocks at load (`_gguf_q2_0_experts_sanitize`): the codes are already in
  MLX's 2-bit order, scale = d and bias = -d, so the load is exact and needs
  no custom kernel.
* the n-gram table: NOT copied. config.json points the sidecar reader at its
  IQ4_NL rows inside split 2; rows are decoded after the gather.
* the MTP head: the GGUF has none. Put the Optimized-Speed pack's
  ``mtp.safetensors`` (Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed) in
  the output dir; attach_mtp loads it as is.

Layout differences from the HF checkpoint are the ones the ds4 GGUF had
(pinned again 2026-11-01 against the Optimized-Speed pack): GDN value heads
in llama.cpp's tiled order (GGUF head r*16+kh is HF head kh*3+r), ssm_a =
-exp(A_log), conv weights [C, k] with value channels tiled, norms already
one-centered.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path

import numpy as np

GGML_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 6: "Q5_0", 8: "Q8_0", 11: "Q3_K",
              12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 20: "IQ4_NL", 23: "IQ4_XS",
              30: "BF16", 42: "Q2_0"}
BLOCK = {"F32": (1, 4), "F16": (1, 2), "BF16": (1, 2), "Q4_0": (32, 18),
         "Q5_0": (32, 22), "Q8_0": (32, 34), "Q3_K": (256, 110), "Q4_K": (256, 144),
         "Q5_K": (256, 176), "Q6_K": (256, 210), "IQ4_NL": (32, 18),
         "IQ4_XS": (256, 136), "Q2_0": (64, 18)}
KVALUES_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10,
                          1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32)

N_LAYER = 48
N_V_HEADS, N_K_HEADS, V_DIM = 48, 16, 128
QK_ROWS = 2 * N_K_HEADS * 128  # q and k rows of in_proj_qkv / conv channels
DENSE_BITS, DENSE_GROUP = 8, 64
EXPERT_BITS, EXPERT_GROUP = 2, 64


# ------------------------------------------------------------------ GGUF reading
class Gguf:
    """Tensors of one or more GGUF splits, by name."""

    def __init__(self, paths):
        self.meta: dict = {}
        self.tensors: dict = {}
        for path in paths:
            self._read(Path(path))

    def _read(self, path: Path):
        with open(path, "rb") as f:
            def u(fmt):
                return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]

            def s():
                return f.read(u("Q")).decode("utf-8", "replace")

            def val(t):
                m = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
                     10: "Q", 11: "q", 12: "d"}
                if t in m:
                    return u(m[t])
                if t == 8:
                    return s()
                if t == 9:
                    et, n = u("I"), u("Q")
                    return [val(et) for _ in range(n)]
                raise ValueError(t)

            assert f.read(4) == b"GGUF"
            u("I")
            nt, nk = u("Q"), u("Q")
            meta = {}
            for _ in range(nk):
                k = s()
                meta[k] = val(u("I"))
            for k, v in meta.items():
                self.meta.setdefault(k, v)
            align = meta.get("general.alignment", 32)
            entries = []
            for _ in range(nt):
                name = s()
                dims = [u("Q") for _ in range(u("I"))]
                typ = GGML_TYPES[u("I")]
                entries.append((name, dims, typ, u("Q")))
            data0 = (f.tell() + align - 1) // align * align
        for name, dims, typ, off in entries:
            be, bb = BLOCK[typ]
            self.tensors[name] = {"path": path, "dims": dims, "type": typ,
                                  "offset": data0 + off,
                                  "nbytes": int(np.prod(dims)) // be * bb}

    def raw(self, name) -> np.ndarray:
        t = self.tensors[name]
        return np.memmap(t["path"], dtype=np.uint8, mode="r", offset=t["offset"],
                         shape=(t["nbytes"],))

    def float(self, name) -> np.ndarray:
        """Any supported tensor as float32 in torch order (dims reversed)."""
        t = self.tensors[name]
        shape = t["dims"][::-1]
        return dequantize(np.asarray(self.raw(name)), t["type"]).reshape(shape)


# ------------------------------------------------------------------ dequantization
def _f16(b: np.ndarray) -> np.ndarray:
    """[..., 2] uint8 -> [...] float32."""
    return np.ascontiguousarray(b).view(np.float16)[..., 0].astype(np.float32)


def _k_scale_min(sc: np.ndarray):
    """Q4_K/Q5_K 12-byte packed 6-bit scales and mins -> ([.., 8], [.., 8])."""
    sc = sc.astype(np.uint32)
    scale = np.empty(sc.shape[:-1] + (8,), np.float32)
    mins = np.empty_like(scale)
    for j in range(8):
        if j < 4:
            scale[..., j] = sc[..., j] & 63
            mins[..., j] = sc[..., j + 4] & 63
        else:
            scale[..., j] = (sc[..., j + 4] & 0xF) | ((sc[..., j - 4] >> 6) << 4)
            mins[..., j] = (sc[..., j + 4] >> 4) | ((sc[..., j] >> 6) << 4)
    return scale, mins


def dequantize(buf: np.ndarray, typ: str) -> np.ndarray:
    """Flat GGUF tensor bytes -> flat float32 values (llama.cpp's dequantize_row_*)."""
    if typ == "F32":
        return buf.view(np.float32).copy()
    if typ == "F16":
        return buf.view(np.float16).astype(np.float32)
    if typ == "BF16":
        return (buf.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    be, bb = BLOCK[typ]
    b = buf.reshape(-1, bb)
    if typ == "Q8_0":
        return (_f16(b[:, :2])[:, None] * b[:, 2:].view(np.int8)).reshape(-1)
    if typ == "Q2_0":
        q = np.stack([(b[:, 2:] >> (2 * k)) & 3 for k in range(4)], axis=-1).reshape(-1, 64)
        return (_f16(b[:, :2])[:, None] * (q.astype(np.float32) - 1.0)).reshape(-1)
    if typ in ("Q4_0", "IQ4_NL"):
        qs = b[:, 2:18]
        q = np.concatenate([qs & 0xF, qs >> 4], axis=-1)
        v = q.astype(np.float32) - 8.0 if typ == "Q4_0" else KVALUES_IQ4NL[q]
        return (_f16(b[:, :2])[:, None] * v).reshape(-1)
    if typ == "Q5_0":
        qh = b[:, 2:6].copy().view(np.uint32)[:, 0]
        qs = b[:, 6:22]
        j = np.arange(16, dtype=np.uint32)
        h0 = ((qh[:, None] >> j) << 4) & 0x10
        h1 = (qh[:, None] >> (j + 12)) & 0x10
        q = np.concatenate([(qs & 0xF) | h0, (qs >> 4) | h1], axis=-1)
        return (_f16(b[:, :2])[:, None] * (q.astype(np.float32) - 16.0)).reshape(-1)
    if typ == "Q4_K":
        d, dmin = _f16(b[:, 0:2]), _f16(b[:, 2:4])
        scale, mins = _k_scale_min(b[:, 4:16])
        qs = b[:, 16:144].reshape(-1, 4, 32)
        q = np.stack([qs & 0xF, qs >> 4], axis=2).reshape(-1, 8, 32)
        y = d[:, None, None] * scale[..., None] * q - dmin[:, None, None] * mins[..., None]
        return y.reshape(-1)
    if typ == "Q5_K":
        d, dmin = _f16(b[:, 0:2]), _f16(b[:, 2:4])
        scale, mins = _k_scale_min(b[:, 4:16])
        qh = b[:, 16:48]
        qs = b[:, 48:176].reshape(-1, 4, 32)
        q = np.empty((b.shape[0], 8, 32), np.float32)
        for j in range(4):
            q[:, 2 * j] = (qs[:, j] & 0xF) + 16 * ((qh >> (2 * j)) & 1)
            q[:, 2 * j + 1] = (qs[:, j] >> 4) + 16 * ((qh >> (2 * j + 1)) & 1)
        y = d[:, None, None] * scale[..., None] * q - dmin[:, None, None] * mins[..., None]
        return y.reshape(-1)
    if typ == "Q6_K":
        ql = b[:, 0:128].reshape(-1, 2, 64)
        qh = b[:, 128:192].reshape(-1, 2, 32)
        sc = b[:, 192:208].view(np.int8).astype(np.float32).reshape(-1, 2, 8)
        d = _f16(b[:, 208:210])
        q = np.empty((b.shape[0], 2, 4, 32), np.float32)
        q[:, :, 0] = (ql[:, :, :32] & 0xF) | (((qh >> 0) & 3) << 4)
        q[:, :, 1] = (ql[:, :, 32:] & 0xF) | (((qh >> 2) & 3) << 4)
        q[:, :, 2] = (ql[:, :, :32] >> 4) | (((qh >> 4) & 3) << 4)
        q[:, :, 3] = (ql[:, :, 32:] >> 4) | (((qh >> 6) & 3) << 4)
        q -= 32.0
        # scale index: is = l/16, then +0/+2/+4/+6 for the four quarters
        s = sc.reshape(-1, 2, 4, 2)  # [.., quarter, half-of-32]
        y = d[:, None, None, None, None] * s[..., None] * q.reshape(-1, 2, 4, 2, 16)
        return y.reshape(-1)
    if typ == "Q3_K":
        hm = b[:, 0:32]
        qs = b[:, 32:96].reshape(-1, 2, 32)
        raw = b[:, 96:108].copy().view(np.uint32)  # [n, 3]
        d = _f16(b[:, 108:110])
        a0, a1, tmp = raw[:, 0], raw[:, 1], raw[:, 2]
        k1, k2 = np.uint32(0x03030303), np.uint32(0x0F0F0F0F)
        aux = np.stack([
            (a0 & k2) | (((tmp >> 0) & k1) << 4),
            (a1 & k2) | (((tmp >> 2) & k1) << 4),
            ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4),
            ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4),
        ], axis=-1)
        scales = aux.view(np.int8).astype(np.float32) - 32.0  # [n, 16]
        y = np.empty((b.shape[0], 2, 4, 2, 16), np.float32)
        for n in range(2):
            for j in range(4):
                m = 1 << (4 * n + j)
                for half in range(2):
                    lo = (qs[:, n, 16 * half:16 * half + 16] >> (2 * j)) & 3
                    hb = (hm[:, 16 * half:16 * half + 16] & m) != 0
                    q = lo.astype(np.float32) - np.where(hb, 0.0, 4.0)
                    y[:, n, j, half] = (d * scales[:, 8 * n + 2 * j + half])[:, None] * q
        return y.reshape(-1)
    if typ == "IQ4_XS":
        d = _f16(b[:, 0:2])
        sh = b[:, 2:4].copy().view(np.uint16)[:, 0].astype(np.uint32)
        sl = b[:, 4:8].astype(np.uint32)
        qs = b[:, 8:136].reshape(-1, 8, 16)
        ib = np.arange(8)
        ls = ((sl[:, ib // 2] >> (4 * (ib % 2))) & 0xF) | (((sh[:, None] >> (2 * ib)) & 3) << 4)
        dl = d[:, None] * (ls.astype(np.float32) - 32.0)
        q = np.concatenate([qs & 0xF, qs >> 4], axis=-1)  # [n, 8, 32]
        return (dl[..., None] * KVALUES_IQ4NL[q]).reshape(-1)
    raise ValueError(f"no dequantization for {typ}")


# ------------------------------------------------------------------ conversion
def affine(a: np.ndarray, bits=DENSE_BITS, group=DENSE_GROUP):
    import mlx.core as mx

    w, s, b = mx.quantize(mx.array(a.astype(np.float32)), group_size=group, bits=bits)
    return (np.array(w), np.array(s.astype(mx.float16)), np.array(b.astype(mx.float16)))


def v_head_perm() -> np.ndarray:
    """pack/HF value head for each GGUF head: GGUF r*16+kh -> HF kh*3+r."""
    r = N_V_HEADS // N_K_HEADS
    return np.array([(i % N_K_HEADS) * r + i // N_K_HEADS for i in range(N_V_HEADS)])


def gguf_rows_for_hf_v() -> np.ndarray:
    """Row gather that puts GGUF value channels in HF order."""
    inv = np.argsort(v_head_perm())  # HF head h lives at GGUF head inv[h]
    return (inv[:, None] * V_DIM + np.arange(V_DIM)).reshape(-1)


class Writer:
    def __init__(self, out: Path, dtype):
        self.out = out
        self.dtype = dtype
        self.cur: dict[str, np.ndarray] = {}
        self.index: dict[str, str] = {}
        self.files = 0

    def f(self, name, a: np.ndarray):
        self.cur[name] = np.ascontiguousarray(a.astype(self.dtype))

    def raw(self, name, a: np.ndarray):
        self.cur[name] = np.ascontiguousarray(a)

    def q(self, prefix, triple):
        w, s, b = triple
        self.raw(prefix + ".weight", w)
        self.raw(prefix + ".scales", s)
        self.raw(prefix + ".biases", b)

    def flush(self):
        import mlx.core as mx

        if not self.cur:
            return
        self.files += 1
        fname = f"model-{self.files:05d}.safetensors"
        mx.save_safetensors(str(self.out / fname), {k: mx.array(v) for k, v in self.cur.items()},
                            metadata={"format": "mlx"})
        for k in self.cur:
            self.index[k] = fname
        self.cur = {}


HC_BITS = 0  # 0 keeps the hyper-connection mixers fp16 (--hc-bits)


def write_hc(w: Writer, prefix: str, down, up, inject=None):
    """Hyper-connection mixer projections: fp16, or affine HC_BITS g64."""
    mats = {"input_mix_weight_down": down, "input_mix_weight_up": up}
    if inject is not None:
        mats["block_inject_weight"] = inject
    for name, a in mats.items():
        if HC_BITS:
            w.q(f"{prefix}.{name}", affine(a, bits=HC_BITS, group=DENSE_GROUP))
        else:
            w.f(f"{prefix}.{name}.weight", a)


def write_block(w: Writer, g: Gguf, il: int, prefix: str):
    B = f"blk.{il}."
    for part, hc in (("attn_hyper_connection", "hc_attn"), ("mlp_hyper_connection", "hc_ffn")):
        w.f(f"{prefix}.{part}.hc_norm.weight", g.float(B + f"{hc}_norm.weight"))
        write_hc(w, f"{prefix}.{part}", g.float(B + f"{hc}_down.weight"),
                 g.float(B + f"{hc}_up.weight"), g.float(B + f"{hc}_inject.weight"))

    if B + "attn_qkv.weight" in g.tensors:  # GatedDeltaNet
        L = f"{prefix}.linear_attn"
        vrows = gguf_rows_for_hf_v()
        qkv_rows = np.concatenate([np.arange(QK_ROWS), QK_ROWS + vrows])
        heads = np.argsort(v_head_perm())
        w.q(L + ".in_proj_qkv", affine(g.float(B + "attn_qkv.weight")[qkv_rows]))
        w.q(L + ".in_proj_z", affine(g.float(B + "attn_gate.weight")[vrows]))
        w.q(L + ".in_proj_a", affine(g.float(B + "ssm_alpha.weight")[heads]))
        w.q(L + ".in_proj_b", affine(g.float(B + "ssm_beta.weight")[heads]))
        w.q(L + ".out_proj", affine(g.float(B + "ssm_out.weight")[:, vrows]))
        w.raw(L + ".A_log", np.log(-g.float(B + "ssm_a"))[heads].astype(np.float32))
        w.f(L + ".dt_bias", g.float(B + "ssm_dt.bias")[heads])
        w.f(L + ".norm.weight", g.float(B + "ssm_norm.weight"))
        w.f(L + ".conv1d.weight", g.float(B + "ssm_conv1d.weight")[qkv_rows][..., None])
    else:  # gated GQA + QSA indexer
        A = f"{prefix}.self_attn"
        for proj, t in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"),
                        ("o_proj", "attn_output")):
            w.q(f"{A}.{proj}", affine(g.float(B + t + ".weight")))
        w.f(A + ".q_norm.weight", g.float(B + "attn_q_norm.weight"))
        w.f(A + ".k_norm.weight", g.float(B + "attn_k_norm.weight"))
        w.q(A + ".indexer.index_qk_proj", affine(np.concatenate(
            [g.float(B + "indexer.q_proj.weight"), g.float(B + "indexer.k_proj.weight")])))
        w.f(A + ".indexer.q_layernorm.weight", g.float(B + "indexer.q_norm.weight"))
        w.f(A + ".indexer.k_layernorm.weight", g.float(B + "indexer.k_norm.weight"))

    if B + "ple_key.weight" in g.tensors:
        P = f"{prefix}.ple"
        w.f(P + ".key_proj.weight", g.float(B + "ple_key.weight"))
        w.f(P + ".value_proj.weight", g.float(B + "ple_value.weight"))
        w.f(P + ".norm_key.weight", g.float(B + "ple_norm_key.weight"))
        w.f(P + ".norm_query.weight", g.float(B + "ple_norm_query.weight"))
        w.f(P + ".norm_conv.weight", g.float(B + "ple_norm_conv.weight"))
        w.f(P + ".conv_weight", g.float(B + "ple_conv1d.weight")[..., None])
        m = g.meta
        E = P + ".ple_embedding"
        w.raw(E + ".layer_multipliers", np.array(m["qwen4exp.ple.layer_multipliers"], np.int64))
        w.raw(E + ".ngram_heads_offsets", np.array(m["qwen4exp.ple.head_offsets"], np.int64))
        w.raw(E + ".ngram_heads_vocab_sizes", np.array(m["qwen4exp.ple.head_vocab_sizes"], np.int64))

    M = f"{prefix}.mlp"
    w.f(M + ".gate.weight", g.float(B + "ffn_gate_inp.weight"))
    w.f(M + ".shared_expert_gate.weight", g.float(B + "ffn_gate_inp_shexp.weight")[None, :])
    for proj, t in (("gate_proj", "ffn_gate_shexp"), ("up_proj", "ffn_up_shexp"),
                    ("down_proj", "ffn_down_shexp")):
        w.q(f"{M}.shared_expert.{proj}", affine(g.float(B + t + ".weight")))


def expert_spec(g: Gguf, il: int) -> dict:
    """GGUF location of one layer's routed experts, for the load-time builder."""
    spec = {}
    for proj, t in (("gate_proj", "ffn_gate_exps"), ("up_proj", "ffn_up_exps"),
                    ("down_proj", "ffn_down_exps")):
        info = g.tensors[f"blk.{il}.{t}.weight"]
        assert info["type"] == "Q2_0", (t, info["type"])
        cols, rows, n_exp = info["dims"]
        spec[proj] = {"path": str(Path(info["path"]).resolve()), "offset": info["offset"],
                      "experts": n_exp, "rows": rows, "cols": cols}
    return spec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", type=Path, nargs="+", required=True,
                    help="both splits (trunk and n-gram table)")
    ap.add_argument("--ref", type=Path, required=True,
                    help="dir with the Optimized-Speed pack's config.json, mtplx_runtime.json, "
                         "tokenizer and chat template")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", type=int, default=N_LAYER, help="debug: convert only the first N trunk layers")
    ap.add_argument("--hc-bits", type=int, choices=(0, 8), default=0,
                    help="hyper-connection mixers: 0 = fp16, 8 = affine 8-bit g64")
    args = ap.parse_args()
    global HC_BITS
    HC_BITS = args.hc_bits

    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    g = Gguf(args.gguf)
    assert g.meta["general.architecture"] == "qwen4exp"
    args.out.mkdir(parents=True, exist_ok=True)
    w = Writer(args.out, np.float16)

    LM = "language_model"
    w.q(f"{LM}.model.embed_tokens", affine(g.float("token_embd.weight")))
    w.q(f"{LM}.lm_head", affine(g.float("output.weight")))
    w.f(f"{LM}.model.hyper_connection_mixer.hc_norm.weight", g.float("output_hc_norm.weight"))
    write_hc(w, f"{LM}.model.hyper_connection_mixer", g.float("output_hc_down.weight"),
             g.float("output_hc_up.weight"))
    w.flush()
    experts = {}
    for il in range(args.layers):
        write_block(w, g, il, f"{LM}.model.layers.{il}")
        experts[str(il)] = expert_spec(g, il)
        if il % 4 == 3:
            w.flush()
        print(f"layer {il} done", flush=True)
    w.flush()

    json.dump({"metadata": {"total_tensors": len(w.index)}, "weight_map": w.index},
              open(args.out / "model.safetensors.index.json", "w"), indent=1)
    cfg = json.load(open(args.ref / "config.json"))
    tc = cfg["text_config"]
    tc["dtype"] = "float16"
    tc["routed_experts_gguf"] = {"format": "Q2_0", "layers": experts}
    table = g.tensors["per_layer_token_embd.weight"]
    assert table["type"] == "IQ4_NL", table["type"]
    dim, rows = table["dims"]
    tc["ngram_gguf"] = {"path": str(Path(table["path"]).resolve()), "offset": table["offset"],
                        "rows": rows, "dim": dim, "dtype": "IQ4_NL", "out_dtype": "float16"}
    if args.layers != N_LAYER:
        tc["num_hidden_layers"] = args.layers
        tc.pop("layer_types", None)
    # Unquantized modules keep the reference pack's False entries (hc mixers,
    # conv, PLE projections); routed experts are 2-bit g64; the rest 8-bit g64.
    quant = {k: v for k, v in cfg["quantization"].items()
             if v is False and not (HC_BITS and "hyper_connection" in k)}
    quant.update({"group_size": DENSE_GROUP, "bits": DENSE_BITS, "mode": "affine"})
    for il in range(args.layers):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            quant[f"{LM}.model.layers.{il}.mlp.switch_mlp.{proj}"] = {
                "group_size": EXPERT_GROUP, "bits": EXPERT_BITS, "mode": "affine"}
        for gate in ("mlp.gate", "mlp.shared_expert_gate"):
            quant[f"{LM}.model.layers.{il}.{gate}"] = False
    cfg["quantization"] = quant
    cfg["quantization_config"] = dict(quant)
    cfg["mlx_lm_extra_tensors"] = {"mtp_file": "mtp.safetensors"}
    cfg["language_model_only"] = True
    cfg.pop("vision_config", None)
    cfg["mtplx_recipe"] = {"source": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF Q2_0",
                           "gguf": [p.name for p in args.gguf],
                           "experts": "Q2_0 read from the GGUF at load as affine 2-bit g64 (exact)",
                           "dense": "dequantized -> affine 8-bit g64",
                           "ngram": "IQ4_NL rows read from the GGUF",
                           "mtp": "Optimized-Speed pack mtp.safetensors", "dtype": "float16"}
    json.dump(cfg, open(args.out / "config.json", "w"), indent=1)
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
              "chat_template.jinja", "generation_config.json", "LICENSE"):
        if (args.ref / f).exists():
            shutil.copy(args.ref / f, args.out / f)
    runtime = json.load(open(args.ref / "mtplx_runtime.json"))
    runtime["public_model_id"] = "mtplx-flash-next-ista-q2_0"
    runtime["artifact_role"] = "converted-local"
    runtime.pop("speed_evidence", None)
    runtime["mtp_depth_default"] = 2
    json.dump(runtime, open(args.out / "mtplx_runtime.json", "w"), indent=2)
    if not (args.out / "mtp.safetensors").exists():
        print("NOTE: copy the Optimized-Speed pack's mtp.safetensors into the output dir")
    print(f"wrote {w.files} shards to {args.out}")


if __name__ == "__main__":
    main()
