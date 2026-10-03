"""The dense attention gate's numerical contract, eager and compiled alike.

MLX 0.32.2 lowers ``sigmoid`` differently in a fused (JIT-compiled) kernel
than in its standalone precompiled kernel: the JIT library resolves the
unqualified ``metal::exp`` in MLX's Sigmoid to the fast exp, the precompiled
metallib (built with -fno-fast-math) to the precise one. Inside a verify
trace the gate's sigmoid fuses with its multiply, so an eager verify forward
that ran the stock expression would disagree with the compiled verifier in
the last bit of some gated outputs, in bfloat16 and in float32 (float16 agrees
bit for bit). On decode_verify the eager forward therefore runs the same
compiled expression; prefill and plain decode keep the stock one.
"""

import mlx.core as mx

from .attention_context import current_attention_phase


@mx.compile
def _bf16_attention_gate(output: mx.array, gate: mx.array) -> mx.array:
    # Quantized projections reduce in fp32 and cast to bf16. In MLX 0.32.2,
    # fusing that cast with sigmoid differs from the standalone bf16 sigmoid
    # at some negative inputs (for example -6.84375). Use the same lowering
    # on eager forwards. The round trip is value-preserving and folds into
    # the existing cast inside an outer verify trace, preserving its result.
    # Computing sigmoid in fp32 and then casting would change that contract.
    return output * mx.sigmoid(gate.astype(mx.float32).astype(mx.bfloat16))


@mx.compile
def _f32_attention_gate(output: mx.array, gate: mx.array) -> mx.array:
    # The fused pair a float32 verify trace runs. A GEMM with TF32 inputs
    # (float32 on a Metal 4 tensor-unit GPU) rounds the one-ulp differences
    # of the stock expression away before o_proj on an M5; the M1 to M4
    # float32 kernels read every bit.
    return output * mx.sigmoid(gate)


def attention_gate(output: mx.array, gate: mx.array) -> mx.array:
    if current_attention_phase() == "decode_verify":
        if gate.dtype == mx.bfloat16:
            return _bf16_attention_gate(output, gate)
        if gate.dtype == mx.float32:
            return _f32_attention_gate(output, gate)
    return output * mx.sigmoid(gate)
