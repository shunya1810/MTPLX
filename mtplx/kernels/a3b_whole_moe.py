"""Fixed Metal entrypoints for exact A3B whole-MoE small-row stages."""

from __future__ import annotations

from typing import Any

import mlx.core as mx


HIDDEN = 2048
EXPERTS = 256
TOP_K = 8
INTERMEDIATE = 512
ACTIVATION_SLOTS = 9
STAGE1_THREADS = 256
STAGE1_PROJECTION_THREADGROUPS = EXPERTS // (8 * 4)
TILED_THREADS = 128
STAGE2_THREADGROUPS = ACTIVATION_SLOTS * (INTERMEDIATE // 16)
STAGE3_THREADGROUPS = HIDDEN // 16

_KERNELS: dict[str, Any] = {}


def _source_preamble(*, rows: int) -> str:
    return f"""
        using namespace metal;

        constexpr uint HIDDEN = {HIDDEN};
        constexpr uint EXPERTS = {EXPERTS};
        constexpr uint TOP_K = {TOP_K};
        constexpr uint INTERMEDIATE = {INTERMEDIATE};
        constexpr uint ACTIVATION_SLOTS = {ACTIVATION_SLOTS};
        constexpr uint ROWS = {rows};
    """


def _fixed_source(*, stage: int, rows: int, variant: str) -> str:
    common = _source_preamble(rows=rows)
    if stage == 1:
        return common + _stage1_source(
            target=variant.startswith("target"),
            rows=rows,
        )
    if stage == 2:
        return common + _stage2_source(
            target=variant.startswith("target"),
            rows=rows,
        )
    return common + _stage3_source(
        target=variant.startswith("target"),
        rows=rows,
    )


def _stage1_source(*, target: bool, rows: int) -> str:
    if target and rows in (2, 3):
        # Verify routes (2-row K1, 3-row k=2) share the split projection +
        # finalizer path; both sources loop over ROWS internally.
        return _target_m2_stage1_projection_source()
    projection = _target_stage1_projection() if target else _mtp_stage1_projection()
    prologue = """
        uint tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint row = threadgroup_position_in_grid.x;

        threadgroup bfloat router_logits[ROWS * EXPERTS];
        threadgroup bfloat probabilities[ROWS * EXPERTS];
        threadgroup float simd_max_values[8];
        threadgroup float simd_sum_values[8];
        threadgroup float local_probabilities[64];
        threadgroup int local_indices[64];
        threadgroup float merged_probabilities[TOP_K];
        threadgroup int merged_indices[TOP_K];
    """
    finalize = """
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float local_logit = float(router_logits[row * EXPERTS + tid]);
        // Every simdgroup reduces the eight partials itself (same lanes,
        // same order, so the same value); the old broadcast through
        // simd_values[0] raced with simdgroups still reading partial 0.
        float local_maximum = simd_max(local_logit);
        if (lane == 0) {
            simd_max_values[simd_gid] = local_maximum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float maximum_candidate = lane < 8 ? simd_max_values[lane] : -INFINITY;
        float row_maximum = simd_max(maximum_candidate);

        float probability = metal::exp(local_logit - row_maximum);
        float local_sum = simd_sum(probability);
        if (lane == 0) {
            simd_sum_values[simd_gid] = local_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sum_candidate = lane < 8 ? simd_sum_values[lane] : 0.0f;
        float row_sum = simd_sum(sum_candidate);
        probabilities[row * EXPERTS + tid] = bfloat(
            probability / row_sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float candidate_probability = float(
            probabilities[row * EXPERTS + tid]);
        int candidate_index = int(tid);
        for (int rank = 0; rank < int(TOP_K); ++rank) {
            float winner_probability = simd_max(candidate_probability);
            float winner_index_value = simd_max(
                candidate_probability == winner_probability
                    ? float(candidate_index)
                    : -1.0f);
            int winner_index = int(winner_index_value);
            if (lane == 0) {
                int destination = int(simd_gid * TOP_K) + rank;
                local_probabilities[destination] = winner_probability;
                local_indices[destination] = winner_index;
            }
            if (candidate_index == winner_index) {
                candidate_probability = -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_gid == 0) {
            int slot0 = int(lane);
            int slot1 = int(lane) + 32;
            float probability0 = local_probabilities[slot0];
            float probability1 = local_probabilities[slot1];
            int index0 = local_indices[slot0];
            int index1 = local_indices[slot1];

            for (int rank = 0; rank < int(TOP_K); ++rank) {
                bool take1 = probability1 > probability0
                    || (probability1 == probability0 && index1 > index0);
                float lane_probability = take1 ? probability1 : probability0;
                int lane_index = take1 ? index1 : index0;
                float winner_probability = simd_max(lane_probability);
                float winner_index_value = simd_max(
                    lane_probability == winner_probability
                        ? float(lane_index)
                        : -1.0f);
                int winner_index = int(winner_index_value);
                if (lane == 0) {
                    merged_probabilities[rank] = winner_probability;
                    merged_indices[rank] = winner_index;
                }
                if (lane_index == winner_index) {
                    if (take1) {
                        probability1 = -INFINITY;
                    } else {
                        probability0 = -INFINITY;
                    }
                }
            }

            if (lane == 0) {
                bfloat rounded_denominator = bfloat(0.0f);
                for (int output_rank = 0; output_rank < int(TOP_K); ++output_rank) {
                    rounded_denominator = bfloat(
                        float(rounded_denominator)
                        + merged_probabilities[TOP_K - 1 - output_rank]);
                }
                for (int output_rank = 0; output_rank < int(TOP_K); ++output_rank) {
                    int source_rank = int(TOP_K) - 1 - output_rank;
                    int destination = int(row * TOP_K) + output_rank;
                    expert_ids[destination] = uint(merged_indices[source_rank]);
                    route_scores[destination] = bfloat(
                        merged_probabilities[source_rank]
                        / float(rounded_denominator));
                }
            }
        }
    """
    return prologue + projection + finalize


def _target_m2_stage1_projection_source() -> str:
    return """
        constexpr uint ROUTER_GROUP = 64;
        constexpr uint Q8_VALUES_PER_LANE = 8;
        constexpr uint Q8_BLOCK = Q8_VALUES_PER_LANE * 32;
        constexpr uint OUTPUT_EXPERTS_PER_GROUP = 8 * 4;

        uint lane = thread_index_in_simdgroup;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint expert_tile = threadgroup_position_in_grid.x;

        // qdot8_affine: each output-column threadgroup owns 32 experts and
        // both exact M2 rows, so every router weight feeds both row results.
        {
            const device uchar* weights =
                reinterpret_cast<const device uchar*>(router_weight);
            float router_result[ROWS][4];
            for (uint row = 0; row < ROWS; ++row) {
                for (uint result_index = 0; result_index < 4; ++result_index) {
                    router_result[row][result_index] = 0.0f;
                }
            }
            uint expert_base = expert_tile * OUTPUT_EXPERTS_PER_GROUP
                + simd_gid * 4;
            for (uint k_block = 0; k_block < HIDDEN; k_block += Q8_BLOCK) {
                uint k_lane = k_block + lane * Q8_VALUES_PER_LANE;
                float input_values[ROWS][Q8_VALUES_PER_LANE];
                float input_sum[ROWS] = {};
                for (uint row = 0; row < ROWS; ++row) {
                    for (uint item = 0; item < Q8_VALUES_PER_LANE; ++item) {
                        float input_value = float(
                            value[row * HIDDEN + k_lane + item]);
                        input_values[row][item] = input_value;
                        input_sum[row] += input_value;
                    }
                }
                for (uint result_index = 0; result_index < 4; ++result_index) {
                    uint expert = expert_base + result_index;
                    uint weight_base = expert * HIDDEN + k_lane;
                    uint metadata_index = expert * (HIDDEN / ROUTER_GROUP)
                        + k_lane / ROUTER_GROUP;
                    float scale = float(router_scales[metadata_index]);
                    float bias = float(router_biases[metadata_index]);
                    float quantized_dot[ROWS] = {};
                    for (uint item = 0; item < Q8_VALUES_PER_LANE; ++item) {
                        uchar packed_weight = weights[weight_base + item];
                        for (uint row = 0; row < ROWS; ++row) {
                            quantized_dot[row] += input_values[row][item]
                                * float(packed_weight);
                        }
                    }
                    for (uint row = 0; row < ROWS; ++row) {
                        router_result[row][result_index] +=
                            scale * quantized_dot[row] + input_sum[row] * bias;
                    }
                }
            }
            for (uint row = 0; row < ROWS; ++row) {
                for (uint result_index = 0; result_index < 4; ++result_index) {
                    float reduced = simd_sum(router_result[row][result_index]);
                    if (lane == 0) {
                        uint expert = expert_base + result_index;
                        router_logits[row * EXPERTS + expert] = bfloat(reduced);
                    }
                }
            }
        }

        // The shared scalar projection has the same weights for both rows.
        // Keep both partials live and consume each q8 byte once.
        {
            const device uchar* weights =
                reinterpret_cast<const device uchar*>(shared_gate_weight);
            float shared_partial[ROWS] = {};
            if (expert_tile == 0 && simd_gid == 0) {
                for (uint k_block = 0; k_block < HIDDEN; k_block += Q8_BLOCK) {
                    uint k_lane = k_block + lane * Q8_VALUES_PER_LANE;
                    float input_sum[ROWS] = {};
                    float quantized_dot[ROWS] = {};
                    uint metadata_index = k_lane / ROUTER_GROUP;
                    float scale = float(shared_gate_scales[metadata_index]);
                    float bias = float(shared_gate_biases[metadata_index]);
                    for (uint item = 0; item < Q8_VALUES_PER_LANE; ++item) {
                        uchar packed_weight = weights[k_lane + item];
                        for (uint row = 0; row < ROWS; ++row) {
                            float input_value = float(
                                value[row * HIDDEN + k_lane + item]);
                            input_sum[row] += input_value;
                            quantized_dot[row] += input_value * float(packed_weight);
                        }
                    }
                    for (uint row = 0; row < ROWS; ++row) {
                        shared_partial[row] +=
                            scale * quantized_dot[row] + input_sum[row] * bias;
                    }
                }
                for (uint row = 0; row < ROWS; ++row) {
                    float shared_reduced = simd_sum(shared_partial[row]);
                    if (lane == 0) {
                        shared_gate[row] = bfloat(shared_reduced);
                    }
                }
            }
        }

    """


def _target_m2_stage1_finalizer_source() -> str:
    return """
        uint tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint simd_gid = simdgroup_index_in_threadgroup;

        threadgroup bfloat probabilities[ROWS * EXPERTS];
        threadgroup float simd_max_values[8];
        threadgroup float simd_sum_values[8];
        threadgroup float local_probabilities[64];
        threadgroup int local_indices[64];
        threadgroup float merged_probabilities[TOP_K];
        threadgroup int merged_indices[TOP_K];

        // The top-k scratch is reused sequentially by the two rows. The loop is
        // uniform across the threadgroup, so the row-boundary barrier is safe.
        for (uint row = 0; row < ROWS; ++row) {
            float local_logit = float(router_logits[row * EXPERTS + tid]);
            // Every simdgroup reduces the eight partials itself (same lanes,
            // same order, so the same value); the old broadcast through
            // simd_values[0] raced with simdgroups still reading partial 0.
            float local_maximum = simd_max(local_logit);
            if (lane == 0) {
                simd_max_values[simd_gid] = local_maximum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float maximum_candidate = lane < 8 ? simd_max_values[lane] : -INFINITY;
            float row_maximum = simd_max(maximum_candidate);

            float probability = metal::exp(local_logit - row_maximum);
            float local_sum = simd_sum(probability);
            if (lane == 0) {
                simd_sum_values[simd_gid] = local_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float sum_candidate = lane < 8 ? simd_sum_values[lane] : 0.0f;
            float row_sum = simd_sum(sum_candidate);
            probabilities[row * EXPERTS + tid] = bfloat(
                probability / row_sum);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float candidate_probability = float(
                probabilities[row * EXPERTS + tid]);
            int candidate_index = int(tid);
            for (int rank = 0; rank < int(TOP_K); ++rank) {
                float winner_probability = simd_max(candidate_probability);
                float winner_index_value = simd_max(
                    candidate_probability == winner_probability
                        ? float(candidate_index)
                        : -1.0f);
                int winner_index = int(winner_index_value);
                if (lane == 0) {
                    int destination = int(simd_gid * TOP_K) + rank;
                    local_probabilities[destination] = winner_probability;
                    local_indices[destination] = winner_index;
                }
                if (candidate_index == winner_index) {
                    candidate_probability = -INFINITY;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (simd_gid == 0) {
                int slot0 = int(lane);
                int slot1 = int(lane) + 32;
                float probability0 = local_probabilities[slot0];
                float probability1 = local_probabilities[slot1];
                int index0 = local_indices[slot0];
                int index1 = local_indices[slot1];

                for (int rank = 0; rank < int(TOP_K); ++rank) {
                    bool take1 = probability1 > probability0
                        || (probability1 == probability0 && index1 > index0);
                    float lane_probability = take1 ? probability1 : probability0;
                    int lane_index = take1 ? index1 : index0;
                    float winner_probability = simd_max(lane_probability);
                    float winner_index_value = simd_max(
                        lane_probability == winner_probability
                            ? float(lane_index)
                            : -1.0f);
                    int winner_index = int(winner_index_value);
                    if (lane == 0) {
                        merged_probabilities[rank] = winner_probability;
                        merged_indices[rank] = winner_index;
                    }
                    if (lane_index == winner_index) {
                        if (take1) {
                            probability1 = -INFINITY;
                        } else {
                            probability0 = -INFINITY;
                        }
                    }
                }

                if (lane == 0) {
                    bfloat rounded_denominator = bfloat(0.0f);
                    for (int output_rank = 0; output_rank < int(TOP_K); ++output_rank) {
                        rounded_denominator = bfloat(
                            float(rounded_denominator)
                            + merged_probabilities[TOP_K - 1 - output_rank]);
                    }
                    for (int output_rank = 0; output_rank < int(TOP_K); ++output_rank) {
                        int source_rank = int(TOP_K) - 1 - output_rank;
                        int destination = int(row * TOP_K) + output_rank;
                        expert_ids[destination] = uint(merged_indices[source_rank]);
                        route_scores[destination] = bfloat(
                            merged_probabilities[source_rank]
                            / float(rounded_denominator));
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    """


def _target_stage1_projection() -> str:
    return """
        constexpr uint ROUTER_GROUP = 64;
        constexpr uint Q8_VALUES_PER_LANE = 8;
        constexpr uint Q8_BLOCK = Q8_VALUES_PER_LANE * 32;

        // qdot8_affine: preserve the accepted q8 QMV lane decomposition.
        float router_result[4];
        for (uint subtile = 0; subtile < 8; ++subtile) {
            for (uint result_index = 0; result_index < 4; ++result_index) {
                router_result[result_index] = 0.0f;
            }
            uint expert_base = subtile * 32 + simd_gid * 4;
            for (uint k_block = 0; k_block < HIDDEN; k_block += Q8_BLOCK) {
                uint k_lane = k_block + lane * Q8_VALUES_PER_LANE;
                float input_values[Q8_VALUES_PER_LANE];
                float input_sum = 0.0f;
                for (uint item = 0; item < Q8_VALUES_PER_LANE; ++item) {
                    float input_value = float(value[row * HIDDEN + k_lane + item]);
                    input_values[item] = input_value;
                    input_sum += input_value;
                }
                for (uint result_index = 0; result_index < 4; ++result_index) {
                    uint expert = expert_base + result_index;
                    uint weight_base = expert * HIDDEN + k_lane;
                    uint metadata_index = expert * (HIDDEN / ROUTER_GROUP)
                        + k_lane / ROUTER_GROUP;
                    float scale = float(router_scales[metadata_index]);
                    float bias = float(router_biases[metadata_index]);
                    float quantized_dot = 0.0f;
                    const device uchar* weights =
                        reinterpret_cast<const device uchar*>(router_weight);
                    for (uint item = 0; item < Q8_VALUES_PER_LANE; ++item) {
                        quantized_dot += input_values[item]
                            * float(weights[weight_base + item]);
                    }
                    router_result[result_index] +=
                        scale * quantized_dot + input_sum * bias;
                }
            }
            for (uint result_index = 0; result_index < 4; ++result_index) {
                float reduced = simd_sum(router_result[result_index]);
                if (lane == 0) {
                    uint expert = expert_base + result_index;
                    router_logits[row * EXPERTS + expert] = bfloat(reduced);
                }
            }
        }

        float shared_partial = 0.0f;
        if (simd_gid == 0) {
            for (uint k_block = 0; k_block < HIDDEN; k_block += Q8_BLOCK) {
                uint k_lane = k_block + lane * Q8_VALUES_PER_LANE;
                float input_sum = 0.0f;
                float quantized_dot = 0.0f;
                uint metadata_index = k_lane / ROUTER_GROUP;
                float scale = float(shared_gate_scales[metadata_index]);
                float bias = float(shared_gate_biases[metadata_index]);
                const device uchar* weights =
                    reinterpret_cast<const device uchar*>(shared_gate_weight);
                for (uint item = 0; item < Q8_VALUES_PER_LANE; ++item) {
                    float input_value = float(value[row * HIDDEN + k_lane + item]);
                    input_sum += input_value;
                    quantized_dot += input_value * float(weights[k_lane + item]);
                }
                shared_partial += scale * quantized_dot + input_sum * bias;
            }
            float shared_reduced = simd_sum(shared_partial);
            if (lane == 0) {
                shared_gate[row] = bfloat(shared_reduced);
            }
        }

    """


def _mtp_stage1_projection() -> str:
    return """
        // dense_bf16_dot: one SIMDgroup owns four router outputs at a time.
        float router_result[4];
        for (uint subtile = 0; subtile < 8; ++subtile) {
            for (uint result_index = 0; result_index < 4; ++result_index) {
                router_result[result_index] = 0.0f;
            }
            uint expert_base = subtile * 32 + simd_gid * 4;
            for (uint k_lane = lane; k_lane < HIDDEN; k_lane += 32) {
                float input_value = float(value[row * HIDDEN + k_lane]);
                for (uint result_index = 0; result_index < 4; ++result_index) {
                    uint expert = expert_base + result_index;
                    router_result[result_index] += input_value
                        * float(router_weight[expert * HIDDEN + k_lane]);
                }
            }
            for (uint result_index = 0; result_index < 4; ++result_index) {
                float reduced = simd_sum(router_result[result_index]);
                if (lane == 0) {
                    uint expert = expert_base + result_index;
                    router_logits[row * EXPERTS + expert] = bfloat(reduced);
                }
            }
        }

        float shared_partial = 0.0f;
        if (simd_gid == 0) {
            for (uint k_lane = lane; k_lane < HIDDEN; k_lane += 32) {
                shared_partial += float(value[row * HIDDEN + k_lane])
                    * float(shared_gate_weight[k_lane]);
            }
            float shared_reduced = simd_sum(shared_partial);
            if (lane == 0) {
                shared_gate[row] = bfloat(shared_reduced);
            }
        }

    """


def _stage2_source(*, target: bool, rows: int) -> str:
    if target:
        if rows == 2:
            return _target_m2_stage2_source()
        return _target_stage2_source()
    return _mtp_stage2_source()


def _target_stage2_source() -> str:
    return """
        constexpr uint ROUTED_GROUP = 64;
        constexpr uint VALUES_PER_LANE = 16;
        constexpr uint K_BLOCK = VALUES_PER_LANE * 32;

        uint group = threadgroup_position_in_grid.x;
        uint slot = group / 32;
        uint tile = group - slot * 32;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint output_base = tile * 16 + simd_gid * 4;

        for (uint row = 0; row < ROWS; ++row) {
            uint expert = slot < TOP_K
                ? expert_ids[row * TOP_K + slot]
                : uint(0);
            const device uint* gate_words = slot < TOP_K
                ? routed_gate_up_weight
                    + expert * 2 * INTERMEDIATE * (HIDDEN / 8)
                : shared_gate_up_weight;
            const device uint* up_words = gate_words
                + INTERMEDIATE * (HIDDEN / 8);
            const device bfloat* gate_scale_values = slot < TOP_K
                ? routed_gate_up_scales
                    + expert * 2 * INTERMEDIATE * (HIDDEN / ROUTED_GROUP)
                : shared_gate_up_scales;
            const device bfloat* gate_bias_values = slot < TOP_K
                ? routed_gate_up_biases
                    + expert * 2 * INTERMEDIATE * (HIDDEN / ROUTED_GROUP)
                : shared_gate_up_biases;
            const device bfloat* up_scale_values = gate_scale_values
                + INTERMEDIATE * (HIDDEN / ROUTED_GROUP);
            const device bfloat* up_bias_values = gate_bias_values
                + INTERMEDIATE * (HIDDEN / ROUTED_GROUP);
            const device uchar* gate_bytes =
                reinterpret_cast<const device uchar*>(gate_words);
            const device uchar* up_bytes =
                reinterpret_cast<const device uchar*>(up_words);

            float gate_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float up_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint k_block = 0; k_block < HIDDEN; k_block += K_BLOCK) {
                uint k_lane = k_block + lane * VALUES_PER_LANE;
                float input_values[VALUES_PER_LANE];
                float input_sum = 0.0f;
                for (uint item = 0; item < VALUES_PER_LANE; item += 4) {
                    float x0 = float(value[row * HIDDEN + k_lane + item]);
                    float x1 = float(value[row * HIDDEN + k_lane + item + 1]);
                    float x2 = float(value[row * HIDDEN + k_lane + item + 2]);
                    float x3 = float(value[row * HIDDEN + k_lane + item + 3]);
                    input_sum += x0 + x1 + x2 + x3;
                    input_values[item] = x0;
                    input_values[item + 1] = x1 / 16.0f;
                    input_values[item + 2] = x2 / 256.0f;
                    input_values[item + 3] = x3 / 4096.0f;
                }
                for (uint result_index = 0; result_index < 4; ++result_index) {
                    uint output_column = output_base + result_index;
                    uint weight_offset = output_column * (HIDDEN / 2) + k_lane / 2;
                    const device ushort* gate_packed =
                        reinterpret_cast<const device ushort*>(
                            gate_bytes + weight_offset);
                    const device ushort* up_packed =
                        reinterpret_cast<const device ushort*>(
                            up_bytes + weight_offset);
                    float gate_quantized_dot = 0.0f;
                    float up_quantized_dot = 0.0f;
                    // qdot4_affine
                    for (uint piece = 0; piece < VALUES_PER_LANE / 4; ++piece) {
                        ushort gate_bits = gate_packed[piece];
                        ushort up_bits = up_packed[piece];
                        uint item = piece * 4;
                        gate_quantized_dot +=
                            input_values[item] * float(gate_bits & 0x000f)
                            + input_values[item + 1] * float(gate_bits & 0x00f0)
                            + input_values[item + 2] * float(gate_bits & 0x0f00)
                            + input_values[item + 3] * float(gate_bits & 0xf000);
                        up_quantized_dot +=
                            input_values[item] * float(up_bits & 0x000f)
                            + input_values[item + 1] * float(up_bits & 0x00f0)
                            + input_values[item + 2] * float(up_bits & 0x0f00)
                            + input_values[item + 3] * float(up_bits & 0xf000);
                    }
                    uint metadata_index = output_column * (HIDDEN / ROUTED_GROUP)
                        + k_lane / ROUTED_GROUP;
                    gate_result[result_index] +=
                        float(gate_scale_values[metadata_index]) * gate_quantized_dot
                        + input_sum * float(gate_bias_values[metadata_index]);
                    up_result[result_index] +=
                        float(up_scale_values[metadata_index]) * up_quantized_dot
                        + input_sum * float(up_bias_values[metadata_index]);
                }
            }

            for (uint result_index = 0; result_index < 4; ++result_index) {
                float gate_sum = simd_sum(gate_result[result_index]);
                float up_sum = simd_sum(up_result[result_index]);
                if (lane == 0) {
                    bfloat gate_value = bfloat(gate_sum);
                    bfloat up_value = bfloat(up_sum);
                    auto sigmoid_y = 1 / (
                        1 + metal::exp(metal::abs(gate_value)));
                    bfloat sigmoid_mlx_exact = gate_value < bfloat(0.0f)
                        ? bfloat(sigmoid_y)
                        : bfloat(1 - sigmoid_y);
                    bfloat silu = bfloat(gate_value * sigmoid_mlx_exact);
                    uint output_column = output_base + result_index;
                    uint output_index =
                        (row * ACTIVATION_SLOTS + slot) * INTERMEDIATE
                        + output_column;
                    activations[output_index] = bfloat(silu * up_value);
                }
            }
        }
    """


def _target_m2_stage2_source() -> str:
    # Activations stay unscaled in 16-bit registers and take their 1/16,
    # 1/256, 1/4096 nibble scale in fp32 at use: power-of-two scaling is
    # exact, so every product equals the fp32 pre-scaled one, with half the
    # registers (A3B verify forward -0.3 ms, outputs bit-identical).
    return """
        constexpr uint ROUTED_GROUP = 64;
        constexpr uint VALUES_PER_LANE = 16;
        constexpr uint K_BLOCK = VALUES_PER_LANE * 32;

        uint group = threadgroup_position_in_grid.x;
        uint slot = group / 32;
        uint tile = group - slot * 32;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint output_base = tile * 16 + simd_gid * 4;

        if (slot < TOP_K) {
            // row-specific selected expert path
            for (uint row = 0; row < ROWS; ++row) {
                uint expert = expert_ids[row * TOP_K + slot];
                const device uint* gate_words = routed_gate_up_weight
                    + expert * 2 * INTERMEDIATE * (HIDDEN / 8);
                const device uint* up_words = gate_words
                    + INTERMEDIATE * (HIDDEN / 8);
                const device bfloat* gate_scale_values = routed_gate_up_scales
                    + expert * 2 * INTERMEDIATE * (HIDDEN / ROUTED_GROUP);
                const device bfloat* gate_bias_values = routed_gate_up_biases
                    + expert * 2 * INTERMEDIATE * (HIDDEN / ROUTED_GROUP);
                const device bfloat* up_scale_values = gate_scale_values
                    + INTERMEDIATE * (HIDDEN / ROUTED_GROUP);
                const device bfloat* up_bias_values = gate_bias_values
                    + INTERMEDIATE * (HIDDEN / ROUTED_GROUP);
                const device uchar* gate_bytes =
                    reinterpret_cast<const device uchar*>(gate_words);
                const device uchar* up_bytes =
                    reinterpret_cast<const device uchar*>(up_words);

                float gate_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                float up_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (uint k_block = 0; k_block < HIDDEN; k_block += K_BLOCK) {
                    uint k_lane = k_block + lane * VALUES_PER_LANE;
                    bfloat input_values[VALUES_PER_LANE];
                    float input_sum = 0.0f;
                    for (uint item = 0; item < VALUES_PER_LANE; item += 4) {
                        float x0 = float(value[row * HIDDEN + k_lane + item]);
                        float x1 = float(value[row * HIDDEN + k_lane + item + 1]);
                        float x2 = float(value[row * HIDDEN + k_lane + item + 2]);
                        float x3 = float(value[row * HIDDEN + k_lane + item + 3]);
                        input_sum += x0 + x1 + x2 + x3;
                        input_values[item] = x0;
                        input_values[item + 1] = x1;
                        input_values[item + 2] = x2;
                        input_values[item + 3] = x3;
                    }
                    for (uint result_index = 0; result_index < 4; ++result_index) {
                        uint output_column = output_base + result_index;
                        uint weight_offset =
                            output_column * (HIDDEN / 2) + k_lane / 2;
                        const device ushort* gate_packed =
                            reinterpret_cast<const device ushort*>(
                                gate_bytes + weight_offset);
                        const device ushort* up_packed =
                            reinterpret_cast<const device ushort*>(
                                up_bytes + weight_offset);
                        float gate_quantized_dot = 0.0f;
                        float up_quantized_dot = 0.0f;
                        // qdot4_affine
                        for (uint piece = 0;
                             piece < VALUES_PER_LANE / 4;
                             ++piece) {
                            ushort gate_bits = gate_packed[piece];
                            ushort up_bits = up_packed[piece];
                            uint item = piece * 4;
                            gate_quantized_dot +=
                                float(input_values[item]) * float(gate_bits & 0x000f)
                                + (float(input_values[item + 1]) / 16.0f) * float(gate_bits & 0x00f0)
                                + (float(input_values[item + 2]) / 256.0f) * float(gate_bits & 0x0f00)
                                + (float(input_values[item + 3]) / 4096.0f) * float(gate_bits & 0xf000);
                            up_quantized_dot +=
                                float(input_values[item]) * float(up_bits & 0x000f)
                                + (float(input_values[item + 1]) / 16.0f) * float(up_bits & 0x00f0)
                                + (float(input_values[item + 2]) / 256.0f) * float(up_bits & 0x0f00)
                                + (float(input_values[item + 3]) / 4096.0f) * float(up_bits & 0xf000);
                        }
                        uint metadata_index =
                            output_column * (HIDDEN / ROUTED_GROUP)
                            + k_lane / ROUTED_GROUP;
                        gate_result[result_index] +=
                            float(gate_scale_values[metadata_index])
                                * gate_quantized_dot
                            + input_sum * float(gate_bias_values[metadata_index]);
                        up_result[result_index] +=
                            float(up_scale_values[metadata_index])
                                * up_quantized_dot
                            + input_sum * float(up_bias_values[metadata_index]);
                    }
                }

                for (uint result_index = 0; result_index < 4; ++result_index) {
                    float gate_sum = simd_sum(gate_result[result_index]);
                    float up_sum = simd_sum(up_result[result_index]);
                    if (lane == 0) {
                        bfloat gate_value = bfloat(gate_sum);
                        bfloat up_value = bfloat(up_sum);
                        auto sigmoid_y = 1 / (
                            1 + metal::exp(metal::abs(gate_value)));
                        bfloat sigmoid_mlx_exact = gate_value < bfloat(0.0f)
                            ? bfloat(sigmoid_y)
                            : bfloat(1 - sigmoid_y);
                        bfloat silu = bfloat(gate_value * sigmoid_mlx_exact);
                        uint output_column = output_base + result_index;
                        uint output_index =
                            (row * ACTIVATION_SLOTS + slot) * INTERMEDIATE
                            + output_column;
                        activations[output_index] = bfloat(silu * up_value);
                    }
                }
            }
        } else {
            // row-paired fixed shared expert path
            const device uint* shared_up_words = shared_gate_up_weight
                + INTERMEDIATE * (HIDDEN / 8);
            const device bfloat* shared_up_scale_values = shared_gate_up_scales
                + INTERMEDIATE * (HIDDEN / ROUTED_GROUP);
            const device bfloat* shared_up_bias_values = shared_gate_up_biases
                + INTERMEDIATE * (HIDDEN / ROUTED_GROUP);
            const device uchar* shared_gate_bytes =
                reinterpret_cast<const device uchar*>(shared_gate_up_weight);
            const device uchar* shared_up_bytes =
                reinterpret_cast<const device uchar*>(shared_up_words);

            float shared_gate_result[ROWS][4];
            float shared_up_result[ROWS][4];
            for (uint row = 0; row < ROWS; ++row) {
                for (uint result_index = 0; result_index < 4; ++result_index) {
                    shared_gate_result[row][result_index] = 0.0f;
                    shared_up_result[row][result_index] = 0.0f;
                }
            }

            for (uint k_block = 0; k_block < HIDDEN; k_block += K_BLOCK) {
                uint k_lane = k_block + lane * VALUES_PER_LANE;
                bfloat input_values[ROWS][VALUES_PER_LANE];
                float input_sum[ROWS] = {};
                for (uint row = 0; row < ROWS; ++row) {
                    for (uint item = 0; item < VALUES_PER_LANE; item += 4) {
                        float x0 = float(value[row * HIDDEN + k_lane + item]);
                        float x1 = float(value[row * HIDDEN + k_lane + item + 1]);
                        float x2 = float(value[row * HIDDEN + k_lane + item + 2]);
                        float x3 = float(value[row * HIDDEN + k_lane + item + 3]);
                        input_sum[row] += x0 + x1 + x2 + x3;
                        input_values[row][item] = x0;
                        input_values[row][item + 1] = x1;
                        input_values[row][item + 2] = x2;
                        input_values[row][item + 3] = x3;
                    }
                }
                for (uint result_index = 0; result_index < 4; ++result_index) {
                    uint output_column = output_base + result_index;
                    uint weight_offset =
                        output_column * (HIDDEN / 2) + k_lane / 2;
                    const device ushort* shared_gate_packed =
                        reinterpret_cast<const device ushort*>(
                            shared_gate_bytes + weight_offset);
                    const device ushort* shared_up_packed =
                        reinterpret_cast<const device ushort*>(
                            shared_up_bytes + weight_offset);
                    float gate_quantized_dot[ROWS] = {};
                    float up_quantized_dot[ROWS] = {};
                    // qdot4_affine shared one-read M2
                    for (uint piece = 0;
                         piece < VALUES_PER_LANE / 4;
                         ++piece) {
                        ushort shared_gate_bits = shared_gate_packed[piece];
                        ushort shared_up_bits = shared_up_packed[piece];
                        uint item = piece * 4;
                        for (uint row = 0; row < ROWS; ++row) {
                            gate_quantized_dot[row] +=
                                input_values[row][item]
                                    * float(shared_gate_bits & 0x000f)
                                + (float(input_values[row][item + 1]) / 16.0f) * float(shared_gate_bits & 0x00f0)
                                + (float(input_values[row][item + 2]) / 256.0f) * float(shared_gate_bits & 0x0f00)
                                + (float(input_values[row][item + 3]) / 4096.0f) * float(shared_gate_bits & 0xf000);
                            up_quantized_dot[row] +=
                                input_values[row][item]
                                    * float(shared_up_bits & 0x000f)
                                + (float(input_values[row][item + 1]) / 16.0f) * float(shared_up_bits & 0x00f0)
                                + (float(input_values[row][item + 2]) / 256.0f) * float(shared_up_bits & 0x0f00)
                                + (float(input_values[row][item + 3]) / 4096.0f) * float(shared_up_bits & 0xf000);
                        }
                    }
                    uint metadata_index =
                        output_column * (HIDDEN / ROUTED_GROUP)
                        + k_lane / ROUTED_GROUP;
                    float gate_scale = float(
                        shared_gate_up_scales[metadata_index]);
                    float gate_bias = float(
                        shared_gate_up_biases[metadata_index]);
                    float up_scale = float(
                        shared_up_scale_values[metadata_index]);
                    float up_bias = float(
                        shared_up_bias_values[metadata_index]);
                    for (uint row = 0; row < ROWS; ++row) {
                        shared_gate_result[row][result_index] +=
                            gate_scale * gate_quantized_dot[row]
                            + input_sum[row] * gate_bias;
                        shared_up_result[row][result_index] +=
                            up_scale * up_quantized_dot[row]
                            + input_sum[row] * up_bias;
                    }
                }
            }

            for (uint row = 0; row < ROWS; ++row) {
                for (uint result_index = 0; result_index < 4; ++result_index) {
                    float gate_sum = simd_sum(
                        shared_gate_result[row][result_index]);
                    float up_sum = simd_sum(
                        shared_up_result[row][result_index]);
                    if (lane == 0) {
                        bfloat gate_value = bfloat(gate_sum);
                        bfloat up_value = bfloat(up_sum);
                        auto sigmoid_y = 1 / (
                            1 + metal::exp(metal::abs(gate_value)));
                        bfloat sigmoid_mlx_exact = gate_value < bfloat(0.0f)
                            ? bfloat(sigmoid_y)
                            : bfloat(1 - sigmoid_y);
                        bfloat silu = bfloat(gate_value * sigmoid_mlx_exact);
                        uint output_column = output_base + result_index;
                        uint output_index =
                            (row * ACTIVATION_SLOTS + slot) * INTERMEDIATE
                            + output_column;
                        activations[output_index] = bfloat(silu * up_value);
                    }
                }
            }
        }
    """


def _target_m2_stage3_rowsplit_source() -> str:
    """M2 stage3 with one threadgroup per (output tile, row).

    The paired kernel walks both rows' 16 routed experts in one threadgroup;
    here the row comes from the grid's y index and each threadgroup runs the
    M2R1 per-row chain (textually the paired kernel's row-0 arithmetic), so
    the output is bit-identical with twice the threadgroups in flight. M1 Max
    A3B verify forward: 16.09 -> 15.81 ms.
    """

    src = _target_m2r1_stage3_source()
    src = src.replace(
        "uint tile = threadgroup_position_in_grid.x;",
        "uint tile = threadgroup_position_in_grid.x;\n"
        "        uint split_row = threadgroup_position_in_grid.y;",
        1,
    )
    for old, new in (
        ("expert_ids[0 * TOP_K + slot]", "expert_ids[split_row * TOP_K + slot]"),
        ("route_scores[0 * TOP_K + slot]", "route_scores[split_row * TOP_K + slot]"),
        ("(0 * ACTIVATION_SLOTS + slot)", "(split_row * ACTIVATION_SLOTS + slot)"),
        ("(0 * ACTIVATION_SLOTS + TOP_K)", "(split_row * ACTIVATION_SLOTS + TOP_K)"),
        ("bfloat gate_value0 = shared_gate[0];", "bfloat gate_value0 = shared_gate[split_row];"),
        ("output[output_column] = bfloat(", "output[split_row * HIDDEN + output_column] = bfloat("),
    ):
        assert old in src, old
        src = src.replace(old, new)
    return src


def _mtp_stage2_source() -> str:
    return """
        constexpr uint ROUTED_GROUP = 32;
        constexpr uint VALUES_PER_LANE = 16;
        constexpr uint K_BLOCK = VALUES_PER_LANE * 32;

        uint group = threadgroup_position_in_grid.x;
        uint slot = group / 32;
        uint tile = group - slot * 32;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint output_base = tile * 16 + simd_gid * 4;

        for (uint row = 0; row < ROWS; ++row) {
            float gate_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float up_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (slot < TOP_K) {
                uint expert = expert_ids[row * TOP_K + slot];
                const device uint* gate_words = routed_gate_up_weight
                    + expert * 2 * INTERMEDIATE * (HIDDEN / 8);
                const device uchar* gate_bytes =
                    reinterpret_cast<const device uchar*>(gate_words);
                const device uchar* up_bytes =
                    reinterpret_cast<const device uchar*>(
                        gate_words + INTERMEDIATE * (HIDDEN / 8));
                const device bfloat* gate_scale_values = routed_gate_up_scales
                    + expert * 2 * INTERMEDIATE * (HIDDEN / ROUTED_GROUP);
                const device bfloat* gate_bias_values = routed_gate_up_biases
                    + expert * 2 * INTERMEDIATE * (HIDDEN / ROUTED_GROUP);
                const device bfloat* up_scale_values = gate_scale_values
                    + INTERMEDIATE * (HIDDEN / ROUTED_GROUP);
                const device bfloat* up_bias_values = gate_bias_values
                    + INTERMEDIATE * (HIDDEN / ROUTED_GROUP);
                for (uint k_block = 0; k_block < HIDDEN; k_block += K_BLOCK) {
                    uint k_lane = k_block + lane * VALUES_PER_LANE;
                    float input_values[VALUES_PER_LANE];
                    float input_sum = 0.0f;
                    for (uint item = 0; item < VALUES_PER_LANE; item += 4) {
                        float x0 = float(value[row * HIDDEN + k_lane + item]);
                        float x1 = float(value[row * HIDDEN + k_lane + item + 1]);
                        float x2 = float(value[row * HIDDEN + k_lane + item + 2]);
                        float x3 = float(value[row * HIDDEN + k_lane + item + 3]);
                        input_sum += x0 + x1 + x2 + x3;
                        input_values[item] = x0;
                        input_values[item + 1] = x1 / 16.0f;
                        input_values[item + 2] = x2 / 256.0f;
                        input_values[item + 3] = x3 / 4096.0f;
                    }
                    for (uint result_index = 0; result_index < 4; ++result_index) {
                        uint output_column = output_base + result_index;
                        uint weight_offset =
                            output_column * (HIDDEN / 2) + k_lane / 2;
                        const device ushort* gate_packed =
                            reinterpret_cast<const device ushort*>(
                                gate_bytes + weight_offset);
                        const device ushort* up_packed =
                            reinterpret_cast<const device ushort*>(
                                up_bytes + weight_offset);
                        float gate_quantized_dot = 0.0f;
                        float up_quantized_dot = 0.0f;
                        // qdot4_affine
                        for (uint piece = 0;
                             piece < VALUES_PER_LANE / 4;
                             ++piece) {
                            ushort gate_bits = gate_packed[piece];
                            ushort up_bits = up_packed[piece];
                            uint item = piece * 4;
                            gate_quantized_dot +=
                                input_values[item] * float(gate_bits & 0x000f)
                                + input_values[item + 1] * float(gate_bits & 0x00f0)
                                + input_values[item + 2] * float(gate_bits & 0x0f00)
                                + input_values[item + 3] * float(gate_bits & 0xf000);
                            up_quantized_dot +=
                                input_values[item] * float(up_bits & 0x000f)
                                + input_values[item + 1] * float(up_bits & 0x00f0)
                                + input_values[item + 2] * float(up_bits & 0x0f00)
                                + input_values[item + 3] * float(up_bits & 0xf000);
                        }
                        uint metadata_index =
                            output_column * (HIDDEN / ROUTED_GROUP)
                            + k_lane / ROUTED_GROUP;
                        gate_result[result_index] +=
                            float(gate_scale_values[metadata_index])
                                * gate_quantized_dot
                            + input_sum * float(gate_bias_values[metadata_index]);
                        up_result[result_index] +=
                            float(up_scale_values[metadata_index])
                                * up_quantized_dot
                            + input_sum * float(up_bias_values[metadata_index]);
                    }
                }
            } else {
                // dense_shared_dot
                for (uint k_lane = lane; k_lane < HIDDEN; k_lane += 32) {
                    float input_value = float(value[row * HIDDEN + k_lane]);
                    for (uint result_index = 0; result_index < 4; ++result_index) {
                        uint output_column = output_base + result_index;
                        gate_result[result_index] += input_value * float(
                            shared_gate_up_weight[
                                output_column * HIDDEN + k_lane]);
                        up_result[result_index] += input_value * float(
                            shared_gate_up_weight[
                                (INTERMEDIATE + output_column) * HIDDEN + k_lane]);
                    }
                }
            }

            for (uint result_index = 0; result_index < 4; ++result_index) {
                float gate_sum = simd_sum(gate_result[result_index]);
                float up_sum = simd_sum(up_result[result_index]);
                if (lane == 0) {
                    bfloat gate_value = bfloat(gate_sum);
                    bfloat up_value = bfloat(up_sum);
                    auto sigmoid_y = 1 / (
                        1 + metal::exp(metal::abs(gate_value)));
                    bfloat sigmoid_mlx_exact = gate_value < bfloat(0.0f)
                        ? bfloat(sigmoid_y)
                        : bfloat(1 - sigmoid_y);
                    bfloat silu = bfloat(gate_value * sigmoid_mlx_exact);
                    uint output_column = output_base + result_index;
                    uint output_index =
                        (row * ACTIVATION_SLOTS + slot) * INTERMEDIATE
                        + output_column;
                    activations[output_index] = bfloat(silu * up_value);
                }
            }
        }
    """


def _stage3_source(*, target: bool, rows: int) -> str:
    if target:
        if rows == 2:
            return _target_m2_stage3_source()
        if rows == 3:
            return _target_m3_stage3_source()
        return _target_stage3_source()
    return _mtp_stage3_source()


def _target_m2_routed_row_source(row: int) -> str:
    accumulator = f"routed_accumulator{row}"
    return f"""
        for (uint slot = 0; slot < TOP_K; ++slot) {{
            uint expert = expert_ids[{row} * TOP_K + slot];
            const device uchar* down_bytes =
                reinterpret_cast<const device uchar*>(
                    routed_down_weight
                    + expert * HIDDEN * (INTERMEDIATE / 8));
            const device bfloat* down_scale_values = routed_down_scales
                + expert * HIDDEN * (INTERMEDIATE / ROUTED_GROUP);
            const device bfloat* down_bias_values = routed_down_biases
                + expert * HIDDEN * (INTERMEDIATE / ROUTED_GROUP);
            uint k_lane = lane * VALUES_PER_LANE;
            float input_values[VALUES_PER_LANE];
            float input_sum = 0.0f;
            for (uint item = 0; item < VALUES_PER_LANE; item += 4) {{
                uint activation_base =
                    ({row} * ACTIVATION_SLOTS + slot) * INTERMEDIATE
                    + k_lane + item;
                float x0 = float(activations[activation_base]);
                float x1 = float(activations[activation_base + 1]);
                float x2 = float(activations[activation_base + 2]);
                float x3 = float(activations[activation_base + 3]);
                input_sum += x0 + x1 + x2 + x3;
                input_values[item] = x0;
                input_values[item + 1] = x1 / 16.0f;
                input_values[item + 2] = x2 / 256.0f;
                input_values[item + 3] = x3 / 4096.0f;
            }}
            float down_result[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
            for (uint result_index = 0; result_index < 4; ++result_index) {{
                uint output_column = output_base + result_index;
                uint weight_offset =
                    output_column * (INTERMEDIATE / 2) + k_lane / 2;
                const device ushort* down_packed =
                    reinterpret_cast<const device ushort*>(
                        down_bytes + weight_offset);
                float quantized_dot = 0.0f;
                // qdot4_affine routed row {row}
                for (uint piece = 0; piece < VALUES_PER_LANE / 4; ++piece) {{
                    ushort packed = down_packed[piece];
                    uint item = piece * 4;
                    quantized_dot +=
                        input_values[item] * float(packed & 0x000f)
                        + input_values[item + 1] * float(packed & 0x00f0)
                        + input_values[item + 2] * float(packed & 0x0f00)
                        + input_values[item + 3] * float(packed & 0xf000);
                }}
                uint metadata_index =
                    output_column * (INTERMEDIATE / ROUTED_GROUP)
                    + k_lane / ROUTED_GROUP;
                down_result[result_index] =
                    float(down_scale_values[metadata_index]) * quantized_dot
                    + input_sum * float(down_bias_values[metadata_index]);
            }}
            for (uint result_index = 0; result_index < 4; ++result_index) {{
                float down_sum = simd_sum(down_result[result_index]);
                if (lane == 0) {{
                    bfloat down_value = bfloat(down_sum);
                    bfloat route_product = bfloat(
                        float(down_value)
                        * float(route_scores[{row} * TOP_K + slot]));
                    {accumulator}[result_index] = bfloat(
                        float({accumulator}[result_index])
                        + float(route_product));
                }}
            }}
        }}
    """


def _target_m2_stage3_source() -> str:
    """Pair exact M2 rows so shared-down storage is consumed once per tile."""

    return (
        """
        constexpr uint ROUTED_GROUP = 64;
        constexpr uint VALUES_PER_LANE = 16;
        constexpr uint OUTPUT_TILES = HIDDEN / 16;
        constexpr uint SIMD_GROUPS = 4;

        uint tile = threadgroup_position_in_grid.x;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint output_base = tile * 16 + simd_gid * 4;

        bfloat routed_accumulator0[4] = {
            bfloat(0.0f), bfloat(0.0f), bfloat(0.0f), bfloat(0.0f)};
        bfloat routed_accumulator1[4] = {
            bfloat(0.0f), bfloat(0.0f), bfloat(0.0f), bfloat(0.0f)};
        """
        + _target_m2_routed_row_source(0)
        + _target_m2_routed_row_source(1)
        + """
        threadgroup bfloat shared_inputs[ROWS * 4 * INTERMEDIATE];
        uint shared_k_lane = lane * VALUES_PER_LANE;
        uint shared_input_base0 =
            (0 * SIMD_GROUPS + simd_gid) * INTERMEDIATE + shared_k_lane;
        uint shared_input_base1 =
            (1 * SIMD_GROUPS + simd_gid) * INTERMEDIATE + shared_k_lane;
        float shared_input_sum0 = 0.0f;
        float shared_input_sum1 = 0.0f;
        for (uint item = 0; item < VALUES_PER_LANE; item += 4) {
            uint activation_base0 =
                (0 * ACTIVATION_SLOTS + TOP_K) * INTERMEDIATE
                + shared_k_lane + item;
            uint activation_base1 =
                (1 * ACTIVATION_SLOTS + TOP_K) * INTERMEDIATE
                + shared_k_lane + item;
            bfloat x00 = activations[activation_base0];
            bfloat x01 = activations[activation_base0 + 1];
            bfloat x02 = activations[activation_base0 + 2];
            bfloat x03 = activations[activation_base0 + 3];
            bfloat x10 = activations[activation_base1];
            bfloat x11 = activations[activation_base1 + 1];
            bfloat x12 = activations[activation_base1 + 2];
            bfloat x13 = activations[activation_base1 + 3];
            shared_input_sum0 +=
                float(x00) + float(x01) + float(x02) + float(x03);
            shared_input_sum1 +=
                float(x10) + float(x11) + float(x12) + float(x13);
            shared_inputs[shared_input_base0 + item] = x00;
            shared_inputs[shared_input_base0 + item + 1] = x01;
            shared_inputs[shared_input_base0 + item + 2] = x02;
            shared_inputs[shared_input_base0 + item + 3] = x03;
            shared_inputs[shared_input_base1 + item] = x10;
            shared_inputs[shared_input_base1 + item + 1] = x11;
            shared_inputs[shared_input_base1 + item + 2] = x12;
            shared_inputs[shared_input_base1 + item + 3] = x13;
        }

        const device uchar* shared_down_bytes =
            reinterpret_cast<const device uchar*>(shared_down_weight);
        float shared_result0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float shared_result1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint result_index = 0; result_index < 4; ++result_index) {
            uint output_column = output_base + result_index;
            uint weight_offset =
                output_column * (INTERMEDIATE / 2) + shared_k_lane / 2;
            const device ushort* shared_down_packed =
                reinterpret_cast<const device ushort*>(
                    shared_down_bytes + weight_offset);
            float shared_quantized_dot0 = 0.0f;
            float shared_quantized_dot1 = 0.0f;
            // qdot4_affine shared, one packed load for both rows
            for (uint piece = 0; piece < VALUES_PER_LANE / 4; ++piece) {
                ushort packed = shared_down_packed[piece];
                uint item = piece * 4;
                shared_quantized_dot0 +=
                    float(shared_inputs[shared_input_base0 + item])
                        * float(packed & 0x000f)
                    + float(shared_inputs[shared_input_base0 + item + 1]) / 16.0f
                        * float(packed & 0x00f0)
                    + float(shared_inputs[shared_input_base0 + item + 2]) / 256.0f
                        * float(packed & 0x0f00)
                    + float(shared_inputs[shared_input_base0 + item + 3]) / 4096.0f
                        * float(packed & 0xf000);
                shared_quantized_dot1 +=
                    float(shared_inputs[shared_input_base1 + item])
                        * float(packed & 0x000f)
                    + float(shared_inputs[shared_input_base1 + item + 1]) / 16.0f
                        * float(packed & 0x00f0)
                    + float(shared_inputs[shared_input_base1 + item + 2]) / 256.0f
                        * float(packed & 0x0f00)
                    + float(shared_inputs[shared_input_base1 + item + 3]) / 4096.0f
                        * float(packed & 0xf000);
            }
            uint metadata_index =
                output_column * (INTERMEDIATE / ROUTED_GROUP)
                + shared_k_lane / ROUTED_GROUP;
            float shared_scale = float(shared_down_scales[metadata_index]);
            float shared_bias = float(shared_down_biases[metadata_index]);
            shared_result0[result_index] =
                shared_scale * shared_quantized_dot0
                + shared_input_sum0 * shared_bias;
            shared_result1[result_index] =
                shared_scale * shared_quantized_dot1
                + shared_input_sum1 * shared_bias;
        }

        for (uint result_index = 0; result_index < 4; ++result_index) {
            float shared_sum0 = simd_sum(shared_result0[result_index]);
            float shared_sum1 = simd_sum(shared_result1[result_index]);
            if (lane == 0) {
                bfloat gate_value0 = shared_gate[0];
                auto sigmoid_y0 = 1 / (
                    1 + metal::exp(metal::abs(gate_value0)));
                bfloat sigmoid_mlx_exact0 = gate_value0 < bfloat(0.0f)
                    ? bfloat(sigmoid_y0)
                    : bfloat(1 - sigmoid_y0);
                bfloat shared_value0 = bfloat(shared_sum0);
                bfloat gated_shared0 = bfloat(
                    sigmoid_mlx_exact0 * shared_value0);
                bfloat gate_value1 = shared_gate[1];
                auto sigmoid_y1 = 1 / (
                    1 + metal::exp(metal::abs(gate_value1)));
                bfloat sigmoid_mlx_exact1 = gate_value1 < bfloat(0.0f)
                    ? bfloat(sigmoid_y1)
                    : bfloat(1 - sigmoid_y1);
                bfloat shared_value1 = bfloat(shared_sum1);
                bfloat gated_shared1 = bfloat(
                    sigmoid_mlx_exact1 * shared_value1);
                uint output_column = output_base + result_index;
                output[output_column] = bfloat(
                    float(routed_accumulator0[result_index])
                    + float(gated_shared0));
                output[HIDDEN + output_column] = bfloat(
                    float(routed_accumulator1[result_index])
                    + float(gated_shared1));
            }
        }
    """
    )


def _target_m3_stage3_source() -> str:
    """Row-tripled M2 stage3 for the k=2 verify ``[primary, d1, d2]``.

    Byte-parity extension of the row-paired 2-row kernel: rows 0 and 1 are
    textually identical to ``_target_m2_stage3_source`` (routed rows reuse
    ``_target_m2_routed_row_source``; the shared per-row arithmetic is
    replicated verbatim), so a 3-row verify's first two rows bit-match the
    shipped M2 kernel, and every row bit-matches the single-row M1 route --
    the AR-exactness contract (install parity lane, limit 0.0).  Row 2 adds a
    third routed accumulator and a third shared partial that consumes the
    same single shared-down packed load already read for rows 0 and 1, so the
    weight read is amortized across all three verify rows.
    """

    return (
        """
        constexpr uint ROUTED_GROUP = 64;
        constexpr uint VALUES_PER_LANE = 16;
        constexpr uint OUTPUT_TILES = HIDDEN / 16;
        constexpr uint SIMD_GROUPS = 4;

        uint tile = threadgroup_position_in_grid.x;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint output_base = tile * 16 + simd_gid * 4;

        bfloat routed_accumulator0[4] = {
            bfloat(0.0f), bfloat(0.0f), bfloat(0.0f), bfloat(0.0f)};
        bfloat routed_accumulator1[4] = {
            bfloat(0.0f), bfloat(0.0f), bfloat(0.0f), bfloat(0.0f)};
        bfloat routed_accumulator2[4] = {
            bfloat(0.0f), bfloat(0.0f), bfloat(0.0f), bfloat(0.0f)};
        """
        + _target_m2_routed_row_source(0)
        + _target_m2_routed_row_source(1)
        + _target_m2_routed_row_source(2)
        + """
        threadgroup bfloat shared_inputs[ROWS * 4 * INTERMEDIATE];
        uint shared_k_lane = lane * VALUES_PER_LANE;
        uint shared_input_base0 =
            (0 * SIMD_GROUPS + simd_gid) * INTERMEDIATE + shared_k_lane;
        uint shared_input_base1 =
            (1 * SIMD_GROUPS + simd_gid) * INTERMEDIATE + shared_k_lane;
        uint shared_input_base2 =
            (2 * SIMD_GROUPS + simd_gid) * INTERMEDIATE + shared_k_lane;
        float shared_input_sum0 = 0.0f;
        float shared_input_sum1 = 0.0f;
        float shared_input_sum2 = 0.0f;
        for (uint item = 0; item < VALUES_PER_LANE; item += 4) {
            uint activation_base0 =
                (0 * ACTIVATION_SLOTS + TOP_K) * INTERMEDIATE
                + shared_k_lane + item;
            uint activation_base1 =
                (1 * ACTIVATION_SLOTS + TOP_K) * INTERMEDIATE
                + shared_k_lane + item;
            uint activation_base2 =
                (2 * ACTIVATION_SLOTS + TOP_K) * INTERMEDIATE
                + shared_k_lane + item;
            bfloat x00 = activations[activation_base0];
            bfloat x01 = activations[activation_base0 + 1];
            bfloat x02 = activations[activation_base0 + 2];
            bfloat x03 = activations[activation_base0 + 3];
            bfloat x10 = activations[activation_base1];
            bfloat x11 = activations[activation_base1 + 1];
            bfloat x12 = activations[activation_base1 + 2];
            bfloat x13 = activations[activation_base1 + 3];
            bfloat x20 = activations[activation_base2];
            bfloat x21 = activations[activation_base2 + 1];
            bfloat x22 = activations[activation_base2 + 2];
            bfloat x23 = activations[activation_base2 + 3];
            shared_input_sum0 +=
                float(x00) + float(x01) + float(x02) + float(x03);
            shared_input_sum1 +=
                float(x10) + float(x11) + float(x12) + float(x13);
            shared_input_sum2 +=
                float(x20) + float(x21) + float(x22) + float(x23);
            shared_inputs[shared_input_base0 + item] = x00;
            shared_inputs[shared_input_base0 + item + 1] = x01;
            shared_inputs[shared_input_base0 + item + 2] = x02;
            shared_inputs[shared_input_base0 + item + 3] = x03;
            shared_inputs[shared_input_base1 + item] = x10;
            shared_inputs[shared_input_base1 + item + 1] = x11;
            shared_inputs[shared_input_base1 + item + 2] = x12;
            shared_inputs[shared_input_base1 + item + 3] = x13;
            shared_inputs[shared_input_base2 + item] = x20;
            shared_inputs[shared_input_base2 + item + 1] = x21;
            shared_inputs[shared_input_base2 + item + 2] = x22;
            shared_inputs[shared_input_base2 + item + 3] = x23;
        }

        const device uchar* shared_down_bytes =
            reinterpret_cast<const device uchar*>(shared_down_weight);
        float shared_result0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float shared_result1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float shared_result2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint result_index = 0; result_index < 4; ++result_index) {
            uint output_column = output_base + result_index;
            uint weight_offset =
                output_column * (INTERMEDIATE / 2) + shared_k_lane / 2;
            const device ushort* shared_down_packed =
                reinterpret_cast<const device ushort*>(
                    shared_down_bytes + weight_offset);
            float shared_quantized_dot0 = 0.0f;
            float shared_quantized_dot1 = 0.0f;
            float shared_quantized_dot2 = 0.0f;
            // qdot4_affine shared, one packed load for all three rows
            for (uint piece = 0; piece < VALUES_PER_LANE / 4; ++piece) {
                ushort packed = shared_down_packed[piece];
                uint item = piece * 4;
                shared_quantized_dot0 +=
                    float(shared_inputs[shared_input_base0 + item])
                        * float(packed & 0x000f)
                    + float(shared_inputs[shared_input_base0 + item + 1]) / 16.0f
                        * float(packed & 0x00f0)
                    + float(shared_inputs[shared_input_base0 + item + 2]) / 256.0f
                        * float(packed & 0x0f00)
                    + float(shared_inputs[shared_input_base0 + item + 3]) / 4096.0f
                        * float(packed & 0xf000);
                shared_quantized_dot1 +=
                    float(shared_inputs[shared_input_base1 + item])
                        * float(packed & 0x000f)
                    + float(shared_inputs[shared_input_base1 + item + 1]) / 16.0f
                        * float(packed & 0x00f0)
                    + float(shared_inputs[shared_input_base1 + item + 2]) / 256.0f
                        * float(packed & 0x0f00)
                    + float(shared_inputs[shared_input_base1 + item + 3]) / 4096.0f
                        * float(packed & 0xf000);
                shared_quantized_dot2 +=
                    float(shared_inputs[shared_input_base2 + item])
                        * float(packed & 0x000f)
                    + float(shared_inputs[shared_input_base2 + item + 1]) / 16.0f
                        * float(packed & 0x00f0)
                    + float(shared_inputs[shared_input_base2 + item + 2]) / 256.0f
                        * float(packed & 0x0f00)
                    + float(shared_inputs[shared_input_base2 + item + 3]) / 4096.0f
                        * float(packed & 0xf000);
            }
            uint metadata_index =
                output_column * (INTERMEDIATE / ROUTED_GROUP)
                + shared_k_lane / ROUTED_GROUP;
            float shared_scale = float(shared_down_scales[metadata_index]);
            float shared_bias = float(shared_down_biases[metadata_index]);
            shared_result0[result_index] =
                shared_scale * shared_quantized_dot0
                + shared_input_sum0 * shared_bias;
            shared_result1[result_index] =
                shared_scale * shared_quantized_dot1
                + shared_input_sum1 * shared_bias;
            shared_result2[result_index] =
                shared_scale * shared_quantized_dot2
                + shared_input_sum2 * shared_bias;
        }

        for (uint result_index = 0; result_index < 4; ++result_index) {
            float shared_sum0 = simd_sum(shared_result0[result_index]);
            float shared_sum1 = simd_sum(shared_result1[result_index]);
            float shared_sum2 = simd_sum(shared_result2[result_index]);
            if (lane == 0) {
                bfloat gate_value0 = shared_gate[0];
                auto sigmoid_y0 = 1 / (
                    1 + metal::exp(metal::abs(gate_value0)));
                bfloat sigmoid_mlx_exact0 = gate_value0 < bfloat(0.0f)
                    ? bfloat(sigmoid_y0)
                    : bfloat(1 - sigmoid_y0);
                bfloat shared_value0 = bfloat(shared_sum0);
                bfloat gated_shared0 = bfloat(
                    sigmoid_mlx_exact0 * shared_value0);
                bfloat gate_value1 = shared_gate[1];
                auto sigmoid_y1 = 1 / (
                    1 + metal::exp(metal::abs(gate_value1)));
                bfloat sigmoid_mlx_exact1 = gate_value1 < bfloat(0.0f)
                    ? bfloat(sigmoid_y1)
                    : bfloat(1 - sigmoid_y1);
                bfloat shared_value1 = bfloat(shared_sum1);
                bfloat gated_shared1 = bfloat(
                    sigmoid_mlx_exact1 * shared_value1);
                bfloat gate_value2 = shared_gate[2];
                auto sigmoid_y2 = 1 / (
                    1 + metal::exp(metal::abs(gate_value2)));
                bfloat sigmoid_mlx_exact2 = gate_value2 < bfloat(0.0f)
                    ? bfloat(sigmoid_y2)
                    : bfloat(1 - sigmoid_y2);
                bfloat shared_value2 = bfloat(shared_sum2);
                bfloat gated_shared2 = bfloat(
                    sigmoid_mlx_exact2 * shared_value2);
                uint output_column = output_base + result_index;
                output[output_column] = bfloat(
                    float(routed_accumulator0[result_index])
                    + float(gated_shared0));
                output[HIDDEN + output_column] = bfloat(
                    float(routed_accumulator1[result_index])
                    + float(gated_shared1));
                output[2 * HIDDEN + output_column] = bfloat(
                    float(routed_accumulator2[result_index])
                    + float(gated_shared2));
            }
        }
    """
    )


def _target_m2r1_stage3_source() -> str:
    """Row-0 of the paired M2 stage3: verbatim per-row arithmetic at ROWS=1.

    Every row-0 expression is textually identical to the M2 source (the
    routed part literally reuses `_target_m2_routed_row_source(0)`), so the
    compiled per-row arithmetic chain bit-matches the M2 kernel's row 0 --
    the M1 decode route must be indistinguishable per row from the M2
    verify route for the AR-exactness gate.  Row-1 lines are dropped.
    """

    return (
        """
        constexpr uint ROUTED_GROUP = 64;
        constexpr uint VALUES_PER_LANE = 16;
        constexpr uint OUTPUT_TILES = HIDDEN / 16;
        constexpr uint SIMD_GROUPS = 4;

        uint tile = threadgroup_position_in_grid.x;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint output_base = tile * 16 + simd_gid * 4;

        bfloat routed_accumulator0[4] = {
            bfloat(0.0f), bfloat(0.0f), bfloat(0.0f), bfloat(0.0f)};
        """
        + _target_m2_routed_row_source(0)
        + """
        threadgroup bfloat shared_inputs[ROWS * 4 * INTERMEDIATE];
        uint shared_k_lane = lane * VALUES_PER_LANE;
        uint shared_input_base0 =
            (0 * SIMD_GROUPS + simd_gid) * INTERMEDIATE + shared_k_lane;
        float shared_input_sum0 = 0.0f;
        for (uint item = 0; item < VALUES_PER_LANE; item += 4) {
            uint activation_base0 =
                (0 * ACTIVATION_SLOTS + TOP_K) * INTERMEDIATE
                + shared_k_lane + item;
            bfloat x00 = activations[activation_base0];
            bfloat x01 = activations[activation_base0 + 1];
            bfloat x02 = activations[activation_base0 + 2];
            bfloat x03 = activations[activation_base0 + 3];
            shared_input_sum0 +=
                float(x00) + float(x01) + float(x02) + float(x03);
            shared_inputs[shared_input_base0 + item] = x00;
            shared_inputs[shared_input_base0 + item + 1] = x01;
            shared_inputs[shared_input_base0 + item + 2] = x02;
            shared_inputs[shared_input_base0 + item + 3] = x03;
        }

        const device uchar* shared_down_bytes =
            reinterpret_cast<const device uchar*>(shared_down_weight);
        float shared_result0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint result_index = 0; result_index < 4; ++result_index) {
            uint output_column = output_base + result_index;
            uint weight_offset =
                output_column * (INTERMEDIATE / 2) + shared_k_lane / 2;
            const device ushort* shared_down_packed =
                reinterpret_cast<const device ushort*>(
                    shared_down_bytes + weight_offset);
            float shared_quantized_dot0 = 0.0f;
            // qdot4_affine shared, row 0 of the paired M2 arithmetic
            for (uint piece = 0; piece < VALUES_PER_LANE / 4; ++piece) {
                ushort packed = shared_down_packed[piece];
                uint item = piece * 4;
                shared_quantized_dot0 +=
                    float(shared_inputs[shared_input_base0 + item])
                        * float(packed & 0x000f)
                    + float(shared_inputs[shared_input_base0 + item + 1]) / 16.0f
                        * float(packed & 0x00f0)
                    + float(shared_inputs[shared_input_base0 + item + 2]) / 256.0f
                        * float(packed & 0x0f00)
                    + float(shared_inputs[shared_input_base0 + item + 3]) / 4096.0f
                        * float(packed & 0xf000);
            }
            uint metadata_index =
                output_column * (INTERMEDIATE / ROUTED_GROUP)
                + shared_k_lane / ROUTED_GROUP;
            float shared_scale = float(shared_down_scales[metadata_index]);
            float shared_bias = float(shared_down_biases[metadata_index]);
            shared_result0[result_index] =
                shared_scale * shared_quantized_dot0
                + shared_input_sum0 * shared_bias;
        }

        for (uint result_index = 0; result_index < 4; ++result_index) {
            float shared_sum0 = simd_sum(shared_result0[result_index]);
            if (lane == 0) {
                bfloat gate_value0 = shared_gate[0];
                auto sigmoid_y0 = 1 / (
                    1 + metal::exp(metal::abs(gate_value0)));
                bfloat sigmoid_mlx_exact0 = gate_value0 < bfloat(0.0f)
                    ? bfloat(sigmoid_y0)
                    : bfloat(1 - sigmoid_y0);
                bfloat shared_value0 = bfloat(shared_sum0);
                bfloat gated_shared0 = bfloat(
                    sigmoid_mlx_exact0 * shared_value0);
                uint output_column = output_base + result_index;
                output[output_column] = bfloat(
                    float(routed_accumulator0[result_index])
                    + float(gated_shared0));
            }
        }
    """
    )


def _target_stage3_source() -> str:
    return """
        constexpr uint ROUTED_GROUP = 64;
        constexpr uint VALUES_PER_LANE = 16;
        constexpr uint OUTPUT_TILES = HIDDEN / 16;

        uint group = threadgroup_position_in_grid.x;
        uint row = group / OUTPUT_TILES;
        uint tile = group - row * OUTPUT_TILES;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint output_base = tile * 16 + simd_gid * 4;

        bfloat routed_accumulator[4] = {
            bfloat(0.0f), bfloat(0.0f), bfloat(0.0f), bfloat(0.0f)};
            for (uint slot = 0; slot < TOP_K; ++slot) {
                uint expert = expert_ids[row * TOP_K + slot];
                const device uchar* down_bytes =
                    reinterpret_cast<const device uchar*>(
                        routed_down_weight
                        + expert * HIDDEN * (INTERMEDIATE / 8));
                const device bfloat* down_scale_values = routed_down_scales
                    + expert * HIDDEN * (INTERMEDIATE / ROUTED_GROUP);
                const device bfloat* down_bias_values = routed_down_biases
                    + expert * HIDDEN * (INTERMEDIATE / ROUTED_GROUP);
                uint k_lane = lane * VALUES_PER_LANE;
                float input_values[VALUES_PER_LANE];
                float input_sum = 0.0f;
                for (uint item = 0; item < VALUES_PER_LANE; item += 4) {
                    uint activation_base =
                        (row * ACTIVATION_SLOTS + slot) * INTERMEDIATE
                        + k_lane + item;
                    float x0 = float(activations[activation_base]);
                    float x1 = float(activations[activation_base + 1]);
                    float x2 = float(activations[activation_base + 2]);
                    float x3 = float(activations[activation_base + 3]);
                    input_sum += x0 + x1 + x2 + x3;
                    input_values[item] = x0;
                    input_values[item + 1] = x1 / 16.0f;
                    input_values[item + 2] = x2 / 256.0f;
                    input_values[item + 3] = x3 / 4096.0f;
                }
                float down_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (uint result_index = 0; result_index < 4; ++result_index) {
                    uint output_column = output_base + result_index;
                    uint weight_offset =
                        output_column * (INTERMEDIATE / 2) + k_lane / 2;
                    const device ushort* down_packed =
                        reinterpret_cast<const device ushort*>(
                            down_bytes + weight_offset);
                    float quantized_dot = 0.0f;
                    // qdot4_affine
                    for (uint piece = 0; piece < VALUES_PER_LANE / 4; ++piece) {
                        ushort packed = down_packed[piece];
                        uint item = piece * 4;
                        quantized_dot +=
                            input_values[item] * float(packed & 0x000f)
                            + input_values[item + 1] * float(packed & 0x00f0)
                            + input_values[item + 2] * float(packed & 0x0f00)
                            + input_values[item + 3] * float(packed & 0xf000);
                    }
                    uint metadata_index =
                        output_column * (INTERMEDIATE / ROUTED_GROUP)
                        + k_lane / ROUTED_GROUP;
                    down_result[result_index] =
                        float(down_scale_values[metadata_index]) * quantized_dot
                        + input_sum * float(down_bias_values[metadata_index]);
                }
                for (uint result_index = 0; result_index < 4; ++result_index) {
                    float down_sum = simd_sum(down_result[result_index]);
                    if (lane == 0) {
                        bfloat down_value = bfloat(down_sum);
                        bfloat route_product = bfloat(
                            float(down_value)
                            * float(route_scores[row * TOP_K + slot]));
                        routed_accumulator[result_index] = bfloat(
                            float(routed_accumulator[result_index])
                            + float(route_product));
                    }
                }
            }

            uint shared_k_lane = lane * VALUES_PER_LANE;
            float shared_inputs[VALUES_PER_LANE];
            float shared_input_sum = 0.0f;
            for (uint item = 0; item < VALUES_PER_LANE; item += 4) {
                uint activation_base =
                    (row * ACTIVATION_SLOTS + TOP_K) * INTERMEDIATE
                    + shared_k_lane + item;
                float x0 = float(activations[activation_base]);
                float x1 = float(activations[activation_base + 1]);
                float x2 = float(activations[activation_base + 2]);
                float x3 = float(activations[activation_base + 3]);
                shared_input_sum += x0 + x1 + x2 + x3;
                shared_inputs[item] = x0;
                shared_inputs[item + 1] = x1 / 16.0f;
                shared_inputs[item + 2] = x2 / 256.0f;
                shared_inputs[item + 3] = x3 / 4096.0f;
            }
            const device uchar* shared_down_bytes =
                reinterpret_cast<const device uchar*>(shared_down_weight);
            float shared_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint result_index = 0; result_index < 4; ++result_index) {
                uint output_column = output_base + result_index;
                uint weight_offset =
                    output_column * (INTERMEDIATE / 2) + shared_k_lane / 2;
                const device ushort* down_packed =
                    reinterpret_cast<const device ushort*>(
                        shared_down_bytes + weight_offset);
                float quantized_dot = 0.0f;
                // qdot4_affine shared
                for (uint piece = 0; piece < VALUES_PER_LANE / 4; ++piece) {
                    ushort packed = down_packed[piece];
                    uint item = piece * 4;
                    quantized_dot +=
                        shared_inputs[item] * float(packed & 0x000f)
                        + shared_inputs[item + 1] * float(packed & 0x00f0)
                        + shared_inputs[item + 2] * float(packed & 0x0f00)
                        + shared_inputs[item + 3] * float(packed & 0xf000);
                }
                uint metadata_index =
                    output_column * (INTERMEDIATE / ROUTED_GROUP)
                    + shared_k_lane / ROUTED_GROUP;
                shared_result[result_index] =
                    float(shared_down_scales[metadata_index]) * quantized_dot
                    + shared_input_sum * float(shared_down_biases[metadata_index]);
            }

            for (uint result_index = 0; result_index < 4; ++result_index) {
                float shared_sum = simd_sum(shared_result[result_index]);
                if (lane == 0) {
                    bfloat shared_value = bfloat(shared_sum);
                    bfloat gate_value = shared_gate[row];
                    auto sigmoid_y = 1 / (
                        1 + metal::exp(metal::abs(gate_value)));
                    bfloat sigmoid_mlx_exact = gate_value < bfloat(0.0f)
                        ? bfloat(sigmoid_y)
                        : bfloat(1 - sigmoid_y);
                    bfloat gated_shared = bfloat(
                        sigmoid_mlx_exact * shared_value);
                    uint output_column = output_base + result_index;
                    uint output_index = row * HIDDEN + output_column;
                    output[output_index] = bfloat(
                        float(routed_accumulator[result_index])
                        + float(gated_shared));
                }
        }
    """


def _mtp_stage3_source() -> str:
    return """
        constexpr uint ROUTED_GROUP = 32;
        constexpr uint VALUES_PER_LANE = 16;
        constexpr uint OUTPUT_TILES = HIDDEN / 16;

        uint group = threadgroup_position_in_grid.x;
        uint row = group / OUTPUT_TILES;
        uint tile = group - row * OUTPUT_TILES;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint output_base = tile * 16 + simd_gid * 4;

        bfloat routed_accumulator[4] = {
            bfloat(0.0f), bfloat(0.0f), bfloat(0.0f), bfloat(0.0f)};
            for (uint slot = 0; slot < TOP_K; ++slot) {
                uint expert = expert_ids[row * TOP_K + slot];
                const device uchar* down_bytes =
                    reinterpret_cast<const device uchar*>(
                        routed_down_weight
                        + expert * HIDDEN * (INTERMEDIATE / 8));
                const device bfloat* down_scale_values = routed_down_scales
                    + expert * HIDDEN * (INTERMEDIATE / ROUTED_GROUP);
                const device bfloat* down_bias_values = routed_down_biases
                    + expert * HIDDEN * (INTERMEDIATE / ROUTED_GROUP);
                uint k_lane = lane * VALUES_PER_LANE;
                float input_values[VALUES_PER_LANE];
                float input_sum = 0.0f;
                for (uint item = 0; item < VALUES_PER_LANE; item += 4) {
                    uint activation_base =
                        (row * ACTIVATION_SLOTS + slot) * INTERMEDIATE
                        + k_lane + item;
                    float x0 = float(activations[activation_base]);
                    float x1 = float(activations[activation_base + 1]);
                    float x2 = float(activations[activation_base + 2]);
                    float x3 = float(activations[activation_base + 3]);
                    input_sum += x0 + x1 + x2 + x3;
                    input_values[item] = x0;
                    input_values[item + 1] = x1 / 16.0f;
                    input_values[item + 2] = x2 / 256.0f;
                    input_values[item + 3] = x3 / 4096.0f;
                }
                float down_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (uint result_index = 0; result_index < 4; ++result_index) {
                    uint output_column = output_base + result_index;
                    uint weight_offset =
                        output_column * (INTERMEDIATE / 2) + k_lane / 2;
                    const device ushort* down_packed =
                        reinterpret_cast<const device ushort*>(
                            down_bytes + weight_offset);
                    float quantized_dot = 0.0f;
                    // qdot4_affine
                    for (uint piece = 0; piece < VALUES_PER_LANE / 4; ++piece) {
                        ushort packed = down_packed[piece];
                        uint item = piece * 4;
                        quantized_dot +=
                            input_values[item] * float(packed & 0x000f)
                            + input_values[item + 1] * float(packed & 0x00f0)
                            + input_values[item + 2] * float(packed & 0x0f00)
                            + input_values[item + 3] * float(packed & 0xf000);
                    }
                    uint metadata_index =
                        output_column * (INTERMEDIATE / ROUTED_GROUP)
                        + k_lane / ROUTED_GROUP;
                    down_result[result_index] =
                        float(down_scale_values[metadata_index]) * quantized_dot
                        + input_sum * float(down_bias_values[metadata_index]);
                }
                for (uint result_index = 0; result_index < 4; ++result_index) {
                    float down_sum = simd_sum(down_result[result_index]);
                    if (lane == 0) {
                        bfloat down_value = bfloat(down_sum);
                        bfloat route_product = bfloat(
                            float(down_value)
                            * float(route_scores[row * TOP_K + slot]));
                        routed_accumulator[result_index] = bfloat(
                            float(routed_accumulator[result_index])
                            + float(route_product));
                    }
                }
            }

            // dense_shared_down
            float shared_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint k_lane = lane; k_lane < INTERMEDIATE; k_lane += 32) {
                float input_value = float(
                    activations[(row * ACTIVATION_SLOTS + TOP_K) * INTERMEDIATE
                        + k_lane]);
                for (uint result_index = 0; result_index < 4; ++result_index) {
                    uint output_column = output_base + result_index;
                    shared_result[result_index] += input_value * float(
                        shared_down_weight[output_column * INTERMEDIATE + k_lane]);
                }
            }
            for (uint result_index = 0; result_index < 4; ++result_index) {
                float shared_sum = simd_sum(shared_result[result_index]);
                if (lane == 0) {
                    bfloat shared_value = bfloat(shared_sum);
                    bfloat gate_value = shared_gate[row];
                    auto sigmoid_y = 1 / (
                        1 + metal::exp(metal::abs(gate_value)));
                    bfloat sigmoid_mlx_exact = gate_value < bfloat(0.0f)
                        ? bfloat(sigmoid_y)
                        : bfloat(1 - sigmoid_y);
                    bfloat gated_shared = bfloat(
                        sigmoid_mlx_exact * shared_value);
                    uint output_column = output_base + result_index;
                    uint output_index = row * HIDDEN + output_column;
                    output[output_index] = bfloat(
                        float(routed_accumulator[result_index])
                        + float(gated_shared));
                }
        }
    """


def all_whole_moe_sources() -> dict[str, str]:
    """Return all fixed sources for construction tests and self-checks."""

    return {
        "target_m1_stage1": _fixed_source(stage=1, rows=1, variant="target_q8g64"),
        "target_m2_stage1_projection": _fixed_source(
            stage=1, rows=2, variant="target_q8g64"
        ),
        "target_m2_stage1_finalizer": _source_preamble(rows=2)
        + _target_m2_stage1_finalizer_source(),
        "mtp_m1_stage1": _fixed_source(stage=1, rows=1, variant="mtp_dense"),
        "target_m1_stage2": _fixed_source(stage=2, rows=1, variant="target_q4g64"),
        "target_m2_stage2": _fixed_source(stage=2, rows=2, variant="target_q4g64"),
        "mtp_m1_stage2": _fixed_source(stage=2, rows=1, variant="mtp_q4g32_dense"),
        "target_m1_stage3": _fixed_source(stage=3, rows=1, variant="target_q4g64"),
        "target_m2_stage3": _fixed_source(stage=3, rows=2, variant="target_q4g64"),
        "mtp_m1_stage3": _fixed_source(stage=3, rows=1, variant="mtp_q4g32_dense"),
        # M2-arithmetic at ROWS=1: the single-row decode route whose per-row
        # chains bit-match the M2 verify kernels (AR-exactness contract).
        "target_m2r1_stage2": _source_preamble(rows=1) + _target_m2_stage2_source(),
        "target_m2r1_stage3": _source_preamble(rows=1) + _target_m2r1_stage3_source(),
        # M3 (k=2) verify route: same split stage1 + row-generic stage2 as M2,
        # recompiled at ROWS=3, and the row-tripled paired stage3.  Rows 0/1
        # bit-match the M2 kernels; every row bit-matches the M1 route.
        "target_m3_stage1_projection": _fixed_source(
            stage=1, rows=3, variant="target_q8g64"
        ),
        "target_m3_stage1_finalizer": _source_preamble(rows=3)
        + _target_m2_stage1_finalizer_source(),
        "target_m3_stage2": _fixed_source(stage=2, rows=3, variant="target_q4g64"),
        "target_m3_stage3": _fixed_source(stage=3, rows=3, variant="target_q4g64"),
        "target_m2_stage3_rowsplit": _source_preamble(rows=1)
        + _target_m2_stage3_rowsplit_source(),
    }


def whole_moe_launch_table() -> dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]]:
    """Describe the fixed grids without probing a runtime tensor."""

    return {
        "target_m1_stage1": ((256, 1, 1), (256, 1, 1)),
        "target_m2_stage1_projection": (
            (STAGE1_PROJECTION_THREADGROUPS * STAGE1_THREADS, 1, 1),
            (STAGE1_THREADS, 1, 1),
        ),
        "target_m2_stage1_finalizer": (
            (STAGE1_THREADS, 1, 1),
            (STAGE1_THREADS, 1, 1),
        ),
        "mtp_m1_stage1": ((256, 1, 1), (256, 1, 1)),
        "target_m1_stage2": ((STAGE2_THREADGROUPS * 128, 1, 1), (128, 1, 1)),
        "target_m2_stage2": ((STAGE2_THREADGROUPS * 128, 1, 1), (128, 1, 1)),
        "mtp_m1_stage2": ((STAGE2_THREADGROUPS * 128, 1, 1), (128, 1, 1)),
        "target_m1_stage3": ((STAGE3_THREADGROUPS * 128, 1, 1), (128, 1, 1)),
        "target_m2_stage3": ((STAGE3_THREADGROUPS * 128, 1, 1), (128, 1, 1)),
        "mtp_m1_stage3": ((STAGE3_THREADGROUPS * 128, 1, 1), (128, 1, 1)),
        "target_m2r1_stage2": ((STAGE2_THREADGROUPS * 128, 1, 1), (128, 1, 1)),
        "target_m2r1_stage3": ((STAGE3_THREADGROUPS * 128, 1, 1), (128, 1, 1)),
        "target_m3_stage1_projection": (
            (STAGE1_PROJECTION_THREADGROUPS * STAGE1_THREADS, 1, 1),
            (STAGE1_THREADS, 1, 1),
        ),
        "target_m3_stage1_finalizer": (
            (STAGE1_THREADS, 1, 1),
            (STAGE1_THREADS, 1, 1),
        ),
        "target_m3_stage2": ((STAGE2_THREADGROUPS * 128, 1, 1), (128, 1, 1)),
        "target_m3_stage3": ((STAGE3_THREADGROUPS * 128, 1, 1), (128, 1, 1)),
    }


def _scalar_dtype():
    from . import moe_scalar

    return moe_scalar.scalar()


def _build_kernel(
    key: str,
    *,
    input_names: list[str],
    output_names: list[str],
):
    from . import moe_scalar

    cache_key = key + moe_scalar.suffix()
    kernel = _KERNELS.get(cache_key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"mtplx_a3b_whole_moe_{key}{moe_scalar.suffix()}",
            input_names=input_names,
            output_names=output_names,
            source=moe_scalar.metal_source(all_whole_moe_sources()[key]),
            ensure_row_contiguous=True,
        )
        _KERNELS[cache_key] = kernel
    return kernel


_TARGET_STAGE1_INPUT_NAMES = [
    "value",
    "router_weight",
    "router_scales",
    "router_biases",
    "shared_gate_weight",
    "shared_gate_scales",
    "shared_gate_biases",
]
_MTP_STAGE1_INPUT_NAMES = ["value", "router_weight", "shared_gate_weight"]
_STAGE1_OUTPUT_NAMES = ["expert_ids", "route_scores", "shared_gate"]
_TARGET_M2_STAGE1_PROJECTION_OUTPUT_NAMES = ["router_logits", "shared_gate"]
_TARGET_M2_STAGE1_FINALIZER_INPUT_NAMES = ["router_logits"]
_TARGET_M2_STAGE1_FINALIZER_OUTPUT_NAMES = ["expert_ids", "route_scores"]
_TARGET_STAGE2_INPUT_NAMES = [
    "value",
    "expert_ids",
    "routed_gate_up_weight",
    "routed_gate_up_scales",
    "routed_gate_up_biases",
    "shared_gate_up_weight",
    "shared_gate_up_scales",
    "shared_gate_up_biases",
]
_MTP_STAGE2_INPUT_NAMES = [
    "value",
    "expert_ids",
    "routed_gate_up_weight",
    "routed_gate_up_scales",
    "routed_gate_up_biases",
    "shared_gate_up_weight",
]
_STAGE2_OUTPUT_NAMES = ["activations"]
_TARGET_STAGE3_INPUT_NAMES = [
    "activations",
    "expert_ids",
    "route_scores",
    "shared_gate",
    "routed_down_weight",
    "routed_down_scales",
    "routed_down_biases",
    "shared_down_weight",
    "shared_down_scales",
    "shared_down_biases",
]
_MTP_STAGE3_INPUT_NAMES = [
    "activations",
    "expert_ids",
    "route_scores",
    "shared_gate",
    "routed_down_weight",
    "routed_down_scales",
    "routed_down_biases",
    "shared_down_weight",
]
_STAGE3_OUTPUT_NAMES = ["output"]


def _build_target_m1_stage1_kernel():
    return _build_kernel(
        "target_m1_stage1",
        input_names=_TARGET_STAGE1_INPUT_NAMES,
        output_names=_STAGE1_OUTPUT_NAMES,
    )


def _build_target_m2_stage1_projection_kernel():
    return _build_kernel(
        "target_m2_stage1_projection",
        input_names=_TARGET_STAGE1_INPUT_NAMES,
        output_names=_TARGET_M2_STAGE1_PROJECTION_OUTPUT_NAMES,
    )


def _build_target_m2_stage1_finalizer_kernel():
    return _build_kernel(
        "target_m2_stage1_finalizer",
        input_names=_TARGET_M2_STAGE1_FINALIZER_INPUT_NAMES,
        output_names=_TARGET_M2_STAGE1_FINALIZER_OUTPUT_NAMES,
    )


def _build_mtp_m1_stage1_kernel():
    return _build_kernel(
        "mtp_m1_stage1",
        input_names=_MTP_STAGE1_INPUT_NAMES,
        output_names=_STAGE1_OUTPUT_NAMES,
    )


def _build_target_m1_stage2_kernel():
    return _build_kernel(
        "target_m1_stage2",
        input_names=_TARGET_STAGE2_INPUT_NAMES,
        output_names=_STAGE2_OUTPUT_NAMES,
    )


def _build_target_m2_stage2_kernel():
    return _build_kernel(
        "target_m2_stage2",
        input_names=_TARGET_STAGE2_INPUT_NAMES,
        output_names=_STAGE2_OUTPUT_NAMES,
    )


def _build_target_m2_stage3_rowsplit_kernel():
    return _build_kernel(
        "target_m2_stage3_rowsplit",
        input_names=_TARGET_STAGE3_INPUT_NAMES,
        output_names=_STAGE3_OUTPUT_NAMES,
    )


def _launch_target_m2_stage3_rowsplit(
    kernel: Any,
    activations: Any,
    expert_ids: Any,
    route_scores: Any,
    shared_gate: Any,
    binding: Any,
):
    routed_down = binding.routed_down
    shared_down = binding.shared_down
    (output,) = kernel(
        inputs=[
            activations,
            expert_ids,
            route_scores,
            shared_gate,
            routed_down.weight,
            routed_down.scales,
            routed_down.biases,
            shared_down.weight,
            shared_down.scales,
            shared_down.biases,
        ],
        grid=(STAGE3_THREADGROUPS * TILED_THREADS, 2, 1),
        threadgroup=(TILED_THREADS, 1, 1),
        output_shapes=[(2, HIDDEN)],
        output_dtypes=[_scalar_dtype()],
    )
    return output


def _build_target_m2r1_stage2_kernel():
    return _build_kernel(
        "target_m2r1_stage2",
        input_names=_TARGET_STAGE2_INPUT_NAMES,
        output_names=_STAGE2_OUTPUT_NAMES,
    )


def _build_mtp_m1_stage2_kernel():
    return _build_kernel(
        "mtp_m1_stage2",
        input_names=_MTP_STAGE2_INPUT_NAMES,
        output_names=_STAGE2_OUTPUT_NAMES,
    )


def _build_target_m1_stage3_kernel():
    return _build_kernel(
        "target_m1_stage3",
        input_names=_TARGET_STAGE3_INPUT_NAMES,
        output_names=_STAGE3_OUTPUT_NAMES,
    )


def _build_target_m2_stage3_kernel():
    return _build_kernel(
        "target_m2_stage3",
        input_names=_TARGET_STAGE3_INPUT_NAMES,
        output_names=_STAGE3_OUTPUT_NAMES,
    )


def _build_target_m2r1_stage3_kernel():
    return _build_kernel(
        "target_m2r1_stage3",
        input_names=_TARGET_STAGE3_INPUT_NAMES,
        output_names=_STAGE3_OUTPUT_NAMES,
    )


def _build_mtp_m1_stage3_kernel():
    return _build_kernel(
        "mtp_m1_stage3",
        input_names=_MTP_STAGE3_INPUT_NAMES,
        output_names=_STAGE3_OUTPUT_NAMES,
    )


def _build_target_m3_stage1_projection_kernel():
    return _build_kernel(
        "target_m3_stage1_projection",
        input_names=_TARGET_STAGE1_INPUT_NAMES,
        output_names=_TARGET_M2_STAGE1_PROJECTION_OUTPUT_NAMES,
    )


def _build_target_m3_stage1_finalizer_kernel():
    return _build_kernel(
        "target_m3_stage1_finalizer",
        input_names=_TARGET_M2_STAGE1_FINALIZER_INPUT_NAMES,
        output_names=_TARGET_M2_STAGE1_FINALIZER_OUTPUT_NAMES,
    )


def _build_target_m3_stage2_kernel():
    return _build_kernel(
        "target_m3_stage2",
        input_names=_TARGET_STAGE2_INPUT_NAMES,
        output_names=_STAGE2_OUTPUT_NAMES,
    )


def _build_target_m3_stage3_kernel():
    return _build_kernel(
        "target_m3_stage3",
        input_names=_TARGET_STAGE3_INPUT_NAMES,
        output_names=_STAGE3_OUTPUT_NAMES,
    )


def _launch_target_stage1(
    kernel: Any,
    value: Any,
    binding: Any,
    *,
    rows: int,
):
    router = binding.router
    shared_gate = binding.shared_scalar_gate
    return kernel(
        inputs=[
            value,
            router.weight,
            router.scales,
            router.biases,
            shared_gate.weight,
            shared_gate.scales,
            shared_gate.biases,
        ],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, 8), (rows, 8), (rows, 1)],
        output_dtypes=[mx.uint32, _scalar_dtype(), _scalar_dtype()],
    )


def _launch_mtp_stage1(kernel: Any, value: Any, binding: Any):
    return kernel(
        inputs=[
            value,
            binding.router.weight,
            binding.shared_scalar_gate.weight,
        ],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, 8), (1, 8), (1, 1)],
        output_dtypes=[mx.uint32, _scalar_dtype(), _scalar_dtype()],
    )


def _launch_target_m2_stage1(
    projection_kernel: Any,
    finalizer_kernel: Any,
    value: Any,
    binding: Any,
    *,
    rows: int = 2,
):
    router = binding.router
    shared_gate_projection = binding.shared_scalar_gate
    router_logits, shared_gate = projection_kernel(
        inputs=[
            value,
            router.weight,
            router.scales,
            router.biases,
            shared_gate_projection.weight,
            shared_gate_projection.scales,
            shared_gate_projection.biases,
        ],
        grid=(STAGE1_PROJECTION_THREADGROUPS * STAGE1_THREADS, 1, 1),
        threadgroup=(STAGE1_THREADS, 1, 1),
        output_shapes=[(rows, EXPERTS), (rows, 1)],
        output_dtypes=[_scalar_dtype(), _scalar_dtype()],
    )
    expert_ids, route_scores = finalizer_kernel(
        inputs=[router_logits],
        grid=(STAGE1_THREADS, 1, 1),
        threadgroup=(STAGE1_THREADS, 1, 1),
        output_shapes=[(rows, TOP_K), (rows, TOP_K)],
        output_dtypes=[mx.uint32, _scalar_dtype()],
    )
    return expert_ids, route_scores, shared_gate


def target_m1_stage1(value: Any, binding: Any):
    """Launch fixed target M1 route and shared-gate ownership."""

    return _launch_target_stage1(
        _build_target_m1_stage1_kernel(),
        value,
        binding,
        rows=1,
    )


def target_m2_stage1(value: Any, binding: Any):
    """Launch fixed target M2 route and shared-gate ownership."""

    return _launch_target_m2_stage1(
        _build_target_m2_stage1_projection_kernel(),
        _build_target_m2_stage1_finalizer_kernel(),
        value,
        binding,
    )


def target_m3_stage1(value: Any, binding: Any):
    """Launch the 3-row (k=2 verify) target route and shared-gate ownership."""

    return _launch_target_m2_stage1(
        _build_target_m3_stage1_projection_kernel(),
        _build_target_m3_stage1_finalizer_kernel(),
        value,
        binding,
        rows=3,
    )


def mtp_m1_stage1(value: Any, binding: Any):
    """Launch fixed MTP M1 route and shared-gate ownership."""

    return _launch_mtp_stage1(_build_mtp_m1_stage1_kernel(), value, binding)


def _target_stage2_inputs(value: Any, expert_ids: Any, binding: Any) -> list[Any]:
    routed_gate_up = binding.routed_gate_up
    shared_gate_up = binding.shared_gate_up
    return [
        value,
        expert_ids,
        routed_gate_up.weight,
        routed_gate_up.scales,
        routed_gate_up.biases,
        shared_gate_up.weight,
        shared_gate_up.scales,
        shared_gate_up.biases,
    ]


def _launch_target_stage2(
    kernel: Any,
    value: Any,
    expert_ids: Any,
    binding: Any,
    *,
    rows: int,
):
    (activations,) = kernel(
        inputs=_target_stage2_inputs(value, expert_ids, binding),
        grid=(STAGE2_THREADGROUPS * TILED_THREADS, 1, 1),
        threadgroup=(TILED_THREADS, 1, 1),
        output_shapes=[(rows, ACTIVATION_SLOTS, INTERMEDIATE)],
        output_dtypes=[_scalar_dtype()],
    )
    return activations


def _launch_mtp_stage2(
    kernel: Any,
    value: Any,
    expert_ids: Any,
    binding: Any,
):
    routed_gate_up = binding.routed_gate_up
    (activations,) = kernel(
        inputs=[
            value,
            expert_ids,
            routed_gate_up.weight,
            routed_gate_up.scales,
            routed_gate_up.biases,
            binding.shared_gate_up.weight,
        ],
        grid=(STAGE2_THREADGROUPS * TILED_THREADS, 1, 1),
        threadgroup=(TILED_THREADS, 1, 1),
        output_shapes=[(1, ACTIVATION_SLOTS, INTERMEDIATE)],
        output_dtypes=[_scalar_dtype()],
    )
    return activations


def target_m1_stage2(value: Any, expert_ids: Any, binding: Any):
    """Launch fixed BF16 target M1 `[1,9,512]` activation ownership."""

    return _launch_target_stage2(
        _build_target_m1_stage2_kernel(), value, expert_ids, binding, rows=1
    )


def target_m2_stage2(value: Any, expert_ids: Any, binding: Any):
    """Launch fixed BF16 target M2 `[2,9,512]` activation ownership."""

    return _launch_target_stage2(
        _build_target_m2_stage2_kernel(), value, expert_ids, binding, rows=2
    )


def target_m2r1_stage2(value: Any, expert_ids: Any, binding: Any):
    """Launch the M2-arithmetic stage2 at ROWS=1 (bit-parity decode route)."""

    return _launch_target_stage2(
        _build_target_m2r1_stage2_kernel(), value, expert_ids, binding, rows=1
    )


def target_m3_stage2(value: Any, expert_ids: Any, binding: Any):
    """Launch fixed BF16 target M3 `[3,9,512]` activation ownership (k=2)."""

    return _launch_target_stage2(
        _build_target_m3_stage2_kernel(), value, expert_ids, binding, rows=3
    )


def mtp_m1_stage2(value: Any, expert_ids: Any, binding: Any):
    """Launch fixed BF16 MTP M1 `[1,9,512]` activation ownership."""

    return _launch_mtp_stage2(
        _build_mtp_m1_stage2_kernel(), value, expert_ids, binding
    )


def _target_stage3_inputs(
    activations: Any,
    expert_ids: Any,
    route_scores: Any,
    shared_gate: Any,
    binding: Any,
) -> list[Any]:
    routed_down = binding.routed_down
    shared_down = binding.shared_down
    return [
        activations,
        expert_ids,
        route_scores,
        shared_gate,
        routed_down.weight,
        routed_down.scales,
        routed_down.biases,
        shared_down.weight,
        shared_down.scales,
        shared_down.biases,
    ]


def _launch_target_stage3(
    kernel: Any,
    activations: Any,
    expert_ids: Any,
    route_scores: Any,
    shared_gate: Any,
    binding: Any,
    *,
    rows: int,
):
    (output,) = kernel(
        inputs=_target_stage3_inputs(
            activations, expert_ids, route_scores, shared_gate, binding
        ),
        grid=(STAGE3_THREADGROUPS * TILED_THREADS, 1, 1),
        threadgroup=(TILED_THREADS, 1, 1),
        output_shapes=[(rows, HIDDEN)],
        output_dtypes=[_scalar_dtype()],
    )
    return output


def _launch_mtp_stage3(
    kernel: Any,
    activations: Any,
    expert_ids: Any,
    route_scores: Any,
    shared_gate: Any,
    binding: Any,
):
    routed_down = binding.routed_down
    (output,) = kernel(
        inputs=[
            activations,
            expert_ids,
            route_scores,
            shared_gate,
            routed_down.weight,
            routed_down.scales,
            routed_down.biases,
            binding.shared_down.weight,
        ],
        grid=(STAGE3_THREADGROUPS * TILED_THREADS, 1, 1),
        threadgroup=(TILED_THREADS, 1, 1),
        output_shapes=[(1, HIDDEN)],
        output_dtypes=[_scalar_dtype()],
    )
    return output


def target_m1_stage3(
    activations: Any,
    expert_ids: Any,
    route_scores: Any,
    shared_gate: Any,
    binding: Any,
):
    """Launch fixed target M1 output ownership."""

    return _launch_target_stage3(
        _build_target_m1_stage3_kernel(),
        activations,
        expert_ids,
        route_scores,
        shared_gate,
        binding,
        rows=1,
    )


def target_m2_stage3(
    activations: Any,
    expert_ids: Any,
    route_scores: Any,
    shared_gate: Any,
    binding: Any,
):
    """Launch fixed row-paired target M2 output ownership."""

    return _launch_target_stage3(
        _build_target_m2_stage3_kernel(),
        activations,
        expert_ids,
        route_scores,
        shared_gate,
        binding,
        rows=2,
    )


def target_m3_stage3(
    activations: Any,
    expert_ids: Any,
    route_scores: Any,
    shared_gate: Any,
    binding: Any,
):
    """Launch the row-tripled target M3 output ownership (k=2 verify)."""

    return _launch_target_stage3(
        _build_target_m3_stage3_kernel(),
        activations,
        expert_ids,
        route_scores,
        shared_gate,
        binding,
        rows=3,
    )


def target_m2r1_stage3(
    activations: Any,
    expert_ids: Any,
    route_scores: Any,
    shared_gate: Any,
    binding: Any,
):
    """Launch the M2-arithmetic stage3 at ROWS=1 (bit-parity decode route)."""

    return _launch_target_stage3(
        _build_target_m2r1_stage3_kernel(),
        activations,
        expert_ids,
        route_scores,
        shared_gate,
        binding,
        rows=1,
    )


def mtp_m1_stage3(
    activations: Any,
    expert_ids: Any,
    route_scores: Any,
    shared_gate: Any,
    binding: Any,
):
    """Launch fixed MTP M1 output ownership."""

    return _launch_mtp_stage3(
        _build_mtp_m1_stage3_kernel(),
        activations,
        expert_ids,
        route_scores,
        shared_gate,
        binding,
    )


def bind_target_m1_stage3(binding: Any):
    """Prebind the exact target M1 fused-down kernel at installation."""

    kernel = _build_target_m1_stage3_kernel()

    def call(activations, expert_ids, route_scores, shared_gate):
        return _launch_target_stage3(
            kernel,
            activations,
            expert_ids,
            route_scores,
            shared_gate,
            binding,
            rows=1,
        )

    return call


def bind_target_m2_stage3(binding: Any):
    """Prebind the exact row-paired target M2 fused-down kernel."""

    kernel = _build_target_m2_stage3_kernel()

    def call(activations, expert_ids, route_scores, shared_gate):
        return _launch_target_stage3(
            kernel,
            activations,
            expert_ids,
            route_scores,
            shared_gate,
            binding,
            rows=2,
        )

    return call


def bind_target_m3_stage3(binding: Any):
    """Prebind the row-tripled target M3 fused-down kernel (k=2 verify)."""

    kernel = _build_target_m3_stage3_kernel()

    def call(activations, expert_ids, route_scores, shared_gate):
        return _launch_target_stage3(
            kernel,
            activations,
            expert_ids,
            route_scores,
            shared_gate,
            binding,
            rows=3,
        )

    return call


def bind_mtp_m1_stage3(binding: Any):
    """Prebind the exact MTP M1 fused-down kernel at installation."""

    kernel = _build_mtp_m1_stage3_kernel()

    def call(activations, expert_ids, route_scores, shared_gate):
        return _launch_mtp_stage3(
            kernel,
            activations,
            expert_ids,
            route_scores,
            shared_gate,
            binding,
        )

    return call


def bind_target_m1(binding: Any):
    """Bind the three fixed target M1 kernels once at installation."""

    stage1_kernel = _build_target_m1_stage1_kernel()
    stage2_kernel = _build_target_m1_stage2_kernel()
    stage3_kernel = _build_target_m1_stage3_kernel()

    def call(value: Any):
        expert_ids, route_scores, shared_gate = _launch_target_stage1(
            stage1_kernel, value, binding, rows=1
        )
        activations = _launch_target_stage2(
            stage2_kernel, value, expert_ids, binding, rows=1
        )
        output = _launch_target_stage3(
            stage3_kernel,
            activations,
            expert_ids,
            route_scores,
            shared_gate,
            binding,
            rows=1,
        )
        return output.reshape(*value.shape)

    return call


def bind_target_m2(binding: Any):
    """Bind the four fixed row-paired target M2 kernels once at installation."""

    stage1_projection_kernel = _build_target_m2_stage1_projection_kernel()
    stage1_finalizer_kernel = _build_target_m2_stage1_finalizer_kernel()
    stage2_kernel = _build_target_m2_stage2_kernel()
    stage3_kernel = _build_target_m2_stage3_rowsplit_kernel()

    def call(value: Any):
        expert_ids, route_scores, shared_gate = _launch_target_m2_stage1(
            stage1_projection_kernel,
            stage1_finalizer_kernel,
            value,
            binding,
        )
        activations = _launch_target_stage2(
            stage2_kernel, value, expert_ids, binding, rows=2
        )
        output = _launch_target_m2_stage3_rowsplit(
            stage3_kernel,
            activations,
            expert_ids,
            route_scores,
            shared_gate,
            binding,
        )
        return output.reshape(1, 2, 2048)

    return call


def bind_target_m3(binding: Any):
    """Bind the four row-tripled target M3 kernels once (k=2, 3-row verify)."""

    stage1_projection_kernel = _build_target_m3_stage1_projection_kernel()
    stage1_finalizer_kernel = _build_target_m3_stage1_finalizer_kernel()
    stage2_kernel = _build_target_m3_stage2_kernel()
    stage3_kernel = _build_target_m3_stage3_kernel()

    def call(value: Any):
        expert_ids, route_scores, shared_gate = _launch_target_m2_stage1(
            stage1_projection_kernel,
            stage1_finalizer_kernel,
            value,
            binding,
            rows=3,
        )
        activations = _launch_target_stage2(
            stage2_kernel, value, expert_ids, binding, rows=3
        )
        output = _launch_target_stage3(
            stage3_kernel,
            activations,
            expert_ids,
            route_scores,
            shared_gate,
            binding,
            rows=3,
        )
        return output.reshape(1, 3, 2048)

    return call


def bind_mtp_m1(binding: Any):
    """Bind the three fixed MTP M1 kernels once at installation."""

    stage1_kernel = _build_mtp_m1_stage1_kernel()
    stage2_kernel = _build_mtp_m1_stage2_kernel()
    stage3_kernel = _build_mtp_m1_stage3_kernel()

    def call(value: Any):
        expert_ids, route_scores, shared_gate = _launch_mtp_stage1(
            stage1_kernel, value, binding
        )
        activations = _launch_mtp_stage2(
            stage2_kernel, value, expert_ids, binding
        )
        output = _launch_mtp_stage3(
            stage3_kernel,
            activations,
            expert_ids,
            route_scores,
            shared_gate,
            binding,
        )
        return output.reshape(*value.shape)

    return call
