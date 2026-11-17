"""MLX-free structural gates for the large-S QSA prefill backend."""

from __future__ import annotations

import ast
import random
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "mtplx/kernels/qsa_indexer_prefill.py"
COMPILE_BACKEND = ROOT / "mtplx/kernels/qsa_indexer_compile.py"
MPP_RUNTIME_GATE = ROOT / "tests/test_qsa_indexer_prefill_mpp_runtime.py"


def _tree() -> ast.Module:
    return ast.parse(BACKEND.read_text())


def _function(name: str) -> ast.FunctionDef:
    for node in _tree().body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def _string_constant(name: str) -> str:
    for node in _tree().body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            assert isinstance(value, str)
            return value
    raise AssertionError(f"missing string constant {name}")


def _load_pure_function(name: str):
    node = _function(name)
    dependencies = (
        [_function("qsa_indexer_prefill_score_chunk_rows")]
        if name == "qsa_indexer_prefill_chunk_rows"
        else []
    )
    module = ast.Module(body=[*dependencies, node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {
        "DEFAULT_PREFILL_SCORE_WORKSPACE_BYTES": 128 * 1024 * 1024,
        "PREFILL_ROW_ALIGNMENT": 32,
        "QSAPrefillScoreProducer": str,
    }
    exec(compile(module, str(BACKEND), "exec"), namespace)  # noqa: S102
    return namespace[name]


def _adaptive_threshold(keys: list[int], k: int) -> tuple[int, int, int, int]:
    """Pure mirror of the Metal hybrid threshold resolution.

    Returns ``(threshold, candidate_count, insertion_comparisons,
    candidate_radix_visits)`` so tests can gate both exactness and work bounds.
    """

    assert 1 <= k <= len(keys)
    prefix = 0
    rank = k - 1
    candidate_count = len(keys)
    for pass_index, (shift, bits) in enumerate(
        ((53, 11), (42, 11), (31, 11), (20, 11), (9, 11), (0, 9))
    ):
        matches = [
            key for key in keys if pass_index == 0 or key >> (shift + bits) == prefix
        ]
        histogram = [0] * (1 << bits)
        for key in matches:
            histogram[(key >> shift) & ((1 << bits) - 1)] += 1
        for digit in range(len(histogram) - 1, -1, -1):
            count = histogram[digit]
            if rank < count:
                prefix = (prefix << bits) | digit
                candidate_count = count
                break
            rank -= count
        else:  # pragma: no cover - the rank invariant makes this unreachable
            raise AssertionError("radix rank escaped its prefix bucket")
        if candidate_count <= 2_048:
            candidates = [key for key in keys if key >> shift == prefix]
            if candidate_count <= 64:
                threshold = next(
                    key
                    for key in candidates
                    if sum(other > key for other in candidates) == rank
                )
                return threshold, len(candidates), len(candidates) ** 2, 0

            candidate_radix_visits = 0
            for next_shift, next_bits in (
                (53, 11),
                (42, 11),
                (31, 11),
                (20, 11),
                (9, 11),
                (0, 9),
            )[pass_index + 1 :]:
                candidate_radix_visits += len(candidates)
                matching = [
                    key
                    for key in candidates
                    if key >> (next_shift + next_bits) == prefix
                ]
                histogram = [0] * (1 << next_bits)
                for key in matching:
                    histogram[(key >> next_shift) & ((1 << next_bits) - 1)] += 1
                for digit in range(len(histogram) - 1, -1, -1):
                    count = histogram[digit]
                    if rank < count:
                        prefix = (prefix << next_bits) | digit
                        break
                    rank -= count
                else:  # pragma: no cover - same rank invariant as phase one
                    raise AssertionError("candidate radix rank escaped its prefix")
            return prefix, len(candidates), 0, candidate_radix_visits
    raise AssertionError("a strict 64-bit key must terminate by the final digit")


def test_prefill_backend_is_separate_and_does_not_modify_integration_contract():
    source = BACKEND.read_text()
    assert "qwen4_exp" not in source
    assert "MTPLX_FUSED_QSA_INDEXER" not in source
    assert "mx.eval" not in source
    assert ".item(" not in source


@pytest.mark.parametrize(
    "rows,heads,blocks,budget,expected",
    [
        (2_048, 4, 65_536, 128 * 1024 * 1024, 96),
        (2_048, 4, 32_768, 128 * 1024 * 1024, 192),
        (17, 4, 1_024, 128 * 1024 * 1024, 17),
        (100, 1, 1, 1, 1),
        (64, 4, 1_024, 1_024 * 4 * 5 * 33, 32),
    ],
)
def test_chunk_planner_counts_head_logits_and_reduced_plane(
    rows: int,
    heads: int,
    blocks: int,
    budget: int,
    expected: int,
):
    planner = _load_pure_function("qsa_indexer_prefill_chunk_rows")
    assert planner(rows, heads, blocks, budget) == expected


@pytest.mark.parametrize(
    "args,match",
    [
        ((0, 4, 8, 1024), "rows"),
        ((1, 0, 8, 1024), "heads"),
        ((1, 4, 0, 1024), "backing_blocks"),
        ((1, 4, 8, 0), "workspace_bytes"),
    ],
)
def test_chunk_planner_rejects_invalid_geometry(args, match: str):
    planner = _load_pure_function("qsa_indexer_prefill_chunk_rows")
    with pytest.raises(ValueError, match=match):
        planner(*args)


def test_producer_aware_chunk_planner_charges_only_materialized_score_planes():
    planner = _load_pure_function("qsa_indexer_prefill_score_chunk_rows")
    budget = 128 * 1024 * 1024
    assert planner(2_048, 4, 65_536, budget, producer="mpp") == 512
    assert planner(2_048, 4, 65_536, budget, producer="mlx") == 96
    assert planner(2_048, 4, 131_072, budget, producer="mpp") == 256
    assert planner(2_048, 4, 262_144, budget, producer="mpp") == 128
    assert 512 * 65_536 * 4 == budget
    assert 256 * 131_072 * 4 == budget
    assert 128 * 262_144 * 4 == budget
    assert 96 * 65_536 * 4 * 5 <= budget
    with pytest.raises(ValueError, match="producer"):
        planner(2_048, 4, 65_536, budget, producer="unknown")


def test_score_oracle_is_one_vectorized_mlx_matmul_without_head_dispatch_loop():
    function = _function("qsa_indexer_prefill_scores")
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert sum(call.func.attr == "matmul" for call in calls) == 1
    assert sum(call.func.attr == "maximum" for call in calls) == 1
    assert any(call.func.attr == "sum" for call in calls)
    assert not any(
        isinstance(node, (ast.For, ast.While)) for node in ast.walk(function)
    )

    source = ast.get_source_segment(BACKEND.read_text(), function)
    assert source is not None
    assert "q_chunk.astype(mx.float32)" in source
    assert "mx.maximum(per_head, 0.0).sum(axis=2)" in source
    assert "/ math.sqrt(dim)" in source


@pytest.mark.parametrize(
    "q_shape,pooled_shape,expected",
    [
        ((1, 96, 4, 128), (1, 65_536, 128), True),
        ((1, 17, 4, 128), (1, 33, 128), True),
        ((2, 96, 4, 128), (1, 65_536, 128), False),
        ((1, 96, 3, 128), (1, 65_536, 128), False),
        ((1, 96, 4, 64), (1, 65_536, 64), False),
        ((1, 0, 4, 128), (1, 65_536, 128), False),
        ((1, 96, 4, 128), (1, 0, 128), False),
    ],
)
def test_mpp_score_geometry_gate_is_shape_specific_and_layout_agnostic(
    q_shape: tuple[int, ...],
    pooled_shape: tuple[int, ...],
    expected: bool,
):
    supported = _load_pure_function("_mpp_score_geometry_supported")
    assert supported(q_shape, pooled_shape) is expected


def _nax_coordinate(lane: int) -> tuple[int, int]:
    qid = lane >> 2
    row = (qid & 4) | ((lane >> 1) & 3)
    column = ((qid & 2) | (lane & 1)) * 4
    return column, row


def test_mpp_fragment_layout_covers_each_4_query_by_32_key_tile_once():
    covered: set[tuple[int, int]] = set()
    for lane in range(32):
        column, row = _nax_coordinate(lane)
        if lane & 16:
            continue
        assert 0 <= row < 4
        partner_column, partner_row = _nax_coordinate(lane ^ 16)
        assert partner_column == column
        assert partner_row == row + 4
        # This lane owns h0/h2 at rows q/q+8; lane^16 owns h1/h3 at
        # q+4/q+12. All four therefore reduce to this one logical query.
        assert [row, partner_row, row + 8, partner_row + 8] == [
            row,
            row + 4,
            row + 8,
            row + 12,
        ]
        for key_half in range(2):
            for elem in range(4):
                location = (row, key_half * 16 + column + elem)
                assert location not in covered
                covered.add(location)
    assert covered == {(row, column) for row in range(4) for column in range(32)}
    threadgroup_covered = {
        (simdgroup * 4 + row, column)
        for simdgroup in range(4)
        for row, column in covered
    }
    assert threadgroup_covered == {
        (row, column) for row in range(16) for column in range(32)
    }


def test_mpp_score_kernel_is_a_fused_16x32_tensorops_tile():
    source = _string_constant("_MPP_SCORE_HEADER") + _string_constant(
        "_MPP_SCORE_SOURCE"
    )
    assert (
        "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>" in source
    )
    assert "QSA_SCORE_QUERY_TILE = 16" in source
    assert "QSA_SCORE_KEY_TILE = 32" in source
    assert "QSA_SCORE_SIMDGROUPS = 4" in source
    assert "threadgroup InT pooled_tile[" in source
    assert "mpp::tensor_ops::matmul2d_descriptor(" in source
    assert "16, 32, 16, false, true, true" in source
    assert "k_frag < QSA_SCORE_K_FRAGMENTS" in source
    assert "matmul.run(left, right, accumulator);" in source
    assert "simd_shuffle_xor(head0_or_1, ushort(16))" in source
    assert "simd_shuffle_xor(head2_or_3, ushort(16))" in source
    assert "head_sum / QSA_SCORE_SQRT_HEAD_DIM" in source
    assert "topk" not in source.lower()
    assert "radix" not in source.lower()
    assert "logical_blocks" not in source
    assert "1.0e-12" not in source

    dispatch = ast.get_source_segment(
        BACKEND.read_text(), _function("qsa_indexer_prefill_scores_mpp")
    )
    assert dispatch is not None
    assert "output_shapes=[(rows, blocks)]" in dispatch
    assert "output_dtypes=[mx.float32]" in dispatch


def test_mpp_score_kernel_honors_input_strides_and_tail_bounds():
    source = _string_constant("_MPP_SCORE_SOURCE")
    assert "int64_t(query) * q_strides[1]" in source
    assert "int64_t(head) * q_strides[2]" in source
    assert "int64_t(elem) * q_strides[3]" in source
    assert "int64_t(block) * pooled_strides[1]" in source
    assert "int64_t(elem) * pooled_strides[2]" in source
    # Device inputs are scalar-gathered through MLX's runtime strides.  A
    # device vec4 reinterpret would silently require alignment that Python
    # arrays do not expose.  The shared-memory TensorOp tile may remain vec4.
    assert "reinterpret_cast<const device vec" not in source
    assert "if (block < blocks)" in source
    assert "if (query < rows && block < blocks)" in source
    dispatch = BACKEND.read_text()
    assert "query_tiles = (rows + 15) // 16" in dispatch
    assert "key_tiles = (blocks + 31) // 32" in dispatch
    assert "q_dtype in (mx.float16, mx.bfloat16)" in dispatch
    assert "pooled_dtype == q_dtype" in dispatch


def test_mpp_support_does_not_probe_unavailable_python_stride_metadata():
    for name in (
        "qsa_indexer_prefill_scores_mpp_supported",
        "qsa_indexer_prefill_prepared_scores_mpp_supported",
        "_mpp_score_signature_supported",
    ):
        function = ast.get_source_segment(BACKEND.read_text(), _function(name))
        assert function is not None
        assert "getattr(" not in function
        assert '"strides"' not in function


def test_compiled_graph_key_matches_mlx_shape_dtype_layout_contract():
    source = COMPILE_BACKEND.read_text()
    assert "_array_strides" not in source
    assert "input_strides" not in source
    assert "raw_strides" not in source
    assert "pooled_strides" not in source
    assert "custom kernels bind" in source
    assert "actual C++ strides" in source


def test_supported_mpp_dispatch_is_fail_closed_and_stride_explicit():
    source = BACKEND.read_text()
    factory = ast.get_source_segment(source, _function("_prefill_mpp_score_kernel"))
    dispatch = ast.get_source_segment(
        source, _function("qsa_indexer_prefill_scores_mpp")
    )
    assert factory is not None and dispatch is not None
    assert "ensure_row_contiguous=False" in factory
    assert "try:" not in factory
    assert "except" not in factory
    assert "try:" not in dispatch
    assert "except" not in dispatch
    assert "return None" not in dispatch
    assert "raise ValueError" in dispatch


def test_mpp_runtime_parity_gate_is_prepared_but_not_imported_here():
    source = MPP_RUNTIME_GATE.read_text()
    assert "test_mpp_scores_match_vectorized_mlx_oracle" in source
    assert "test_mpp_scores_preserve_exact_selected_indices" in source
    assert "test_mpp_zero_relu_ties_keep_the_exact_selector_contract" in source
    assert "test_non_production_geometry_stays_on_the_vectorized_oracle" in source
    assert "<= 1e-4" in source
    assert "pytest.skip" not in source
    assert "is None" not in source


def test_two_stage_chunk_loop_feeds_scores_to_dedicated_topk():
    function = _function("qsa_indexer_prefill_metal")
    calls = {
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "qsa_indexer_prefill_score_chunk_rows" in calls
    assert "qsa_indexer_prefill_scores" in calls
    assert "qsa_indexer_prefill_scores_mpp_supported" in calls
    assert "qsa_indexer_prefill_scores_mpp" in calls
    assert "qsa_indexer_prefill_topk_metal" in calls
    assert any(isinstance(node, ast.For) for node in ast.walk(function))

    source = ast.get_source_segment(BACKEND.read_text(), function)
    assert source is not None
    assert "score_producer: QSAPrefillScoreProducer = (" in source
    assert '"mpp" if use_mpp_scores else "simd" if use_simd_scores else "mlx"' in source
    assert "if use_mpp_scores:" in source
    assert "if scores is None:" not in source
    assert "pooled_t = mx.swapaxes(" in source


def test_topk_kernel_preserves_mask_adjustment_and_strict_stable_key():
    source = BACKEND.read_text()
    assert "const uint valid_count = metal::min(logical, complete);" in source
    assert "scores[score_base + block] - float(block) * 1.0e-12f" in source
    assert "qsa_composite_key(adjusted, block)" in source
    assert "_INSERTION_CANDIDATES = 64" in source
    assert (
        "constant constexpr uint INSERTION_CANDIDATES = {_INSERTION_CANDIDATES};"
        in source
    )
    assert "constant constexpr uint RADIX_PASSES = 6;" in source
    assert "for (uint pass = 0; pass < RADIX_PASSES; ++pass)" in source
    assert "qsa_prefill_radix_shift(pass)" in source
    assert "qsa_prefill_radix_bits(pass)" in source
    assert "chosen_count <= FINAL_CANDIDATES" in source
    assert "final_candidate_keys[FINAL_CANDIDATES]" in source
    assert "const uint candidate_count = final_candidate_total;" in source
    assert "if (candidate_count <= INSERTION_CANDIDATES)" in source
    assert "greater_rank +=" in source
    assert "final_candidate_keys[other] > key ? 1u : 0u;" in source
    assert "if (greater_rank == radix_rank)" in source
    assert "const uint first_refine_pass = radix_next_pass;" in source
    assert "item < candidate_count" in source
    assert "threshold_key = radix_prefix;" in source
    assert "sequence <= FINAL_CANDIDATES" not in source
    assert "qsa_row_score_before" in source
    assert "qsa_index_before" in source


@pytest.mark.parametrize("count,k", [(2_049, 1), (8_193, 512), (65_536, 512)])
def test_adaptive_prefix_threshold_is_exact_for_random_strict_keys(count: int, k: int):
    rng = random.Random(0x51A + count + k)
    key_set: set[int] = set()
    while len(key_set) < count:
        key_set.add(rng.getrandbits(64))
    keys = list(key_set)
    threshold, candidates, insertion_work, radix_work = _adaptive_threshold(keys, k)
    assert candidates <= 2_048
    assert insertion_work <= 64**2
    assert radix_work <= 5 * 2_048
    assert threshold == sorted(keys, reverse=True)[k - 1]
    assert sum(key >= threshold for key in keys) == k


def test_adaptive_prefix_handles_a_large_shared_score_prefix():
    # This is the hard shape for exact QSA ties: many keys can share all score
    # bits, leaving the appended block id as the only strict ordering field.
    score_order_key = 0x8000_0000
    keys = [(score_order_key << 32) | block for block in range(65_536)]
    threshold, candidates, insertion_work, radix_work = _adaptive_threshold(keys, 512)
    assert candidates <= 2_048
    assert insertion_work <= 64**2
    assert radix_work <= 5 * 2_048
    assert threshold == keys[-512]
    assert sum(key >= threshold for key in keys) == 512


def test_large_final_bucket_uses_bounded_candidate_radix_not_quadratic_rank():
    high_bin = 0x6A5
    lower_bin = high_bin - 1
    candidates = [
        (high_bin << 53) | ((block * 0x1F12_3BB5) & ((1 << 53) - 1))
        for block in range(2_048)
    ]
    lower = [(lower_bin << 53) | block for block in range(2_048)]
    keys = candidates + lower
    threshold, final_count, insertion_work, radix_work = _adaptive_threshold(keys, 512)
    assert final_count == 2_048
    assert insertion_work == 0
    assert radix_work == 5 * 2_048
    assert threshold == sorted(keys, reverse=True)[511]
    assert sum(key >= threshold for key in keys) == 512


@pytest.mark.parametrize(
    "bucket_size,expect_insertion",
    [(64, True), (65, False)],
)
def test_candidate_resolution_switches_at_the_bounded_insertion_cutoff(
    bucket_size: int,
    expect_insertion: bool,
):
    high_bin = 0x521
    keys = [
        (high_bin << 53) | ((block * 0x10_001) & ((1 << 53) - 1))
        for block in range(bucket_size)
    ]
    keys.extend(((high_bin - 1) << 53) | block for block in range(bucket_size))
    threshold, final_count, insertion_work, radix_work = _adaptive_threshold(keys, 32)
    assert final_count == bucket_size
    assert threshold == sorted(keys, reverse=True)[31]
    if expect_insertion:
        assert insertion_work == 64**2
        assert radix_work == 0
    else:
        assert insertion_work == 0
        assert radix_work == 5 * 65


def test_topk_is_one_threadgroup_per_row_and_supports_all_output_contracts():
    source = BACKEND.read_text()
    assert "const uint row = threadgroup_position_in_grid.x;" in source
    assert "grid=(rows * width, 1, 1)" in source
    assert "threadgroup=(width, 1, 1)" in source
    for mode in ("blocks", "dense_mask", "row_tokens"):
        assert f'"{mode}"' in source


def test_compiled_core_has_a_distinct_prefill_blocks_signature():
    source = COMPILE_BACKEND.read_text()
    assert '"prefill_blocks"' in source
    assert "qsa_indexer_prefill_blocks_metal" in source
    assert "qsa_indexer_prefill_score_chunk_rows" in source
    assert "qsa_indexer_prefill_prepared_scores_mpp_supported" in source
    assert "prefill_score_workspace_bytes" in source

    tree = ast.parse(source)
    mode_alias = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "QSACompiledMode"
            for target in node.targets
        )
    )
    assert "prefill_blocks" in ast.unparse(mode_alias.value)


def test_compiled_prefill_mode_is_separate_from_existing_selector_modes():
    source = COMPILE_BACKEND.read_text()
    tree = ast.parse(source)
    core = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "QSACompiledIndexerCore"
    )
    selector = next(
        node
        for node in core.body
        if isinstance(node, ast.FunctionDef) and node.name == "_selector"
    )
    selector_source = ast.get_source_segment(source, selector)
    assert selector_source is not None
    assert 'if mode == "prefill_blocks"' in selector_source
    assert "return qsa_indexer_prefill_blocks_metal(" in selector_source
    assert 'if mode == "blocks"' in selector_source
    assert "qsa_indexer_select_blocks_metal(" in selector_source
    assert 'elif mode == "dense_mask"' in selector_source
    assert 'elif mode == "row_tokens"' in selector_source


def test_compiled_prefill_receipts_and_leaf_contract_are_explicit():
    source = COMPILE_BACKEND.read_text()
    assert '"prefill_blocks": 0' in source
    assert '"prefill_blocks": 3' in source
    assert 'mode == "prefill_blocks" and rows <= 1' in source
    assert 'report["prefill_score_workspace_bytes"]' in source
    assert "prefill_score_producer: str" in source
    assert '"prefill_score_producers": {"mpp": 0, "mlx": 0}' in source
    assert 'prefill_score_producer = "mpp" if use_mpp_scores else "mlx"' in source
    assert "producer=prefill_score_producer" in source
