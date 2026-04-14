"""测试 seq_len 分桶函数的边界行为。"""
import pytest


def test_bucket_short_sequences():
    from turboquant_vllm.triton.fused_paged_tq4_attention import _seq_len_bucket
    assert _seq_len_bucket(1) == 0
    assert _seq_len_bucket(512) == 0
    assert _seq_len_bucket(2048) == 0


def test_bucket_medium_sequences():
    from turboquant_vllm.triton.fused_paged_tq4_attention import _seq_len_bucket
    assert _seq_len_bucket(2049) == 1
    assert _seq_len_bucket(4096) == 1
    assert _seq_len_bucket(8192) == 1


def test_bucket_long_sequences():
    from turboquant_vllm.triton.fused_paged_tq4_attention import _seq_len_bucket
    assert _seq_len_bucket(8193) == 2
    assert _seq_len_bucket(32768) == 2
    assert _seq_len_bucket(131072) == 2


def test_bucket_boundary_2048():
    from turboquant_vllm.triton.fused_paged_tq4_attention import _seq_len_bucket
    assert _seq_len_bucket(2048) == 0
    assert _seq_len_bucket(2049) == 1


def test_bucket_boundary_8192():
    from turboquant_vllm.triton.fused_paged_tq4_attention import _seq_len_bucket
    assert _seq_len_bucket(8192) == 1
    assert _seq_len_bucket(8193) == 2


def test_autotune_configs_include_block_n_128():
    from turboquant_vllm.triton.fused_paged_tq4_attention import _FUSED_DECODE_CONFIGS
    block_ns = {cfg.kwargs["BLOCK_N"] for cfg in _FUSED_DECODE_CONFIGS}
    assert 128 in block_ns


def test_autotune_configs_count():
    from turboquant_vllm.triton.fused_paged_tq4_attention import _FUSED_DECODE_CONFIGS
    # 3 BLOCK_N (32, 64, 128) × 2 stages × 2 warps = 12
    assert len(_FUSED_DECODE_CONFIGS) == 12
