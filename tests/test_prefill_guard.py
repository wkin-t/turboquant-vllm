"""测试 prefill 容量守卫：块数超出缓冲区时触发 fallback。"""
import sys
import os
import types
import importlib
import pytest
import torch
from unittest.mock import MagicMock, patch, PropertyMock


# ---------------------------------------------------------------------------
# 在导入 tq4_backend 之前，先 mock 掉所有外部依赖（vllm、triton），
# 使测试可在 CPU-only / 无 vllm 环境中运行。
# ---------------------------------------------------------------------------
def _stub(name, **attrs):
    """创建或更新 sys.modules 中的 stub module，设置指定属性。"""
    mod = sys.modules.get(name) or types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _make_all_mocks():
    # ---- vllm ----
    _stub("vllm")
    _stub("vllm.v1")
    _stub("vllm.v1.attention")
    _stub("vllm.v1.attention.backends")
    _stub(
        "vllm.v1.attention.backends.flash_attn",
        FlashAttentionBackend=type("FlashAttentionBackend", (), {}),
        FlashAttentionImpl=type("FlashAttentionImpl", (), {}),
        FlashAttentionMetadataBuilder=type("FlashAttentionMetadataBuilder", (), {}),
        FlashAttentionMetadata=type("FlashAttentionMetadata", (), {}),
    )
    _stub(
        "vllm.v1.attention.backends.registry",
        AttentionBackendEnum=type("AttentionBackendEnum", (), {}),
        register_backend=lambda *a, **kw: None,
    )
    _stub(
        "vllm.v1.kv_cache_interface",
        FullAttentionSpec=type("FullAttentionSpec", (), {}),
        SlidingWindowSpec=type("SlidingWindowSpec", (), {}),
    )
    _stub("vllm.v1.attention.backend")
    # ---- turboquant_vllm.quantizer / triton ----
    _stub(
        "turboquant_vllm.quantizer",
        TurboQuantMSE=type("TurboQuantMSE", (), {}),
        TurboQuantProd=type("TurboQuantProd", (), {}),
    )
    _stub("turboquant_vllm.triton")
    _stub("turboquant_vllm.triton.tq4_compress", tq4_compress=MagicMock())
    _stub("turboquant_vllm.triton.tq4_decompress", tq4_decompress=MagicMock())
    # ---- turboquant_vllm 顶层包：设置 __path__ 让子包解析走文件系统 ----
    src_root = os.path.join(
        os.path.dirname(__file__), "..", "src"
    )
    pkg_dir = os.path.normpath(os.path.join(src_root, "turboquant_vllm"))
    tq_top = types.ModuleType("turboquant_vllm")
    tq_top.__path__ = [pkg_dir]
    tq_top.__package__ = "turboquant_vllm"
    tq_top.__spec__ = importlib.util.spec_from_file_location(
        "turboquant_vllm",
        os.path.join(pkg_dir, "__init__.py"),
        submodule_search_locations=[pkg_dir],
    )
    sys.modules["turboquant_vllm"] = tq_top

    # ---- turboquant_vllm.vllm 子包：同样给 __path__ ----
    vllm_sub_dir = os.path.join(pkg_dir, "vllm")
    tq_vllm = types.ModuleType("turboquant_vllm.vllm")
    tq_vllm.__path__ = [vllm_sub_dir]
    tq_vllm.__package__ = "turboquant_vllm.vllm"
    sys.modules["turboquant_vllm.vllm"] = tq_vllm


_make_all_mocks()


def make_backend_with_buffer(max_blocks: int, block_size: int = 16):
    backend = MagicMock()
    backend._cg_prefill_k = torch.zeros(max_blocks * block_size, 8, 128)
    backend._cg_prefill_v = torch.zeros(max_blocks * block_size, 8, 128)
    backend._block_size = block_size
    return backend


def make_attn_metadata(num_seqs: int, seq_len: int, block_size: int = 16):
    meta = MagicMock()
    meta.seq_lens = torch.tensor([seq_len] * num_seqs)
    num_blocks = (seq_len + block_size - 1) // block_size
    meta.block_table = torch.arange(
        num_seqs * num_blocks
    ).reshape(num_seqs, num_blocks)
    return meta


def test_capacity_ok_returns_false_when_exceeded():
    from turboquant_vllm.vllm.tq4_backend import TQ4AttentionImpl
    backend = make_backend_with_buffer(max_blocks=10, block_size=16)
    meta = make_attn_metadata(num_seqs=1, seq_len=200, block_size=16)
    result = TQ4AttentionImpl._check_prefill_capacity(backend, meta)
    assert result is False


def test_capacity_ok_returns_true_when_within():
    from turboquant_vllm.vllm.tq4_backend import TQ4AttentionImpl
    backend = make_backend_with_buffer(max_blocks=20, block_size=16)
    meta = make_attn_metadata(num_seqs=1, seq_len=100, block_size=16)
    result = TQ4AttentionImpl._check_prefill_capacity(backend, meta)
    assert result is True


def test_tq4_prefill_calls_fallback_when_capacity_exceeded():
    from turboquant_vllm.vllm.tq4_backend import TQ4AttentionImpl
    backend = make_backend_with_buffer(max_blocks=5, block_size=16)
    meta = make_attn_metadata(num_seqs=1, seq_len=200, block_size=16)
    backend._fallback_prefill = MagicMock(return_value="fallback_result")
    backend._check_prefill_capacity = MagicMock(return_value=False)
    backend._decompress_cache_paged = MagicMock()
    result = TQ4AttentionImpl._tq4_prefill(
        backend,
        query=torch.zeros(200, 8, 128),
        key=torch.zeros(200, 8, 128),
        value=torch.zeros(200, 8, 128),
        kv_cache=MagicMock(),
        attn_metadata=meta,
    )
    backend._fallback_prefill.assert_called_once()
    backend._decompress_cache_paged.assert_not_called()
    assert result == "fallback_result"
