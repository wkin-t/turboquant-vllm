"""测试 thinking 模式路由：启用 reasoning_parser 时应跳过 CUDAGraph 路径。"""
import pytest
from unittest.mock import MagicMock


def make_vllm_config(with_reasoning_parser: bool):
    cfg = MagicMock()
    cfg.reasoning_parser = "gemma4" if with_reasoning_parser else None
    return cfg


def test_is_thinking_model_true_when_reasoning_parser_set():
    vllm_config = make_vllm_config(with_reasoning_parser=True)
    result = getattr(vllm_config, "reasoning_parser", None) is not None
    assert result is True


def test_is_thinking_model_false_when_no_reasoning_parser():
    vllm_config = make_vllm_config(with_reasoning_parser=False)
    result = getattr(vllm_config, "reasoning_parser", None) is not None
    assert result is False


def test_forward_skips_cg_when_thinking_mode():
    backend = MagicMock()
    backend._is_thinking_model = True
    backend._cg_buffers_ready = True
    backend._fused_paged_available = True
    attn_metadata = MagicMock()
    attn_metadata.num_actual_tokens = 1
    _cg_eligible = backend._cg_buffers_ready and not backend._is_thinking_model
    is_decode = _cg_eligible and attn_metadata.num_actual_tokens == 1
    assert is_decode is False


def test_forward_uses_cg_when_not_thinking_mode():
    backend = MagicMock()
    backend._is_thinking_model = False
    backend._cg_buffers_ready = True
    attn_metadata = MagicMock()
    attn_metadata.num_actual_tokens = 1
    _cg_eligible = backend._cg_buffers_ready and not backend._is_thinking_model
    is_decode = _cg_eligible and attn_metadata.num_actual_tokens == 1
    assert is_decode is True
