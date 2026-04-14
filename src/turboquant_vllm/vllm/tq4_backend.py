"""TQ4 compressed KV cache attention backend for vLLM.

Phase 3c: Packed TQ4 cache layout with real VRAM savings.

The KV cache is stored as uint8 bytes in a packed TQ4 format (68 bytes
per token per head per K/V = 136 bytes total vs 512 bytes FP16 = 3.76x
compression).  Buffer allocation uses a custom ``TQ4FullAttentionSpec``
that overrides ``page_size_bytes`` so the block allocator provisions
3.76x more blocks in the same VRAM budget.  Each ``forward()`` call
decompresses the relevant blocks to FP16 and delegates to Flash Attention.

Implementation phases:
    3a (done): Passthrough skeleton -- validated plugin wiring.
    3b (done): Compress-decompress round-trip in standard FP16 cache.
    3c (this): Packed uint8 cache with real VRAM savings.
    3d: Production benchmark against vLLM baseline.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from dataclasses import fields as dc_fields
from typing import TYPE_CHECKING

import torch
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum,
    register_backend,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

from turboquant_vllm.quantizer import TurboQuantMSE
from turboquant_vllm.triton.tq4_compress import tq4_compress
from turboquant_vllm.triton.tq4_decompress import tq4_decompress

if TYPE_CHECKING:
    from vllm.v1.attention.backend import (
        AttentionCGSupport,
        AttentionImplBase,
        AttentionMetadataBuilder,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TQ4 constants
# ---------------------------------------------------------------------------

TQ4_BITS = 4
TQ4_SEED = 42

# Per-token per-head storage: head_dim/2 bytes (nibble-packed) + 4 bytes (fp32 norm)
# For head_dim=128: 64 + 4 = 68 bytes vs 256 bytes FP16 = 3.76x compression
TQ4_NORM_BYTES = 4  # fp32

# FlashAttention 2 hard limit on head dimension (SM89 Ada / SM80 Ampere).
# Layers with head_dim > this use PyTorch SDPA fallback.
_FA2_MAX_HEAD_DIM = 256


def _packed_index_size(bits: int, head_dim: int) -> int:
    """Index byte count for one token/head: nibble-packed at 4-bit, else unpacked.

    Args:
        bits: Quantization bits per coordinate.
        head_dim: Dimension of each attention head.

    Returns:
        Number of uint8 bytes needed for one token's indices.
    """
    if bits == 4:
        return head_dim // 2
    return head_dim


def _tq4_bytes_per_token(head_dim: int, bits: int = TQ4_BITS) -> int:
    """Packed byte count for one token, one KV head, one of K or V.

    Args:
        head_dim: Dimension of each attention head.
        bits: Quantization bits per coordinate.

    Returns:
        Byte count: index bytes + 4 (fp32 norm).
    """
    return _packed_index_size(bits, head_dim) + TQ4_NORM_BYTES


def _padded_slot_bytes(head_dim: int) -> int:
    """Per-head per-token byte count aligned for integer page-size ratios.

    Replaces ``next_power_of_2`` (which wasted ~49 % of KV cache memory) with
    a next-multiple-of-264 alignment that wastes < 2 % for Gemma 4.

    Background: vLLM's ``unify_kv_cache_spec_page_size`` requires
    ``max_page_size % any_page_size == 0``.  With power-of-2 rounding,
    SWA (512 B/head) and global (1024 B/head) already satisfy ratio=2, but
    waste 48–49 % of each block.  By switching to the SWA base unit (264 B =
    ``_tq4_bytes_per_token_kv(256)``), any two sizes are multiples of 264 and
    keep an integer ratio without the power-of-2 overhead.

    Gemma 4 result::

        head_dim=256 (SWA)  → 264 B/head  (0 % waste, was 48.5 %)
        head_dim=512 (full) → 528 B/head  (1.5 % waste, was 49.2 %)

    Expected context improvement: ~107 k → ~210 k tokens on RTX 4090.

    NOTE: For models with Mamba layers (e.g. Qwen3.5 MoE) that require
    power-of-2 alignment, revert to ``next_power_of_2`` by overriding this
    function.

    Args:
        head_dim: Dimension of each attention head.

    Returns:
        Byte count per token per KV head, aligned for integer page ratios.
    """
    actual = _tq4_bytes_per_token_kv(head_dim)
    # _BASE = _tq4_bytes_per_token_kv(256) = 264 for standard 4-bit TQ4.
    # Any two multiples of _BASE have an integer ratio.
    _BASE = 264
    return (actual + _BASE - 1) // _BASE * _BASE


def _tq4_bytes_per_token_kv(
    head_dim: int, k_bits: int = TQ4_BITS, v_bits: int = TQ4_BITS
) -> int:
    """Total packed bytes per token per KV head (K + V combined).

    Args:
        head_dim: Dimension of each attention head.
        k_bits: Key quantization bits.
        v_bits: Value quantization bits.

    Returns:
        Combined byte count for K and V.
    """
    return _tq4_bytes_per_token(head_dim, k_bits) + _tq4_bytes_per_token(
        head_dim, v_bits
    )


# ---------------------------------------------------------------------------
# Fused paged decode feature gate (Story 6.3)
# ---------------------------------------------------------------------------

# Try importing the fused kernel at module load time.  If Triton is missing
# or the kernel's JIT compilation fails at import, the flag stays False and
# the decompress-all path is used unconditionally.
_fused_paged_kernel_available = False
_fused_paged_tq4_decode_fn = None
try:
    from turboquant_vllm.triton.fused_paged_tq4_attention import (
        fused_paged_tq4_decode as _fused_paged_tq4_decode_fn,
    )

    _fused_paged_kernel_available = True
except (ImportError, RuntimeError) as exc:
    logger.info("Fused paged TQ4 decode kernel unavailable: %s", exc)


def _parse_fused_paged_env() -> bool:
    """Parse ``TQ4_USE_FUSED_PAGED`` environment variable.

    Returns:
        ``True`` when the env var is set to a truthy value
        (``"1"``, ``"true"``, ``"yes"``; case-insensitive).
        ``False`` for everything else including absent.
    """
    return os.environ.get("TQ4_USE_FUSED_PAGED", "").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# INT8 prefill feature gate (Story 6.4)
# ---------------------------------------------------------------------------

_int8_prefill_kernel_available = False
_fused_paged_tq4_int8_prefill_fn = None
try:
    from turboquant_vllm.triton.fused_paged_tq4_int8_prefill import (
        fused_paged_tq4_int8_prefill as _fused_paged_tq4_int8_prefill_fn,
    )

    _int8_prefill_kernel_available = True
except (ImportError, RuntimeError) as exc:
    logger.info("INT8 prefill kernel unavailable: %s", exc)


_VALID_BITS = frozenset({2, 3, 4, 5})


def _parse_kv_bits_env() -> tuple[int, int]:
    """Parse ``TQ4_K_BITS`` and ``TQ4_V_BITS`` environment variables.

    Returns:
        ``(k_bits, v_bits)`` — defaults to ``(TQ4_BITS, TQ4_BITS)`` when
        env vars are absent.

    Raises:
        ValueError: If either env var value is not a valid integer or
            is not in {2, 3, 4, 5}.
    """
    raw = {}
    try:
        raw["TQ4_K_BITS"] = os.environ.get("TQ4_K_BITS", str(TQ4_BITS))
        k_bits = int(raw["TQ4_K_BITS"])
        raw["TQ4_V_BITS"] = os.environ.get("TQ4_V_BITS", str(TQ4_BITS))
        v_bits = int(raw["TQ4_V_BITS"])
    except ValueError as exc:
        msg = f"invalid env var value (expected integer): {raw} — {exc}"
        raise ValueError(msg) from exc
    for name, val in (("TQ4_K_BITS", k_bits), ("TQ4_V_BITS", v_bits)):
        if val not in _VALID_BITS:
            msg = f"{name}={val} is invalid; must be one of {sorted(_VALID_BITS)}"
            raise ValueError(msg)
    return k_bits, v_bits


def _parse_int8_prefill_env() -> bool:
    """Parse ``TQ4_USE_INT8_PREFILL`` environment variable.

    Returns:
        ``True`` when the env var is set to a truthy value
        (``"1"``, ``"true"``, ``"yes"``; case-insensitive).
        ``False`` for everything else including absent.
    """
    return os.environ.get("TQ4_USE_INT8_PREFILL", "").lower() in (
        "1",
        "true",
        "yes",
    )


# ---------------------------------------------------------------------------
# KV cache spec (3c.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class TQ4FullAttentionSpec(FullAttentionSpec):
    """KV cache spec with TQ4 packed page size.

    Overrides ``real_page_size_bytes`` so the block allocator provisions
    buffers sized for the packed TQ4 format. Supports asymmetric K/V
    bit-widths via ``TQ4_K_BITS`` / ``TQ4_V_BITS`` env vars.
    Follows the same pattern as ``MLAAttentionSpec`` which overrides
    page size for the 656-byte FlashMLA format.
    """

    @property
    def real_page_size_bytes(self) -> int:  # noqa: D102
        # Padded slot ensures page-size divisibility with Mamba layers
        # in hybrid models (Qwen3.5). Padding bytes are unused by kernels.
        return self.block_size * self.num_kv_heads * _padded_slot_bytes(self.head_size)


@dataclass(frozen=True, kw_only=True)
class TQ4SlidingWindowSpec(SlidingWindowSpec):
    """KV cache spec for sliding-window layers with TQ4 packed page size.

    Mirrors :class:`TQ4FullAttentionSpec` but inherits from
    ``SlidingWindowSpec`` so the monkey-patch in
    :func:`register_tq4_backend` covers models that mix full and
    sliding-window attention (e.g. Gemma 4).
    """

    @property
    def real_page_size_bytes(self) -> int:  # noqa: D102
        return self.block_size * self.num_kv_heads * _padded_slot_bytes(self.head_size)


# ---------------------------------------------------------------------------
# Backend (3c.2 - 3c.3)
# ---------------------------------------------------------------------------


class TQ4MetadataBuilder(FlashAttentionMetadataBuilder):
    """Metadata builder for TQ4 with conditional CUDA graph support.

    CUDA graphs are supported for single-token decode only when the fused
    paged kernel is available; otherwise CG support is NEVER (the paged
    decompress path has dynamic allocations).  Inherits all metadata-building
    logic from Flash Attention; only the CUDA graph support level differs.
    """

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: object,
        kv_cache_spec: object,
    ) -> AttentionCGSupport:
        """Report CUDA graph support: single-token decode when fused available.

        When fused paged decode is available, decode goes through
        ``_fused_decode_path`` (CG-safe).  Otherwise, decode uses
        ``_decompress_cache_paged`` which has 10+ non-CG-safe operations
        (torch.unique, boolean indexing, dynamic allocations).
        """
        from vllm.v1.attention.backend import AttentionCGSupport

        k_bits, v_bits = _parse_kv_bits_env()
        if (
            _parse_fused_paged_env()
            and _fused_paged_kernel_available
            and k_bits == v_bits
        ):
            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
        return AttentionCGSupport.NEVER


class TQ4AttentionBackend(FlashAttentionBackend):
    """TQ4 compressed KV cache attention backend.

    Phase 3c: packed uint8 cache layout with real VRAM savings.
    The cache stores nibble-packed TQ4 indices + fp32 norms as raw bytes.
    ``get_kv_cache_shape()`` returns a 3D ``(NB, BS, bytes_per_token)``
    layout matching the packed format.
    """

    forward_includes_kv_cache_update = True

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        """Required for VLMs like Molmo2 with bidirectional visual tokens."""
        return True

    @staticmethod
    def supports_head_size(head_size: int) -> bool:
        """TQ4 uses its own Triton kernels, not FlashAttention CUDA kernels.

        Only requires head_dim to be even (nibble-packing constraint).
        Supports head_dim=512 needed by Gemma 4 global attention layers.
        """
        return head_size % 2 == 0 and head_size > 0

    @staticmethod
    def get_name() -> str:
        """Must return ``"CUSTOM"`` to match ``AttentionBackendEnum.CUSTOM``."""
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[AttentionImplBase]:
        """Return :class:`TQ4AttentionImpl`."""
        return TQ4AttentionImpl

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
        """Return :class:`TQ4MetadataBuilder` for CUDA graph support."""
        return TQ4MetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """Packed TQ4 cache: ``(num_blocks, block_size, padded_bytes)``.

        The last dimension packs K and V data for all heads as raw bytes
        with padding for hybrid model page alignment. Only the first
        ``num_kv_heads * _tq4_bytes_per_token_kv(head_size)`` bytes per
        token contain packed data; trailing bytes are unused padding.
        """
        total_bytes = num_kv_heads * _padded_slot_bytes(head_size)
        return (num_blocks, block_size, total_bytes)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        """Raise to trigger identity fallback in reshape.

        The inherited FlashAttentionBackend returns a 5-element stride
        order for the standard ``(2, NB, BS, H, D)`` shape. Our 3D
        packed layout ``(NB, BS, total_bytes)`` needs identity ordering.
        Raising ``NotImplementedError`` triggers the fallback in
        ``_reshape_kv_cache_tensors`` (same pattern as FlashMLA which
        does not implement this method at all).
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Attention implementation (3c.4 - 3c.5)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Shared scratch buffer registry for enforce-eager (non-CG) mode.
# In eager mode, transformer layers execute sequentially, so decompress
# buffers from layer i are no longer live when layer i+1 runs.  All layers
# with the same (num_kv_heads, head_size) shape share a single pair of
# buffers, reducing scratch VRAM from O(num_layers) to O(unique_shapes).
#
# Keys: (device_str, compute_dtype, num_kv_heads, head_size)
# Values: (k_tensor, v_tensor)
#
# NOT used when CUDA graphs are enabled — graph capture records tensor
# addresses, so all layer buffers must be independently addressable.
# ---------------------------------------------------------------------------
_TQ4_SHARED_DECOMPRESS_BUFS: dict[tuple, tuple["torch.Tensor", "torch.Tensor"]] = {}
_TQ4_SHARED_PREFILL_BUFS: dict[tuple, tuple["torch.Tensor", "torch.Tensor"]] = {}

class TQ4AttentionImpl(FlashAttentionImpl):
    """TQ4 attention: compress -> store -> decompress -> Flash Attention.

    Phase 3c: stores packed TQ4 bytes in a uint8 cache for real VRAM
    savings.  Each ``forward()`` call:

    1. Compresses incoming K/V tokens to TQ4 packed bytes.
    2. Scatter-writes packed bytes to the uint8 cache via ``slot_mapping``.
    3. Decompresses the full cache to FP16 for Flash Attention.
    4. Calls ``flash_attn_varlen_func`` directly with the FP16 data.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize TQ4 attention with compression primitives."""
        super().__init__(*args, **kwargs)

        # Use attributes set by super().__init__()
        head_size = self.head_size
        num_kv_heads = self.num_kv_heads

        # Resolve per-component bit-widths from env vars
        k_bits, v_bits = _parse_kv_bits_env()
        self._k_bits = k_bits
        self._v_bits = v_bits

        # TQ4 compression primitives (deterministic from seed, shared across layers)
        # Rotation matrix is dim-dependent only (not bits-dependent), so shared.
        k_quantizer = TurboQuantMSE(head_size, k_bits, seed=TQ4_SEED)
        v_quantizer = (
            k_quantizer
            if v_bits == k_bits
            else TurboQuantMSE(head_size, v_bits, seed=TQ4_SEED)
        )

        # Eagerly move primitives to the target device (D7 mod 5).
        # FlashAttentionImpl.__init__ doesn't expose device, but
        # vLLM's global config is available during model construction.
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        device = (
            vllm_config.device_config.device
            if vllm_config is not None
            else torch.device("cpu")
        )

        # Shared rotation (dim-dependent only, same seed → identical matrix)
        self._tq4_rotation = k_quantizer.rotation.to(device)  # (D, D) fp32
        # Pre-split rotation.T for fused compress kernel (contiguous loads)
        rot_t = k_quantizer.rotation.T.contiguous()
        self._tq4_rot_T_even = rot_t[:, 0::2].contiguous().to(device)  # (D, D//2) fp32
        self._tq4_rot_T_odd = rot_t[:, 1::2].contiguous().to(device)  # (D, D//2) fp32

        # Per-component codebooks (may differ when k_bits != v_bits)
        self._k_centroids = k_quantizer.codebook.centroids.to(device)
        self._k_boundaries = k_quantizer.codebook.boundaries.to(device)
        self._v_centroids = v_quantizer.codebook.centroids.to(device)
        self._v_boundaries = v_quantizer.codebook.boundaries.to(device)

        # Byte layout offsets within the last dimension of the packed cache.
        # Layout: [K_indices | K_norms | V_indices | V_norms]
        # The Triton compress kernel always nibble-packs (head_dim // 2 bytes
        # per component), regardless of bit-width. Different codebook sizes
        # provide quality improvement, not storage savings in the vLLM path.
        k_idx_size = head_size // 2
        v_idx_size = head_size // 2
        self._k_idx_size = k_idx_size
        self._v_idx_size = v_idx_size
        self._k_idx_end = num_kv_heads * k_idx_size
        self._k_norm_end = self._k_idx_end + num_kv_heads * TQ4_NORM_BYTES
        self._v_idx_end = self._k_norm_end + num_kv_heads * v_idx_size
        self._total_bytes = self._v_idx_end + num_kv_heads * TQ4_NORM_BYTES

        # CUDA graph scratch buffers (D7 mod 2) — lazy-allocated on first
        # forward() from kv_cache.shape, which is stable for engine lifetime.
        # First forward runs during vLLM warmup, before graph capture.
        self._cg_buffers_ready = False

        # Fused paged decode feature gate (Story 6.3, AC 1+6).
        # Explicit opt-in via TQ4_USE_FUSED_PAGED env var AND successful
        # kernel import.  Disabled for asymmetric configs — fused kernel
        # uses a single codebook and cannot handle k_bits != v_bits.
        self._fused_paged_available = (
            _parse_fused_paged_env()
            and _fused_paged_kernel_available
            and k_bits == v_bits
        )

        # INT8 prefill gate (Story 6.4): requires fused decode gate + its own
        # env var + successful kernel import.
        self._int8_prefill_available = (
            self._fused_paged_available
            and _parse_int8_prefill_env()
            and _int8_prefill_kernel_available
        )

        # Buffer downsizing source: scheduler knows its own max prefill length.
        # Fallback 2048 matches vLLM's default max_num_batched_tokens for
        # chunked prefill.
        self._max_prefill_len = (
            vllm_config.scheduler_config.max_num_batched_tokens
            if vllm_config is not None
            else 2048
        )

        # Decode buffer bound: max_model_len caps decompress buffer instead
        # of full cache capacity.  Fallback 6144 matches Molmo2 default.
        self._max_model_len = (
            vllm_config.model_config.max_model_len if vllm_config is not None else 6144
        )

        # Shared scratch buffers: since TQ4 attention is NEVER captured in
        # CUDAGraphs (AttentionCGSupport.NEVER), the runtime always executes
        # attention eagerly regardless of cudagraph_mode.  Layers execute
        # sequentially at runtime, so buffers for layer i are free when
        # layer i+1 starts — sharing is always safe for TQ4.
        # This also fixes PIECEWISE CUDAGraph mode: without sharing, each of
        # the 8 global-attention layers allocates ~440 MB independently
        # (~3.5 GB total), causing OOM on 24 GB GPUs at max_model_len=107k.
        self._use_shared_buffers = vllm_config is not None
        # Thinking 模式检测：reasoning_parser 存在则为 thinking 模型
        # thinking 请求走 eager 路径避免 CUDAGraph 重捕导致的延迟方差
        self._is_thinking_model = (
            getattr(vllm_config, "reasoning_parser", None) is not None
            if vllm_config is not None
            else False
        )

        logger.info(
            "TQ4AttentionImpl: %d KV heads, head_size=%d, k_bits=%d, v_bits=%d, "
            "%d bytes/token (%.2fx compression vs FP16)",
            num_kv_heads,
            head_size,
            k_bits,
            v_bits,
            self._total_bytes,
            (2 * num_kv_heads * head_size * 2) / self._total_bytes,
        )
        logger.info(
            "Fused paged TQ4 decode: %s",
            "enabled" if self._fused_paged_available else "disabled",
        )
        logger.info(
            "INT8 prefill path: %s",
            "enabled" if self._int8_prefill_available else "disabled",
        )
        logger.info(
            "TQ4 shared scratch buffers: %s",
            "enabled (enforce-eager)" if self._use_shared_buffers else "disabled (CG mode)",
        )

        # SDPA fallback for head_dim > FA2 limit (e.g. Gemma 4 global layers
        # with head_dim=512 on SM89 where only FA2 is available).
        self._use_sdpa_fallback = self.head_size > _FA2_MAX_HEAD_DIM
        if self._use_sdpa_fallback:
            logger.info(
                "TQ4: head_size=%d > %d (FA2 limit), using PyTorch SDPA "
                "fallback for attention computation",
                self.head_size,
                _FA2_MAX_HEAD_DIM,
            )

    def _init_cg_buffers(
        self, kv_cache: torch.Tensor, compute_dtype: torch.dtype
    ) -> None:
        """Pre-allocate CUDA graph scratch buffers from kv_cache shape.

        Called once during vLLM warmup (first forward), before CUDA graph
        capture.  Uses max-size + slicing (D7 pattern), NOT per-batch
        allocations.

        In enforce-eager mode, decompress/prefill buffers are shared across
        all layers with the same (num_kv_heads, head_size) shape via the
        module-level ``_TQ4_SHARED_*`` registries.  Transformer layers run
        sequentially, so their buffers are never live simultaneously — this
        reduces scratch VRAM from O(num_layers) to O(unique_shapes).

        SWA layers only attend to the last ``sliding_window`` tokens, so their
        decompress buffer is capped at ``min(sliding_window, max_model_len)``
        rather than the full ``max_model_len``, saving additional VRAM.

        Args:
            kv_cache: ``(num_blocks, block_size, total_bytes)`` uint8 cache.
            compute_dtype: Model compute dtype (e.g. ``torch.bfloat16``).
        """
        num_blocks, block_size, _ = kv_cache.shape
        self._block_size = block_size
        max_tokens = num_blocks * block_size
        device = kv_cache.device
        H = self.num_kv_heads
        D = self.head_size

        # SWA layers only attend to the last ``sliding_window`` tokens; cap the
        # decompress buffer to that window size.  Global attention layers have
        # sliding_window=None or 0 and use the full max_model_len.
        sliding_window = getattr(self, "sliding_window", None)
        # vLLM ≥0.8 may store sliding_window as a tuple (e.g. Gemma4 SWA layers).
        # Normalise to a plain int so subsequent comparisons don't TypeError.
        if isinstance(sliding_window, (tuple, list)):
            sliding_window = sliding_window[0] if sliding_window else None
        effective_max = (
            min(sliding_window, self._max_model_len)
            if (sliding_window is not None and sliding_window > 0)
            else self._max_model_len
        )
        decompress_tokens = min(effective_max, max_tokens)

        # --- Decompress buffers (decode path) ---
        if self._use_shared_buffers:
            buf_key = (str(device), compute_dtype, H, D)
            existing = _TQ4_SHARED_DECOMPRESS_BUFS.get(buf_key)
            if existing is None or existing[0].shape[0] < decompress_tokens:
                k = torch.empty(
                    decompress_tokens, H, D, dtype=compute_dtype, device=device
                )
                v = torch.empty_like(k)
                _TQ4_SHARED_DECOMPRESS_BUFS[buf_key] = (k, v)
            self._cg_decompress_k, self._cg_decompress_v = (
                _TQ4_SHARED_DECOMPRESS_BUFS[buf_key]
            )
        else:
            self._cg_decompress_k = torch.empty(
                decompress_tokens, H, D, dtype=compute_dtype, device=device
            )
            self._cg_decompress_v = torch.empty_like(self._cg_decompress_k)

        # --- Prefill scratch buffers ---
        prefill_tokens = min(self._max_prefill_len, max_tokens)
        if self._use_shared_buffers:
            pfill_key = (str(device), compute_dtype, H, D, "prefill")
            existing = _TQ4_SHARED_PREFILL_BUFS.get(pfill_key)
            if existing is None or existing[0].shape[0] < prefill_tokens:
                k = torch.empty(
                    prefill_tokens, H, D, dtype=compute_dtype, device=device
                )
                v = torch.empty_like(k)
                _TQ4_SHARED_PREFILL_BUFS[pfill_key] = (k, v)
            self._cg_prefill_k, self._cg_prefill_v = (
                _TQ4_SHARED_PREFILL_BUFS[pfill_key]
            )
        else:
            self._cg_prefill_k = torch.empty(
                prefill_tokens, H, D, dtype=compute_dtype, device=device
            )
            self._cg_prefill_v = torch.empty_like(self._cg_prefill_k)

        self._max_prefill_blocks = prefill_tokens // block_size

        # Compress output buffers for one decode step (single token).
        # K and V may have different index sizes — allocate for the larger.
        max_idx_size = max(self._k_idx_size, self._v_idx_size)
        self._cg_compress_packed = torch.empty(
            1, H, max_idx_size, dtype=torch.uint8, device=device
        )
        self._cg_compress_norms = torch.empty(
            1, H, 1, dtype=torch.float32, device=device
        )

        # Q rotation buffer for decode (single token, fp32 for precision)
        self._cg_q_rot = torch.empty(
            1, self.num_heads, D, dtype=torch.float32, device=device
        )

        # Q rotation cast buffer (compute dtype for Flash Attention input)
        self._cg_q_rot_cast = torch.empty(
            1, self.num_heads, D, dtype=compute_dtype, device=device
        )

        # Compress row assembly buffer for _compress_and_store
        self._cg_compress_row = torch.empty(
            1, self._total_bytes, dtype=torch.uint8, device=device
        )

        self._cg_buffers_ready = True
        dtype_bytes = self._cg_decompress_k.element_size()
        shared_tag = " [shared]" if self._use_shared_buffers else ""
        swa_tag = (
            f" [SWA-capped={sliding_window}]"
            if (sliding_window is not None and sliding_window > 0)
            else ""
        )
        prefill_mib = prefill_tokens * H * D * dtype_bytes / (1024 * 1024)
        logger.info(
            "TQ4 CUDA graph buffers allocated: decompress=%s"
            " (tokens=%d%s%s, 2×%.1f MiB), "
            "prefill=%s (2×%.1f MiB, %d blocks), "
            "compress+row+q_rot=%.1f KiB",
            self._cg_decompress_k.shape,
            decompress_tokens,
            swa_tag,
            shared_tag,
            decompress_tokens * H * D * dtype_bytes / (1024 * 1024),
            self._cg_prefill_k.shape,
            prefill_mib,
            self._max_prefill_blocks,
            (max_idx_size * H + 4 * H + self.num_heads * D * 4 + self._total_bytes)
            / 1024,
        )

    # ----- packed cache operations (3c.4 - 3c.5) -----

    def _compress_and_store(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        *,
        compress_out: tuple[torch.Tensor, torch.Tensor] | None = None,
        row_out: torch.Tensor | None = None,
    ) -> None:
        """Compress K/V and scatter-write TQ4 bytes to packed cache.

        Args:
            key: ``(N, H, D)`` new key tokens.
            value: ``(N, H, D)`` new value tokens.
            kv_cache: ``(NB, BS, padded_bytes)`` uint8 packed cache.
                Only ``[:, :, :_total_bytes]`` contains packed data.
            slot_mapping: ``(num_actual_tokens,)`` flat slot indices.
            compress_out: Optional pre-allocated ``(packed, norms)`` buffers
                for tq4_compress (D7 CUDA graph decode path).
            row_out: Optional pre-allocated row assembly buffer ``(N, total_bytes)``
                uint8 (D7 CUDA graph decode path).
        """
        N = key.shape[0]
        H = self.num_kv_heads

        # Build packed byte row per token: [K_idx | K_norm | V_idx | V_norm]
        # When compress_out is shared between K and V, we must copy K's
        # result into the row before V overwrites the shared buffer.
        row = (
            row_out[:N]
            if row_out is not None
            else torch.empty(N, self._total_bytes, dtype=torch.uint8, device=key.device)
        )

        k_packed, k_norms = tq4_compress(
            key,
            self._tq4_rot_T_even,
            self._tq4_rot_T_odd,
            self._k_boundaries,
            out=compress_out,
        )
        row[:, : self._k_idx_end] = k_packed.reshape(N, -1)
        row[:, self._k_idx_end : self._k_norm_end] = (
            k_norms.reshape(N, H).contiguous().view(torch.uint8)
        )

        v_packed, v_norms = tq4_compress(
            value,
            self._tq4_rot_T_even,
            self._tq4_rot_T_odd,
            self._v_boundaries,
            out=compress_out,
        )
        row[:, self._k_norm_end : self._v_idx_end] = v_packed.reshape(N, -1)
        row[:, self._v_idx_end :] = v_norms.reshape(N, H).contiguous().view(torch.uint8)

        # Scatter-write to flat cache using slot_mapping.
        # Cache may be padded (hybrid model alignment), so use actual
        # last dim and write only the packed data columns.
        num_actual = slot_mapping.shape[0]
        flat_cache = kv_cache.view(-1, kv_cache.shape[-1])
        flat_cache[slot_mapping[:num_actual], : self._total_bytes] = row[:num_actual]

    def _decompress_cache(
        self,
        kv_cache: torch.Tensor,
        compute_dtype: torch.dtype,
        *,
        apply_rotation: bool = True,
        out_k: torch.Tensor | None = None,
        out_v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompress packed uint8 cache -> key_cache, value_cache.

        Uses the fused Triton kernel (Phase 3c.8) for decompress.  When
        ``apply_rotation=False``, output stays in rotated space and the
        caller must pre-rotate Q by ``Pi^T`` and post-rotate the output
        by ``Pi``.  Default ``True`` applies unrotation for backward
        compatibility with tests.

        Args:
            kv_cache: ``(NB, BS, padded_bytes)`` uint8 packed cache.
                Only ``[:, :, :_total_bytes]`` contains packed data.
            compute_dtype: Output dtype (e.g., ``torch.bfloat16``).
            apply_rotation: If ``True`` (default), apply unrotation to
                return tensors in original space.  ``False`` returns
                rotated-space tensors for the optimized forward path.
            out_k: Optional pre-allocated ``(max_tokens, H, D)`` buffer for
                decompressed keys (D7 CUDA graph decode path).
            out_v: Optional pre-allocated ``(max_tokens, H, D)`` buffer for
                decompressed values (D7 CUDA graph decode path).

        Returns:
            key_cache: ``(NB, BS, H, D)`` in ``compute_dtype``.
            value_cache: ``(NB, BS, H, D)`` in ``compute_dtype``.
        """
        NB, BS, _ = kv_cache.shape
        H = self.num_kv_heads
        k_idx_size = self._k_idx_size
        v_idx_size = self._v_idx_size
        D = self.head_size

        flat = kv_cache.reshape(NB * BS, -1)

        # Extract K regions
        k_packed = flat[:, : self._k_idx_end].contiguous().reshape(-1, H, k_idx_size)
        k_norms = (
            flat[:, self._k_idx_end : self._k_norm_end]
            .contiguous()
            .view(torch.float32)
            .reshape(-1, H, 1)
        )

        # Extract V regions
        v_packed = (
            flat[:, self._k_norm_end : self._v_idx_end]
            .contiguous()
            .reshape(-1, H, v_idx_size)
        )
        v_norms = (
            flat[:, self._v_idx_end : self._total_bytes]
            .contiguous()
            .view(torch.float32)
            .reshape(-1, H, 1)
        )

        # Fused Triton decompress (no rotation applied)
        key_out = tq4_decompress(
            k_packed, k_norms, self._k_centroids, compute_dtype, out=out_k
        )
        value_out = tq4_decompress(
            v_packed, v_norms, self._v_centroids, compute_dtype, out=out_v
        )

        # Optionally unrotate (backward compat for tests; forward() skips this)
        if apply_rotation:
            key_out = (key_out.float() @ self._tq4_rotation).to(compute_dtype)
            value_out = (value_out.float() @ self._tq4_rotation).to(compute_dtype)

        return key_out.reshape(NB, BS, H, D), value_out.reshape(NB, BS, H, D)

    def _decompress_cache_paged(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        compute_dtype: torch.dtype,
        *,
        out_k: torch.Tensor,
        out_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decompress only the physical blocks referenced by block_table.

        Not CUDA-graph-safe: uses ``torch.unique`` (variable-length output)
        and conditional branching on runtime tensor values.

        Instead of decompressing the entire cache (``NB*BS`` tokens), this
        extracts the unique physical blocks actually referenced by the
        current batch's ``block_table``, decompresses them contiguously,
        and returns a remapped block table for Flash Attention.

        Args:
            kv_cache: ``(NB, BS, padded_bytes)`` uint8 packed cache.
                Only ``[:, :, :_total_bytes]`` contains packed data.
            block_table: ``(batch, max_blocks_per_seq)`` int32 block table.
            seq_lens: ``(batch,)`` int32 sequence lengths.
            compute_dtype: Output dtype (e.g., ``torch.bfloat16``).
            out_k: Pre-allocated ``(max_tokens, H, D)`` buffer for keys.
            out_v: Pre-allocated ``(max_tokens, H, D)`` buffer for values.

        Returns:
            ``(key_cache, value_cache, remapped_block_table)`` where
            key/value are ``(num_compact_blocks, BS, H, D)`` and
            remapped_block_table maps logical blocks to compact indices.
        """
        NB, BS, _ = kv_cache.shape
        H = self.num_kv_heads
        k_idx_size = self._k_idx_size
        v_idx_size = self._v_idx_size
        D = self.head_size

        # Extract valid block indices from block_table using seq_lens
        max_blocks_per_seq = block_table.shape[1]
        blocks_needed = (seq_lens + BS - 1) // BS  # ceil division
        # Build mask of valid entries
        col_idx = torch.arange(max_blocks_per_seq, device=block_table.device).unsqueeze(
            0
        )
        valid_mask = col_idx < blocks_needed.unsqueeze(1)
        valid_block_indices = block_table[valid_mask]

        unique_blocks = torch.unique(valid_block_indices, sorted=True)
        num_unique = unique_blocks.numel()

        # Capacity check: derive from buffer shape so method works with
        # any pre-allocated buffer (prefill or decode sized).
        max_blocks_capacity = out_k.shape[0] // BS
        if num_unique <= max_blocks_capacity:
            k_buf = out_k
            v_buf = out_v
        else:
            logger.warning(
                "Paged decompress: %d unique blocks exceed "
                "pre-allocated capacity (%d blocks), using dynamic fallback",
                num_unique,
                max_blocks_capacity,
            )
            fallback_tokens = num_unique * BS
            k_buf = torch.empty(
                fallback_tokens, H, D, dtype=compute_dtype, device=kv_cache.device
            )
            v_buf = torch.empty_like(k_buf)

        # Gather referenced blocks and decompress
        selected = kv_cache[unique_blocks]  # (num_unique, BS, padded_bytes)
        flat = selected.reshape(num_unique * BS, -1)

        k_packed = flat[:, : self._k_idx_end].contiguous().reshape(-1, H, k_idx_size)
        k_norms = (
            flat[:, self._k_idx_end : self._k_norm_end]
            .contiguous()
            .view(torch.float32)
            .reshape(-1, H, 1)
        )
        v_packed = (
            flat[:, self._k_norm_end : self._v_idx_end]
            .contiguous()
            .reshape(-1, H, v_idx_size)
        )
        v_norms = (
            flat[:, self._v_idx_end : self._total_bytes]
            .contiguous()
            .view(torch.float32)
            .reshape(-1, H, 1)
        )

        # Slice output buffers to exact size needed
        k_out_slice = k_buf[: num_unique * BS]
        v_out_slice = v_buf[: num_unique * BS]

        key_out = tq4_decompress(
            k_packed, k_norms, self._k_centroids, compute_dtype, out=k_out_slice
        )
        value_out = tq4_decompress(
            v_packed, v_norms, self._v_centroids, compute_dtype, out=v_out_slice
        )

        key_cache = key_out.reshape(num_unique, BS, H, D)
        value_cache = value_out.reshape(num_unique, BS, H, D)

        # Build remapped block table: old physical → compact 0..N-1
        remap = torch.zeros(NB, dtype=block_table.dtype, device=block_table.device)
        remap[unique_blocks] = torch.arange(
            num_unique, dtype=block_table.dtype, device=block_table.device
        )
        remapped_block_table = remap[block_table]

        return key_cache, value_cache, remapped_block_table

    # ----- TQ4 encode / decode helpers -----

    def _tq4_decode(
        self, query, key, value, kv_cache, attn_metadata
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode path: compress, rotate Q, paged decompress with bounded buffers."""
        if key is not None and value is not None:
            self._compress_and_store(
                key,
                value,
                kv_cache,
                attn_metadata.slot_mapping,
                compress_out=(self._cg_compress_packed, self._cg_compress_norms),
                row_out=self._cg_compress_row,
            )

        q_slice = query[: attn_metadata.num_actual_tokens]
        q_rot_buf = self._cg_q_rot[:1]
        torch.matmul(q_slice.float(), self._tq4_rotation.T, out=q_rot_buf)
        self._cg_q_rot_cast[:1].copy_(q_rot_buf)

        key_cache, value_cache, remapped_bt = self._decompress_cache_paged(
            kv_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            query.dtype,
            out_k=self._cg_decompress_k,
            out_v=self._cg_decompress_v,
        )
        return self._cg_q_rot_cast[:1], key_cache, value_cache, remapped_bt

    def _check_prefill_capacity(self, attn_metadata) -> bool:
        """检查当前批次 KV 块数是否在静态缓冲区容量以内。
        True → 可走 TQ4 路径；False → 需 fallback。
        """
        block_table = attn_metadata.block_table
        seq_lens = attn_metadata.seq_lens
        BS = self._block_size

        blocks_needed = (seq_lens + BS - 1) // BS
        max_cols = block_table.shape[1]
        col_idx = torch.arange(max_cols, device=block_table.device)
        valid_mask = col_idx.unsqueeze(0) < blocks_needed.unsqueeze(1)
        valid_block_indices = block_table[valid_mask]
        num_unique = torch.unique(valid_block_indices).numel()

        max_blocks_capacity = self._cg_prefill_k.shape[0] // BS
        return num_unique <= max_blocks_capacity

    def _fallback_prefill(self, query, key, value, kv_cache, attn_metadata):
        """容量超出时 fallback 到标准 attention，避免触发大型动态内存分配。"""
        import logging
        import torch.nn.functional as F
        logging.getLogger(__name__).warning(
            "TQ4 prefill buffer capacity exceeded — falling back to standard "
            "attention. To fix permanently, see C-medium plan."
        )
        # 直接在原始 Q/K/V 上做 scaled_dot_product_attention，
        # 完全绕过 _decompress_cache_paged 的动态分配路径。
        num_actual_tokens = attn_metadata.num_actual_tokens
        q = query[:num_actual_tokens]
        # 转为 (batch=1, heads, seq, head_dim) 格式
        q_4d = q.unsqueeze(0).transpose(1, 2)
        k_4d = key[:num_actual_tokens].unsqueeze(0).transpose(1, 2)
        v_4d = value[:num_actual_tokens].unsqueeze(0).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q_4d, k_4d, v_4d,
            scale=self.scale,
            is_causal=True,
            enable_gqa=self.num_heads != self.num_kv_heads,
        )
        # 返回与 _tq4_prefill 相同的四元组格式，后两项置 None 表示无需解压缓存
        return out.squeeze(0).transpose(0, 1), None, None, None

    def _tq4_prefill(
        self, query, key, value, kv_cache, attn_metadata
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prefill path: compress, rotate Q, paged decompress with bounded buffers."""
        # --- 容量守卫（短期 256K 防崩溃）---
        if kv_cache is not None and not self._check_prefill_capacity(attn_metadata):
            return self._fallback_prefill(query, key, value, kv_cache, attn_metadata)
        # --- 原有逻辑继续（保持原样）---
        num_actual_tokens = attn_metadata.num_actual_tokens
        if kv_cache is not None and key is not None and value is not None:
            self._compress_and_store(key, value, kv_cache, attn_metadata.slot_mapping)

        q_slice = query[:num_actual_tokens]
        q_rot = (q_slice.float() @ self._tq4_rotation.T).to(q_slice.dtype)

        key_cache, value_cache, remapped_bt = self._decompress_cache_paged(
            kv_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            query.dtype,
            out_k=self._cg_prefill_k,
            out_v=self._cg_prefill_v,
        )
        return q_rot, key_cache, value_cache, remapped_bt

    # ----- fused decode path (Story 6.3) -----

    def _fused_decode_path(
        self, query, key, value, kv_cache, attn_metadata, output
    ) -> torch.Tensor:
        """Fused paged decode: compress → fused kernel (in-place attention + rotation).

        Replaces the decompress-all → FlashAttn → post-rotate pipeline
        with a single ``fused_paged_tq4_decode()`` call.  The fused
        wrapper handles Q pre-rotation, in-tile TQ4 decompression,
        attention scoring, and output post-rotation.
        """
        num_actual_tokens = attn_metadata.num_actual_tokens

        # Step 1: Compress and store new tokens (same as decompress-all path)
        if key is not None and value is not None:
            self._compress_and_store(
                key,
                value,
                kv_cache,
                attn_metadata.slot_mapping,
                compress_out=(self._cg_compress_packed, self._cg_compress_norms),
                row_out=self._cg_compress_row,
            )

        # Step 2: Fused kernel — handles Q rotation, paged decompression,
        # attention, and output post-rotation in one call.
        q_slice = query[:num_actual_tokens]
        # Guaranteed non-None: _fused_paged_available requires successful import.
        assert _fused_paged_tq4_decode_fn is not None
        _fused_paged_tq4_decode_fn(
            q_slice,
            kv_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            self._k_centroids,
            self._tq4_rotation,
            self.num_kv_heads,
            self.head_size,
            kv_cache.shape[1],  # block_size
            self.scale,
            out=output[:num_actual_tokens],
        )

        return output

    # ----- INT8 prefill path (Story 6.4) -----

    def _int8_prefill_path(
        self, query, key, value, kv_cache, attn_metadata, output
    ) -> torch.Tensor:
        """Fused paged INT8 prefill: compress → INT8 fused kernel.

        Uses IMMA tensor cores for Q@K^T (INT8) while P@V stays FP16.
        Same compression pipeline as decompress-all prefill — no changes
        to ``_compress_and_store()``.
        """
        num_actual_tokens = attn_metadata.num_actual_tokens

        # Step 1: Compress and store (same as decompress-all path)
        if key is not None and value is not None:
            self._compress_and_store(
                key,
                value,
                kv_cache,
                attn_metadata.slot_mapping,
            )

        # Step 2: INT8 fused kernel
        q_slice = query[:num_actual_tokens]
        assert _fused_paged_tq4_int8_prefill_fn is not None
        _fused_paged_tq4_int8_prefill_fn(
            q_slice,
            kv_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            self._k_centroids,
            self._tq4_rotation,
            self.num_kv_heads,
            self.head_size,
            kv_cache.shape[1],  # block_size
            self.scale,
            out=output[:num_actual_tokens],
        )

        return output

    # ----- SDPA fallback for head_dim > FA2 limit -----

    def _sdpa_varlen_attention(
        self,
        q: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        output: torch.Tensor,
        attn_metadata,
    ) -> None:
        """SDPA fallback for head_dim > FA2 limit (e.g. 512).

        Gathers per-sequence KV from paged block layout and calls
        ``torch.nn.functional.scaled_dot_product_attention`` per sequence.
        Slower than FlashAttention but supports arbitrary head dimensions.

        Args:
            q: ``(num_actual_tokens, num_heads, head_size)`` rotated query.
            key_cache: ``(num_compact_blocks, block_size, num_kv_heads, head_size)``.
            value_cache: same shape as key_cache.
            block_table: ``(batch_size, max_blocks_per_seq)`` remapped indices.
            output: ``(num_actual_tokens, num_heads, head_size)`` output buffer.
            attn_metadata: Attention metadata with query_start_loc, seq_lens.
        """
        import torch.nn.functional as F

        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        block_size = key_cache.shape[1]
        batch_size = seq_lens.shape[0]
        is_causal = attn_metadata.causal
        has_softcap = self.logits_soft_cap > 0

        for i in range(batch_size):
            q_start = query_start_loc[i].item()
            q_end = query_start_loc[i + 1].item()
            q_len = q_end - q_start
            seq_len = seq_lens[i].item()

            q_seq = q[q_start:q_end]

            num_blocks_needed = (seq_len + block_size - 1) // block_size
            blk_idx = block_table[i, :num_blocks_needed]
            k_seq = key_cache[blk_idx].reshape(
                -1, self.num_kv_heads, self.head_size
            )[:seq_len]
            v_seq = value_cache[blk_idx].reshape(
                -1, self.num_kv_heads, self.head_size
            )[:seq_len]

            q_4d = q_seq.unsqueeze(0).transpose(1, 2)
            k_4d = k_seq.unsqueeze(0).transpose(1, 2)
            v_4d = v_seq.unsqueeze(0).transpose(1, 2)

            if has_softcap:
                num_groups = self.num_heads // self.num_kv_heads
                if num_groups > 1:
                    k_4d = k_4d.repeat_interleave(num_groups, dim=1)
                    v_4d = v_4d.repeat_interleave(num_groups, dim=1)

                scores = (
                    torch.matmul(q_4d, k_4d.transpose(-2, -1)) * self.scale
                )
                scores = (
                    torch.tanh(scores / self.logits_soft_cap)
                    * self.logits_soft_cap
                )

                if is_causal and q_len > 1:
                    row_idx = torch.arange(q_len, device=q.device).unsqueeze(1)
                    col_idx = torch.arange(seq_len, device=q.device).unsqueeze(0)
                    mask = col_idx <= (row_idx + seq_len - q_len)
                    scores = scores.masked_fill(
                        ~mask.unsqueeze(0).unsqueeze(0), float("-inf")
                    )

                attn_weights = torch.softmax(
                    scores, dim=-1, dtype=torch.float32
                ).to(q.dtype)
                out_seq = torch.matmul(attn_weights, v_4d)
            else:
                out_seq = F.scaled_dot_product_attention(
                    q_4d,
                    k_4d,
                    v_4d,
                    scale=self.scale,
                    is_causal=is_causal and q_len > 1,
                    enable_gqa=self.num_heads != self.num_kv_heads,
                )

            output[q_start:q_end] = out_seq.squeeze(0).transpose(0, 1)

    # ----- forward -----

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
    ):
        """TQ4 attention: compress -> store -> pre-rotate Q -> decompress -> FA -> post-rotate.

        Phase 3c.8: Uses fused Triton decompress (no rotation). The
        rotation is applied to Q before attention and to the output
        after, saving O(cache_len) matmuls per decode step.
        """
        assert output is not None

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "Fused output quantization is not supported with TQ4 backend"
            )

        # Profiling mode
        if attn_metadata is None:
            output.zero_()
            return output

        # Warmup with no cache allocated yet
        if kv_cache is None:
            output.zero_()
            return output

        # Encoder attention: no TQ4, delegate to parent
        # (VIT uses a separate backend, but guard just in case)
        from vllm.v1.attention.backend import AttentionType

        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return self._forward_encoder_attention(
                query[: attn_metadata.num_actual_tokens],
                key[: attn_metadata.num_actual_tokens],
                value[: attn_metadata.num_actual_tokens],
                output[: attn_metadata.num_actual_tokens],
                attn_metadata,
                layer,
            )

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Lazy-init CUDA graph buffers on first forward (during warmup)
        if not self._cg_buffers_ready and kv_cache is not None:
            self._init_cg_buffers(kv_cache, compute_dtype=query.dtype)

        # Steps 1-3: compress, rotate Q, decompress (decode vs prefill path)
        # thinking 模式不走 CUDAGraph：thinking 序列长度持续增长会触发图重捕
        _cg_eligible = self._cg_buffers_ready and not self._is_thinking_model
        is_decode = _cg_eligible and num_actual_tokens == 1

        # Fused paged decode (Story 6.3): single kernel replaces
        # decompress + FlashAttn + post-rotate for decode steps.
        if self._fused_paged_available and is_decode:
            return self._fused_decode_path(
                query, key, value, kv_cache, attn_metadata, output
            )

        # INT8 prefill (Story 6.4): IMMA tensor core Q@K^T for prefill.
        # Guard: kernel is single-sequence only; fall back for multi-sequence
        # batches (vLLM scheduler may combine multiple requests).
        if (
            self._int8_prefill_available
            and not is_decode
            and attn_metadata.seq_lens.shape[0] == 1
        ):
            return self._int8_prefill_path(
                query, key, value, kv_cache, attn_metadata, output
            )

        if is_decode:
            q_rot, key_cache, value_cache, fa_block_table = self._tq4_decode(
                query, key, value, kv_cache, attn_metadata
            )
        else:
            q_rot, key_cache, value_cache, fa_block_table = self._tq4_prefill(
                query, key, value, kv_cache, attn_metadata
            )

        # Step 4: Run attention with rotated Q and rotated KV
        if attn_metadata.use_cascade:
            raise NotImplementedError("TQ4 does not yet support cascade attention")

        if self._use_sdpa_fallback:
            # head_dim > FA2 limit -- use PyTorch SDPA
            self._sdpa_varlen_attention(
                q_rot, key_cache, value_cache, fa_block_table,
                output[:num_actual_tokens], attn_metadata,
            )
        else:
            # head_dim <= FA2 limit -- use FlashAttention
            from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

            descale_shape = (
                attn_metadata.query_start_loc.shape[0] - 1,
                self.num_kv_heads,
            )
            q_descale = layer._q_scale.expand(descale_shape)
            k_descale = layer._k_scale.expand(descale_shape)
            v_descale = layer._v_scale.expand(descale_shape)

            flash_attn_varlen_func(
                q=q_rot,
                k=key_cache,
                v=value_cache,
                out=output[:num_actual_tokens],
                cu_seqlens_q=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                seqused_k=attn_metadata.seq_lens,
                max_seqlen_k=attn_metadata.max_seq_len,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                alibi_slopes=self.alibi_slopes,
                window_size=list(self.sliding_window)
                if self.sliding_window is not None
                else None,
                block_table=fa_block_table,
                softcap=self.logits_soft_cap,
                scheduler_metadata=attn_metadata.scheduler_metadata,
                fa_version=self.vllm_flash_attn_version,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                num_splits=attn_metadata.max_num_splits,
                s_aux=self.sinks,
            )

        # Step 5: Post-rotate output by Pi (undo rotation space)
        out_slice = output[:num_actual_tokens]
        output[:num_actual_tokens] = (out_slice.float() @ self._tq4_rotation).to(
            out_slice.dtype
        )

        return output


# ---------------------------------------------------------------------------
# Registration (3c.1 -- monkey-patch for TQ4 page size)
# ---------------------------------------------------------------------------

_original_get_kv_cache_spec = None


def register_tq4_backend() -> None:
    """Register TQ4 as the CUSTOM attention backend.

    In addition to registering the backend class, this monkey-patches
    ``Attention.get_kv_cache_spec`` so that decoder attention layers
    return :class:`TQ4FullAttentionSpec` or :class:`TQ4SlidingWindowSpec`
    (with ``dtype=torch.uint8`` and TQ4-sized pages) instead of the
    standard ``FullAttentionSpec`` / ``SlidingWindowSpec``.

    Called automatically by the ``vllm.general_plugins`` entry point,
    or manually before starting vLLM::

        from turboquant_vllm.vllm import register_tq4_backend

        register_tq4_backend()
        # then start vLLM with --attention-backend CUSTOM
    """
    global _original_get_kv_cache_spec  # noqa: PLW0603

    register_backend(
        AttentionBackendEnum.CUSTOM,
        "turboquant_vllm.vllm.tq4_backend.TQ4AttentionBackend",
    )

    # Register TQ4 specs in the KV cache manager mapping.
    # vLLM uses exact type() match, not isinstance(), so subclasses
    # of FullAttentionSpec / SlidingWindowSpec must be explicitly added.
    from vllm.v1.core.single_type_kv_cache_manager import spec_manager_map

    if TQ4FullAttentionSpec not in spec_manager_map:
        spec_manager_map[TQ4FullAttentionSpec] = spec_manager_map[FullAttentionSpec]

    if TQ4SlidingWindowSpec not in spec_manager_map:
        spec_manager_map[TQ4SlidingWindowSpec] = spec_manager_map[SlidingWindowSpec]

    # Monkey-patch Attention.get_kv_cache_spec to return TQ4 spec
    from vllm.model_executor.layers.attention.attention import Attention

    if _original_get_kv_cache_spec is None:
        _original_get_kv_cache_spec = Attention.get_kv_cache_spec

    def _tq4_get_kv_cache_spec(self, vllm_config):
        spec = _original_get_kv_cache_spec(self, vllm_config)
        if isinstance(spec, FullAttentionSpec) and not isinstance(
            spec, TQ4FullAttentionSpec
        ):
            kwargs = {f.name: getattr(spec, f.name) for f in dc_fields(spec)}
            kwargs["dtype"] = torch.uint8
            return TQ4FullAttentionSpec(**kwargs)
        if isinstance(spec, SlidingWindowSpec) and not isinstance(
            spec, TQ4SlidingWindowSpec
        ):
            kwargs = {f.name: getattr(spec, f.name) for f in dc_fields(spec)}
            kwargs["dtype"] = torch.uint8
            return TQ4SlidingWindowSpec(**kwargs)
        return spec

    Attention.get_kv_cache_spec = _tq4_get_kv_cache_spec
    logger.info("TQ4 attention backend registered as CUSTOM (packed cache)")
