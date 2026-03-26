"""Tests for TurboQuant KV cache wrapper integration."""

import pytest
import torch
import torch.nn.functional as F

from turboquant_consumer.kv_cache import CompressedDynamicCache, TurboQuantKVCache

from .conftest import BITS, DIM


@pytest.mark.unit
class TestKVCache:
    """Validate KV cache wrapper integration with HuggingFace DynamicCache."""

    def test_basic_update(self) -> None:
        """Wrapped cache should accept and return correct tensor shapes."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = TurboQuantKVCache(cache, head_dim=DIM, bits=BITS)

        keys = torch.randn(1, 8, 1, DIM)
        values = torch.randn(1, 8, 1, DIM)

        out_k, out_v = cache.update(keys, values, layer_idx=0)
        assert out_k.shape[-1] == DIM
        assert out_v.shape[-1] == DIM

    def test_disable_passthrough(self) -> None:
        """Disabling compression should pass through unmodified tensors."""
        from transformers import DynamicCache

        head_dim = 64
        cache = DynamicCache()
        tq_cache = TurboQuantKVCache(cache, head_dim=head_dim, bits=BITS)

        tq_cache.disable()
        keys = torch.randn(1, 4, 1, head_dim)
        values = torch.randn(1, 4, 1, head_dim)

        out_k, out_v = cache.update(keys, values, layer_idx=0)
        torch.testing.assert_close(out_k[:, :, -1:, :], keys)
        torch.testing.assert_close(out_v[:, :, -1:, :], values)

    def test_restore_unwraps(self) -> None:
        """Restore should remove the wrapper entirely."""
        from transformers import DynamicCache

        cache = DynamicCache()
        original_update = cache.update
        tq_cache = TurboQuantKVCache(cache, head_dim=64, bits=BITS)
        assert cache.update != original_update

        tq_cache.restore()
        assert cache.update == original_update

    def test_keys_only_mode(self) -> None:
        """compress_values=False should only compress keys."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = TurboQuantKVCache(cache, head_dim=DIM, bits=BITS, compress_values=False)

        keys = torch.randn(1, 4, 1, DIM)
        values = torch.randn(1, 4, 1, DIM)

        out_k, out_v = cache.update(keys, values, layer_idx=0)
        # Values should be unchanged (no compression)
        torch.testing.assert_close(out_v[:, :, -1:, :], values)
        # Keys should be different (compressed then decompressed)
        assert not torch.allclose(out_k[:, :, -1:, :], keys, atol=1e-6)

    def test_values_only_mode(self) -> None:
        """compress_keys=False should only compress values."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = TurboQuantKVCache(cache, head_dim=DIM, bits=BITS, compress_keys=False)

        keys = torch.randn(1, 4, 1, DIM)
        values = torch.randn(1, 4, 1, DIM)

        out_k, out_v = cache.update(keys, values, layer_idx=0)
        # Keys should be unchanged
        torch.testing.assert_close(out_k[:, :, -1:, :], keys)
        # Values should be different (compressed then decompressed)
        assert not torch.allclose(out_v[:, :, -1:, :], values, atol=1e-6)

    def test_multi_layer_accumulation(self) -> None:
        """Cache should accumulate tokens across multiple update calls."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = TurboQuantKVCache(cache, head_dim=DIM, bits=BITS)

        for _ in range(3):
            keys = torch.randn(1, 4, 1, DIM)
            values = torch.randn(1, 4, 1, DIM)
            out_k, out_v = cache.update(keys, values, layer_idx=0)

        assert out_k.shape[2] == 3
        assert out_v.shape[2] == 3

    def test_enable_after_disable(self) -> None:
        """Re-enabling compression after disable should resume compressing."""
        from transformers import DynamicCache

        cache = DynamicCache()
        tq_cache = TurboQuantKVCache(cache, head_dim=DIM, bits=BITS)

        # Disable, add uncompressed token
        tq_cache.disable()
        cache.update(torch.randn(1, 4, 1, DIM), torch.randn(1, 4, 1, DIM), layer_idx=0)

        # Re-enable, add compressed token
        tq_cache.enable()
        out_k, out_v = cache.update(
            torch.randn(1, 4, 1, DIM), torch.randn(1, 4, 1, DIM), layer_idx=0
        )

        assert out_k.shape[2] == 2  # Both tokens accumulated


@pytest.mark.unit
class TestCompressedDynamicCache:
    """Validate CompressedDynamicCache with real VRAM savings."""

    def test_basic_update(self) -> None:
        """Wrapped cache should accept and return correct tensor shapes."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        keys = torch.randn(1, 8, 1, DIM)
        values = torch.randn(1, 8, 1, DIM)

        out_k, out_v = cache.update(keys, values, layer_idx=0)
        assert out_k.shape == (1, 8, 1, DIM)
        assert out_v.shape == (1, 8, 1, DIM)

    def test_multi_token_prefill(self) -> None:
        """Prefill with multiple tokens should compress and dequantize all."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        seq_len = 64
        keys = torch.randn(1, 4, seq_len, DIM)
        values = torch.randn(1, 4, seq_len, DIM)

        out_k, out_v = cache.update(keys, values, layer_idx=0)
        assert out_k.shape == (1, 4, seq_len, DIM)
        assert out_v.shape == (1, 4, seq_len, DIM)

    def test_output_dtype_matches_input(self) -> None:
        """Decompressed output should match the input dtype."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        keys = torch.randn(1, 4, 1, DIM, dtype=torch.bfloat16)
        values = torch.randn(1, 4, 1, DIM, dtype=torch.bfloat16)

        out_k, out_v = cache.update(keys, values, layer_idx=0)
        assert out_k.dtype == torch.bfloat16
        assert out_v.dtype == torch.bfloat16

    def test_stores_uint8_indices(self) -> None:
        """Compressed storage should use uint8 for indices."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        cache.update(torch.randn(1, 4, 1, DIM), torch.randn(1, 4, 1, DIM), layer_idx=0)

        assert cc._compressed_keys[0].indices.dtype == torch.uint8
        assert cc._compressed_values[0].indices.dtype == torch.uint8

    def test_stores_fp32_norms(self) -> None:
        """Compressed storage should use fp32 for norms (fp16 causes degradation)."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        cache.update(torch.randn(1, 4, 1, DIM), torch.randn(1, 4, 1, DIM), layer_idx=0)

        assert cc._compressed_keys[0].norms.dtype == torch.float32
        assert cc._compressed_values[0].norms.dtype == torch.float32

    def test_vram_savings(self) -> None:
        """Compressed storage should be smaller than FP16 baseline."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        # Simulate 100 tokens across 4 layers
        for layer in range(4):
            cache.update(
                torch.randn(1, 8, 100, DIM),
                torch.randn(1, 8, 100, DIM),
                layer_idx=layer,
            )

        compressed_bytes = cc.vram_bytes()
        baseline_bytes = cc.baseline_vram_bytes()

        # uint8 (1 byte) + fp32 norm (4 bytes per vector) vs fp16 (2 bytes per element)
        # 132 bytes vs 256 bytes per token per head → ~1.94x compression
        assert compressed_bytes < baseline_bytes
        ratio = baseline_bytes / compressed_bytes
        assert ratio > 1.9, f"Expected >1.9x compression, got {ratio:.2f}x"

    def test_compression_stats(self) -> None:
        """compression_stats() should return valid statistics."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        cache.update(
            torch.randn(1, 4, 50, DIM), torch.randn(1, 4, 50, DIM), layer_idx=0
        )

        stats = cc.compression_stats()
        assert stats["num_layers"] == 1
        assert stats["seq_len"] == 50
        assert stats["head_dim"] == DIM
        assert stats["bits"] == BITS
        assert stats["compression_ratio"] > 1.9
        assert stats["savings_mib"] > 0

    def test_multi_layer_accumulation(self) -> None:
        """Cache should accumulate tokens across multiple update calls."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        for _ in range(3):
            keys = torch.randn(1, 4, 1, DIM)
            values = torch.randn(1, 4, 1, DIM)
            out_k, out_v = cache.update(keys, values, layer_idx=0)

        assert out_k.shape[2] == 3
        assert out_v.shape[2] == 3

    def test_get_seq_length(self) -> None:
        """get_seq_length() should reflect compressed storage."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        assert cache.get_seq_length() == 0

        cache.update(
            torch.randn(1, 4, 10, DIM), torch.randn(1, 4, 10, DIM), layer_idx=0
        )
        assert cache.get_seq_length() == 10

        cache.update(torch.randn(1, 4, 5, DIM), torch.randn(1, 4, 5, DIM), layer_idx=0)
        assert cache.get_seq_length() == 15

    def test_disable_passthrough(self) -> None:
        """Disabling compression should pass through unmodified tensors."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        cc.disable()
        keys = torch.randn(1, 4, 1, DIM)
        values = torch.randn(1, 4, 1, DIM)

        out_k, out_v = cache.update(keys, values, layer_idx=0)
        torch.testing.assert_close(out_k[:, :, -1:, :], keys)
        torch.testing.assert_close(out_v[:, :, -1:, :], values)

    def test_restore_unwraps(self) -> None:
        """Restore should remove the wrapper entirely."""
        from transformers import DynamicCache

        cache = DynamicCache()
        original_update = cache.update
        original_get_seq = cache.get_seq_length
        cc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)
        assert cache.update != original_update

        cc.restore()
        assert cache.update == original_update
        assert cache.get_seq_length == original_get_seq

    def test_accuracy_parity_with_accuracy_only(self) -> None:
        """Compressed cache should produce same values as accuracy-only cache.

        Both compress with TurboQuantCompressorMSE at the same bit-width,
        so the decompressed outputs should be numerically identical.
        """
        from transformers import DynamicCache

        keys = torch.randn(1, 4, 10, DIM)
        values = torch.randn(1, 4, 10, DIM)

        # Accuracy-only mode
        cache_a = DynamicCache()
        _ = TurboQuantKVCache(cache_a, head_dim=DIM, bits=BITS)
        out_k_a, out_v_a = cache_a.update(keys.clone(), values.clone(), layer_idx=0)

        # Compressed mode
        cache_c = DynamicCache()
        _ = CompressedDynamicCache(cache_c, head_dim=DIM, bits=BITS)
        out_k_c, out_v_c = cache_c.update(keys.clone(), values.clone(), layer_idx=0)

        # Both go through the same compressor, so reconstructions should match.
        # Small difference due to uint8/fp16 storage precision in compressed mode.
        cos_k = torch.nn.functional.cosine_similarity(
            out_k_a.flatten(), out_k_c.flatten(), dim=0
        )
        cos_v = torch.nn.functional.cosine_similarity(
            out_v_a.flatten(), out_v_c.flatten(), dim=0
        )
        assert cos_k > 0.999, f"Key cosine similarity {cos_k:.4f} too low"
        assert cos_v > 0.999, f"Value cosine similarity {cos_v:.4f} too low"

    def test_previous_layer_freed(self) -> None:
        """Decompressed tensors from previous layers should be freed."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        # Layer 0
        cache.update(
            torch.randn(1, 4, 10, DIM), torch.randn(1, 4, 10, DIM), layer_idx=0
        )
        layer0_keys = cache.layers[0].keys
        assert layer0_keys is not None
        assert layer0_keys.numel() > 0  # Layer 0 is decompressed

        # Layer 1 should free layer 0's decompressed data
        cache.update(
            torch.randn(1, 4, 10, DIM), torch.randn(1, 4, 10, DIM), layer_idx=1
        )
        layer0_keys_after = cache.layers[0].keys
        layer1_keys = cache.layers[1].keys
        assert layer0_keys_after is not None
        assert layer0_keys_after.numel() == 0  # Layer 0 freed
        assert layer1_keys is not None
        assert layer1_keys.numel() > 0  # Layer 1 is decompressed


@pytest.mark.unit
class TestLongSequenceRegression:
    """Regression tests for precision at realistic sequence lengths.

    These tests simulate multi-layer transformer KV cache usage at
    1000+ tokens across 36 layers to catch precision bugs (like the
    fp16 norms issue) that don't manifest in short-sequence unit tests.
    """

    NUM_LAYERS = 36
    NUM_HEADS = 8
    PREFILL_LEN = 1024
    GEN_STEPS = 32

    def _run_prefill_and_generate(
        self,
        wrapper_cls: type,
        bits: int = BITS,
    ) -> list[torch.Tensor]:
        """Simulate prefill + generation through all layers.

        Args:
            wrapper_cls: TurboQuantKVCache or CompressedDynamicCache.
            bits: Quantization bit width.

        Returns:
            List of decompressed key tensors (one per layer) after
            prefill + generation steps, for the final generation step.
        """
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = wrapper_cls(cache, head_dim=DIM, bits=bits)

        # Prefill: push PREFILL_LEN tokens through all layers
        for layer_idx in range(self.NUM_LAYERS):
            k = torch.randn(1, self.NUM_HEADS, self.PREFILL_LEN, DIM)
            v = torch.randn(1, self.NUM_HEADS, self.PREFILL_LEN, DIM)
            cache.update(k, v, layer_idx=layer_idx)

        # Generation: push 1 token at a time through all layers
        final_keys: list[torch.Tensor] = []
        for step in range(self.GEN_STEPS):
            step_keys = []
            for layer_idx in range(self.NUM_LAYERS):
                k = torch.randn(1, self.NUM_HEADS, 1, DIM)
                v = torch.randn(1, self.NUM_HEADS, 1, DIM)
                out_k, _ = cache.update(k, v, layer_idx=layer_idx)
                step_keys.append(out_k)
            final_keys = step_keys

        return final_keys

    def test_compressed_matches_accuracy_only(self) -> None:
        """CompressedDynamicCache should closely match TurboQuantKVCache.

        Both use TurboQuantCompressorMSE at the same bit-width. The
        only difference is storage format (uint8+fp32 vs fp32 round-trip).
        After 36 layers and 1000+ tokens, the outputs should still be
        nearly identical. A cosine similarity below 0.999 per layer
        would indicate a precision regression.
        """
        torch.manual_seed(42)
        keys_accuracy = self._run_prefill_and_generate(TurboQuantKVCache)

        torch.manual_seed(42)
        keys_compressed = self._run_prefill_and_generate(CompressedDynamicCache)

        assert len(keys_accuracy) == self.NUM_LAYERS
        assert len(keys_compressed) == self.NUM_LAYERS

        for layer_idx in range(self.NUM_LAYERS):
            ka = keys_accuracy[layer_idx]
            kc = keys_compressed[layer_idx]
            assert ka.shape == kc.shape, (
                f"Layer {layer_idx}: shape mismatch {ka.shape} vs {kc.shape}"
            )

            cos = F.cosine_similarity(ka.flatten(), kc.flatten(), dim=0)
            assert cos > 0.999, (
                f"Layer {layer_idx}: cosine similarity {cos:.6f} < 0.999 — "
                f"precision regression detected"
            )

    def test_no_nan_or_inf_at_scale(self) -> None:
        """No NaN or Inf should appear after many layers and tokens."""
        torch.manual_seed(42)
        keys = self._run_prefill_and_generate(CompressedDynamicCache)

        for layer_idx, k in enumerate(keys):
            assert not k.isnan().any(), f"Layer {layer_idx}: NaN in keys"
            assert not k.isinf().any(), f"Layer {layer_idx}: Inf in keys"

    def test_seq_length_tracks_correctly(self) -> None:
        """get_seq_length should return prefill + generation tokens."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        # Prefill
        for layer_idx in range(self.NUM_LAYERS):
            cache.update(
                torch.randn(1, self.NUM_HEADS, self.PREFILL_LEN, DIM),
                torch.randn(1, self.NUM_HEADS, self.PREFILL_LEN, DIM),
                layer_idx=layer_idx,
            )

        assert cache.get_seq_length(0) == self.PREFILL_LEN

        # Generate 10 tokens
        for _ in range(10):
            for layer_idx in range(self.NUM_LAYERS):
                cache.update(
                    torch.randn(1, self.NUM_HEADS, 1, DIM),
                    torch.randn(1, self.NUM_HEADS, 1, DIM),
                    layer_idx=layer_idx,
                )

        assert cache.get_seq_length(0) == self.PREFILL_LEN + 10

    def test_compression_stats_at_scale(self) -> None:
        """Compression stats should reflect realistic cache sizes."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        for layer_idx in range(self.NUM_LAYERS):
            cache.update(
                torch.randn(1, self.NUM_HEADS, self.PREFILL_LEN, DIM),
                torch.randn(1, self.NUM_HEADS, self.PREFILL_LEN, DIM),
                layer_idx=layer_idx,
            )

        stats = cc.compression_stats()
        assert stats["num_layers"] == self.NUM_LAYERS
        assert stats["seq_len"] == self.PREFILL_LEN
        assert stats["num_heads"] == self.NUM_HEADS
        assert stats["compression_ratio"] > 1.9
        # At 36 layers × 8 heads × 1024 tokens, savings should be substantial
        assert stats["savings_mib"] > 50
