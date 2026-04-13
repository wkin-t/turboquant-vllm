"""Test TQ4 compression with heterogeneous head_dim (Gemma 4 scenario)."""
import pytest
import torch
from turboquant_vllm import TurboQuantMSE
from turboquant_vllm.triton.tq4_compress import tq4_compress
from turboquant_vllm.triton.tq4_decompress import tq4_decompress

TQ4_SEED = 42


@pytest.mark.parametrize("head_dim", [256, 512])
def test_compress_decompress_roundtrip(head_dim):
    """Verify compress->decompress preserves signal for Gemma 4 head dims."""
    N, H = 4, 8
    x = torch.randn(N, H, head_dim, device="cuda", dtype=torch.float16)

    quantizer = TurboQuantMSE(head_dim, bits=4, seed=TQ4_SEED)
    rot_t = quantizer.rotation.T.to("cuda")
    rot_t_even = rot_t[:, 0::2].contiguous()
    rot_t_odd = rot_t[:, 1::2].contiguous()
    boundaries = quantizer.codebook.boundaries.to("cuda")
    centroids = quantizer.codebook.centroids.to("cuda")

    packed, norms = tq4_compress(x, rot_t_even, rot_t_odd, boundaries)
    assert packed.shape == (N, H, head_dim // 2)
    assert norms.shape == (N, H, 1)

    recovered = tq4_decompress(packed, norms, centroids, dtype=torch.float16)
    recovered = recovered @ quantizer.rotation.to("cuda")

    cosine_sim = torch.nn.functional.cosine_similarity(
        x.flatten(0, 1), recovered.flatten(0, 1), dim=-1
    ).mean()
    assert cosine_sim > 0.95, f"Cosine sim {cosine_sim:.4f} too low for head_dim={head_dim}"


def test_different_head_dims_independent():
    """Simulate Gemma 4: some layers use 256, others use 512."""
    for head_dim in [256, 512]:
        x = torch.randn(2, 4, head_dim, device="cuda", dtype=torch.float16)
        q = TurboQuantMSE(head_dim, bits=4, seed=TQ4_SEED)
        rot_t = q.rotation.T.to("cuda")
        packed, norms = tq4_compress(
            x, rot_t[:, 0::2].contiguous(), rot_t[:, 1::2].contiguous(),
            q.codebook.boundaries.to("cuda"),
        )
        assert packed.shape[-1] == head_dim // 2


def test_codebook_differs_per_dim():
    """Lloyd-Max codebooks are distinct for different head_dims."""
    from turboquant_vllm.lloyd_max import solve_lloyd_max
    c256, _ = solve_lloyd_max(256, 4)
    c512, _ = solve_lloyd_max(512, 4)
    assert not torch.equal(c256, c512), "Codebooks should differ for different dims"


def test_supports_head_size_override():
    """TQ4AttentionBackend.supports_head_size accepts 256 and 512."""
    from turboquant_vllm.vllm.tq4_backend import TQ4AttentionBackend
    assert TQ4AttentionBackend.supports_head_size(128)
    assert TQ4AttentionBackend.supports_head_size(256)
    assert TQ4AttentionBackend.supports_head_size(512)
    assert not TQ4AttentionBackend.supports_head_size(0)
    assert not TQ4AttentionBackend.supports_head_size(3)  # odd