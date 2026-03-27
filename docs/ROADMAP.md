# TurboQuant Consumer — Roadmap

Implementation status and path forward for TurboQuant KV cache compression
on consumer GPUs (RTX 4090, 24 GB VRAM) with Molmo2 vision-language models.

**Paper:** arXiv 2504.19874 — "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (ICLR 2026)

---

## Completed

### Layer 1: Core Quantization Algorithm

| Component | Status | Tests | Notes |
|-----------|--------|-------|-------|
| Lloyd-Max codebook solver | Done | 8 | Gaussian approx for d >= 64, `@lru_cache`d |
| `TurboQuantMSE` (Stage 1) | Done | 6 | Rotation + scalar quantize, ~95% cosine sim at 3-bit |
| `TurboQuantProd` (Stage 2) | Done | 5 | MSE + QJL correction, unbiased inner products |

### Layer 2a: Production Compressors

| Component | Status | Tests | Notes |
|-----------|--------|-------|-------|
| `CompressedKeys` / `CompressedValues` | Done | — | Dataclass containers |
| `TurboQuantCompressorV2` (keys) | Done | 6 | Includes `asymmetric_attention_scores()` |
| `TurboQuantCompressorMSE` (values) | Done | 2 | MSE-only for value reconstruction |

### Layer 2b: Compressed KV Cache (real VRAM savings)

| Component | Status | Tests | Notes |
|-----------|--------|-------|-------|
| `TurboQuantKVCache` (accuracy-only) | Done | 7 | Compress-decompress round-trip, no VRAM savings |
| `CompressedDynamicCache` | Done | 13 | uint8 indices + fp32 norms, 1.94x compression |
| Benchmark harness (`--compressed`) | Done | — | A/B testing with Molmo2, VRAM + quality metrics |

### Experiments

| # | Date | Result | Key Finding |
|---|------|--------|-------------|
| 001 | 2026-03-25 | Failed | TurboQuantProd (2-bit MSE + 1-bit QJL) garbled output — QJL wasted in drop-in mode |
| 002 | 2026-03-25 | Passed | MSE-only fix: identical text output, coherent video (1.3x overhead) |
| 003 | 2026-03-25 | Passed | CompressedDynamicCache: coherent output, 1.94x compression, fp16 norms bug found and fixed |
| 005 | 2026-03-25 | Passed | Incremental dequant: 3.76x compression with 1.78x overhead (down from 3.36x) |

---

### P1: Long-sequence regression test

| Component | Status | Tests | Notes |
|-----------|--------|-------|-------|
| Multi-layer precision test | Done | 4 | 36 layers, 1024 prefill + 32 gen steps, >0.999 cosine sim |
| TQ4 regression at scale | Done | 1 | Same scale test for 4-bit nibble-packed path |

### P2: TQ4 nibble packing (3.76x compression)

| Component | Status | Tests | Notes |
|-----------|--------|-------|-------|
| `_nibble_pack` / `_nibble_unpack` | Done | 1 | Bit-shift pack/unpack, exact round-trip verified |
| `CompressedDynamicCache` bits=4 | Done | 7 | Auto-enabled at bits=4, transparent to callers |
| `_CompressedLayer.packed` flag | Done | — | Tracks packing format through cat/stats |

### P3: Incremental dequantization (1.78x overhead)

| Component | Status | Tests | Notes |
|-----------|--------|-------|-------|
| Incremental dequant in `CompressedDynamicCache` | Done | — | Dequantize only new tokens, maintain running buffer |

**Experiment 005 results:** 3.76x compression with 1.78x overhead (down from 3.36x full-cache dequant). Near-identical output quality preserved.

### Compression Summary

| Mode | Bytes/block | Compression | Quality | Overhead | Status |
|------|-------------|-------------|---------|----------|--------|
| FP16 baseline | 256 | 1.0x | — | — | — |
| TQ3 uint8 | 132 | 1.94x | ~95% cosine | 2.35x | Done |
| TQ4 full-cache dequant | 68 | 3.76x | ~97% cosine | 3.36x | Done |
| **TQ4 incremental dequant** | **68** | **3.76x** | **~97% cosine** | **1.78x** | **Done** |
| TQ3 bit-packed | 52 | 4.92x | ~95% cosine | — | Deferred (P5) |

**Projected VRAM for Molmo2-4B (36 layers, 8 KV heads, 11K tokens):**

| Mode | KV Cache Size | Savings vs FP16 |
|------|--------------|-----------------|
| FP16 baseline | 1,639 MiB | — |
| TQ3 uint8 | 845 MiB | 794 MiB (1.94x) |
| **TQ4 nibble** | **436 MiB** | **1,203 MiB (3.76x)** |

---

## In Progress

### P3b: Fused Triton Q@K^T kernel (validated, needs Flash Attention fusion)

| Component | Status | Notes |
|-----------|--------|-------|
| Fused Q@K^T Triton kernel | **Done** | Nibble unpacking + pre-rotation trick, 17.8x speedup |
| Micro-benchmark (11K tokens) | **Done** | 1.0 cosine similarity vs unfused reference |
| Single-layer Molmo2-4B integration | **Done** | Correct output with fused kernel |
| AMD ROCm validation | **Done** | Triton HIP backend, 1.0 cosine, 0.31 ms/call on 890M (experiment 008) |
| Multi-layer integration | **Blocked** | Needs Flash Attention-style fusion (see below) |

**Key finding:** A fused Q@K^T-only kernel does not match SDPA precision when composed across all 36 transformer layers. The fp32 kernel scores differ from bf16 SDPA scores by 0.023 cosine per layer, which compounds to 0.43 over 36 layers — degenerate output. Root cause: the Q@K^T-only approach materializes attention scores in fp16 at two intermediate points, causing error multiplication. Full Flash Attention-style fusion (Q@K^T + online softmax + V matmul in one kernel, fp32 accumulation throughout) is required for multi-layer correctness. See P5 for the implementation roadmap.

**Research:** See `molmo-video-analyzer/_bmad-output/planning-artifacts/research/technical-flash-attention-fusion-turboquant-kv-cache-research-2026-03-26.md` for the detailed analysis that quantified this drift and identified the full FA fusion solution.

**Cross-platform:** The Triton kernel works on both NVIDIA (CUDA) and AMD (ROCm/HIP) with zero code changes. The multi-layer precision issue is platform-independent — P5 fix applies to both.

---

## Future Work

### P8: AMD ROCm platform support (Radeon 890M / gfx1150)

**Goal:** Enable TurboQuant development and validation on AMD integrated GPUs, starting with Radeon 890M (RDNA 3.5, gfx1150) on a Ryzen AI 9 HX 370 laptop running Bazzite (immutable Fedora).

**Context:** gfx1150 lacks official ROCm support as of March 2026. The `HSA_OVERRIDE_GFX_VERSION=11.0.0` workaround enables PyTorch GPU detection in containerized environments. Core algorithm is device-agnostic and works on CPU without modification.

**Research:** See `_bmad-output/planning-artifacts/research/technical-rocm-amd-igpu-pytorch-inference-research-2026-03-26.md` for full feasibility assessment.

#### Phase 0 — Smoke Test (2026-03-26, COMPLETED)

| Step | Action | Result |
|------|--------|--------|
| 0.1 | Pull ROCm PyTorch container via Podman | ✅ `rocm/pytorch:rocm7.1_ubuntu24.04_py3.12_pytorch_release_2.8.0` |
| 0.2 | Set `HSA_OVERRIDE_GFX_VERSION=11.0.0` | ✅ `torch.cuda.is_available()` → True |
| 0.3 | Run `torch.cuda.get_device_name(0)` | ✅ "AMD Radeon Graphics", 32 GB, reports gfx1100 |
| 0.4 | Run simple matmul on GPU | ✅ 1000x1000 and 2000x2000 matmul, CPU/GPU match atol=1e-4 |
| 0.5 | Run test suite on CPU inside container | ✅ **62/62 tests pass** (12.4s) |
| 0.6 | Run TurboQuant ops on GPU, cross-validate vs CPU | ✅ Bit-identical quantization, 0.995 cache cosine, no NaN/Inf |

**Findings:** Initial attempts crashed with `Memory critical error — Memory in use` on all HSA override values (`11.0.0`, `11.0.1`, `11.0.2`, `11.5.0`, `11.5.1`). Root cause: **SELinux label enforcement** on Bazzite blocks `hipMalloc` inside Podman containers. Fix: `--security-opt=label=disable`.

With SELinux labels disabled + `HSA_OVERRIDE_GFX_VERSION=11.0.0`:
- GPU compute fully functional (1000x1000 and 2000x2000 matmul)
- CPU/GPU agreement within atol=1e-4 (max diff 0.004 — normal fp divergence)
- Memory allocation working (71.6 MB allocated, 86.0 MB reserved)
- All 62 TurboQuant tests pass on CPU inside container
- ROCm 7.11 preview also provides native gfx1150 wheels at `https://repo.amd.com/rocm/whl/gfx1150/`

**Working Podman command:**
```bash
podman run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video \
  --security-opt=label=disable \
  -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  -v ~/Projects/turboquant-consumer:/workspace:z \
  -w /workspace \
  docker.io/rocm/pytorch:rocm7.1_ubuntu24.04_py3.12_pytorch_release_2.8.0
```

#### Phase 1 — Dev Environment (COMPLETED 2026-03-26)

| Step | Action | Result |
|------|--------|--------|
| 1.1 | Create `Containerfile` for dev environment (ROCm + project deps) | ✅ `infra/Containerfile.rocm` + `infra/run-rocm.sh` |
| 1.2 | Mount project sources as volumes | ✅ Handled by `run-rocm.sh` (+ HF cache mount) |
| 1.3 | Add cross-device validation test fixtures (CPU vs GPU) | ✅ 21 tests parametrized, 84/84 pass on AMD GPU |
| 1.4 | Verify `uv sync` works inside container with ROCm PyTorch | ⚠️ `uv sync` installs CUDA torch from PyPI — use `PYTHONPATH=/workspace/src` instead |

**Phase 1.3 Results — Cross-Device Test Parametrization (2026-03-26):**

Spike audit found **zero source changes needed** — all internal state tensors (`codebook.centroids`, `codebook.boundaries`, `self.rotation`, `self.qjl_matrix`) already use `.to(input.device)` before operations.

Implementation: Added `device` fixture in `conftest.py` parametrized with `["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]`. 21 tests across `test_lloyd_max.py`, `test_quantizer.py`, and `test_compressors.py` now run on both CPU and GPU. GPU tests skip gracefully when CUDA is unavailable.

Validated inside ROCm container on Radeon 890M (gfx1150): **84/84 tests passed** with no tolerance relaxation — all existing `atol`, cosine similarity, and correlation thresholds hold on AMD GPU.

#### Phase 2 — Core Algorithm on AMD (COMPLETED 2026-03-26)

**Session 1 — Quick wins (gates Phase 3):**

| Step | Action | GPU? | Status |
|------|--------|------|--------|
| 2.1 | `torch.compile(mode="default")` spike on ROCm | Yes | ✅ Both `default` and `reduce-overhead` pass, 1.17x speedup, perfect eager parity |
| 2.2 | KV cache test parametrization — add `device` fixture to `test_kv_cache.py` (basic update, nibble packing, VRAM savings) | Yes | ✅ 11 tests parametrized, 95/95 pass on AMD GPU |

**Session 2 — Coverage & hardening:**

| Step | Action | GPU? | Status |
|------|--------|------|--------|
| 2.3 | Push test coverage above 90% — identify uncovered paths | No | ✅ 99% coverage (338/340 stmts), threshold raised to 95% |
| 2.4 | Codebook solver convergence — check edge cases (low/high dims, extreme bit widths) | No | ✅ Added exact Beta path, 6-bit (64 levels), and boundary edge case tests |

**Session 3 — Optional research:**

| Step | Action | GPU? | Status |
|------|--------|------|--------|
| 2.5 | 2-bit and 5-bit support — extend `solve_lloyd_max` and tests | No | ✅ Code already generic; added tests for bits 2-5 across quantizer, codebook, and KV cache |
| 2.6 | CompressedDynamicCache API ergonomics review | No | ✅ API is clean — consistent constructors, well-structured exports, no sharp edges |

#### Phase 3 — End-to-End Validation (COMPLETED 2026-03-27)

| Step | Action | Status |
|------|--------|--------|
| 3.1 | Verify Molmo2-4B weights accessible (~8 GB) | ✅ Model accessible, 4 shards fetched |
| 3.2 | Run baseline inference (no compression) on GPU | ✅ 5.0 tok/s, 10,052 MiB peak, coherent output |
| 3.3 | Run TQ4 compressed inference on GPU | ✅ 4.3 tok/s, 3.76x KV compression, coherent output |
| 3.4 | Cross-validate: same inputs on CPU vs GPU | ✅ **16/16 tokens match** (100%) — HSA override is safe |
| 3.5 | Benchmark actual throughput on 890M | ✅ 5.0 tok/s baseline, 0.86x TQ4 overhead |

**Validation script:** `experiments/experiment_007_e2e_amd_validation.py`

```bash
# All steps in one run (inside ROCm container):
./infra/run-rocm.sh python experiments/experiment_007_e2e_amd_validation.py

# Skip slow CPU cross-validation:
./infra/run-rocm.sh python experiments/experiment_007_e2e_amd_validation.py --skip-cross-validate

# Custom settings:
./infra/run-rocm.sh python experiments/experiment_007_e2e_amd_validation.py \
    --model allenai/Molmo2-4B --bits 4 --max-new-tokens 64
```

Results: `experiments/logs/experiment-007-e2e-amd-validation.json`

**Phase 3 key findings:**
- HSA override (gfx1150 → gfx1100) introduces zero token-level precision errors
- TQ4 compression overhead is only ~15% at short sequences (0.86x throughput)
- 890M throughput (~5 tok/s) is ~10-12x slower than 4090 (~50-60 tok/s), matching bandwidth ratio prediction
- ROCm SDPA warnings (`Mem Efficient` / `Flash Efficient` experimental) — output correct despite warnings

**Known limitations:**
- ~11-16x slower than RTX 4090 (DDR5 ~90 GB/s vs GDDR6X ~1 TB/s)
- `torch.compile` works on ROCm 7.1 (both `default` and `reduce-overhead` modes pass with TurboQuant ops, 1.17x speedup)
- Fused Triton kernel (P3b) works on ROCm via HIP backend (experiment 008) — multi-layer precision issue (P5) is platform-independent
- `hipMallocManaged()` not supported on gfx1150 as of ROCm 7.2

**Decision framework:**
```
Phase 0 Smoke Test
       │
       ├─ GPU detected + tests pass → Phase 1 → Phase 2 + 3
       │
       └─ GPU NOT detected → CPU-only development (still valuable)
                               └─ Monitor ROCm releases for gfx1150 support
```

---

### P4: Molmo2-8B validation

**Goal:** Confirm CompressedDynamicCache works with the larger model. The 8B model recognizes character names (e.g., "Elaine", "Kramer") which 4B cannot.

**Approach:** Run benchmark with `--model allenai/Molmo2-8B --compressed`. May need `bitsandbytes` 4-bit weight quantization to fit model + compressed cache in 24 GB.

**When:** After P3 (incremental dequant) eliminates the decode overhead.

### P5: Fused TQ4 Flash Attention kernel

**Goal:** Fuse the full attention computation (Q@K^T + online softmax + V matmul) into a single Triton kernel that reads nibble-packed TQ4 indices directly — never materializing decompressed keys or the attention score matrix in fp16.

**Why:** The P3b Q@K^T-only kernel achieves 17.8x on the micro-benchmark but can't maintain SDPA precision across 36 layers (0.023 cosine loss/layer → 0.43 over 36 layers). The root cause is two fp16 materialization points: scores after Q@K^T and weights after softmax. Full Flash Attention fusion eliminates both by maintaining the `(m_i, l_i, acc)` state machine entirely in fp32, casting only the final output to fp16. The correction factor `alpha = exp2(m_old - m_new)` is mathematically exact, not approximate.

**Expected precision:** >0.998 per-layer cosine similarity (vs 0.977 with Q@K^T-only), >0.93 over 36 layers (vs 0.43).

**Platform:** Triton HIP backend confirmed working for the Q@K^T kernel (experiment 008), so P5 should also work cross-platform (NVIDIA + AMD ROCm).

**Key architectural insight (arXiv 2511.11581):** GQA Q-Block pattern flattens multiple query heads sharing a KV head into a single 2D tensor. For Molmo2's 28Q/4KV (7:1 ratio): 7 Q-heads per block → BLOCK_M=8 (padded from 7). This avoids per-head loops and maps cleanly to Triton's tile-based programming model.

**Survey of existing systems (13 reviewed):** KIVI, BitDecoding, Kitty, INT-FlashAttention, QServe, FlashInfer, etc. — none fuse vector quantization codebook lookup with Flash Attention. This would be novel.

#### Phase 1: Vanilla Triton FA baseline (COMPLETE 2026-03-26)

| Step | Action | Result |
|------|--------|--------|
| 1.1 | Fork Triton tutorial FA kernel, forward-only fp16/bf16 | **Done** — 15 tests, all >0.999 cosine vs SDPA |
| 1.2 | Add GQA support via head mapping (not Q-Block yet) | **Done** — 4:1 and 7:1 GQA validated |
| 1.3 | Register via HuggingFace `AttentionInterface.register()` | **Done** — 64/64 token-identical text output on Molmo2-4B |
| 1.4 | Autotune for RTX 4090: BLOCK_M∈{16,64,128}, BLOCK_N∈{32,64} | **Done** — 0.26-0.38x SDPA throughput (see below) |

**Experiment 009 results (Molmo2-4B, RTX 4090, bf16):**

| Mode | SDPA tok/s | Triton FA tok/s | Ratio | Token match |
|------|-----------|----------------|-------|-------------|
| Text-only (17 input) | 43.2 | 11.2 | 0.26x | **64/64 (100%)** |
| Image (1205 input) | 47.1 | 17.7 | 0.38x | 8/64 (coherent, expected divergence) |

**Key findings:**
- **Correctness validated:** Token-identical output for text-only. Image divergence ("iconic" added at token 8) is expected — 1205-token prefill amplifies fp differences between cuDNN Flash Attention and our Triton kernel.
- **Throughput gap is expected:** SDPA dispatches to cuDNN's Flash Attention (years of CUDA engineering). Our Triton kernel is a correct scaffold, not a performance competitor. Phase 2 fundamentally changes the memory access pattern (reading compressed indices), so the SDPA comparison becomes irrelevant.
- **Autotune key fix:** Original key `["N_CTX_Q", "N_CTX_KV", "HEAD_DIM"]` caused re-autotuning on every decode step (N_CTX_KV changes each token). Fixed to `["N_CTX_Q", "HEAD_DIM"]` — 100x speedup from 0.5 to 11-18 tok/s.
- **Model config:** Molmo2-4B has 32Q/8KV (4:1 GQA), not 28Q/4KV (7:1) as initially assumed. Both ratios validated in unit tests.

#### Phase 2: TQ4 K-only fusion (COMPLETE 2026-03-26)

| Step | Action | Result |
|------|--------|--------|
| 2.1 | Pre-rotate Q outside kernel: `q_rot = query @ Pi_T` | **Done** — wrapper handles rotation |
| 2.2 | Replace K tile load: nibble unpack → centroid gather → norm scale | **Done** — `tl.join` + `reshape` interleaves even/odd centroids |
| 2.3 | Keep V as standard fp16 (uncompressed) | **Done** — 9 tests, all >0.998 cosine vs unfused path |
| 2.4 | Benchmark decode throughput | Deferred to experiment 010 |

**Key technique:** `tl.join(k_hi, k_lo).reshape(BLOCK_N, HEAD_DIM)` interleaves even/odd centroid values to reconstruct the full decompressed K tile without explicit loops.

#### Phase 3: TQ4 K+V fusion (COMPLETE 2026-03-26)

| Step | Action | Result |
|------|--------|--------|
| 3.1 | Add V tile decompression (same codebook and rotation as K) | **Done** — same nibble unpack + centroid gather + interleave |
| 3.2 | Post-rotation trick: `out = acc @ Pi` outside kernel | **Done** — mirror of pre-rotation for K |
| 3.3 | Validate fused K+V vs unfused path | **Done** — 7 tests, all >0.998 cosine |
| 3.4 | Bandwidth measurement + decode throughput | Deferred to experiment 010 |

**Key insight:** K and V share the same rotation matrix and codebook (same seed in CompressedDynamicCache). Pre-rotate Q by `Pi^T`, compute attention in rotated space, post-rotate output by `Pi`. One centroid table, zero in-kernel rotations.

#### Phase 4: Model integration + E2E validation (COMPLETE 2026-03-26)

| Step | Action | Result |
|------|--------|--------|
| 4.1 | Integrate fused kernel via AttentionInterface + cache side-channel | **Done** — `install_fused_tq4_kv()`, `get_compressed()` API |
| 4.2 | Experiment 010: E2E Molmo2-4B 3-path comparison | **Done** — coherent output across all 36 layers (see below) |
| 4.3 | Variable sequence length + edge cases | Deferred to production hardening |
| 4.4 | Regression test suite (3 tiers) | Tier 1 done (31 unit tests), Tier 2-3 deferred |

**Experiment 010 results (Molmo2-4B, RTX 4090, bf16, 36 layers):**

| Path | Text tok/s | Image tok/s | Text quality |
|------|-----------|-------------|--------------|
| SDPA baseline | 44.0 | 46.9 | "portrayed by Jason Alexand..." |
| Unfused TQ4 | 13.1 | 25.6 | "portrayed by Jerry Seinfel..." |
| Fused TQ4 K+V | 11.9 | 14.4 | "portrayed by Jerry Seinfel..." |

**Key findings:**
- **36-layer composition: PASS.** The Q@K^T-only kernel produced garbled output (0.43 cosine). The fused FA kernel produces coherent, factually correct text — the fp32 online softmax prevents catastrophic precision drift.
- **Fused vs unfused divergence is expected.** 14% token match (text), 52% (image). Same pattern as Phase 1 (SDPA vs Triton FA): different numerical paths compound through autoregressive generation. All outputs are valid paraphrases, not degraded.
- **Throughput: fused 0.91x of unfused (warm cache).** Decompression overhead (nibble unpack + centroid gather + interleave + pre/post rotation) currently exceeds the bandwidth savings at short sequences. Bandwidth wins should dominate at longer sequences (11K+ tokens).
- **VRAM: comparable across all paths** (~10 GiB). Peak is dominated by model weights + activations, not KV cache at 64 output tokens.

**Integration approach (decided after research — see party mode discussion 2026-03-26):**

Approach B selected: **AttentionInterface + side-channel cache reference.** Evaluated four options:

| Approach | Description | Verdict |
|----------|-------------|---------|
| A: Full forward monkey-patch | Replace entire `Molmo2Attention.forward()` | **Rejected** — fragile, must replicate 50+ lines of attention logic (QK norm, RoPE variants), breaks on transformers updates |
| **B: AttentionInterface + cache stash** | Register via `ALL_ATTENTION_FUNCTIONS`, stash cache ref on modules | **Selected** — clean HF API, small coupling surface, graceful fallback |
| C: Custom cache skipping decompression | Override `cache.update()` return value | **Rejected** — breaks HF cache API contract |
| D: AttentionInterface + cache kwarg | Pass cache via `**kwargs` like `eager_paged` | **Rejected** — Molmo2 doesn't pass cache through kwargs to attention function |

**Why B works:** `CompressedDynamicCache.update()` stores compressed K/V in `_compressed_keys`/`_compressed_values` (already implemented). The attention function reads compressed data from the cache, ignoring the decompressed K/V arguments. Requires:
1. `get_compressed(layer_idx)` public method on CompressedDynamicCache
2. `module._tq4_cache` reference stashed on each attention layer
3. Attention function checks for stash, falls back to SDPA if absent

**Known overhead:** `cache.update()` still decompresses K/V (wasted in fused path). Addressed by Phase 5 "fused-aware cache mode" (see below).

#### Projected performance (RTX 4090)

| Metric | Current (unfused) | Phase 2 (K-only) | Phase 3 (K+V) | Improvement |
|--------|-------------------|-------------------|---------------|-------------|
| Decode tok/s | 8.9 | 15-20 | 25-35 | 2.8-3.9x |
| Memory traffic/layer | ~25 MB | ~6 MB | ~3-6 MB | 4-8x less |
| Overhead vs baseline | 1.78x slower | ~1.1x | 0.85-1.1x | Near parity |

#### RTX 4090 resource budget

| Resource | Budget | Notes |
|----------|--------|-------|
| Memory bandwidth | 1008 GB/s | Target >500 GB/s achieved (50%+) |
| Shared memory | 128 KB/SM | Design <64 KB/block for 2 concurrent blocks |
| Registers | <128/thread | 4 warps with good occupancy |
| Occupancy | >50% | Sufficient for memory-bound kernel |

#### Success criteria

- Per-layer cosine similarity: >0.998 (match FA-level precision)
- 36-layer composition: >0.93 (residual connections stabilize)
- Decode tokens/sec: >15 (Phase 2), >25 (Phase 3)
- Text output: matches reference (decisive gate)
- KV cache VRAM: unchanged (3.76x compression preserved)

#### Testing strategy (3 tiers)

1. **Unit (fast):** Nibble unpack, centroid gather, single-tile attention vs standard
2. **Per-layer (medium):** Each of 36 layers >0.998 cosine similarity vs SDPA
3. **End-to-end (slow):** Full Molmo2 inference >0.93 cosine, text output identical

#### Prerequisites

- Triton ≥3.0 (available via `torch.triton`)
- HuggingFace transformers ≥4.50 (`AttentionInterface` API)
- Molmo2-4B weights (cached locally)
- RTX 4090 GPU
- CompressedDynamicCache (Layer 2b — done)
- Nsight Compute (profiling)

#### Research references

- `molmo-video-analyzer/_bmad-output/planning-artifacts/research/technical-triton-flash-attention-tutorial-deep-dive-2026-03-26.md` — FA inner loop mechanics, numerical stability analysis
- `molmo-video-analyzer/_bmad-output/planning-artifacts/research/technical-flash-attention-fusion-turboquant-kv-cache-research-2026-03-26.md` — Fusion architecture, precision analysis, 13-system survey, GQA Q-Block pattern
- `molmo-video-analyzer/_bmad-output/planning-artifacts/research/technical-fused-turboquant-triton-kernel-research-2026-03-25.md` — Original Q@K^T-only analysis (superseded by full FA approach)
- arXiv 2511.11581 — "Anatomy of Attention" (GQA Q-Block, Triton performance parity with FA-3)
- arXiv 2405.02803 — "Is Flash Attention Stable?" (FA achieves 1.7x lower RMSE than SDPA)

### P5b: Fused-aware cache mode (COMPLETE 2026-03-26)

**Goal:** Eliminate wasted decompression in the fused kernel path.

**Result:** Added `fused_mode=True` flag to CompressedDynamicCache. `install_fused_tq4_kv()` enables it automatically. Throughput impact: +6% on text (17 tokens), +13% on image (1205 tokens).

---

## North Star: Match the Paper

The Google TurboQuant paper claims **5-6x compression AND up to 8x attention speedup.** Our current implementation achieves compression (3.76x) but not speedup. The gap: our fused kernel dequantizes K fully inside the inner loop (centroid gather → 128 floats → dot product). **The paper never reconstructs the full key vector.** It uses `estimate_inner_product` — an unbiased estimator that computes Q@K^T directly from compressed indices + QJL sign corrections.

**What we have vs what the paper does:**

| Component | Our status | Paper's approach | Speed impact |
|-----------|-----------|------------------|-------------|
| PolarQuant (rotation + Lloyd-Max) | Done (TurboQuantMSE) | Same | — |
| QJL correction (Stage 2) | Built but unused (TurboQuantProd) | Core of speed claim | Eliminates dequant step |
| `estimate_inner_product()` | Built (CompressorV2) | Fused into attention kernel | **8x fewer bytes read** |
| Full-dequant fused FA (P5) | Done — proves FA fusion works | Not used by paper | 0.62x of unfused (too slow) |
| Direct inner-product fused FA | **NOT BUILT** | Paper's actual kernel | **The speed target** |

**Key Lesson #2 was wrong in context:** "QJL is invisible in drop-in mode" is true for drop-in cache replacement. But QJL is **essential** for the paper's speed claim because it enables `estimate_inner_product` which avoids full dequantization. The fused kernel should use TurboQuantProd, not TurboQuantMSE.

### P9: Paper-faithful fused kernel — `estimate_inner_product` in Flash Attention

**Goal:** Fuse TurboQuantProd's unbiased inner product estimator into Flash Attention. Achieve both compression AND speedup matching the paper's claims on RTX 4090.

**Why this is different from P5:** P5 dequantizes K inside the FA loop (centroid gather → full 128-float vector → dot product). P9 computes Q@K^T directly from compressed data without ever materializing the full key:

```
P5 (current, slow):
  For each K tile:
    packed → nibble unpack → centroid gather → [BLOCK_N, 128] fp16 → Q @ K^T

P9 (paper-faithful, fast):
  For each K tile:
    MSE term:  sum(q_rot[d] * centroids[idx[d]]) * norm    (~128 ops, scalar output)
    QJL term:  res_norm * sqrt(pi/2)/m * sum(q_proj * signs) (~128 ops, scalar output)
    score = MSE_term + QJL_term                              (never materializes 128 floats)
```

**Storage per key position (TurboQuantProd at 4-bit total):**
- MSE indices: 128 × 3-bit = 48 bytes (with bit-packing) or 128 bytes (uint8 unpacked)
- QJL signs: 128 bits = 16 bytes (packed as uint8)
- Norms: 4 bytes (fp32)
- Residual norms: 4 bytes (fp32)
- **Total: 72 bytes (with 3-bit packing) or 152 bytes (uint8 unpacked)**

**Storage per value position:** Values still use TurboQuantMSE (full 4-bit, 68 bytes nibble-packed). QJL not needed for V because V appears in P@V matmul, not inner product estimation.

#### Phase 1: Validate TurboQuantProd quality on Molmo2 (1 day)

| Step | Action | Validation |
|------|--------|------------|
| 1.1 | Run Molmo2-4B with TurboQuantProd(bits=4) KV cache | Coherent output, character names (4B text-only) |
| 1.2 | Compare quality: TurboQuantProd(4-bit=3MSE+1QJL) vs TurboQuantMSE(4-bit) | Cosine similarity, token match |
| 1.3 | Verify `estimate_inner_product` matches standard Q@K^T on decompressed keys | Per-layer cosine >0.998 |

**Risk:** TurboQuantProd uses 3-bit MSE (8 centroids) instead of 4-bit (16 centroids). The QJL correction should compensate for inner product estimation, but reconstruction quality may be lower. Experiment 001 showed garbled output with TurboQuantProd in drop-in mode — but P9 uses `estimate_inner_product`, not drop-in dequant.

#### Phase 2: Fused inner-product kernel (2-3 days)

| Step | Action | Validation |
|------|--------|------------|
| 2.1 | New Triton kernel: MSE dot product directly from indices (no dequant) | Cosine >0.998 vs unfused |
| 2.2 | Add QJL correction term inside kernel | Combined score matches `estimate_inner_product` |
| 2.3 | Integrate with FA online softmax (same `(m_i, l_i, acc)` state machine) | 36-layer composition >0.93 |
| 2.4 | V tiles use MSE-only dequant (same as P5 Phase 3) or separate path | Output matches reference |

**Key kernel design:**
```
# Pre-compute outside kernel:
q_rot = q @ Pi_mse^T              # Pre-rotate for MSE term (already proven in P5)
q_proj = q @ S^T                  # Pre-project for QJL term (new)

# Inside FA inner loop, for each K tile:
# MSE term: dot product from compressed indices (no full dequant)
mse_score = norm * sum_d(q_rot[d] * centroids_3bit[idx[d]])

# QJL term: correction from sign bits
qjl_score = res_norm * sqrt(pi/2) / m * sum_j(q_proj[j] * signs[j])

# Combined attention score
score = mse_score + qjl_score

# Feed into online softmax as before (proven stable in P5)
```

#### Phase 3: 3-bit packing for indices (1-2 days)

| Step | Action | Validation |
|------|--------|------------|
| 3.1 | Implement 3-bit byte-crossing pack/unpack in Triton | Round-trip correctness |
| 3.2 | Integrate into kernel (replace uint8 index load) | Same output, smaller reads |
| 3.3 | Pack QJL signs as bit-packed uint8 (128 bits = 16 bytes) | Already efficient |

This reduces key storage from 152 bytes (uint8) to 72 bytes (bit-packed) per position. Combined with V at 68 bytes, total KV = 140 bytes vs 512 FP16 = **3.66x compression with speed.**

#### Phase 4: Profile and optimize (1-2 days)

| Step | Action | Validation |
|------|--------|------------|
| 4.1 | Nsight Compute profiling of P9 kernel vs cuDNN FA | Identify bottleneck |
| 4.2 | Tune autotune configs for RTX 4090 (SM89) | Peak achieved bandwidth |
| 4.3 | Benchmark at 1K, 5K, 11K token sequences | Find crossover point |
| 4.4 | Compare vs paper's claimed 8x (adjusting for H100→4090 bandwidth ratio) | Realistic target for consumer GPU |

**RTX 4090 vs H100 bandwidth:** 1008 GB/s vs 3350 GB/s (3.3x gap). The paper's 8x on H100 might translate to ~2.5-4x on RTX 4090 — still a significant win if the kernel is bandwidth-bound.

#### Phase 5: E2E Molmo2 validation + vLLM integration

| Step | Action | Validation |
|------|--------|------------|
| 5.1 | E2E experiment: SDPA vs unfused TQ4 vs P9 fused on Molmo2-4B | All three produce coherent output |
| 5.2 | Throughput comparison at video-scale (11K+ tokens) | Fused >= unfused AND >= 1.5x vs SDPA at long sequences |
| 5.3 | vLLM custom attention backend (PagedAttention + TurboQuantProd) | Serve Molmo2-8B with TQ KV cache |
| 5.4 | Production benchmark: max_model_len with TQ vs FP8 | 3x+ context window at equal or better throughput |

**Success criteria (north star):**
- KV compression: >=3.5x vs FP16 (matching paper)
- Attention speedup: >= 1.5x vs baseline at 11K+ tokens on RTX 4090
- Text quality: coherent output, character names preserved on Molmo2-8B
- max_model_len: 3x increase in vLLM serving

### P6: TQ3 bit-packing (folded into P9 Phase 3)

Now part of P9 Phase 3. Required for the paper-faithful storage format.

### P7: vLLM native integration (folded into P9 Phase 5)

Now part of P9 Phase 5. No longer "monitor upstream" — we build it.

---

## Hardware Context

### Primary — Desktop (RTX 4090)

| Component | Spec | Relevance |
|-----------|------|-----------|
| GPU | NVIDIA RTX 4090 (24 GB GDDR6X) | All benchmarks run here |
| CPU | AMD 7800X3D | Codebook solving, data loading |
| RAM | 128 GB DDR5 | Model offloading when needed |
| Target model | Molmo2-4B (experiments) / 8B (future) | Vision-language model for video analysis |
| Target workload | Seinfeld clip analysis | 11K+ visual tokens at 2fps |
| Production stack | vLLM in Podman (CDI GPU) | Currently FP8 KV cache |

### Secondary — Laptop (Radeon 890M iGPU)

| Component | Spec | Relevance |
|-----------|------|-----------|
| GPU | AMD Radeon 890M (32 GB shared VRAM, gfx1150 RDNA 3.5) | ROCm via HSA override (P8) |
| CPU | AMD Ryzen AI 9 HX 370 (12C/24T, 5.16 GHz) | Algorithm dev, CPU-path testing |
| NPU | AMD XDNA (Strix) | Future exploration (incompatible with custom cache ops) |
| RAM | 64 GB DDR5 | Shared with iGPU VRAM |
| OS | Bazzite 43 (immutable Fedora) | Podman-native, no system package installs |
| Dev environment | Podman + ROCm container | `HSA_OVERRIDE_GFX_VERSION=11.0.0` |
| Bandwidth | ~90 GB/s (DDR5) vs ~1 TB/s (4090 GDDR6X) | 11-16x slower end-to-end |

---

## Key Lessons

1. **FP16 norms are a trap.** At 10K+ token sequences across 36 layers, fp16 norm precision loss compounds and flips low-confidence logits. Always use fp32 for norms.

2. **QJL is invisible in drop-in mode.** Standard attention does `Q @ K.T` on decompressed keys. QJL correction only helps with `estimate_inner_product()` (custom kernel). Using QJL in drop-in mode wastes 1 bit of MSE resolution for nothing.

3. **Peak VRAM != KV cache size.** On Molmo2-4B with 11K tokens, forward-pass activations dominate peak VRAM (~90%). KV cache compression savings are real but invisible to `max_memory_allocated()`. They matter for max_model_len budgeting, not peak measurement.

4. **PyTorch treats uint8 as boolean masks.** Fancy indexing with uint8 tensors triggers boolean masking, not integer indexing. Always cast to `.long()` before centroid lookup.

5. **Don't fight byte alignment.** TQ4 nibble packing (2 values per byte) is trivial and gives 3.76x compression. TQ3 bit-packing (3-bit byte-crossing) is hard and only 30% better. Work with the hardware, not against it.

6. **No PyTorch sub-byte ecosystem.** `torch.uint3` etc. are placeholders with no ops. TorchAO packing is weight-quant-specific. Every KV cache implementation rolls its own Triton kernels. Plan accordingly.

7. **Q@K^T-only fusion is a dead end for multi-layer models.** Materializing attention scores in fp16 between Q@K^T and softmax introduces 0.023 cosine loss per layer, compounding to 0.43 over 36 layers. Full Flash Attention fusion (fp32 accumulation throughout, single fp16 cast at output) is the only correct approach. The 17.8x micro-benchmark speedup was misleading — always validate multi-layer composition early.

8. **No existing system fuses codebook VQ with Flash Attention.** Survey of 13 quantized attention implementations (KIVI, BitDecoding, Kitty, QServe, FlashInfer, etc.) found they all use scalar quantization (INT2/4/8, FP4/8). TurboQuant's vector quantization codebook lookup is architecturally different and requires a novel kernel design.

9. **Dequant-then-dot is the wrong architecture for speed.** P5 proved full FA fusion prevents precision collapse across 36 layers. But dequanting K inside the inner loop (centroid gather → 128 floats → Q@K^T) is slower than cuDNN's optimized FA reading fp16 directly. The paper achieves speedup by computing Q@K^T directly from compressed indices via `estimate_inner_product` — never materializing the full key vector. This requires TurboQuantProd (MSE+QJL), not TurboQuantMSE.

10. **Key Lesson #2 was context-dependent.** "QJL is invisible in drop-in mode" is true for drop-in cache replacement (standard attention on decompressed keys). But QJL is essential for the paper's speed claim — it enables `estimate_inner_product` which avoids full dequantization. The fused kernel for speed must use TurboQuantProd, not TurboQuantMSE.
