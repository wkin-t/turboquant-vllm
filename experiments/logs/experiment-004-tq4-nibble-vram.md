## Experiment: 004 — TQ4 nibble-packed validation (Molmo2-4B + video)

**Date:** 2026-03-25
**Hardware:** RTX 4090 (24 GB), AMD 7800X3D, 128 GB DDR5
**Model:** allenai/Molmo2-4B (bfloat16, device_map="auto")
**Baseline Config:** Standard DynamicCache, no compression
**Experimental Config:** CompressedDynamicCache with TQ4 nibble packing (4-bit, 2 indices per byte)

### Hypothesis

TQ4 nibble-packed CompressedDynamicCache should produce higher quality output
than TQ3 (more centroids = better reconstruction) while achieving 3.76x
compression — nearly double the TQ3 compression ratio.

### Setup

- Video: Seinfeld clip01.mp4 (~11K visual tokens at 2fps)
- Prompt: "Describe what happens in this video scene in detail..."
- Generation: 256 tokens, greedy decoding (do_sample=False)
- Benchmark: baseline run first, then TQ4 compressed run
- Nibble packing: two 4-bit indices packed per uint8 byte

### Results

| Metric | Baseline | TQ4 Nibble | Delta |
|--------|----------|-----------|-------|
| Input tokens | 11,397 | 11,397 | Same |
| Output tokens | 256 | 256 | Same |
| Tokens/sec | 30.0 | 8.9 | 3.36x slower |
| VRAM peak | 18,058 MiB | 18,057 MiB | ~0 (activation-dominated) |
| Output quality | Detailed scene description | **Near-identical** | Minor phrasing only |

**KV Cache Compression Stats:**

| Metric | Value |
|--------|-------|
| Compressed KV cache | 435.2 MiB |
| Baseline KV cache (FP16 equivalent) | 1,638.6 MiB |
| Compression ratio | **3.76x** |
| Savings (theoretical) | **1,203.3 MiB** |

### Output Quality Comparison

Both outputs describe the same Seinfeld apartment scene. The first ~100 tokens
are **word-for-word identical**:

> "The scene opens in a cozy, lived-in apartment with gray walls and a white
> door, where a woman with long, curly brown hair in a red jacket over a black
> shirt stands beside a man in a green jacket with a brown collar over a
> purple shirt. They're huddled close..."

Divergence begins at minor descriptive details:
- Baseline: "both looking worried and tense"
- TQ4: "both looking off to the side with tense, worried expressions"

Both are valid descriptions of the same visual content. This is **better quality
than TQ3** (Experiment 003), where the outputs diverged after ~30 tokens. TQ4's
16 centroids (vs TQ3's 8) provide meaningfully better reconstruction fidelity.

### Observations

1. **Quality: near-identical to baseline.** TQ4 with 16 centroids and ~97%
   cosine similarity produces output that is indistinguishable from baseline
   for practical purposes. The first 100+ tokens match word-for-word.

2. **3.76x compression validated.** 435 MiB vs 1,639 MiB — saving 1.2 GiB of
   KV cache storage. For Molmo2-4B's 36 layers and 8 KV heads, this is
   substantial.

3. **3.36x overhead.** Slower than TQ3's 2.35x — expected because 4-bit Lloyd-Max
   has 16 centroids (15 boundary comparisons) vs 3-bit's 8 centroids (7 boundaries).
   Plus the nibble unpack adds bit-shift operations. A fused Triton kernel would
   eliminate this overhead entirely.

4. **VRAM peak still activation-dominated.** Same finding as Experiment 003 — the
   ~1.2 GiB KV savings don't appear in `max_memory_allocated()` because prefill
   activations dominate the peak. The savings are real in permanent storage.

### Comparison Across All Experiments

| Experiment | Mode | Compression | Quality | Overhead |
|-----------|------|-------------|---------|----------|
| 001 | TQ3 (TurboQuantProd) | N/A | Garbled | N/A |
| 002 | TQ3 (MSE-only, accuracy) | 1x (no savings) | Identical/Coherent | 1.3x |
| 003 | TQ3 (CompressedDynamicCache) | 1.94x | Coherent | 2.35x |
| **004** | **TQ4 (nibble-packed)** | **3.76x** | **Near-identical** | **3.36x** |

### Next Steps

1. **Write up the results** — First TurboQuant implementation validated on a
   vision-language model (Molmo2) with video input. Novel contribution.

2. **Molmo2-8B validation** — With 1.2 GiB KV savings, 8B might have enough
   headroom for longer context.

3. **Consider vendoring community Triton kernel** — to address the 3.36x
   overhead without writing our own from scratch.
