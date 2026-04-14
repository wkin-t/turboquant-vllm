# Gemma 4 异构 Head Dim 支持实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 turboquant-vllm 支持 Gemma 4 的异构 head_dim（256/512），使 TQ4 KV cache 压缩能在 RTX 4090 上配合 Gemma 4 26B AWQ 使用。

**Architecture:** `TQ4AttentionBackend` 继承自 vLLM 的 `FlashAttentionBackend`，父类的 head_size 校验不包含 512。需要覆盖校验方法接受 256/512，并在测试中验证这两个维度的压缩/解压正确性。TQ4 的核心算法（旋转矩阵、Lloyd-Max codebook、Triton 内核）已经支持任意偶数维度，不需要修改。

**Tech Stack:** Python, Triton, PyTorch, vLLM 0.19.0

---

## 文件结构

| 操作 | 文件 | 职责 |
|------|------|------|
| 修改 | `src/turboquant_vllm/vllm/tq4_backend.py` | 覆盖 head_size 校验方法 |
| 修改 | `tests/test_triton_multi_dim.py` | 添加 256/512 维度测试 |
| 修改 | `tests/helpers/vllm_impl.py` | 参数化 HEAD_SIZE 支持多维度 |
| 新建 | `tests/test_gemma4_head_dim.py` | Gemma 4 特定的异构 head_dim 测试 |

---

### Task 1: 添加 head_dim 256/512 的压缩/解压测试

**Files:**
- Modify: `tests/test_triton_multi_dim.py:20`

- [ ] **Step 1: 在测试 fixture 中添加 256 和 512**

在 `tests/test_triton_multi_dim.py` 第 20 行，将 head_dim fixture 从 `[64, 96, 128]` 扩展为 `[64, 96, 128, 256, 512]`:

```python
@pytest.fixture(params=[64, 96, 128, 256, 512])
def head_dim(request):
    return request.param
```

- [ ] **Step 2: 运行测试验证 256/512 通过**

Run: `cd turboquant-vllm && python -m pytest tests/test_triton_multi_dim.py -v -k "256 or 512"`
Expected: ALL PASS（Triton 内核已支持任意偶数维度）

- [ ] **Step 3: Commit**

```bash
git add tests/test_triton_multi_dim.py
git commit -m "test: add head_dim 256/512 to multi-dim compression tests"
```

---

### Task 2: 覆盖 vLLM FlashAttentionBackend 的 head_size 校验

**Files:**
- Modify: `src/turboquant_vllm/vllm/tq4_backend.py:279-341`

- [ ] **Step 1: 查找 FlashAttentionBackend 的 head_size 校验方法**

在 vLLM 0.19.0 中，`FlashAttentionBackend` 有一个类方法用于校验 head_size 是否支持。需要确认方法名（可能是 `check_backend_compatibility` 或在 `get_impl_cls` 路径中）。在服务器上运行：

```bash
ssh cet_ai_server 'source /data/venvs/vllm/bin/activate && python3 -c "
from vllm.attention.backends.flash_attn import FlashAttentionBackend
# 列出所有与 head_size 相关的方法
for name in dir(FlashAttentionBackend):
    if \"head\" in name.lower() or \"support\" in name.lower() or \"check\" in name.lower():
        print(name, getattr(FlashAttentionBackend, name, None))
"'
```

- [ ] **Step 2: 在 TQ4AttentionBackend 中覆盖校验方法**

在 `TQ4AttentionBackend` 类中添加覆盖方法。具体实现取决于 Step 1 的发现，但预期形式为：

```python
@staticmethod
def get_supported_head_sizes() -> list[int]:
    """TQ4 supports any even head_size (rotation matrix is QR-based)."""
    return [32, 64, 96, 128, 160, 192, 224, 256, 384, 512]
```

或者，如果校验在 `check_backend_compatibility` 中：

```python
@classmethod
def check_head_size(cls, head_size: int) -> bool:
    """TQ4 supports any even head_size."""
    return head_size % 2 == 0
```

- [ ] **Step 3: 运行本地验证**

```bash
cd turboquant-vllm
python -c "
from turboquant_vllm.vllm.tq4_backend import TQ4AttentionBackend
# 确认 512 在支持列表中
print(TQ4AttentionBackend.get_supported_head_sizes())
assert 512 in TQ4AttentionBackend.get_supported_head_sizes()
print('OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add src/turboquant_vllm/vllm/tq4_backend.py
git commit -m "feat: support head_dim 256/512 by overriding FlashAttentionBackend validation"
```

---

### Task 3: 创建 Gemma 4 异构 head_dim 集成测试

**Files:**
- Create: `tests/test_gemma4_head_dim.py`

- [ ] **Step 1: 编写测试验证同一模型中混合 head_dim 的场景**

```python
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
    N, H = 4, 8  # batch, heads
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
    # Apply inverse rotation
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


def test_codebook_cached_per_dim():
    """Lloyd-Max codebooks are cached per head_dim."""
    from turboquant_vllm.lloyd_max import solve_lloyd_max
    c256, b256 = solve_lloyd_max(256, 4)
    c512, b512 = solve_lloyd_max(512, 4)
    assert c256.shape != c512.shape or not torch.equal(c256, c512), \
        "Codebooks should differ for different dims"
```

- [ ] **Step 2: 运行测试**

Run: `cd turboquant-vllm && python -m pytest tests/test_gemma4_head_dim.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_gemma4_head_dim.py
git commit -m "test: add Gemma 4 heterogeneous head_dim integration tests"
```

---

### Task 4: 在 AI 服务器上验证端到端

- [ ] **Step 1: 推送到 fork**

```bash
cd turboquant-vllm
git push origin main
```

- [ ] **Step 2: 在服务器上安装修改版**

```bash
ssh cet_ai_server 'source /data/venvs/vllm/bin/activate && \
  pip install git+https://github.com/wkin-t/turboquant-vllm.git --force-reinstall --no-deps && \
  pip install transformers==5.5.3 huggingface_hub --upgrade --no-deps -q'
```

- [ ] **Step 3: 启动 vLLM + TurboQuant**

```bash
ssh cet_ai_server 'source /data/venvs/vllm/bin/activate && \
  nohup python -m vllm.entrypoints.openai.api_server \
    --host 127.0.0.1 --port 8426 \
    --model /data/hf-models/gemma-4-26B-A4B-it-AWQ-4bit \
    --served-model-name gemma-4-26b-a4b-it-awq \
    --trust-remote-code --dtype float16 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 131072 \
    --max-num-seqs 2 \
    --attention-backend CUSTOM \
    --enable-chunked-prefill --enable-prefix-caching \
    --enforce-eager \
    --reasoning-parser gemma4 \
    --default-chat-template-kwargs "{\"enable_thinking\":true}" \
    >/tmp/vllm.log 2>&1 &'
```

- [ ] **Step 4: 测试推理**

```bash
ssh cet_ai_server 'curl -s -H "Content-Type: application/json" \
  http://127.0.0.1:8426/v1/chat/completions \
  -d "{\"model\":\"gemma-4-26b-a4b-it-awq\",\"messages\":[{\"role\":\"user\",\"content\":\"1+1=?\"}],\"max_tokens\":100}" \
  | python3 -m json.tool'
```

Expected: 正常返回 JSON 响应，KV cache 使用 TQ4 压缩。

- [ ] **Step 5: 验证 KV cache 压缩率**

检查日志中的 TQ4 压缩信息：
```bash
ssh cet_ai_server 'grep "TQ4\|compression\|bytes/token" /tmp/vllm.log'
```

Expected: 日志显示 `~3.76x compression vs FP16`。

- [ ] **Step 6: Commit 测试结果**

```bash
git commit --allow-empty -m "chore: verified Gemma 4 26B AWQ end-to-end on RTX 4090"
```
