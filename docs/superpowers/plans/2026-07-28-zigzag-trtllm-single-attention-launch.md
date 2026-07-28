# Zigzag TRT-LLM Single Attention Launch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the two per-layer TRT-LLM context-attention launches used by zigzag CP with one launch over a synthetic `2 * batch_size` request batch, without changing token order or attention semantics.

**Architecture:** Extend zigzag metadata with combined prev-then-next sequence geometry. Extend TRT-LLM MHA metadata with a page table duplicated in the same prev-then-next request order. `ZigzagCPStrategy.run_attention` keeps the existing two-call path for FlashAttention but uses one combined call for TRT-LLM MHA; the TRT-LLM closure selects the duplicated page table for that call.

**Tech Stack:** PyTorch metadata construction, FlashInfer `trtllm_batch_context_with_kv_cache`, SGLang CP-v2 zigzag, CPU unit tests, GB300 kernel/e2e profiling.

## Global Constraints

- Start only after the BCG exit gate passes.
- Preserve the existing two-call FlashAttention behavior; this change is scoped to TRT-LLM MHA.
- The combined logical request order is:

  ```text
  prev(request 0), ..., prev(request B-1),
  next(request 0), ..., next(request B-1)
  ```

- Do not concatenate metadata or duplicate page tables once per layer. Build them once per forward during metadata initialization.
- Preserve padded CP rows at the end of the attention output.
- Add unit tests for batches with unequal sequence lengths, non-zero prefix lengths, and CP ranks 0–3.
- Store remote evidence under `/scratch/gptoss_bf16_cp4_ep4_opt_20260728/fused_attention/`.

---

## Task 1: Add Combined Zigzag Query/KV Geometry

**Files:**

- Modify: `python/sglang/srt/layers/cp/zigzag.py`
- Extend: `test/registered/cp/test_cp_strategy_unit.py`

**Metadata fields:**

```python
actual_seq_q_combined_tensor: Optional[Any] = None
kv_len_combined_tensor: Optional[Any] = None
cu_seqlens_q_combined_tensor: Optional[Any] = None
cu_seqlens_kv_combined_tensor: Optional[Any] = None
max_seqlen_q_combined: int = 0
```

- [ ] **Step 1: Extend expected-metadata test helpers**

  For every existing zigzag case, derive:

  ```python
  actual_q_combined = actual_seq_q_prev_list + actual_seq_q_next_list
  kv_len_combined = kv_len_prev_list + kv_len_next_list
  cu_q_combined = [0] + list(accumulate(actual_q_combined))
  cu_kv_combined = [0] + list(accumulate(kv_len_combined))
  ```

  Assert tensor dtypes are `torch.int32`, lengths are `2 * bs` or `2 * bs + 1`, and:

  ```python
  cu_q_combined[-1] == total_q_prev_tokens + total_q_next_tokens
  max_seqlen_q_combined == max(actual_q_combined)
  ```

- [ ] **Step 2: Add an explicit non-zero-prefix case**

  Use:

  ```text
  cp_size=4
  seq_lens=[19, 27]
  extend_seq_lens=[11, 13]
  ```

  For every CP rank, verify each combined KV length retains its request's prefix offset and that the second half corresponds to next blocks, not an interleaved request order.

- [ ] **Step 3: Run the metadata tests and confirm fields are absent**

  ```bash
  python3 -m pytest -q \
    test/registered/cp/test_cp_strategy_unit.py
  ```

  Expected: combined-field assertions fail.

- [ ] **Step 4: Build combined metadata once in `build_metadata`**

  After the existing four logical lists are complete, add:

  ```python
  actual_seq_q_combined_list = (
      actual_seq_q_prev_list + actual_seq_q_next_list
  )
  kv_len_combined_list = kv_len_prev_list + kv_len_next_list
  cu_q_combined = [0] + list(accumulate(actual_seq_q_combined_list))
  cu_kv_combined = [0] + list(accumulate(kv_len_combined_list))
  ```

  Materialize the four combined tensors on the same device and with the same dtype as the existing per-half tensors.

- [ ] **Step 5: Run the full CP strategy unit file**

  ```bash
  python3 -m pytest -q \
    test/registered/cp/test_cp_strategy_unit.py
  ```

  Expected: all metadata, split, gather, KV, and existing attention tests pass.

- [ ] **Step 6: Commit the combined geometry**

  ```bash
  git add python/sglang/srt/layers/cp/zigzag.py \
    test/registered/cp/test_cp_strategy_unit.py
  git commit -m "feat: build combined zigzag TRT-LLM metadata"
  ```

---

## Task 2: Make `run_attention` Issue One TRT-LLM Call

**Files:**

- Modify: `python/sglang/srt/layers/cp/zigzag.py`
- Extend: `test/registered/cp/test_cp_strategy_unit.py`

**Callback contract:**

```python
attn_fn(
    q_chunk,
    cu_seqlens_q,
    cache_seqlens,
    max_seqlen_q,
    *,
    cu_seqlens_kv,
    use_zigzag_page_table: bool = False,
)
```

- [ ] **Step 1: Add a failing TRT-LLM one-call dispatch test**

  Build rank-0 CP2 metadata for one eight-token request and a four-row query. Use an attention callback that records arguments and returns `q_chunk + 100`.

  Call:

  ```python
  out = ZigzagCPStrategy(cp_size=2).run_attention(
      q,
      fb,
      device=torch.device("cpu"),
      attn_fn=attn_fn,
      attention_backend=CPAttentionBackendKind.TRTLLM_MHA,
  )
  ```

  Assert:

  - `len(calls) == 1`;
  - the callback query equals `q[:logical_tokens]`;
  - `use_zigzag_page_table is True`;
  - combined Q and KV cumulative tensors match the metadata;
  - output equals `q + 100`.

- [ ] **Step 2: Add a padded-query test**

  Append two physical padding rows to `q`. The callback still receives only logical rows. Assert the returned two tail rows are zero and no second callback occurs.

- [ ] **Step 3: Add a CPU attention-math equivalence test**

  Implement a deterministic pure-PyTorch reference callback that, for every synthetic sequence described by `cu_seqlens_q`, `cu_seqlens_kv`, and `cache_seqlens`, computes scaled dot-product attention with a prefix-aware causal mask:

  ```python
  q_position = cache_seqlen - q_len + torch.arange(q_len)
  allowed = torch.arange(cache_seqlen)[None, :] <= q_position[:, None]
  scores = (q_seq @ k_seq.T) * scale
  out_seq = torch.softmax(scores.masked_fill(~allowed, -torch.inf), -1) @ v_seq
  ```

  For CP ranks 0–3, `seq_lens=[19, 27]`, and `extend_seq_lens=[11, 13]`, compare:

  1. the old reference path that invokes the callback separately with prev metadata and next metadata and concatenates the logical outputs; and
  2. one callback over the combined prev-then-next metadata and combined Q/K/V sequence storage.

  Require `torch.testing.assert_close` at `atol=1e-5`, `rtol=1e-5`. This is the semantic gate; a callback that merely returns `q + 100` is only a dispatch-contract test.

- [ ] **Step 4: Preserve the FlashAttention two-call test**

  Rename the existing test to `test_zigzag_flashattention_dispatch_runs_prev_then_next` and keep:

  ```python
  self.assertEqual(len(calls), 2)
  ```

  This prevents an accidental ABI change to FlashAttention.

- [ ] **Step 5: Run the tests and confirm TRT-LLM still calls twice**

  ```bash
  python3 -m pytest -q \
    test/registered/cp/test_cp_strategy_unit.py
  ```

  Expected: only the new TRT-LLM call-count and combined-argument assertions fail.

- [ ] **Step 6: Implement the backend-specific dispatch**

  In `run_attention`:

  ```python
  logical_tokens = meta.total_q_prev_tokens + meta.total_q_next_tokens
  if attention_backend == CPAttentionBackendKind.TRTLLM_MHA:
      result = attn_fn(
          q[:logical_tokens],
          meta.cu_seqlens_q_combined_tensor,
          meta.kv_len_combined_tensor,
          meta.max_seqlen_q_combined,
          cu_seqlens_kv=meta.cu_seqlens_kv_combined_tensor,
          use_zigzag_page_table=True,
      )
  else:
      # Existing prev and next calls, unchanged.
  ```

  Reuse the existing padding-tail logic after either branch. Do not create `q_prev`, `q_next`, or concatenate results in the TRT-LLM branch.

- [ ] **Step 7: Run the CP strategy tests**

  ```bash
  python3 -m pytest -q \
    test/registered/cp/test_cp_strategy_unit.py
  ```

  Expected: one TRT-LLM call, two FlashAttention calls, and all padding cases pass.

- [ ] **Step 8: Commit the one-call dispatcher**

  ```bash
  git add python/sglang/srt/layers/cp/zigzag.py \
    test/registered/cp/test_cp_strategy_unit.py
  git commit -m "perf: merge zigzag TRT-LLM attention calls"
  ```

---

## Task 3: Duplicate TRT-LLM Page Tables Once per Forward

**Files:**

- Modify: `python/sglang/srt/layers/attention/trtllm_mha_backend.py`
- Create: `test/registered/unit/layers/attention/test_trtllm_mha_zigzag.py`

**Metadata fields and helper:**

```python
zigzag_page_table: torch.Tensor = None
zigzag_swa_page_table: torch.Tensor = None

def _build_zigzag_page_tables(
    self,
    metadata: TRTLLMMHAMetadata,
    forward_batch: ForwardBatch,
) -> None:
    ...

def _get_zigzag_layer_page_table(
    self,
    layer: RadixAttention,
) -> torch.Tensor:
    ...
```

- [ ] **Step 1: Add failing page-table order tests**

  Make the new file inherit from `CustomTestCase` and register it literally:

  ```python
  from sglang.test.ci.ci_register import register_cpu_ci
  from sglang.test.test_utils import CustomTestCase

  register_cpu_ci(est_time=8, suite="base-a-test-cpu")
  ```

  End the file with `unittest.main()`.

  Construct:

  ```python
  metadata.page_table = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
  metadata.swa_page_table = torch.tensor([[30, 31], [40, 41]], dtype=torch.int32)
  ```

  With CP-v2 active, assert:

  ```python
  metadata.zigzag_page_table.tolist() == [
      [10, 11], [20, 21], [10, 11], [20, 21]
  ]
  metadata.zigzag_swa_page_table.tolist() == [
      [30, 31], [40, 41], [30, 31], [40, 41]
  ]
  ```

  With CP-v2 inactive, both fields remain `None`.

- [ ] **Step 2: Add full-vs-SWA selection tests**

  Patch the layer mapping so one layer is full attention and one is SWA. Assert `_get_zigzag_layer_page_table` selects the duplicated table matching `_get_layer_page_table`.

- [ ] **Step 3: Run the new test and confirm the helper is absent**

  ```bash
  python3 -m pytest -q \
    test/registered/unit/layers/attention/test_trtllm_mha_zigzag.py
  ```

  Expected: missing fields/helpers fail.

- [ ] **Step 4: Build duplicated tables immediately after the base page table**

  In eager extend metadata initialization, after `_fill_page_table_device`, call `_build_zigzag_page_tables`.

  Implement duplication as:

  ```python
  metadata.zigzag_page_table = torch.cat(
      [metadata.page_table, metadata.page_table], dim=0
  )
  ```

  Apply the same operation to `swa_page_table` when present. This runs once per forward outside per-layer attention calls.

- [ ] **Step 5: Run the page-table tests**

  ```bash
  python3 -m pytest -q \
    test/registered/unit/layers/attention/test_trtllm_mha_zigzag.py
  ```

  Expected: order and layer selection tests pass.

- [ ] **Step 6: Commit the duplicated metadata**

  ```bash
  git add \
    python/sglang/srt/layers/attention/trtllm_mha_backend.py \
    test/registered/unit/layers/attention/test_trtllm_mha_zigzag.py
  git commit -m "feat: prepare zigzag TRT-LLM page tables"
  ```

---

## Task 4: Select the Combined Page Table in the TRT-LLM Closure

**Files:**

- Modify: `python/sglang/srt/layers/attention/trtllm_mha_backend.py`
- Extend: `test/registered/unit/layers/attention/test_trtllm_mha_zigzag.py`

- [ ] **Step 1: Add a failing closure contract test**

  Patch `flashinfer.prefill.trtllm_batch_context_with_kv_cache` and invoke the closure through `forward_extend` with a fake CP strategy.

  Assert the fake strategy calls the callback with `use_zigzag_page_table=True` and FlashInfer receives:

  - duplicated block table with `2 * bs` rows;
  - `batch_size == 2 * bs`;
  - combined `seq_lens`, `cum_seq_lens_q`, and `cum_seq_lens_kv`;
  - the original single concatenated query tensor.

- [ ] **Step 2: Add a non-CP regression test**

  Invoke the callback without `use_zigzag_page_table`. Assert the original page table, batch size, and metadata are unchanged.

- [ ] **Step 3: Run the tests and confirm the callback rejects the new keyword**

  ```bash
  python3 -m pytest -q \
    test/registered/unit/layers/attention/test_trtllm_mha_zigzag.py
  ```

  Expected: the closure does not accept `use_zigzag_page_table`.

- [ ] **Step 4: Extend `_trtllm_context_attn`**

  Add:

  ```python
  def _trtllm_context_attn(
      q_chunk,
      cu_seqlens_q,
      cache_seqlens,
      max_seqlen_q,
      *,
      cu_seqlens_kv,
      use_zigzag_page_table=False,
  ):
      block_tables = (
          self._get_zigzag_layer_page_table(layer)
          if use_zigzag_page_table
          else page_table
      )
      ...
  ```

  Pass `block_tables` to FlashInfer. Keep all scale, window, sink, output dtype, and workspace arguments unchanged.

- [ ] **Step 5: Run all focused attention/CP tests**

  ```bash
  python3 -m pytest -q \
    test/registered/cp/test_cp_strategy_unit.py \
    test/registered/unit/layers/attention/test_trtllm_mha_zigzag.py
  ```

  Expected: all tests pass.

- [ ] **Step 6: Run formatting and static checks on touched files**

  ```bash
  pre-commit run --files \
    python/sglang/srt/layers/cp/zigzag.py \
    python/sglang/srt/layers/attention/trtllm_mha_backend.py \
    test/registered/cp/test_cp_strategy_unit.py \
    test/registered/unit/layers/attention/test_trtllm_mha_zigzag.py
  ```

  Expected: all hooks pass.

- [ ] **Step 7: Commit the TRT-LLM callback change**

  ```bash
  git add \
    python/sglang/srt/layers/attention/trtllm_mha_backend.py \
    test/registered/unit/layers/attention/test_trtllm_mha_zigzag.py
  git commit -m "perf: launch zigzag TRT-LLM context attention once"
  ```

---

## Task 5: Validate Numerical and Serving Correctness on GB300

**Files:**

- Record remotely: `fused_attention/install.log`
- Record remotely: `fused_attention/server.log`
- Record remotely: `fused_attention/smoke.jsonl`
- Record remotely: `fused_attention/gsm8k.txt`
- Record remotely: `fused_attention/parity.json`

- [ ] **Step 1: Install the exact branch tip**

  Push the optimization branch, switch the remote checkout detached to the same SHA, and reinstall editable. Preserve both SHAs in `fused_attention/commit.txt`.

- [ ] **Step 2: Launch CP4+EP4 with BCG and explicit buckets**

  Use:

  ```text
  SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK=65536
  --tp 4 --ep 4
  --enable-prefill-cp --attn-cp-size 4 --cp-strategy zigzag
  --moe-a2a-backend flashinfer
  --moe-runner-backend flashinfer_trtllm_routed
  --attention-backend trtllm_mha
  --context-length 139264
  --chunked-prefill-size 131072
  --max-prefill-tokens 131072
  --cuda-graph-backend-prefill breakable
  --cuda-graph-bs-prefill 4096 65536 131072
  ```

  Expected: capture and health succeed.

- [ ] **Step 3: Compare deterministic outputs with the parent BCG commit and TP4**

  On the parent BCG commit in both eager-prefill and BCG modes, the one-launch commit, and TP4 on the one-launch commit, run the same random-ID seeds for:

  ```text
  (input, output, concurrency) =
  (4096, 64, 1),
  (131072, 20, 1),
  (131072, 20, 4)
  ```

  Require:

  - identical generated token IDs at temperature zero across eager CP, BCG CP, one-launch CP, and TP4;
  - identical completion lengths;
  - no NaN/Inf or worker error;
  - GSM8K-20 score at least `0.80`.

- [ ] **Step 4: Stop only each run's recorded server PID**

  Preserve logs before stopping.

---

## Task 6: Prove the Kernel Launch Reduction in a Torch Profile

**Files:**

- Record remotely: `fused_attention/profile-before/`
- Record remotely: `fused_attention/profile-after/`
- Record remotely: `fused_attention/attention-launch-summary.json`

- [ ] **Step 1: Capture matched before/after profiles**

  For the parent BCG commit and the one-launch commit, use CP4+EP4, concurrency 1, 128K input, 20 output, and 20 profile steps:

  ```bash
  python3 -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port 30000 \
    --dataset-name random-ids \
    --num-prompts 1 --max-concurrency 1 --request-rate inf \
    --random-input-len 131072 --random-output-len 20 \
    --random-range-ratio 1 --tokenize-prompt \
    --temperature 0 --warmup-requests 1 --flush-cache \
    --profile --profile-steps 20 --profile-activities CPU GPU \
    --profile-output-dir /scratch/gptoss_bf16_cp4_ep4_opt_20260728/fused_attention/profile-after \
    --profile-prefix cp4-ep4-one-launch
  ```

  Use `profile-before` and a matching prefix for the parent commit.

- [ ] **Step 2: Analyze both profiles**

  Use `llm-torch-profiler-analysis`. Count events for the FlashInfer TRT-LLM context-attention kernel and report:

  - total launches;
  - launches per transformer layer;
  - total attention GPU time;
  - CPU launch gaps around attention;
  - end-to-end TTFT.

- [ ] **Step 3: Enforce the one-launch gate**

  Pass only if:

  - context-attention launches per layer fall from two to one;
  - total launch count is within one warmup/profiling boundary of half the parent count;
  - deterministic outputs match;
  - attention GPU time does not regress by more than 3%;
  - median preliminary TTFT is not worse.

- [ ] **Step 4: Diagnose a failed gate before continuing**

  If the kernel internally serializes the `2 * bs` batch or regresses GPU time, inspect grid selection, `max_q_len`, page-table order, and per-request lengths. Do not declare success based only on fewer trace rows.

---

## One-Launch Exit Gate

Do not begin kernel fusion until:

- CPU tests prove combined metadata ordering for all CP4 ranks;
- TRT-LLM dispatch calls attention once and FlashAttention still calls twice;
- duplicated page tables are built once per forward;
- GB300 outputs match the parent BCG commit;
- GSM8K-20 remains at least 0.80;
- profiles show one TRT-LLM context-attention launch per layer;
- total attention time and TTFT are neutral or improved;
- the exact traces and summary JSON are preserved.
