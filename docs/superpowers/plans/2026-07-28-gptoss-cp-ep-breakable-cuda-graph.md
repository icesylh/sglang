# GPT-OSS CP+EP Breakable CUDA Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run GPT-OSS BF16 CP4+EP4 prefill through the breakable CUDA-graph body path while attention remains an eager graph break, preserving CP semantics and FlashInfer A2A correctness.

**Architecture:** Extend `PrefillCudaGraphRunner` with a CP-v2 body-capture path parallel to `EagerRunner._execute_extend_cp_v2`. CP metadata, embedding sharding, and final hidden-state gathering remain outside the captured transformer body; fixed per-bucket CP input buffers feed captured layers; existing `RadixAttention` breakable wrappers keep TRT-LLM attention eager. Replace the broad server-argument bans with a narrow allowlist for full prefill CP plus FlashInfer routed MoE.

**Tech Stack:** SGLang breakable CUDA graph, CP-v2 zigzag, PyTorch CUDA graphs, TRT-LLM MHA, FlashInfer A2A and routed BF16 MoE, four NVIDIA GB300 GPUs.

## Global Constraints

- Start only after the support exit gate in `2026-07-28-gptoss-bf16-flashinfer-cp-ep-support.md` passes.
- Keep attention outside captured graph segments through the existing `breakable_unified_attention_with_output` path.
- Do not graph-capture dynamic Python CP metadata construction.
- Use fixed tensor addresses for every tensor consumed by captured layer-body segments.
- Permit only the validated topology; retain existing BCG bans for partial CP, other CP strategies, other MoE A2A backends, MLA, DCP, and unvalidated multimodal cases.
- Use explicit prefill capture buckets instead of the generated 512-token sweep.
- Store remote evidence under `/scratch/gptoss_bf16_cp4_ep4_opt_20260728/bcg/`.
- Add source commits to `codex/gptoss-bf16-cp4-ep4-opt`; do not add them to the support-only PR.

---

## Task 1: Replace Broad BCG Bans with a Narrow CP+FlashInfer Allowlist

**Files:**

- Modify: `python/sglang/srt/server_args.py`
- Extend: `test/registered/unit/server_args/test_flashinfer_a2a_cp.py`

**Interfaces:**

```python
def _supports_breakable_prefill_cp(self) -> bool:
    ...

def _supports_breakable_moe_a2a(self) -> bool:
    ...
```

- [ ] **Step 1: Add failing server-argument truth-table tests**

  Build `ServerArgs` shells with resolved fields and test:

  | Prefill CP | CP size | TP | Strategy | Prefill attention | A2A | MoE runner | CP allowed | A2A allowed |
  |---|---:|---:|---|---|---|---|---|---|
  | true | 4 | 4 | zigzag | trtllm_mha | flashinfer | flashinfer_trtllm_routed | true | true |
  | true | 2 | 4 | zigzag | trtllm_mha | flashinfer | flashinfer_trtllm_routed | false | false |
  | true | 4 | 4 | interleave | trtllm_mha | flashinfer | flashinfer_trtllm_routed | false | false |
  | true | 4 | 4 | zigzag | flashattention | flashinfer | flashinfer_trtllm_routed | false | false |
  | true | 4 | 4 | zigzag | trtllm_mha | deepep | deep_gemm | true | false |
  | true | 4 | 4 | zigzag | trtllm_mha | flashinfer | triton | true | false |
  | false | 1 | 4 | zigzag | trtllm_mha | none | triton | false | true |

  Set `model_config.hf_config.architectures=["GptOssForCausalLM"]` for the accepted rows. Add a negative row with an otherwise identical topology and `architectures=["LlamaForCausalLM"]`; both support predicates must reject it.

  Also invoke `_disable_breakable_cudagraph_if_incompatible` and assert the exact validated combination leaves `cuda_graph_config.prefill.backend == Backend.BREAKABLE`, while each rejected combination sets it to `Backend.DISABLED`.

- [ ] **Step 2: Run the tests and confirm the current broad bans**

  Run:

  ```bash
  python3 -m pytest -q \
    test/registered/unit/server_args/test_flashinfer_a2a_cp.py
  ```

  Expected: validated CP and FlashInfer A2A still disable BCG.

- [ ] **Step 3: Implement the narrow support predicates**

  `_supports_breakable_prefill_cp` must resolve the effective prefill backend through the same helper used by `ModelRunner.init_attention_backend`, then require:

  ```python
  resolved = self._resolved()
  prefill_attention_backend, _ = self._resolved_attention_backends()
  return (
      self.enable_prefill_cp
      and resolved.attn_cp_size == self.tp_size
      and self.cp_strategy == "zigzag"
      and prefill_attention_backend == "trtllm_mha"
      and self.get_model_config().hf_config.architectures
      == ["GptOssForCausalLM"]
  )
  ```

  Read the architecture through the existing `ServerArgs` model-config accessor rather than constructing a second `ModelConfig`. Accept only `GptOssForCausalLM`; this is a proof-scoped allowlist, not a claim that every MHA model supports CP body capture.

  `_supports_breakable_moe_a2a` must return true for A2A `none`, or require all of:

  ```python
  resolved.moe_a2a_backend == "flashinfer"
  self.moe_runner_backend == "flashinfer_trtllm_routed"
  self._supports_breakable_prefill_cp()
  ```

  Change only the two BCG rules:

  ```python
  (
      "context parallel (attn_cp_size > 1)",
      lambda: resolved.attn_cp_size > 1
      and not self._supports_breakable_prefill_cp(),
  )
  (
      "MoE A2A backend",
      lambda: not self._supports_breakable_moe_a2a(),
  )
  ```

- [ ] **Step 4: Run the focused and adjacent server-argument tests**

  ```bash
  python3 -m pytest -q \
    test/registered/unit/server_args/test_flashinfer_a2a_cp.py \
    test/registered/unit/server_args/test_server_args.py
  ```

  Expected: all tests pass and unvalidated configurations remain disabled.

- [ ] **Step 5: Commit the BCG allowlist**

  ```bash
  git add python/sglang/srt/server_args.py \
    test/registered/unit/server_args/test_flashinfer_a2a_cp.py
  git commit -m "feat: allow breakable graphs for validated CP FlashInfer"
  ```

---

## Task 2: Add Fixed CP Input Buffers to the Prefill Graph Runner

**Files:**

- Modify: `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`
- Create: `test/registered/unit/model_executor/runner/test_prefill_cuda_graph_cp.py`

**State and interfaces:**

```python
self.enable_cp_v2_body_capture: bool
self.cp_input_embeds: Optional[torch.Tensor]
self.cp_positions: Optional[torch.Tensor]
self.cp_bucket_local_tokens: Dict[int, int]

def _prepare_cp_static_inputs(
    self,
    forward_batch: ForwardBatch,
    *,
    static_num_tokens: int,
    capture: bool,
) -> int:
    """Build live CP metadata, shard embeddings/positions, and fill fixed buffers."""
```

- [ ] **Step 1: Write failing helper tests without constructing a GPU runner**

  Make the new file inherit from `CustomTestCase` and register it literally:

  ```python
  from sglang.test.ci.ci_register import register_cpu_ci
  from sglang.test.test_utils import CustomTestCase

  register_cpu_ci(est_time=8, suite="base-a-test-cpu")
  ```

  End the file with `unittest.main()`.

  Allocate a runner with `PrefillCudaGraphRunner.__new__`, inject:

  - a fake model with `get_input_embeddings`;
  - CPU `cp_input_embeds` and `cp_positions` buffers;
  - `cp_bucket_local_tokens={16: 4}`;
  - a rank-0 zigzag strategy with CP4;
  - a `ForwardBatch` containing a 16-token request.

  Patch `prepare_cp_forward` and `cp_split_before_forward` to return a known four-row local embedding and position tensor. Assert `_prepare_cp_static_inputs`:

  - copies exactly four rows;
  - zeroes any tail up to the bucket-local row count;
  - sets `forward_batch.input_embeds` and `forward_batch.positions` to fixed-address slices;
  - returns four;
  - preserves `forward_batch.input_ids` as the global token-ID view.

  Add a replay case where the live local rows are 3 but the captured bucket has 4; require row 3 of both static buffers to be zero.

  Add a second replay case that changes request count from one to four inside the same global bucket and uses unequal per-request extend lengths. Assert `prepare_cp_forward` rebuilds the split/reverse/length metadata, fixed buffer addresses stay unchanged, and no tensor view from the prior request survives.

- [ ] **Step 2: Confirm the helper and state are absent**

  ```bash
  python3 -m pytest -q \
    test/registered/unit/model_executor/runner/test_prefill_cuda_graph_cp.py
  ```

  Expected: the helper is missing.

- [ ] **Step 3: Detect the supported CP-v2 body-capture mode during runner construction**

  Import:

  ```python
  from sglang.srt.layers.cp.utils import (
      cp_gather_after_forward,
      cp_split_before_forward,
      enable_cp_v2,
      prepare_cp_forward,
  )
  ```

  Set `enable_cp_v2_body_capture` only when:

  - backend is `BreakableCudaGraphBackend`;
  - CP-v2 is enabled;
  - server args have full prefill CP;
  - the server-argument support predicate passed.

  For this mode allocate, on the runner device:

  ```python
  self.cp_input_embeds = torch.zeros(
      (self.max_num_tokens, model_runner.model_config.hidden_size),
      dtype=model_runner.dtype,
  )
  self.cp_positions = torch.zeros(
      (self.max_num_tokens,), dtype=torch.int64
  )
  self.cp_bucket_local_tokens = {}
  ```

  Global-sized allocation is intentional: it avoids deriving a fragile upper bound and is still smaller than a full transformer activation set.

- [ ] **Step 4: Implement `_prepare_cp_static_inputs`**

  The helper must:

  1. call `prepare_cp_forward(forward_batch)` to build live, non-captured Python metadata;
  2. obtain full embeddings from `model_runner.model.get_input_embeddings()(forward_batch.input_ids[:raw_tokens])`;
  3. call `cp_split_before_forward(full_embeddings, full_positions, forward_batch)`;
  4. on capture, save `local_embeddings.shape[0]` in `cp_bucket_local_tokens[static_num_tokens]`;
  5. on replay, reject rather than overflow if the live local row count exceeds the saved bucket row count;
  6. copy the local tensors into fixed buffers and zero the tail;
  7. replace `forward_batch.input_embeds` and `forward_batch.positions` with fixed-buffer slices of the captured local size.

  The returned row count is the live physical local row count used to trim captured output before CP gather.

- [ ] **Step 5: Run the helper tests**

  ```bash
  python3 -m pytest -q \
    test/registered/unit/model_executor/runner/test_prefill_cuda_graph_cp.py
  ```

  Expected: all fixed-address, copy, zero-tail, and overflow tests pass.

- [ ] **Step 6: Commit the CP buffer helper**

  ```bash
  git add \
    python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py \
    test/registered/unit/model_executor/runner/test_prefill_cuda_graph_cp.py
  git commit -m "feat: stage CP inputs for breakable prefill graphs"
  ```

---

## Task 3: Capture and Replay the CP-Local Transformer Body

**Files:**

- Modify: `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`
- Extend: `test/registered/unit/model_executor/runner/test_prefill_cuda_graph_cp.py`
- Extend: `test/registered/unit/model_executor/runner/test_prefill_cuda_graph_padding.py`

**Interfaces:**

```python
def _execute_cp_body_capture(
    self,
    forward_batch: ForwardBatch,
    static_forward_batch: ForwardBatch,
    static_num_tokens: int,
    raw_num_tokens: int,
    **kwargs,
) -> Union[LogitsProcessorOutput, PPProxyTensors]:
    ...

def _validate_cp_flashinfer_dispatch_capacity(
    self,
    *,
    global_bucket_tokens: int,
    required_local_tokens: int,
) -> None:
    ...
```

- [ ] **Step 1: Add failing capture-preparation tests**

  Assert that in CP body-capture mode:

  - `capture_prepare(bucket)` creates zigzag metadata for the dummy global bucket;
  - `_prepare_cp_static_inputs(..., capture=True)` runs before attention metadata initialization;
  - `_run_forward` passes CP-local fixed `input_embeds` and positions to `layer_model.forward`;
  - the graph shape key remains the global bucket size;
  - `backend._output_rows` stores the CP-local body output row count.

- [ ] **Step 2: Add failing replay tests**

  Mock `backend.replay` to return a captured local tensor with padded tail. Assert `_execute_cp_body_capture`:

  1. slices to the live physical CP row count;
  2. calls `cp_gather_after_forward`;
  3. calls `model.logits_processor` with gathered hidden states and live global `input_ids`;
  4. never invokes the normal outer `model.forward` monkey-patch path;
  5. propagates auxiliary hidden states using the same tuple rules as `EagerRunner._execute_extend_cp_v2`.

- [ ] **Step 3: Add failing FlashInfer dispatch-capacity tests**

  Inject fake entries into `runner.moe_layers`, each with a `.dispatcher.max_num_tokens`. Assert:

  - capacity equal to the captured CP-local row count passes;
  - capacity greater than the row count passes;
  - capacity smaller than the row count raises a `ValueError` naming the global bucket, required local rows, configured capacity, and `SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK`;
  - non-FlashInfer dispatchers are ignored even if they expose a similarly named field.

  This makes workspace sizing a checked pre-capture invariant rather than an operational guess.

- [ ] **Step 4: Confirm the existing runner has no CP body path**

  ```bash
  python3 -m pytest -q \
    test/registered/unit/model_executor/runner/test_prefill_cuda_graph_cp.py \
    test/registered/unit/model_executor/runner/test_prefill_cuda_graph_padding.py
  ```

  Expected: the CP-specific assertions fail.

- [ ] **Step 5: Prepare CP inputs and validate dispatcher capacity during capture**

  In `capture_one_shape`, after `capture_prepare` and before attention metadata initialization:

  ```python
  if self.enable_cp_v2_body_capture:
      required_local_tokens = self._prepare_cp_static_inputs(
          forward_batch,
          static_num_tokens=num_tokens,
          capture=True,
      )
      self._validate_cp_flashinfer_dispatch_capacity(
          global_bucket_tokens=num_tokens,
          required_local_tokens=required_local_tokens,
      )
  ```

  Implement the validator by iterating `self.moe_layers`, selecting each layer's `.dispatcher` only when it is a `FlashInferDispatcher`, and checking it once per unique dispatcher object. Log one structured summary per capture bucket with configured and required rows.

  The attention backend then initializes against a `ForwardBatch` that already contains live CP metadata and global cache locations.

- [ ] **Step 6: Keep `_run_forward` on the layer-body API**

  In CP body-capture mode, call:

  ```python
  return self.layer_model.forward(
      forward_batch.input_ids,
      forward_batch.positions,
      forward_batch,
      forward_batch.input_embeds,
  )
  ```

  Here `positions` and `input_embeds` are CP-local fixed slices while `input_ids` remains global. Do not call `prepare_cp_forward` inside the graph capture body.

- [ ] **Step 7: Prepare CP inputs and metadata during replay**

  In `load_batch`, after constructing `static_forward_batch` and before `_prepare_forward_metadata_for_replay`, call:

  ```python
  self._cp_live_local_tokens = self._prepare_cp_static_inputs(
      static_forward_batch,
      static_num_tokens=static_num_tokens,
      capture=False,
  )
  ```

  Keep `attn_cp_metadata` on `static_forward_batch`; this is what the eager attention graph-break callback and CP-aware MoE communication read during replay.

- [ ] **Step 8: Implement the CP replay tail**

  `_execute_cp_body_capture` must:

  ```python
  local_hidden = self.backend.replay(
      ShapeKey(size=static_num_tokens), static_forward_batch, **kwargs
  )
  local_hidden = _slice_output_rows(local_hidden, self._cp_live_local_tokens)
  hidden_states = cp_gather_after_forward(
      local_hidden, static_forward_batch, torch.cuda.current_stream()
  )
  return model.logits_processor(
      forward_batch.input_ids,
      hidden_states,
      model.lm_head,
      forward_batch,
      aux_hidden_states,
  )
  ```

  Match the PP-rank and auxiliary-hidden-state branches in `EagerRunner._execute_extend_cp_v2`.

- [ ] **Step 9: Route `execute` to the CP replay tail**

  Use `_execute_cp_body_capture` only when `enable_cp_v2_body_capture` is true. Preserve the existing body-capture path for non-CP BCG and Full CG.

- [ ] **Step 10: Run all runner unit tests**

  ```bash
  python3 -m pytest -q \
    test/registered/unit/model_executor/runner/test_prefill_cuda_graph_cp.py \
    test/registered/unit/model_executor/runner/test_prefill_cuda_graph_padding.py \
    test/registered/cuda_graph/breakable/test_breakable_cuda_graph.py
  ```

  Expected: all CPU-capable tests pass; CUDA-only tests pass on a CUDA host or skip locally.

- [ ] **Step 11: Commit the CP body replay**

  ```bash
  git add \
    python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py \
    test/registered/unit/model_executor/runner/test_prefill_cuda_graph_cp.py \
    test/registered/unit/model_executor/runner/test_prefill_cuda_graph_padding.py
  git commit -m "feat: replay CP-local bodies with breakable CUDA graphs"
  ```

---

## Task 4: Prove Attention Is Eager and Captured Segments Are Reused

**Files:**

- Extend: `test/registered/cuda_graph/breakable/test_breakable_cuda_graph.py`
- Record remotely: `bcg/segment_inventory.json`
- Record remotely: `bcg/capture.log`

- [ ] **Step 1: Add a CUDA unit test with a CP-shaped attention break**

  Keep the file's existing CUDA CI registrations and `CustomTestCase` base; do not create an unregistered standalone test module.

  Construct a simple breakable graph:

  ```python
  q = torch.zeros((4, 8), device="cuda")
  linear_out = q @ weight
  attn_out = eager_on_graph(True)(fake_attention)(linear_out, live_metadata)
  output.copy_(attn_out @ weight2)
  ```

  Replay after changing both `q` and tensor fields inside `live_metadata`. Assert:

  - output reflects the new values;
  - the graph has two captured segments;
  - the fake attention callback runs once per replay;
  - the linear operations do not re-enter Python.

- [ ] **Step 2: Run the CUDA break test on `baizhou-dev`**

  ```bash
  ssh baizhou-dev '
    cd /sgl-workspace/sglang
    python3 -m pytest -q \
      test/registered/cuda_graph/breakable/test_breakable_cuda_graph.py
  '
  ```

  Expected: all breakable graph tests pass on GB300.

- [ ] **Step 3: Enable a debug-only segment inventory**

  Use existing BCG debug logging or inspect `BreakableCUDAGraph._segments` and `_break_fns` after capture. Do not add a permanent user-facing flag. Record, per 4096-token bucket:

  ```json
  {
    "bucket": 4096,
    "segments": 25,
    "break_functions": 24,
    "attention_layers": 24
  }
  ```

  Exact counts may differ with model depth; the invariant is one eager attention break per attention layer and at least one captured compute segment between adjacent breaks.

- [ ] **Step 4: Commit only a test if source code was not required**

  If capture evidence instead shows FlashInfer `dispatch` or `combine` is not graph-safe while the same operation succeeds eagerly, add a focused failing break/replay test and wrap only that operation with the existing `eager_on_graph` mechanism in `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py`. Re-run the segment inventory and require captured transformer segments on both sides of the new A2A break. Never respond by disabling the entire prefill graph.

  ```bash
  git add test/registered/cuda_graph/breakable/test_breakable_cuda_graph.py
  git commit -m "test: cover CP-shaped breakable attention replay"
  ```

  If the measured A2A break was required, stage the test plus `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py` and use commit message `fix: break prefill graph around FlashInfer A2A` instead.

---

## Task 5: Validate Small-Shape CP4+EP4 BCG on GB300

**Files:**

- Record remotely: `bcg/install.log`
- Record remotely: `bcg/server-4096.log`
- Record remotely: `bcg/smoke-4096.jsonl`
- Record remotely: `bcg/gsm8k.txt`

- [ ] **Step 1: Install the exact optimization-branch tip**

  Push the branch, fetch it on `baizhou-dev`, switch detached to the exact remote tip, and reinstall editable. Record local and remote SHAs.

- [ ] **Step 2: Launch one explicit 4096-token BCG bucket**

  ```bash
  SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK=4096 \
  SGLANG_TORCH_PROFILER_DIR=/scratch/gptoss_bf16_cp4_ep4_opt_20260728/bcg \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  python3 -m sglang.launch_server \
    --model-path /scratch/models/gpt-oss-120b-bf16 \
    --host 127.0.0.1 --port 30000 \
    --tp 4 --ep 4 \
    --enable-prefill-cp --attn-cp-size 4 --cp-strategy zigzag \
    --moe-a2a-backend flashinfer \
    --moe-runner-backend flashinfer_trtllm_routed \
    --attention-backend trtllm_mha \
    --context-length 8192 \
    --chunked-prefill-size 4096 --max-prefill-tokens 4096 \
    --cuda-graph-backend-prefill breakable \
    --cuda-graph-bs-prefill 4096
  ```

  Expected: capture completes, health reaches HTTP 200, and server logs do not say prefill CUDA graph was disabled.

- [ ] **Step 3: Run repeated variable-length requests against the same bucket**

  Run input lengths `4096`, `3072`, `2048`, then `4096` again, each with 20 output tokens and cache flush. Expected:

  - all requests succeed;
  - the 3072 and 2048 requests reuse the 4096 bucket;
  - output token counts are correct;
  - no stale metadata, illegal memory access, shape mismatch, or collective hang appears.

- [ ] **Step 4: Run the GSM8K-20 correctness gate**

  Expected score: at least `0.80`.

- [ ] **Step 5: Stop only the PID saved for this launch**

  Verify the recorded PID exits and all four scheduler workers disappear.

---

## Task 6: Validate the 128K Benchmark Shape and Measure CPU-Overhead Removal

**Files:**

- Record remotely: `bcg/server-128k.log`
- Record remotely: `bcg/eager-c1.jsonl`
- Record remotely: `bcg/bcg-c1.jsonl`
- Record remotely: `bcg/eager-c4.jsonl`
- Record remotely: `bcg/bcg-c4.jsonl`
- Record remotely: `bcg/cpu-overhead-summary.json`

- [ ] **Step 1: Launch with explicit small, middle, and 128K capture buckets and validated FlashInfer workspace**

  Use:

  ```text
  SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
  SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK=65536
  --context-length 139264
  --chunked-prefill-size 131072
  --max-prefill-tokens 131072
  --cuda-graph-backend-prefill breakable
  --cuda-graph-bs-prefill 4096 65536 131072
  ```

  For CP4 the largest logical local share is `ceil(131072 / 4) = 32768`; keep the configured `65536` safety capacity and require the new pre-capture validator to log `configured >= required` for every bucket. Before any later 262144-token capture, derive the required capacity again from the logged physical CP-local row count instead of reusing this value blindly.

  Keep all exact CP4+EP4, A2A, routed MoE, and TRT-LLM MHA flags from the small-shape launch.

- [ ] **Step 2: Replay the middle and largest buckets before benchmarking**

  First run the following command with `--random-input-len 65536`, then repeat it with `131072`:

  ```bash
  python3 -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port 30000 \
    --dataset-name random-ids \
    --num-prompts 1 --max-concurrency 1 --request-rate inf \
    --random-input-len 131072 --random-output-len 20 \
    --random-range-ratio 1 --tokenize-prompt \
    --temperature 0 --warmup-requests 1 --flush-cache
  ```

  Expected: BCG replay is used for both explicit buckets, both requests succeed, and the dispatcher-capacity log covers the physical local row count for both.

- [ ] **Step 3: Compare BCG against the same commit with prefill BCG disabled**

  For each mode and each concurrency in `{1, 4}`, run three preliminary repeats with exact input `131072`, output `20`, random IDs, range ratio `1`, temperature `0`, one warmup request, and cache flush.

  Preserve TTFT and server-side CPU trace summaries. The BCG gate is:

  - no correctness regression;
  - identical temperature-zero generated token IDs between eager and BCG for the same seeds at 4096/20 and 131072/20;
  - median TTFT is no worse than eager by more than 1%;
  - Python/operator launch count between attention layers drops by at least 50%;
  - transformer linear, normalization, routing, and MoE kernels appear in captured CUDA-graph segments;
  - TRT-LLM attention remains an eager break.

- [ ] **Step 4: Diagnose instead of masking any BCG failure**

  If capture or replay fails, use `superpowers:systematic-debugging`. Record the first failing bucket, rank, layer, and operation. Do not broaden the allowlist or add a global eager fallback. Fix the smallest unstable address/shape boundary and repeat Tasks 3–6.

- [ ] **Step 5: Commit any GB300-derived correction with its regression test**

  Each correction gets its own test-first commit. Example:

  ```text
  fix: refresh CP metadata before breakable replay
  ```

---

## BCG Exit Gate

Do not begin the one-launch attention work until:

- the exact CP4+EP4 server starts with `--cuda-graph-backend-prefill breakable`;
- 4096 and 128K replay both pass repeated-shape and smaller-in-bucket requests;
- the 65536-token middle bucket also captures and replays successfully;
- GSM8K-20 remains at least 0.80;
- attention is confirmed eager at every layer;
- captured compute segments replay between attention breaks;
- FlashInfer A2A has no workspace overflow or geometry mismatch;
- every capture bucket logs a FlashInfer dispatch capacity at least as large as its physical CP-local row count;
- preliminary c1/c4 TTFT is neutral or improved;
- the exact branch SHA, logs, segment inventory, and comparison JSON are stored in the BCG artifact directory.
