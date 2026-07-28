# GPT-OSS CP+EP Profile, Fusion, Benchmark, and Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove remaining trace-proven CP4+EP4 overhead, demonstrate lower TTFT than TP4 at concurrency 1 and 4 for exact 128K/1K requests, and deliver validated local 256K/20/20-step torch profiles for both modes.

**Architecture:** Use an alternating paired benchmark harness to make TTFT decisions. Profile after BCG and one-launch work, rank bottlenecks by TTFT contribution, and implement only measured fusions. The first concrete fusion candidate replaces zigzag’s split/cat/pad KV gather chain with JIT pack/reorder kernels and precomputed row maps. Continue the measure-test-accept loop until the TTFT goal passes and no remaining low-hanging candidate meets the action threshold.

**Tech Stack:** SGLang online serving benchmark, torch profiler, `llm-torch-profiler-analysis`, SGLang JIT CUDA kernels through TVM-FFI, Python result aggregation, four NVIDIA GB300 GPUs.

## Global Constraints

- Use the exact BF16 checkpoint `/scratch/models/gpt-oss-120b-bf16`.
- Use `random-ids`, exact token lengths, `--tokenize-prompt`, range ratio `1`, temperature `0`, request rate `inf`, one warmup request, and cache flush.
- Official input length is `131072`; official output length is `1024`.
- Concurrency 1 sends one request; concurrency 4 sends four requests.
- Use at least five paired repeats per concurrency and alternate TP-first/CP-first order by pair.
- Acceptance metric is the median across runs of each run’s `mean_ttft_ms`. CP must be strictly lower than TP for both concurrency values. TPOT is reported but is not a pass/fail gate.
- Require CP to win at least three of five same-pair comparisons at each concurrency.
- Use the same model, TRT-LLM MHA backend, context length, capture buckets, random seeds, scheduler limits, and MoE runner for both modes. The TP mode removes only CP/EP/A2A flags.
- Apply these identical resolved-memory/scheduler flags to every matched CP/TP profile and benchmark: `--kv-cache-dtype bf16 --page-size 1 --max-running-requests 4 --mem-fraction-static 0.85`. Record the resolved server values and reject a pair if they differ.
- Preserve every server command, commit SHA, environment value, raw JSONL file, and server log.
- Apply `add-jit-kernel` for any new lightweight CUDA kernel.
- Use `superpowers:systematic-debugging` for crashes, hangs, wrong outputs, or regressions.
- Use `superpowers:verification-before-completion` before the final commit/report.

## Canonical Modes

**TP4:**

```text
--tp 4
--attention-backend trtllm_mha
--moe-runner-backend flashinfer_trtllm_routed
--moe-a2a-backend none
--cuda-graph-backend-prefill breakable
```

**CP4+EP4:**

```text
--tp 4 --ep 4
--enable-prefill-cp --attn-cp-size 4 --cp-strategy zigzag
--attention-backend trtllm_mha
--moe-a2a-backend flashinfer
--moe-runner-backend flashinfer_trtllm_routed
--cuda-graph-backend-prefill breakable
```

**Remote artifact root:**

```text
/scratch/gptoss_bf16_cp4_ep4_opt_20260728/
```

**Final local artifact root:**

```text
/Users/baizhou.zhang/.codex/visualizations/2026/07/28/019fa792-cf6d-75c3-9df1-963782ec22d3/gptoss_bf16_cp4_ep4_final/
```

---

## Task 1: Add a Strict Paired-Result Aggregator

**Files:**

- Create: `benchmark/gpt_oss/compare_bf16_cp_ep_results.py`
- Create: `test/registered/unit/benchmark/test_gptoss_cp_ep_results.py`

**CLI:**

```text
official_results_dir=/scratch/gptoss_bf16_cp4_ep4_opt_20260728/final_benchmark/results
official_summary=/scratch/gptoss_bf16_cp4_ep4_opt_20260728/final_benchmark/summary.json
python3 benchmark/gpt_oss/compare_bf16_cp_ep_results.py \
  --input-dir "$official_results_dir" \
  --metadata-dir /scratch/gptoss_bf16_cp4_ep4_opt_20260728/final_benchmark/run-metadata \
  --output "$official_summary"
```

**Output schema:**

```json
{
  "concurrency_1": {
    "tp_run_mean_ttft_ms": {"min": 0.0, "median": 0.0, "max": 0.0},
    "cp_run_mean_ttft_ms": {"min": 0.0, "median": 0.0, "max": 0.0},
    "tp_request_ttft_ms": {"p50": 0.0, "p99": 0.0},
    "cp_request_ttft_ms": {"p50": 0.0, "p99": 0.0},
    "tp_mean_tpot_ms": 0.0,
    "cp_mean_tpot_ms": 0.0,
    "tp_throughput": {"request_per_s": 0.0, "input_token_per_s": 0.0, "output_token_per_s": 0.0},
    "cp_throughput": {"request_per_s": 0.0, "input_token_per_s": 0.0, "output_token_per_s": 0.0},
    "paired_cp_wins": 0,
    "num_pairs": 5,
    "failures": 0,
    "retries": 0,
    "server_restarts": 0,
    "pass": false
  },
  "concurrency_4": {},
  "overall_pass": false
}
```

- [ ] **Step 1: Write failing parser and acceptance tests**

  Make the new file inherit from `CustomTestCase` and register it literally:

  ```python
  from sglang.test.ci.ci_register import register_cpu_ci
  from sglang.test.test_utils import CustomTestCase

  register_cpu_ci(est_time=5, suite="base-a-test-cpu")
  ```

  End the file with `unittest.main()`.

  Create ten fixture records per concurrency with tags:

  ```text
  pair-01-tp-c1
  pair-01-cp-c1
  ...
  pair-05-tp-c4
  pair-05-cp-c4
  ```

  Test that the aggregator:

  - rejects missing pairs, duplicate tags, non-successful requests, wrong exact input/output lengths, or fewer than five pairs;
  - uses `mean_ttft_ms`, not the within-run `median_ttft_ms`;
  - computes the median across five run means;
  - counts pairwise CP wins;
  - reports min/median/max of run-level mean TTFT, p50/p99 across request-level TTFT samples, mean TPOT, and mean request/input/output throughput for each mode;
  - reads one run-metadata JSON per tag and sums failures, retries, and server restarts;
  - passes only when CP median is strictly lower and paired wins are at least three.

- [ ] **Step 2: Run the test and confirm the module is absent**

  ```bash
  python3 -m pytest -q \
    test/registered/unit/benchmark/test_gptoss_cp_ep_results.py
  ```

  Expected: import or file-not-found failure.

- [ ] **Step 3: Implement the strict aggregator**

  Read every `*.jsonl` file in the directory. Require one JSON object per file and validate:

  ```python
  record["completed"] == concurrency
  record["random_input_len"] == 131072
  record["random_output_len"] == 1024
  record["random_range_ratio"] == 1.0
  len(record["input_lens"]) == concurrency
  set(record["input_lens"]) == {131072}
  set(record["output_lens"]) == {1024}
  not any(record["errors"])
  ```

  Require each tag's metadata record to contain non-negative integer `failure_count`, `retry_count`, and `server_restart_count`, the exact server command, server PID, commit SHA, and seed. Reject a benchmark set with missing metadata or any nonzero failure/retry/restart count; keep those counts in the output so operational instability cannot be hidden.

  The official benchmark must invoke `--output-details` so those arrays are present.

- [ ] **Step 4: Run the aggregator tests**

  ```bash
  python3 -m pytest -q \
    test/registered/unit/benchmark/test_gptoss_cp_ep_results.py
  ```

  Expected: all validation and decision tests pass.

- [ ] **Step 5: Commit the benchmark decision tool**

  ```bash
  git add benchmark/gpt_oss/compare_bf16_cp_ep_results.py \
    test/registered/unit/benchmark/test_gptoss_cp_ep_results.py
  git commit -m "test: codify GPT-OSS CP versus TP TTFT gate"
  ```

---

## Task 2: Collect the First Post-One-Launch Bottleneck Profile

**Files:**

- Record remotely: `baselines/post_one_launch_cp/`
- Record remotely: `baselines/post_one_launch_tp/`
- Record remotely: `baselines/post_one_launch_analysis.md`
- Record remotely: `baselines/post_one_launch_analysis.json`

- [ ] **Step 1: Launch matched CP and TP servers one at a time**

  Common settings:

  ```text
  --context-length 139264
  --chunked-prefill-size 131072
  --max-prefill-tokens 131072
  --cuda-graph-bs-prefill 4096 65536 131072
  ```

  CP environment:

  ```text
  SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK=65536
  ```

  Record `nvidia-smi`, FlashInfer/PyTorch/SGLang versions, branch SHA, server command, server PID, and health response.

- [ ] **Step 2: Capture 20 steps at c1, 128K input, 20 output**

  For each mode:

  ```bash
  mode=cp
  mode_profile_dir=/scratch/gptoss_bf16_cp4_ep4_opt_20260728/baselines/post_one_launch_cp
  python3 -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port 30000 \
    --dataset-name random-ids \
    --num-prompts 1 --max-concurrency 1 --request-rate inf \
    --random-input-len 131072 --random-output-len 20 \
    --random-range-ratio 1 --tokenize-prompt \
    --temperature 0 --warmup-requests 1 --flush-cache \
    --output-details \
    --profile --profile-steps 20 --profile-activities CPU GPU \
    --profile-output-dir "$mode_profile_dir" \
    --profile-prefix "${mode}-post-one-launch"
  ```

  Repeat with `mode=tp` and the corresponding `post_one_launch_tp` directory. Expected: four rank trace files per mode, with at least 20 profiled forward steps and no profiler error.

- [ ] **Step 3: Analyze both profiles**

  Use `llm-torch-profiler-analysis` and produce three tables per mode:

  1. kernel time, launch count, average duration, and share;
  2. CPU/GPU overlap opportunities and exposed gaps;
  3. repeated producer-consumer patterns suitable for fusion.

  Add a CP-vs-TP comparison that explicitly reports:

  - TRT-LLM context-attention count and total time;
  - BCG segment replay time and eager attention gaps;
  - FlashInfer dispatch/combine time;
  - KV all-gather time;
  - `aten::cat`, `aten::split`, `aten::contiguous`, pad, and index/select time;
  - NCCL wait/spin time;
  - CPU self time between layer events.

- [ ] **Step 4: Rank actionable candidates**

  A candidate is actionable if either:

  - its chain contributes at least 1% of total CP GPU time; or
  - exposed CPU gaps around it contribute at least 1% of CP TTFT and the calls repeat per layer.

  Rank by `recoverable_ms = exposed_ms * realistic_recovery_fraction`, not by launch count alone.

---

## Task 3: Fuse Zigzag KV Pack, Pad, Reorder, and Split

This is the first concrete candidate. Execute it only if Task 2 attributes at least 1% of total GPU time or a repeated material CPU launch gap to the current KV gather/reorder chain. If it is below both thresholds, record `"status": "rejected_before_implementation"` with measured time and proceed to Task 5.

**Files:**

- Create: `python/sglang/jit_kernel/csrc/cp/zigzag_reorder.cuh`
- Create: `python/sglang/jit_kernel/zigzag_reorder.py`
- Create: `python/sglang/jit_kernel/tests/test_zigzag_reorder.py`
- Create: `python/sglang/jit_kernel/benchmark/bench_zigzag_reorder.py`
- Modify: `python/sglang/srt/layers/cp/zigzag.py`
- Extend: `test/registered/cp/test_cp_strategy_unit.py`

**Python interfaces:**

```python
def zigzag_pack_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    padded_rows: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Write [K | V] and zero row padding in one launch."""

def zigzag_reorder_rows(
    gathered: torch.Tensor,
    source_rows: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gather logical rows from rank-padded collective output in one launch."""

def zigzag_reorder_split_kv(
    gathered: torch.Tensor,
    source_rows: torch.Tensor,
    key_width: int,
    key_out: Optional[torch.Tensor] = None,
    value_out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorder and split K/V into contiguous outputs in one launch."""
```

**Metadata field:**

```python
gather_source_rows_tensor: Optional[torch.Tensor] = None
```

- [ ] **Step 1: Add failing CPU reference-map tests**

  Build `gather_source_rows` by simulating the existing implementation:

  1. for each rank, take `range(rank * max_len, rank * max_len + logical_rank_len)`;
  2. split that trimmed rank-major list by `reverse_split_len`;
  3. concatenate chunks in `cp_reverse_index` order.

  For CP sizes 2 and 4, batch sizes 1 and 4, unequal lengths, and padded ranks, assert indexing a synthetic gathered tensor with this map exactly matches the current `_all_gather_reorganized` plus split/cat reference.

- [ ] **Step 2: Add failing JIT correctness tests**

  Register the test file with literal:

  ```python
  register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")
  register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-b200")
  ```

  Cover BF16 and FP16, row widths `64`, `128`, `256`, and GPT-OSS KV widths. Include odd logical row counts and non-empty padding. Compare:

  - `zigzag_pack_kv` to `torch.cat` plus `F.pad`;
  - `zigzag_reorder_rows` to `torch.index_select`;
  - `zigzag_reorder_split_kv` to reference reorder, split, and contiguous copies.

- [ ] **Step 3: Run the tests and confirm kernels are absent**

  On `baizhou-dev`:

  ```bash
  python3 -m pytest -q \
    python/sglang/jit_kernel/tests/test_zigzag_reorder.py \
    test/registered/cp/test_cp_strategy_unit.py
  ```

  Expected: import or missing-symbol failure.

- [ ] **Step 4: Implement the JIT CUDA launchers**

  Follow `add-jit-kernel`:

  - validate every tensor with `TensorMatcher`;
  - use `SymbolicSize`, `SymbolicDType`, and `SymbolicDevice`;
  - use `AlignedVector` for row-width copies when alignment permits;
  - use `LaunchKernel` and runtime errors, not raw assertions;
  - add a trailing purpose comment to every `#include <sgl_kernel/...>`;
  - keep row counts and split widths runtime values;
  - use PDL only after the benchmark shows a benefit on SM100.

  `zigzag_reorder_split_kv` must write both outputs from the same source-row read loop so it replaces reorder, split, and two contiguous copies.

- [ ] **Step 5: Add the thin Python wrapper**

  Use `cache_once`, `load_jit`, and `make_cpp_args`. Validate CUDA device and supported dtype in Python, allow caller-provided outputs, and allocate only when outputs are absent.

- [ ] **Step 6: Integrate the source-row map into zigzag metadata**

  Build `gather_source_rows_tensor` once in `build_metadata` using the tested CPU reference algorithm. Use `torch.int32` unless the largest flat row index exceeds `2**31 - 1`.

- [ ] **Step 7: Replace the KV gather chain**

  In `materialize_full_kv`:

  1. call `zigzag_pack_kv(k, v, max_len)`;
  2. all-gather the packed buffer without the intermediate trim/cat;
  3. call `zigzag_reorder_split_kv` with `gather_source_rows_tensor`;
  4. pass the two contiguous outputs directly to `set_kv_buffer`.

  In `gather_hidden_states`, use `zigzag_reorder_rows` on the raw rank-padded all-gather result.

  Retain the old PyTorch reference behind a private test helper, not a runtime fallback.

- [ ] **Step 8: Run correctness tests like CI**

  ```bash
  cd test
  python3 run_suite.py --hw cuda --suite base-b-kernel-unit-1-gpu-large
  python3 run_suite.py --hw cuda --suite base-b-kernel-unit-1-gpu-b200
  ```

  Also run:

  ```bash
  python3 -m pytest -q test/registered/cp/test_cp_strategy_unit.py
  ```

  Expected: all tests pass.

- [ ] **Step 9: Add and run a marker benchmark**

  Register:

  ```python
  register_cuda_ci(
      est_time=8, suite="base-b-kernel-benchmark-1-gpu-large"
  )
  ```

  Benchmark `torch` versus `jit` for CP4 logical per-rank rows corresponding to global lengths `4096`, `131072`, and `262144`, using GPT-OSS BF16 KV widths. Report microseconds and effective GB/s. Use `marker.do_bench`, `memory_args`, and correct `graph_clone_args`.

  Acceptance:

  - fused pack is at least 1.10× faster than cat+pad;
  - fused reorder/split is at least 1.20× faster than split/cat/contiguous;
  - total fused chain saves at least 10% of the trace-measured candidate time.

- [ ] **Step 10: Run e2e parity and short TTFT repeats**

  Require deterministic token parity with the one-launch parent at 4096/64 and 128K/20, GSM8K-20 at least 0.80, and three c1 128K/20 repeats with no TTFT regression.

- [ ] **Step 11: Accept or reject the candidate**

  Accept only if the microbenchmark and e2e gates pass and the new profile reduces the target chain by at least 10% without shifting equal time into synchronization.

  If rejected, remove the uncommitted source changes with `apply_patch`, retain the benchmark/trace decision under the artifact root, and do not commit dead runtime code.

- [ ] **Step 12: Commit an accepted fusion**

  ```bash
  git add \
    python/sglang/jit_kernel/csrc/cp/zigzag_reorder.cuh \
    python/sglang/jit_kernel/zigzag_reorder.py \
    python/sglang/jit_kernel/tests/test_zigzag_reorder.py \
    python/sglang/jit_kernel/benchmark/bench_zigzag_reorder.py \
    python/sglang/srt/layers/cp/zigzag.py \
    test/registered/cp/test_cp_strategy_unit.py
  git commit -m "perf: fuse zigzag CP KV reordering"
  ```

---

## Task 4: Fuse Model-Boundary Zigzag Shard Operations if Trace-Visible

Execute only if embedding/position sharding plus final hidden gather still contributes at least 1% of TTFT after Task 3.

**Files:**

- Extend: `python/sglang/jit_kernel/csrc/cp/zigzag_reorder.cuh`
- Extend: `python/sglang/jit_kernel/zigzag_reorder.py`
- Extend: `python/sglang/jit_kernel/tests/test_zigzag_reorder.py`
- Extend: `python/sglang/jit_kernel/benchmark/bench_zigzag_reorder.py`
- Modify: `python/sglang/srt/layers/cp/zigzag.py`

**Interface:**

```python
def zigzag_shard_rows(
    source: torch.Tensor,
    source_rows: torch.Tensor,
    padded_rows: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gather selected source rows and zero the physical tail in one launch."""
```

- [ ] **Step 1: Add source-index metadata tests**

  Materialize the exact row indices represented by `split_list` plus `zigzag_index`. Assert the reference index-select output matches current `torch.split` plus `torch.cat` for hidden states and positions.

- [ ] **Step 2: Add JIT tests for BF16 two-dimensional rows and int64 positions**

  Cover CP4 ranks 0–3, batch sizes 1 and 4, unequal sequence lengths, and non-empty physical padding.

- [ ] **Step 3: Implement and benchmark the fused shard**

  Use one launch to gather logical rows and zero the tail. Accept only if it saves at least 10% of the measured boundary-shard chain and at least 1% of e2e TTFT in three short repeats.

- [ ] **Step 4: Integrate and run the full correctness gate**

  Run JIT suites, CP strategy tests, deterministic parity, and GSM8K-20.

- [ ] **Step 5: Commit only if accepted**

  ```bash
  git commit -m "perf: fuse zigzag CP input sharding"
  ```

  If below threshold, record a measured rejection and leave runtime code unchanged.

---

## Task 5: Run the Trace-Driven Optimization Loop

**Files:**

- Record remotely per iteration using a numeric, descriptive directory such as `fusion_iterations/01-zigzag-kv-reorder/`; increment the number and use the selected candidate slug for later iterations.
- Maintain remotely: `fusion_iterations/decision-log.jsonl`

- [ ] **Step 1: Re-profile after every accepted structural change**

  Use CP4+EP4, c1, 128K input, 20 output, 20 steps, identical server flags and profiler settings.

- [ ] **Step 2: Select exactly one next candidate**

  Choose the largest recoverable contributor in this order:

  1. exposed CPU gaps between BCG segments and eager attention;
  2. remaining zigzag pack/reorder/pad/copy chains;
  3. FlashInfer A2A dispatch/combine setup or synchronization;
  4. TRT-LLM context-attention launch/grid inefficiency;
  5. repeated routing/top-k producer-consumer chains.

  Record:

  ```json
  {
    "iteration": 1,
    "candidate": "name",
    "baseline_ms": 0.0,
    "share_of_ttft_pct": 0.0,
    "expected_recoverable_ms": 0.0,
    "planned_change": "concrete producer-consumer replacement",
    "acceptance_threshold": "numeric threshold",
    "status": "selected"
  }
  ```

- [ ] **Step 3: Apply test-first implementation and a focused benchmark**

  Every candidate must have:

  - a reference-correctness test;
  - a failure observed before implementation;
  - a focused microbenchmark or trace counter;
  - deterministic e2e parity;
  - a short TTFT check;
  - one commit only after acceptance.

- [ ] **Step 4: Reject regressions explicitly**

  Reject a candidate if it:

  - saves less than 10% of its own chain;
  - regresses c1 or c4 preliminary TTFT by more than 1%;
  - increases memory enough to prevent 256K profiling;
  - adds a synchronization point;
  - changes generated tokens or accuracy.

- [ ] **Step 5: Continue until both completion conditions hold**

  The loop ends only when:

  1. the official CP TTFT gate in Task 6 passes; and
  2. a fresh trace has no low-risk, low-effort candidate above 1% of total GPU time or a material repeated CPU gap, and two consecutive measured candidates are rejected as below threshold.

  TRT-LLM attention math, FlashInfer routed GEMMs, and unavoidable A2A/NCCL transfer time are core costs, not low-hanging fruit, only after their launch/setup overhead has been removed.

  If the official gate still fails when low-hanging candidates are exhausted, do not stop. Select the largest structural gap—attention efficiency, KV communication volume, or A2A serialization—and write a new design delta before implementing it.

---

## Task 6: Run the Official 128K/1K c1 and c4 Benchmark

**Files:**

- Record remotely: `final_benchmark/results/*.jsonl`
- Record remotely: `final_benchmark/server-logs/`
- Record remotely: `final_benchmark/commands/`
- Record remotely: `final_benchmark/run-metadata/`
- Record remotely: `final_benchmark/summary.json`
- Record remotely: `final_benchmark/summary.md`

- [ ] **Step 1: Fix common server settings**

  Use:

  ```text
  --model-path /scratch/models/gpt-oss-120b-bf16
  --attention-backend trtllm_mha
  --moe-runner-backend flashinfer_trtllm_routed
  --context-length 139264
  --chunked-prefill-size 131072
  --max-prefill-tokens 131072
  --cuda-graph-backend-prefill breakable
  --cuda-graph-bs-prefill 4096 65536 131072
  ```

  CP additionally uses:

  ```text
  SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK=65536
  ```

- [ ] **Step 2: Execute five alternating pairs**

  Pair order:

  ```text
  pair 1: TP then CP
  pair 2: CP then TP
  pair 3: TP then CP
  pair 4: CP then TP
  pair 5: TP then CP
  ```

  For each server launch, after health and one warmup, run both c1 and c4. Stop only the saved PID before launching the other mode.

  Before every benchmark command, assert its result and metadata paths do not exist; create a new immutable attempt directory if rerunning a failed official set. For each tag, write a matching metadata JSON containing the exact server command, PID, branch SHA, seed, start/end timestamps, `failure_count`, `retry_count`, and `server_restart_count`.

- [ ] **Step 3: Use the exact c1 command**

  ```bash
  pair_number=01
  mode=tp
  pair_seed=20260729
  result_path=/scratch/gptoss_bf16_cp4_ep4_opt_20260728/final_benchmark/results/pair-01-tp-c1.jsonl
  python3 -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port 30000 \
    --dataset-name random-ids \
    --num-prompts 1 --max-concurrency 1 --request-rate inf \
    --random-input-len 131072 --random-output-len 1024 \
    --random-range-ratio 1 --tokenize-prompt \
    --temperature 0 --seed "$pair_seed" \
    --warmup-requests 1 --flush-cache --output-details \
    --tag "pair-$pair_number-$mode-c1" \
    --output-file "$result_path"
  ```

  For each subsequent run, set `pair_number` to `02` through `05`, set `mode` to `tp` or `cp`, set `pair_seed` to `20260728 + pair_number`, and set `result_path` to the matching unique filename.

- [ ] **Step 4: Use the exact c4 command**

  ```bash
  pair_number=01
  mode=tp
  pair_seed=20260729
  result_path=/scratch/gptoss_bf16_cp4_ep4_opt_20260728/final_benchmark/results/pair-01-tp-c4.jsonl
  python3 -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port 30000 \
    --dataset-name random-ids \
    --num-prompts 4 --max-concurrency 4 --request-rate inf \
    --random-input-len 131072 --random-output-len 1024 \
    --random-range-ratio 1 --tokenize-prompt \
    --temperature 0 --seed "$pair_seed" \
    --warmup-requests 1 --flush-cache --output-details \
    --tag "pair-$pair_number-$mode-c4" \
    --output-file "$result_path"
  ```

  Use the same seed for TP and CP in a pair. Increment the seed between pairs.

- [ ] **Step 5: Aggregate and enforce the gate**

  ```bash
  python3 benchmark/gpt_oss/compare_bf16_cp_ep_results.py \
    --input-dir /scratch/gptoss_bf16_cp4_ep4_opt_20260728/final_benchmark/results \
    --metadata-dir /scratch/gptoss_bf16_cp4_ep4_opt_20260728/final_benchmark/run-metadata \
    --output /scratch/gptoss_bf16_cp4_ep4_opt_20260728/final_benchmark/summary.json
  ```

  Require `overall_pass == true`.

- [ ] **Step 6: Report the complete distribution and operational record without moving the goalposts**

  `summary.md` must show all ten raw run means per concurrency, min/median/max run-mean TTFT, request-level p50/p99 TTFT, paired directions, percentage improvement, mean TPOT, request/input/output throughput, and failure/retry/restart counts. Do not substitute a favorable percentile or throughput metric for the specified median run-mean TTFT gate.

- [ ] **Step 7: If either concurrency fails, return to Task 5**

  Preserve the failed official set as an immutable iteration artifact. Do not overwrite it on the next attempt.

---

## Task 7: Collect the Final 256K/20/20-Step Traces

**Files:**

- Record remotely: `final_profiles/cp4_ep4/`
- Record remotely: `final_profiles/tp4/`
- Download locally to the final artifact root.

- [ ] **Step 1: Launch final-profile servers with the larger context**

  Common:

  ```text
  --context-length 270336
  --chunked-prefill-size 262144
  --max-prefill-tokens 262144
  --cuda-graph-bs-prefill 4096 131072 262144
  ```

  CP environment:

  ```text
  SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK=65536
  ```

  The CP local share of one 262144-token request is about 65536 rows; if physical alignment raises it, set the environment to the exact logged captured local row count before launch.

- [ ] **Step 2: Run a pre-profile 256K/20 correctness request**

  Require one successful request, exact input `262144`, output `20`, and no worker error.

- [ ] **Step 3: Capture CP4+EP4**

  ```bash
  python3 -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port 30000 \
    --dataset-name random-ids \
    --num-prompts 1 --max-concurrency 1 --request-rate inf \
    --random-input-len 262144 --random-output-len 20 \
    --random-range-ratio 1 --tokenize-prompt \
    --temperature 0 --seed 20260728 \
    --warmup-requests 1 --flush-cache --output-details \
    --profile --profile-steps 20 --profile-activities CPU GPU \
    --profile-output-dir /scratch/gptoss_bf16_cp4_ep4_opt_20260728/final_profiles/cp4_ep4 \
    --profile-prefix cp4-ep4-256k
  ```

  Expected: exactly four rank `.trace.json.gz` files plus server args/result metadata.

- [ ] **Step 4: Capture TP4**

  Repeat with the TP4 server and:

  ```text
  --profile-output-dir /scratch/gptoss_bf16_cp4_ep4_opt_20260728/final_profiles/tp4
  --profile-prefix tp4-256k
  ```

  Expected: exactly four rank trace files.

- [ ] **Step 5: Validate remote trace integrity**

  For every trace:

  ```bash
  find /scratch/gptoss_bf16_cp4_ep4_opt_20260728/final_profiles \
    -type f -name '*.trace.json.gz' -print0 |
    while IFS= read -r -d '' trace_file; do
      gzip -t "$trace_file"
    done
  ```

  Load every discovered file as JSON and require a non-empty `traceEvents` array. Record SHA256 checksums, sizes, filenames, server command, branch SHA, benchmark JSON, and profile step count in `manifest.json`.

- [ ] **Step 6: Download traces locally**

  Create:

  ```text
  /Users/baizhou.zhang/.codex/visualizations/2026/07/28/019fa792-cf6d-75c3-9df1-963782ec22d3/gptoss_bf16_cp4_ep4_final/cp4_ep4/
  /Users/baizhou.zhang/.codex/visualizations/2026/07/28/019fa792-cf6d-75c3-9df1-963782ec22d3/gptoss_bf16_cp4_ep4_final/tp4/
  ```

  Use `rsync -av --checksum` from `baizhou-dev`. Re-run local `gzip -t` and compare SHA256 checksums to the remote manifest.

- [ ] **Step 7: Merge rank traces locally**

  Use the GB300 journal-prescribed utilities from [`fzyzcjy/torch_utils`](https://github.com/fzyzcjy/torch_utils). Clone the repository below the final local artifact root as `tools/torch_utils`, check out the inspected commit `53739f1d10165d6441bb1eeaec878505c6295f86`, record that SHA in the local and remote manifests, and install its runtime dependencies:

  ```bash
  python3 -m pip install orjson typer
  ```

  `bench_serving` creates one timestamp child below each `--profile-output-dir`. For each mode, require exactly one such child containing four rank traces, set `mode_dir` to that timestamp directory, and derive `profile_id` as the full filename prefix before `-TP-0`. Then run:

  ```bash
  python3 tools/torch_utils/src/torch_profile_trace_merger/sglang_profiler_trace_merger.py \
    --profile-id "$profile_id" \
    --start-time-ms 0 \
    --end-time-ms 999999999 \
    --dir-data "$mode_dir"

  merged_filename="merged-${profile_id}.trace.json.gz"
  python3 tools/torch_utils/src/convert_to_perfetto_compatible/convert_to_perfetto_compatible.py \
    "$merged_filename" \
    --dir-data "$mode_dir"
  ```

  Preserve:

  - all four original rank traces;
  - `merged-${profile_id}.trace.json.gz`;
  - `perfetto-compatible-merged-${profile_id}.trace.json.gz`.

  Run `gzip -t` on both derived files, load both as JSON, require non-empty `traceEvents`, and record SHA256 checksums. The merge utility must discover exactly TP ranks 0–3; fail if a rank is missing or duplicated.

- [ ] **Step 8: Produce the final three-table trace report**

  Use `llm-torch-profiler-analysis` on both local profile directories. Save:

  ```text
  analysis.md
  analysis.json
  ```

  Include kernel, overlap-opportunity, and fusion-pattern tables; compare CP vs TP; state whether any remaining CP-specific non-core item exceeds the 1% GPU-time or material CPU-gap threshold.

---

## Task 8: Final Verification and Branch Handoff

**Files:**

- Verify every touched source/test/benchmark file.
- Record final: `final_verification.txt`
- Record remotely at the artifact root: `manifest.txt`
- Record remotely at the artifact root: `final_report.md`

- [ ] **Step 1: Run focused local tests**

  ```bash
  python3 -m pytest -q \
    test/registered/unit/layers/test_flashinfer_trtllm_bf16_padding.py \
    test/registered/unit/models/test_gpt_oss_ep_weight_loading.py \
    test/registered/unit/server_args/test_flashinfer_a2a_cp.py \
    test/registered/unit/model_executor/runner/test_prefill_cuda_graph_cp.py \
    test/registered/cp/test_cp_strategy_unit.py \
    test/registered/unit/layers/attention/test_trtllm_mha_zigzag.py \
    test/registered/unit/benchmark/test_gptoss_cp_ep_results.py
  ```

- [ ] **Step 2: Run JIT CUDA suites if a fusion landed**

  ```bash
  cd test
  python3 run_suite.py --hw cuda --suite base-b-kernel-unit-1-gpu-large
  python3 run_suite.py --hw cuda --suite base-b-kernel-unit-1-gpu-b200
  python3 run_suite.py --hw cuda --suite base-b-kernel-benchmark-1-gpu-large
  ```

- [ ] **Step 3: Run formatting and diff checks**

  ```bash
  pre-commit run --all-files
  git diff --check origin/main...HEAD
  git status --short
  ```

  Preserve unrelated user changes; only task files may be staged.

- [ ] **Step 4: Verify completion evidence**

  Require all of:

  - support-only draft PR URL;
  - exact final optimization branch SHA;
  - official `overall_pass == true`;
  - CP median TTFT lower at c1 and c4;
  - correctness and accuracy gates pass;
  - no trace-proven low-hanging item remains;
  - eight rank traces and two merged traces exist locally;
  - local checksums match the remote manifest;
  - final analysis files exist.

- [ ] **Step 5: Write the root manifest and final report**

  Write `/scratch/gptoss_bf16_cp4_ep4_opt_20260728/manifest.txt` with:

  - main base SHA, support PR URL/SHA, and final optimization SHA;
  - model path and checkpoint identity;
  - software/GPU versions and every environment override;
  - exact TP and CP server commands;
  - pointers and checksums for support, baseline, BCG, fused-attention, fusion-iteration, final-benchmark, and final-profile artifacts;
  - the pinned `torch_utils` SHA;
  - local download root and checksum verification result.

  Write `final_report.md` with correctness outcomes, accepted/rejected optimizations, official c1/c4 tables, the pass/fail decision, final trace links, three-table profiler conclusions, and the no-low-hanging-fruit decision. If a remaining bottleneck is externally blocked, name the exact kernel or dependency, its measured share, a minimal reproducer, and why an in-tree low-risk fix is unavailable.

  Copy `manifest.txt`, `final_report.md`, the final benchmark summary, and the final analysis files into the final local artifact root. Verify their checksums after transfer.

- [ ] **Step 6: Commit final harness/report references**

  Commit only source-controlled harness and documentation, not multi-gigabyte trace files:

  ```bash
  git commit -m "perf: optimize GPT-OSS CP4 EP4 prefill"
  ```

- [ ] **Step 7: Push the optimization branch**

  ```bash
  git push origin codex/gptoss-bf16-cp4-ep4-opt
  ```

  Do not open a second PR unless explicitly requested; the user required a PR only for missing backend support.

---

## Final Completion Gate

The work is complete only when:

- the requested BF16 CP4+EP4 FlashInfer combination is supported and its support-only PR is submitted;
- BCG replays the CP-local transformer body and skips attention;
- zigzag TRT-LLM attention launches once per layer;
- accepted kernel fusions have correctness, microbenchmark, and e2e evidence;
- CP4+EP4 median TTFT is strictly below TP4 at c1 and c4 for exact 128K/1K;
- the stop criterion confirms no obvious CP-specific low-hanging bottleneck remains;
- CP4+EP4 and TP4 256K/20/20-step profiles are validated, merged, analyzed, and downloaded to the specified local artifact root.
