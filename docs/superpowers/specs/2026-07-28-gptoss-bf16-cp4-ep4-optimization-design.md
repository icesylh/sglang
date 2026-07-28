# GPT-OSS BF16 CP4+EP4 Optimization Design

## Objective

Optimize SGLang serving for `lmsys/gpt-oss-120b-bf16` on the four-GPU
`baizhou-dev` GB300 devbox. The optimized CP4+EP4 configuration must have lower
TTFT than the TP4 reference at both concurrency 1 and concurrency 4 for the
fixed 128K-input/1K-output benchmark. TPOT is recorded but is not a primary
acceptance metric.

The CP4+EP4 configuration must use:

```text
--tp 4
--ep 4
--enable-prefill-cp
--attn-cp-size 4
--cp-strategy zigzag
--moe-a2a-backend flashinfer
--moe-runner-backend flashinfer_trtllm_routed
```

After optimization, collect torch-profiler traces for CP4+EP4 and TP4 at
concurrency 1 with a 256K-token input, 20 output tokens, and 20 active profile
steps, and download the results locally.

## Starting State and Evidence

The implementation branch is
`codex/gptoss-bf16-cp4-ep4-opt`, based on
`origin/main@fc8b328f5cb526fd1fa2c26cb3dafe3761af8d5d`.

The main branch already contains:

- zigzag CP for GPT-OSS with TRTLLM-MHA;
- FlashInfer MoE A2A with the `flashinfer_trtllm_routed` runner for selected
  quantized models;
- the breakable CUDA graph infrastructure introduced by SGLang PR #22218;
- a blanket rule that disables breakable prefill CUDA graphs when
  `attn_cp_size > 1`;
- a second blanket rule that disables breakable prefill CUDA graphs for every
  non-`none` MoE A2A backend.

The main branch does not yet prove that BF16 GPT-OSS can use FlashInfer A2A,
the routed TRTLLM BF16 MoE kernel, CP4, and EP4 together. A task-scoped dirty
checkout on `baizhou-dev` contains an uncommitted prototype that:

- narrows global expert tensors to the local EP rank;
- pads the 2880-wide GPT-OSS hidden dimension for the SM103 TRTLLM BF16 MoE
  kernel;
- preserves GPT-OSS expert biases and clamped gated activation semantics;
- permits FlashInfer A2A when the full TP group is also the prefill CP group.

That prototype has passed only a TP1 BF16 bring-up and a small GSM8K run. It is
input evidence, not accepted implementation.

An older MXFP4 256K/1K experiment provides a pre-optimization reference:

| Configuration | c1 mean TTFT | c4 mean TTFT |
| --- | ---: | ---: |
| TP4 | 3027 ms | 7597 ms |
| CP4+EP4 with DeepEP/DeepGEMM | 4907 ms | 11487 ms |

The old rank-0 traces show two concrete problems:

- CP launches 576 TRTLLM context-attention kernels versus 285 for TP during
  the captured workload, because zigzag dispatches the early and late query
  segments separately.
- DeepEP dispatch/combine/notification and CP all-gather add substantial
  communication time. The required FlashInfer A2A configuration must replace
  that baseline before drawing conclusions about the final bottleneck.

The old experiment used MXFP4 and different MoE backends. It is useful only
for prioritization and cannot satisfy any BF16 acceptance criterion.

## Scope

The work is one performance program with four ordered deliverables:

1. BF16 GPT-OSS support for the required FlashInfer A2A and routed TRTLLM MoE
   combination under CP4+EP4.
2. Breakable prefill CUDA graph support for that CP4+EP4 combination.
3. One-launch zigzag TRTLLM context attention.
4. Profile-driven fusion or launch-overhead reductions until no clear
   low-risk, low-effort bottleneck remains.

The work does not change model quality, sampling semantics, benchmark lengths,
or the TP4 reference after seeing results. It does not optimize MXFP4, DeepEP,
or DeepGEMM except where shared code must remain correct.

## Repository and Devbox Workflow

The local Codex worktree is the source of truth for commits. GPU execution
happens in `/sgl-workspace/sglang` on `baizhou-dev`, using the local checkpoint
`/scratch/models/gpt-oss-120b-bf16`.

Before altering the devbox checkout:

1. Save its current `git diff`, untracked task tests, branch/ref information,
   launch command, and server log under
   `/scratch/gptoss_bf16_cp4_ep4_opt_20260728/preexisting/`.
2. Terminate only the task-scoped TP1 server identified by its recorded PID.
3. Do not terminate unrelated GPU or CPU processes.
4. Update the devbox checkout to the tested local commit and reinstall the
   editable SGLang package plus current project dependencies before the first
   GPU run on each new base.

No model checkpoint files are modified in place. Secrets and token values are
not written to artifacts.

## Deliverable 1: BF16 FlashInfer A2A and Routed MoE Support

### Weight loading

GPT-OSS BF16 expert checkpoint tensors are global across 128 experts. EP4
loads exactly the contiguous 32-expert range owned by each EP rank. Already
local tensors remain unchanged. Any other expert-axis size fails with a clear
error rather than being silently sliced.

Bias tensors follow the same expert ownership rule. Bias-specific loader paths
must not reuse a stale projection shard dimension.

### TRTLLM BF16 MoE shape and semantics

GPT-OSS has hidden size 2880, while the SM103 TRTLLM BF16 MoE kernel requires
the hidden dimension to be aligned to 128. The runner-facing representation
pads hidden inputs and expert weights to 2944, then slices the final output
back to 2880.

The padded representation preserves:

- the per-expert gate/up projection biases;
- the per-expert down-projection biases;
- GPT-OSS `swiglu_limit`;
- GPT-OSS gate alpha and beta semantics;
- `[up, gate]` weight ordering required by the FlashInfer runner.

Padding constants and any folded bias channels are derived once at weight
finalization. The serving path must not allocate a token-sized bias tensor per
layer.

### Parallelism validation

FlashInfer MoE A2A is accepted when either:

- DP attention owns the full TP group, matching the existing supported path;
  or
- prefill CP is enabled and `attn_cp_size == tp_size`.

Partial CP is rejected for this path. The existing DP-attention behavior and
error messages remain covered by tests.

### Support PR

Once unit tests, BF16 TP1 correctness, and real CP4+EP4 generation pass, create
a support-only branch pointer from the support commit and open a GitHub pull
request. The ongoing optimization branch retains the support commit and
continues independently, so graph and performance experiments do not expand
the support PR.

The PR description records the exact checkpoint path alias, hardware, package
versions, launch command, unit tests, deterministic generation result, and
accuracy result.

## Deliverable 2: Breakable CUDA Graph for CP4+EP4

Breakable graph capture keeps attention eager and captures the surrounding
transformer work in stable CUDA graph segments. Support is enabled narrowly
for the proven GPT-OSS MHA CP and FlashInfer A2A path; incompatible MLA, DCP,
multimodal, and unsupported A2A cases retain their current guards.

### CP-aware capture contract

For each prefill token bucket, capture and replay agree on:

- global logical token count;
- rank-local physical token count after zigzag splitting and padding;
- request count and per-request extend lengths;
- CP split, reverse, and per-rank length metadata;
- stable input, position, KV-location, and graph-output addresses.

Variable serving metadata is refreshed before replay. Captured graph segments
do not retain Python objects or tensor views whose storage can change between
requests.

The CP output gather/reorder either runs at an eager break or writes into a
stable global-sized output buffer. It cannot infer global output rows from a
rank-local tensor shape.

### FlashInfer A2A capture contract

The capture bucket cannot exceed the configured FlashInfer dispatch capacity.
The maximum dispatcher tokens per rank is derived from the largest enabled
prefill bucket and EP size, validated before graph capture, and recorded in the
server manifest.

If an A2A operation cannot be captured safely, it becomes an explicit eager
break while adjacent compute remains captured. The implementation must not
fall back to disabling all prefill graphs.

### BCG validation

Validation covers:

- capture and replay at the smallest, middle, and largest enabled prefill
  buckets;
- two consecutive requests with different real token counts in the same
  bucket;
- CP4+EP4 deterministic output agreement between eager and breakable modes;
- no stale metadata, output-shape corruption, CUDA capture error, or dispatcher
  capacity failure;
- profiler evidence that non-attention CPU launch gaps materially shrink.

## Deliverable 3: One-Launch Zigzag TRTLLM Attention

The current zigzag layout stores all early query segments followed by all late
query segments. The fused dispatch treats them as a single logical batch of
`2 * request_count` rows:

```text
[early request 0 ... early request N-1,
 late request 0 ... late request N-1]
```

For TRTLLM-MHA, the CP strategy constructs one combined metadata view:

- concatenated query sequence lengths and one cumulative query-length tensor;
- concatenated early/late KV lengths and one cumulative KV-length tensor;
- page-table rows repeated in the same early/late order;
- `batch_size = 2 * request_count`;
- `max_q_len = max(max_early_q_len, max_late_q_len)`.

`TRTLLMHAAttnBackend` invokes
`trtllm_batch_context_with_kv_cache` exactly once per layer for a zigzag CP
prefill. The output already matches the original early/late query layout, so
only physical padding is appended.

The generic CP strategy interface retains the existing two-call fallback for
attention backends that cannot represent the combined metadata. FlashAttention
behavior is not changed by the TRTLLM-specific optimization.

Tests verify:

- one callback invocation for TRTLLM zigzag;
- exact combined cumulative lengths, cache lengths, repeated page-table rows,
  batch size, and maximum query length;
- multiple requests with unequal lengths and nonzero prefixes;
- physical padding;
- eager-versus-fused tensor agreement using a reference attention
  implementation;
- real GPT-OSS CP4+EP4 output agreement and accuracy.

The final profiler must show approximately one TRTLLM context-attention launch
per participating layer and profile step, rather than the previous two.

## Deliverable 4: Profile-Driven Kernel and Launch Fusion

After Deliverables 1-3, collect a fresh BF16 mapping trace with readable Python
sites and a formal optimized trace. Analyze kernel share, CPU/GPU gaps,
collective overlap, and repeated producer-consumer sequences.

Candidates are implemented only when the formal trace shows one of:

- at least 1% of total GPU time;
- a repeated small-kernel sequence that creates a material CPU launch gap;
- avoidable synchronization on the critical TTFT path.

Expected zigzag candidates include:

- split/cat/pad during hidden-state and position sharding;
- all-gather postprocessing plus reverse-index reordering;
- query-result concatenation and physical padding;
- repeated device tensor construction for stable CP metadata.

The preferred form is one focused Triton or CUDA kernel per proven
producer-consumer chain. A fusion is rejected if it increases bytes moved,
requires a synchronization, weakens unequal-length request support, or only
improves a microbenchmark without improving the fixed serving workload.

After every accepted fusion, rerun correctness checks and both c1/c4
benchmarks. Revert or leave uncommitted any change that has no stable TTFT
benefit.

## Benchmark Contract

### Shared settings

Both configurations use:

- checkpoint and tokenizer:
  `/scratch/models/gpt-oss-120b-bf16`;
- four GB300 GPUs on the same devbox;
- the same tested git commit and installed dependency versions;
- TRTLLM-MHA attention;
- BF16 weights and the same KV-cache dtype;
- identical context length, page size, chunked-prefill size, maximum prefill
  tokens, maximum running requests, memory fraction, sampling settings, and
  random seeds;
- `random-ids` with tokenized prompts and
  `random_range_ratio = 1.0`;
- input length 131072 and output length 1024;
- temperature 0 and request rate infinity;
- one warmup request before each measured run;
- radix cache flush before every measured run.

The server context length is at least 139264 and uses the existing explicit
long-context override. This provides headroom above input plus output while
keeping the workload lengths fixed.

TP4 uses `--tp 4` without CP or EP. CP4+EP4 uses the exact required flags in
the Objective section. Backend-specific capacity settings are allowed only
when required for correctness and are recorded.

### Repetition and aggregation

For concurrency 1, each measured run sends one request. For concurrency 4,
each measured run sends four simultaneous requests. Each configuration and
concurrency pair is run at least five times with paired ordering alternated to
reduce thermal and temporal bias.

The comparison reports every run plus:

- median of per-run mean TTFT;
- minimum, maximum, and median TTFT;
- p50 and p99 request TTFT where the client reports them;
- mean TPOT and request/input/output throughput;
- failures, retries, and server restarts.

If the result is within normal run-to-run noise, add paired repetitions until
the ordering is stable or the next profiler-guided change is ready.

### Acceptance

CP4+EP4 passes the primary performance gate only when:

- its median per-run mean TTFT is lower than TP4 at concurrency 1;
- its median per-run mean TTFT is lower than TP4 at concurrency 4;
- the direction holds in a majority of paired runs at both concurrencies;
- all requests complete with the requested token counts and valid outputs.

TPOT does not block acceptance unless a regression indicates a correctness,
stall, or scheduling problem.

## Correctness and Regression Tests

The implementation uses test-driven changes. Each behavior change begins with
a focused failing test, then the smallest implementation that passes it.

Required local or CPU-capable tests cover:

- EP expert range selection;
- BF16 TRTLLM weight padding and expert-bias semantics;
- FlashInfer A2A CP validation;
- CP-aware BCG shape and metadata refresh helpers;
- combined zigzag TRTLLM metadata and one-call dispatch;
- unequal sequence lengths, prefixes, and padding.

Required GPU checks cover:

- BF16 TP1 routed-MoE bring-up;
- BF16 CP4+EP4 startup with the exact required flags;
- deterministic prompt output compared with TP4/eager reference;
- a model accuracy sample large enough to detect gross weight-loading or
  activation errors;
- BCG repeated replay across different token counts;
- the full 128K/1K c1/c4 benchmark.

Unrelated repository tests are not weakened or skipped to land the work.

## Final Trace Contract

After the final benchmark winner is fixed, launch fresh TP4 and CP4+EP4
servers from the same final commit and collect separate traces with:

- concurrency 1;
- input length 262144;
- output length 20;
- one warmup request;
- radix-cache flush;
- 20 active profile steps;
- CPU and GPU activities;
- four rank-local `.trace.json.gz` files per configuration.

The run records server args, benchmark args, git SHA, package versions, GPU
inventory, server log, benchmark JSONL, and trace manifest. Every gzip file is
validated.

Copy final artifacts to:

```text
/Users/baizhou.zhang/.codex/visualizations/2026/07/28/
  019fa792-cf6d-75c3-9df1-963782ec22d3/gptoss_bf16_cp4_ep4_final/
```

Keep rank shards and also use the journal-prescribed `torch_utils` scripts to
produce merged and Perfetto-compatible traces when the scripts support the
captured schema. The final report links the local files directly.

## Artifact Layout

Remote artifacts:

```text
/scratch/gptoss_bf16_cp4_ep4_opt_20260728/
  preexisting/
  support/
  baselines/
  bcg/
  fused_attention/
  fusion_iterations/
  final_benchmark/
  final_profiles/
  manifest.txt
  final_report.md
```

Each experiment directory contains the exact launch command, benchmark
command, `server_info`, git SHA, dependency versions, logs, JSONL results,
trace manifest where applicable, and a concise conclusion.

## Stop Conditions

Optimization stops when all of the following are true:

1. The required BF16 CP4+EP4 FlashInfer configuration is correct and covered
   by a submitted support PR.
2. Breakable prefill CUDA graph runs reliably with CP4+EP4 and removes the
   identified non-attention launch overhead.
3. Zigzag TRTLLM context attention launches once per layer.
4. CP4+EP4 passes the c1 and c4 TTFT gates against TP4.
5. A fresh formal trace contains no clear low-risk, low-effort optimization
   above the 1% GPU-time or material CPU-gap threshold.
6. Final TP4 and CP4+EP4 profile shards and derived local artifacts satisfy the
   Final Trace Contract.

If an apparent remaining gap is inside an external FlashInfer or driver kernel,
the final report records the exact kernel, share, reproduction, and why an
SGLang-local change cannot address it safely.
