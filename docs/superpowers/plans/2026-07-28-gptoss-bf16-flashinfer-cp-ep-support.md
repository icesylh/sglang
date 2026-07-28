# GPT-OSS BF16 FlashInfer CP+EP Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `lmsys/gpt-oss-120b-bf16` start, load correctly, and pass correctness checks with prefill CP4, MoE EP4, FlashInfer A2A, and the FlashInfer TRT-LLM routed BF16 MoE runner.

**Architecture:** Keep the compatibility patch narrow. GPT-OSS owns expert-axis checkpoint slicing; `FusedMoE` preserves logical sizes while allocating kernel-padded sizes; the unquantized FlashInfer TRT-LLM method converts canonical BF16 weights into the kernel layout and folds static expert biases into padded channels; the runner pads/slices activations; `ServerArgs` admits only the full-width CP topology required by FlashInfer A2A.

**Tech Stack:** Python, PyTorch BF16, FlashInfer 0.6.x TRT-LLM routed MoE, SGLang CP-v2/EP, `unittest`/`pytest`, four NVIDIA GB300 GPUs.

## Global Constraints

- Work on `codex/gptoss-bf16-cp4-ep4-opt`, based on `origin/main@fc8b328f5cb526fd1fa2c26cb3dafe3761af8d5d`.
- Treat `/scratch/models/gpt-oss-120b-bf16` as read-only.
- Preserve the existing dirty prototype on `baizhou-dev` before changing the remote checkout.
- Stop only the task-owned server PID recorded in the artifact directory; never use a broad `pkill`.
- The supported topology is exactly:

  ```text
  --tp 4 --ep 4
  --enable-prefill-cp --attn-cp-size 4 --cp-strategy zigzag
  --moe-a2a-backend flashinfer
  --moe-runner-backend flashinfer_trtllm_routed
  ```

- Keep this compatibility work separable from CUDA-graph, attention-launch, and fusion work.
- Follow red-green-refactor for every source change.
- Run CPU tests locally; run FlashInfer import, kernel, launch, and accuracy checks on `baizhou-dev`.
- Store remote evidence under `/scratch/gptoss_bf16_cp4_ep4_opt_20260728/support/`.

---

## Task 1: Preserve the Remote Prototype and Establish a Reproducible Baseline

**Files:**

- Inspect: `/sgl-workspace/sglang`
- Create remotely: `/scratch/gptoss_bf16_cp4_ep4_opt_20260728/preexisting/`
- Record remotely: `support/environment.txt`, `support/baseline_failure.txt`

- [ ] **Step 1: Capture the remote repository and process state**

  Run:

  ```bash
  ssh baizhou-dev '
    set -eu
    task_root=/scratch/gptoss_bf16_cp4_ep4_opt_20260728
    mkdir -p "$task_root/preexisting" "$task_root/support"
    cd /sgl-workspace/sglang
    git status --short --branch > "$task_root/preexisting/git-status.txt"
    git rev-parse HEAD > "$task_root/preexisting/head.txt"
    git diff --binary > "$task_root/preexisting/prototype.patch"
    tar -czf "$task_root/preexisting/prototype-tests.tgz" \
      test/registered/unit/layers/test_flashinfer_trtllm_bf16_padding.py \
      test/registered/unit/models/test_gpt_oss_ep_weight_loading.py \
      test/registered/unit/server_args/test_flashinfer_a2a_cp.py
    nvidia-smi --query-gpu=index,name,memory.total,driver_version \
      --format=csv,noheader > "$task_root/support/environment.txt"
    python3 - <<'"'"'PY'"'"' >> "$task_root/support/environment.txt"
import flashinfer
import sglang
print("flashinfer", flashinfer.__version__)
print("sglang", sglang.__file__)
PY
    stash_before=$(git rev-parse --verify refs/stash 2>/dev/null || true)
    git stash push --include-untracked \
      --message codex-gptoss-cp-ep-preexisting-20260728 \
      > "$task_root/preexisting/stash.txt"
    stash_after=$(git rev-parse --verify refs/stash 2>/dev/null || true)
    if [ -n "$stash_after" ] && [ "$stash_after" != "$stash_before" ]; then
      printf "created_stash=%s\n" "$stash_after" >> "$task_root/preexisting/stash.txt"
    else
      printf "created_stash=none\n" >> "$task_root/preexisting/stash.txt"
    fi
  '
  ```

  Expected: the patch, test archive, commit SHA, status, GPU inventory, and package versions all exist below the task artifact root; the remote checkout is clean; and the prior dirty state is recoverable from both the binary patch/archive and the recorded Git stash.

- [ ] **Step 2: Stop only the existing task-owned server**

  Resolve the PID from the saved task log or `ps`, verify its command contains both `/scratch/models/gpt-oss-120b-bf16` and port `30000`, then run:

  ```bash
  ssh baizhou-dev 'kill 41238; for i in $(seq 1 30); do kill -0 41238 2>/dev/null || exit 0; sleep 1; done; exit 1'
  ```

  Expected: PID `41238` exits and no unrelated SGLang process is stopped. If the PID changed, replace `41238` only after the same command-line verification.

- [ ] **Step 3: Reproduce the unsupported CP4+EP4 combination before applying fixes**

  Use the pristine main checkout or a clean worktree at `fc8b328f...`, install it, and launch with the exact topology:

  ```bash
  SGLANG_TORCH_PROFILER_DIR=/scratch/gptoss_bf16_cp4_ep4_opt_20260728/support \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  python3 -m sglang.launch_server \
    --model-path /scratch/models/gpt-oss-120b-bf16 \
    --host 127.0.0.1 --port 30000 \
    --tp 4 --ep 4 \
    --enable-prefill-cp --attn-cp-size 4 --cp-strategy zigzag \
    --moe-a2a-backend flashinfer \
    --moe-runner-backend flashinfer_trtllm_routed \
    --attention-backend trtllm_mha \
    --context-length 139264 \
    --cuda-graph-backend-prefill disabled \
    > /scratch/gptoss_bf16_cp4_ep4_opt_20260728/support/baseline_failure.txt 2>&1
  ```

  Expected: main rejects FlashInfer A2A without DP attention, or proceeds far enough to expose the BF16 EP/load-layout failure. Preserve the full traceback.

---

## Task 2: Add GPT-OSS Expert-Parallel Checkpoint Slicing

**Files:**

- Modify: `python/sglang/srt/models/gpt_oss.py`
- Create: `test/registered/unit/models/test_gpt_oss_ep_weight_loading.py`

**Interface:**

```python
def _narrow_fused_moe_ep_weight(
    loaded_weight: torch.Tensor, num_experts: int
) -> torch.Tensor:
    ...
```

- [ ] **Step 1: Write failing expert-axis slicing tests**

  Make the new file a CPU CI test:

  ```python
  from sglang.test.ci.ci_register import register_cpu_ci
  from sglang.test.test_utils import CustomTestCase

  register_cpu_ci(est_time=5, suite="base-a-test-cpu")
  ```

  Inherit from `CustomTestCase`, and end the file with `unittest.main()`.

  Add tests that patch `sglang.srt.models.gpt_oss.get_parallel` and assert:

  ```python
  get_parallel.return_value = SimpleNamespace(moe_ep_size=4, moe_ep_rank=2)
  weight = torch.arange(128 * 2).view(128, 2)
  actual = _narrow_fused_moe_ep_weight(weight, num_experts=128)
  torch.testing.assert_close(actual, weight[64:96])
  ```

  Also cover EP1 identity, already-local `(32, ...)` input identity, and a malformed first dimension raising:

  ```text
  Expected 128 global or 32 local experts, got 64
  ```

- [ ] **Step 2: Run the new test and confirm the missing symbol**

  Run:

  ```bash
  python3 -m pytest -q test/registered/unit/models/test_gpt_oss_ep_weight_loading.py
  ```

  Expected: collection fails because `_narrow_fused_moe_ep_weight` does not exist.

- [ ] **Step 3: Implement the narrow helper**

  Implement:

  ```python
  def _narrow_fused_moe_ep_weight(loaded_weight, num_experts):
      parallel = get_parallel()
      if parallel.moe_ep_size == 1:
          return loaded_weight
      assert num_experts % parallel.moe_ep_size == 0
      num_local_experts = num_experts // parallel.moe_ep_size
      if loaded_weight.shape[0] == num_local_experts:
          return loaded_weight
      if loaded_weight.shape[0] != num_experts:
          raise ValueError(
              f"Expected {num_experts} global or {num_local_experts} local experts, "
              f"got {loaded_weight.shape[0]}"
          )
      return loaded_weight.narrow(
          0, parallel.moe_ep_rank * num_local_experts, num_local_experts
      )
  ```

  In the normal GPT-OSS `expert_params_mapping` loading branch, call the helper after resolving `param` and before transposing non-bias tensors. This keeps both weights and biases on the same local expert range.

- [ ] **Step 4: Run the focused test**

  Run:

  ```bash
  python3 -m pytest -q test/registered/unit/models/test_gpt_oss_ep_weight_loading.py
  ```

  Expected: all expert-axis tests pass.

- [ ] **Step 5: Commit the EP loading change**

  ```bash
  git add python/sglang/srt/models/gpt_oss.py \
    test/registered/unit/models/test_gpt_oss_ep_weight_loading.py
  git commit -m "fix: shard GPT-OSS BF16 experts for EP loading"
  ```

---

## Task 3: Preserve Logical MoE Sizes and Load the Fused Down-Projection Bias

**Files:**

- Modify: `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`
- Extend: `test/registered/unit/models/test_gpt_oss_ep_weight_loading.py`

**Interface:**

```python
self.intermediate_size_per_partition_unpadded: int
```

- [ ] **Step 1: Add failing tests for logical size preservation and bias loading**

  Construct a `FusedMoE` shell with `use_padded_loading=True`, then call `_load_w2` with two-dimensional expert bias data and the weight-loader supplied `shard_dim=2`:

  ```python
  expert_data = torch.empty(2, 4)
  loaded_weight = torch.arange(8, dtype=torch.float32).view(2, 4)
  layer._load_w2(
      expert_data=expert_data,
      shard_dim=2,
      shard_id="w2",
      loaded_weight=loaded_weight,
      tp_rank=0,
      is_bias=True,
  )
  torch.testing.assert_close(expert_data, loaded_weight)
  ```

  Add a constructor-focused test that verifies a logical intermediate size of `2880` is retained while the FlashInfer TRT-LLM kernel allocation rounds to `2944`.

- [ ] **Step 2: Confirm the bias path fails on the out-of-range shard dimension**

  Run:

  ```bash
  python3 -m pytest -q \
    test/registered/unit/models/test_gpt_oss_ep_weight_loading.py
  ```

  Expected: `_load_w2` fails when it applies `shard_dim=2` to the bias tensor, and the unpadded attribute is absent.

- [ ] **Step 3: Preserve the logical intermediate size and normalize bias sharding**

  Before FlashInfer alignment, set:

  ```python
  self.intermediate_size_per_partition_unpadded = (
      intermediate_size // self.moe_tp_size
  )
  self.intermediate_size_per_partition = (
      self.intermediate_size_per_partition_unpadded
  )
  ```

  In `_load_w2`, set `shard_dim = -1` whenever `is_bias` before calling `narrow_padded_param_and_loaded_weight`.

- [ ] **Step 4: Run the focused tests**

  Run:

  ```bash
  python3 -m pytest -q \
    test/registered/unit/models/test_gpt_oss_ep_weight_loading.py
  ```

  Expected: all tests pass.

- [ ] **Step 5: Commit the logical-size and bias-loader fix**

  ```bash
  git add python/sglang/srt/layers/moe/fused_moe_triton/layer.py \
    test/registered/unit/models/test_gpt_oss_ep_weight_loading.py
  git commit -m "fix: retain logical MoE sizes for padded BF16 kernels"
  ```

---

## Task 4: Prepare Biased BF16 Weights for FlashInfer TRT-LLM

**Files:**

- Modify: `python/sglang/srt/layers/quantization/unquant.py`
- Create: `test/registered/unit/layers/test_flashinfer_trtllm_bf16_padding.py`

**Interfaces:**

```python
def _prepare_flashinfer_trtllm_bf16_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_bias: Optional[torch.Tensor],
    w2_bias: Optional[torch.Tensor],
    *,
    hidden_size: int,
    intermediate_size: int,
    gemm1_alpha: Optional[float],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    ...
```

```python
self._flashinfer_kernel_hidden_size: int
self._flashinfer_input_pad_value: float
self._flashinfer_gemm1_alpha: Optional[torch.Tensor]
self._flashinfer_gemm1_beta: Optional[torch.Tensor]
self._flashinfer_gemm1_clamp_limit: Optional[torch.Tensor]
```

- [ ] **Step 1: Write a failing mathematical equivalence test**

  Make the new file a CPU CI test:

  ```python
  from sglang.test.ci.ci_register import register_cpu_ci
  from sglang.test.test_utils import CustomTestCase

  register_cpu_ci(est_time=8, suite="base-a-test-cpu")
  ```

  Inherit from `CustomTestCase`, and end the file with `unittest.main()`.

  Use two experts, `hidden_size=3`, logical intermediate size `5`, padded intermediate size `128`, BF16 weights, FP32 biases, `alpha=1.702`, and `limit=7.0`.

  Compare the canonical biased computation:

  ```python
  expected = F.linear(
      gpt_oss_activation(F.linear(x, w13[e], b13[e]), alpha, limit),
      w2[e],
      b2[e],
  )
  ```

  against the bias-free padded computation using an input channel filled with `1.0`:

  ```python
  x_padded = F.pad(x, (0, kernel_hidden_size - hidden_size), value=1.0)
  actual = F.linear(
      gpt_oss_activation(F.linear(x_padded, prepared_w13[e]), alpha, limit),
      prepared_w2[e],
  )[:, :hidden_size]
  ```

  Require `atol=0.03`, `rtol=0.03`, output hidden size `128`, and padded intermediate size `128`.

- [ ] **Step 2: Confirm the helper is absent**

  Run:

  ```bash
  python3 -m pytest -q \
    test/registered/unit/layers/test_flashinfer_trtllm_bf16_padding.py
  ```

  Expected: collection fails on the missing helper.

- [ ] **Step 3: Implement hidden padding and bias folding**

  Implement these invariants:

  - Round `hidden_size` to a multiple of 128.
  - Pad W13 on K and W2 on N.
  - Use the first padded hidden channel as a constant-one feature for W13 bias.
  - Use the first padded intermediate channel as a constant-one activated feature for W2 bias.
  - Derive the W13 `up` value so the GPT-OSS activation of the synthetic channel equals exactly one:

    ```python
    gate = 1.0
    up = 1.0 / (gate / (1.0 + math.exp(-gemm1_alpha * gate))) - 1.0
    ```

  - Reject bias folding if there is no padded hidden channel or no padded intermediate channel.

- [ ] **Step 4: Wire the preparation before FlashInfer block-layout conversion**

  In `process_weights_after_loading`:

  1. Read `layer.moe_runner_config.hidden_size`.
  2. Read `layer.intermediate_size_per_partition_unpadded`.
  3. Prepare W13/W2 and rebind the parameters.
  4. Create per-local-expert FP32 `gemm1_alpha`, `gemm1_beta`, and `gemm1_clamp_limit` tensors.
  5. Preserve the existing FlashInfer row permutation and block-layout conversion.

  Change `load_up_proj_weight_first` to:

  ```python
  return self.use_flashinfer_cutlass or self.use_flashinfer_trtllm_moe
  ```

- [ ] **Step 5: Run the mathematical tests**

  Run:

  ```bash
  python3 -m pytest -q \
    test/registered/unit/layers/test_flashinfer_trtllm_bf16_padding.py
  ```

  Expected: bias-folding equivalence and projection-order tests pass.

- [ ] **Step 6: Commit the weight preparation**

  ```bash
  git add python/sglang/srt/layers/quantization/unquant.py \
    test/registered/unit/layers/test_flashinfer_trtllm_bf16_padding.py
  git commit -m "fix: prepare biased BF16 MoE weights for FlashInfer"
  ```

---

## Task 5: Pad Runner Inputs and Pass GPT-OSS Activation Parameters

**Files:**

- Modify: `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`
- Extend: `test/registered/unit/layers/test_flashinfer_trtllm_bf16_padding.py`

**Interface:**

```python
@dataclass
class FlashInferTrtllmBf16MoeQuantInfo(MoeQuantInfo):
    gemm1_weights: torch.Tensor
    gemm2_weights: torch.Tensor
    global_num_experts: int
    local_expert_offset: int
    kernel_hidden_size: int
    input_pad_value: float
    gemm1_alpha: Optional[torch.Tensor] = None
    gemm1_beta: Optional[torch.Tensor] = None
    gemm1_clamp_limit: Optional[torch.Tensor] = None
```

- [ ] **Step 1: Add a mocked runner contract test**

  Patch `flashinfer.fused_moe.trtllm_bf16_routed_moe`, invoke `fused_experts_none_to_flashinfer_trtllm_bf16`, and assert:

  - a `(tokens, 2880)` input becomes `(tokens, 2944)`;
  - the extra 64 values equal `input_pad_value`;
  - the three activation parameter tensors are passed unchanged;
  - a `(tokens, 2944)` kernel result is returned as contiguous `(tokens, 2880)`.

- [ ] **Step 2: Confirm the dataclass/runner contract fails**

  Run:

  ```bash
  python3 -m pytest -q \
    test/registered/unit/layers/test_flashinfer_trtllm_bf16_padding.py
  ```

  Expected: the new dataclass arguments or mocked keyword assertions fail.

- [ ] **Step 3: Extend the quant-info payload and routed/non-routed calls**

  Pad `dispatch_output.hidden_states` with `torch.nn.functional.pad`, pass `gemm1_alpha`, `gemm1_beta`, and `gemm1_clamp_limit` to both BF16 FlashInfer calls, and slice the final result back to the original hidden width.

  In `UnquantizedFusedMoEMethod.forward_cuda`, populate all new fields from the values prepared during weight postprocessing.

- [ ] **Step 4: Run focused and adjacent MoE tests**

  Run:

  ```bash
  python3 -m pytest -q \
    test/registered/unit/layers/test_flashinfer_trtllm_bf16_padding.py \
    test/registered/unit/models/test_gpt_oss_ep_weight_loading.py
  ```

  Expected: all tests pass.

- [ ] **Step 5: Commit the runner contract**

  ```bash
  git add python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py \
    python/sglang/srt/layers/quantization/unquant.py \
    test/registered/unit/layers/test_flashinfer_trtllm_bf16_padding.py
  git commit -m "fix: run padded GPT-OSS BF16 MoE with FlashInfer"
  ```

---

## Task 6: Admit FlashInfer A2A for Full Prefill CP

**Files:**

- Modify: `python/sglang/srt/server_args.py`
- Create: `test/registered/unit/server_args/test_flashinfer_a2a_cp.py`

**Interface:**

```python
def _supports_flashinfer_a2a_parallelism(self) -> bool:
    ...
```

- [ ] **Step 1: Write the topology truth-table tests**

  Make the new file a CPU CI test:

  ```python
  from sglang.test.ci.ci_register import register_cpu_ci
  from sglang.test.test_utils import CustomTestCase

  register_cpu_ci(est_time=5, suite="base-a-test-cpu")
  ```

  Inherit from `CustomTestCase`, and end the file with `unittest.main()`.

  Cover:

  | DP attention | `dp_size` | Prefill CP | `attn_cp_size` | `tp_size` | Expected |
  |---|---:|---|---:|---:|---|
  | true | 4 | false | 1 | 4 | true |
  | false | 1 | true | 4 | 4 | true |
  | false | 1 | true | 2 | 4 | false |
  | false | 1 | false | 1 | 4 | false |

- [ ] **Step 2: Confirm the helper is absent**

  Run:

  ```bash
  python3 -m pytest -q \
    test/registered/unit/server_args/test_flashinfer_a2a_cp.py
  ```

  Expected: the helper is missing.

- [ ] **Step 3: Implement the narrow allowlist**

  Return true for either:

  ```python
  resolved.enable_dp_attention and self.dp_size == self.tp_size
  ```

  or:

  ```python
  self.enable_prefill_cp and resolved.attn_cp_size == self.tp_size
  ```

  Use this helper in `_handle_a2a_moe` and update the assertion text to name both supported modes.

- [ ] **Step 4: Run server-argument regression tests**

  Run:

  ```bash
  python3 -m pytest -q \
    test/registered/unit/server_args/test_flashinfer_a2a_cp.py \
    test/registered/unit/server_args/test_server_args.py
  ```

  Expected: all tests pass.

- [ ] **Step 5: Commit the topology validation**

  ```bash
  git add python/sglang/srt/server_args.py \
    test/registered/unit/server_args/test_flashinfer_a2a_cp.py
  git commit -m "feat: allow FlashInfer A2A with full prefill CP"
  ```

---

## Task 7: Validate the Exact BF16 CP4+EP4 Configuration on GB300

**Files:**

- Record remotely: `support/install.log`
- Record remotely: `support/server.log`
- Record remotely: `support/health.json`
- Record remotely: `support/gsm8k.txt`
- Record remotely: `support/smoke.jsonl`
- Record remotely: `support/tp1-server.log`
- Record remotely: `support/tp1-smoke.jsonl`
- Record remotely: `support/tp1-gsm8k.txt`

- [ ] **Step 1: Push the optimization branch and install the exact commit remotely**

  From the local worktree:

  ```bash
  git push -u origin codex/gptoss-bf16-cp4-ep4-opt
  ```

  On `baizhou-dev`:

  ```bash
  ssh baizhou-dev '
    set -eu
    cd /sgl-workspace/sglang
    git fetch origin codex/gptoss-bf16-cp4-ep4-opt
    git switch --detach origin/codex/gptoss-bf16-cp4-ep4-opt
    python3 -m pip install -e . \
      > /scratch/gptoss_bf16_cp4_ep4_opt_20260728/support/install.log 2>&1
  '
  ```

  Expected: remote `git rev-parse HEAD` equals the local branch tip.

- [ ] **Step 2: Prove the BF16 routed-MoE path at TP1**

  Launch the same checkpoint on one GB300 before adding CP/EP:

  ```bash
  ssh baizhou-dev '
    set -eu
    cd /sgl-workspace/sglang
    task_root=/scratch/gptoss_bf16_cp4_ep4_opt_20260728/support
    CUDA_VISIBLE_DEVICES=0 \
    nohup python3 -m sglang.launch_server \
      --model-path /scratch/models/gpt-oss-120b-bf16 \
      --host 127.0.0.1 --port 30000 \
      --tp 1 \
      --moe-a2a-backend none \
      --moe-runner-backend flashinfer_trtllm_routed \
      --attention-backend trtllm_mha \
      --context-length 8192 \
      --cuda-graph-backend-prefill disabled \
      > "$task_root/tp1-server.log" 2>&1 &
    echo $! > "$task_root/tp1-server.pid"
  '
  ```

  Poll `/health` for up to 20 minutes. Expected: HTTP 200 with no load-layout, bias, padding, or routed-MoE error.

- [ ] **Step 3: Run the TP1 smoke and correctness gates**

  Run the same deterministic 4096/20 random-ID request used for CP, writing `tp1-smoke.jsonl`, then run GSM8K-20 and write `tp1-gsm8k.txt`.

  Expected: exactly one successful 4096/20 request, GSM8K score at least `0.80`, and no malformed output or worker crash. This isolates BF16 routed-MoE correctness before distributed CP/EP is introduced.

- [ ] **Step 4: Stop only the TP1 server PID**

  Read `support/tp1-server.pid`, verify the command still names the BF16 checkpoint and port `30000`, stop only that PID, and wait for it to exit before using the four GPUs.

- [ ] **Step 5: Launch the exact requested CP4+EP4 configuration**

  Launch with:

  ```bash
  ssh baizhou-dev '
    set -eu
    cd /sgl-workspace/sglang
    task_root=/scratch/gptoss_bf16_cp4_ep4_opt_20260728/support
    SGLANG_TORCH_PROFILER_DIR="$task_root" \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    nohup python3 -m sglang.launch_server \
      --model-path /scratch/models/gpt-oss-120b-bf16 \
      --host 127.0.0.1 --port 30000 \
      --tp 4 --ep 4 \
      --enable-prefill-cp --attn-cp-size 4 --cp-strategy zigzag \
      --moe-a2a-backend flashinfer \
      --moe-runner-backend flashinfer_trtllm_routed \
      --attention-backend trtllm_mha \
      --context-length 139264 \
      --cuda-graph-backend-prefill disabled \
      > "$task_root/server.log" 2>&1 &
    echo $! > "$task_root/server.pid"
  '
  ```

  Poll `/health` for up to 20 minutes. Expected: HTTP 200 and no traceback, NCCL hang, invalid expert shape, or unsupported-backend assertion in the log.

- [ ] **Step 6: Run a deterministic CP4+EP4 smoke request**

  ```bash
  ssh baizhou-dev '
    cd /sgl-workspace/sglang
    python3 -m sglang.bench_serving \
      --backend sglang --host 127.0.0.1 --port 30000 \
      --dataset-name random-ids \
      --num-prompts 1 --max-concurrency 1 --request-rate inf \
      --random-input-len 4096 --random-output-len 20 \
      --random-range-ratio 1 --tokenize-prompt \
      --temperature 0 --warmup-requests 1 --flush-cache \
      --output-file /scratch/gptoss_bf16_cp4_ep4_opt_20260728/support/smoke.jsonl
  '
  ```

  Expected: one successful request with exactly 4096 input tokens and 20 generated tokens.

- [ ] **Step 7: Run the CP4+EP4 correctness gate**

  Run:

  ```bash
  ssh baizhou-dev '
    cd /sgl-workspace/sglang
    python3 -m sglang.test.run_eval \
      --host 127.0.0.1 --port 30000 \
      --eval-name gsm8k --num-examples 20 \
      | tee /scratch/gptoss_bf16_cp4_ep4_opt_20260728/support/gsm8k.txt
  '
  ```

  Expected: score at least `0.80`, no malformed output, and no worker crash.

- [ ] **Step 8: Stop only the recorded CP4+EP4 server**

  ```bash
  ssh baizhou-dev '
    task_root=/scratch/gptoss_bf16_cp4_ep4_opt_20260728/support
    server_pid=$(cat "$task_root/server.pid")
    kill "$server_pid"
    for i in $(seq 1 60); do
      kill -0 "$server_pid" 2>/dev/null || exit 0
      sleep 1
    done
    exit 1
  '
  ```

  Expected: the recorded server exits cleanly.

---

## Task 8: Publish a Support-Only Pull Request

**Files:**

- Include only the five support source files and three support test files.
- Exclude the design spec and all later performance work.

- [ ] **Step 1: Run the final local support test set**

  ```bash
  python3 -m pytest -q \
    test/registered/unit/layers/test_flashinfer_trtllm_bf16_padding.py \
    test/registered/unit/models/test_gpt_oss_ep_weight_loading.py \
    test/registered/unit/server_args/test_flashinfer_a2a_cp.py \
    test/registered/cp/test_cp_strategy_unit.py \
    test/registered/unit/server_args/test_server_args.py
  ```

  Expected: all tests pass.

- [ ] **Step 2: Create a clean support branch from current main**

  Create a temporary worktree:

  ```bash
  support_worktree=$(mktemp -d /tmp/sglang-gptoss-support.XXXXXX)
  git worktree add -b codex/gptoss-bf16-flashinfer-support \
    "$support_worktree" origin/main
  ```

  Cherry-pick only the support code commits from Tasks 2–6. Do not cherry-pick the design-spec commit.

- [ ] **Step 3: Verify the support-only diff**

  In the support worktree:

  ```bash
  git diff --stat origin/main...HEAD
  git diff --check origin/main...HEAD
  git log --oneline origin/main..HEAD
  ```

  Expected: no design document, benchmark harness, CUDA-graph work, one-launch work, or fusion work appears.

- [ ] **Step 4: Push and open a draft PR**

  Use the `github:yeet` skill. Push:

  ```bash
  git push -u origin codex/gptoss-bf16-flashinfer-support
  ```

  PR title:

  ```text
  Support GPT-OSS BF16 FlashInfer MoE with prefill CP and EP
  ```

  PR body must include:

  - the exact CP4+EP4 server command;
  - the main-branch failure signature;
  - the BF16 bias-folding equivalence test;
  - all local test commands and results;
  - GB300 health, smoke, and GSM8K results;
  - an explicit note that BCG and performance changes are intentionally excluded.

- [ ] **Step 5: Keep the optimization branch on its original support commits**

  Return to `codex/gptoss-bf16-cp4-ep4-opt`. Do not merge or rebase it onto the support PR branch; it already contains equivalent support commits plus the approved design record.

---

## Support Exit Gate

Do not begin CUDA-graph work until all conditions are true:

- exact CP4+EP4 flags reach HTTP health;
- the TP1 routed-MoE server, 4096/20 smoke, and GSM8K-20 gate pass first;
- the 4096/20 deterministic request succeeds;
- GSM8K-20 score is at least 0.80;
- the focused local test set passes;
- the support-only draft PR exists and excludes performance changes;
- remote logs and the exact tested commit SHA are preserved under the support artifact directory.
