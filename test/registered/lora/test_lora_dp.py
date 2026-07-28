# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Dynamic LoRA load/unload under plain data parallelism (dp_size > 1, no dp attention).

Control requests fan out to every DP replica while generate requests are
round-robin dispatched, so with greedy sampling a prompt can only produce the
same text across repeated requests if all replicas share the same adapter
state. That reproducibility is the cross-replica consistency oracle used here.
"""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=420, stage="base-b", runner_config="2-gpu-large")

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_A = "algoprog/fact-generation-llama-3.1-8b-instruct-lora"  # r = 64
ADAPTER_B = "nvidia/llama-3.1-nemoguard-8b-topic-control"  # r = 4

PROMPTS = [
    "AI is a field of computer science focused on",
    "List three prime numbers:",
]

# Round-robin dispatch alternates DP ranks, so any even number >= 2 exercises
# every replica; 4 also catches state that only breaks after the first pass.
NUM_REQUESTS_PER_CHECK = 4


class TestLoRADynamicUpdateDP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            BASE_MODEL,
            cls.base_url,
            DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--dp-size",
                "2",
                "--enable-lora",
                "--max-loras-per-batch",
                "2",
                "--max-lora-rank",
                "64",
                "--lora-target-modules",
                "all",
                "--random-seed",
                "42",
                "--mem-fraction-static",
                "0.8",
                "--disable-radix-cache",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def _load(self, lora_name: str, lora_path: str) -> requests.Response:
        return requests.post(
            f"{self.base_url}/load_lora_adapter",
            json={"lora_name": lora_name, "lora_path": lora_path},
        )

    def _unload(self, lora_name: str) -> requests.Response:
        return requests.post(
            f"{self.base_url}/unload_lora_adapter",
            json={"lora_name": lora_name},
        )

    def _generate(self, prompt: str, lora_name=None) -> requests.Response:
        payload = {
            "text": prompt,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 24},
        }
        if lora_name is not None:
            payload["lora_path"] = lora_name
        return requests.post(f"{self.base_url}/generate", json=payload)

    def _generate_on_all_ranks(self, prompt: str, lora_name=None) -> str:
        """Send the same greedy request enough times to hit every DP rank; assert
        all succeed and return the unique output text."""
        texts = set()
        for _ in range(NUM_REQUESTS_PER_CHECK):
            response = self._generate(prompt, lora_name=lora_name)
            self.assertEqual(
                response.status_code,
                200,
                f"generation failed (lora={lora_name}): {response.text}",
            )
            texts.add(response.json()["text"])
        self.assertEqual(
            len(texts),
            1,
            f"DP ranks disagree for lora={lora_name!r}, prompt={prompt!r}: {texts}",
        )
        return texts.pop()

    def test_dynamic_lora_dp(self):
        # 1. Dynamic load must fan out to both replicas.
        response = self._load("adapter_a", ADAPTER_A)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["success"], response.text)

        lora_texts = {p: self._generate_on_all_ranks(p, "adapter_a") for p in PROMPTS}
        base_texts = {p: self._generate_on_all_ranks(p) for p in PROMPTS}

        # 2. The adapter must actually change the output on every replica.
        self.assertNotEqual(
            lora_texts,
            base_texts,
            "adapter produced base-model outputs on all prompts",
        )

        # 3. A never-loaded adapter is rejected, not silently served.
        response = self._generate(PROMPTS[0], lora_name="never_loaded")
        self.assertNotEqual(response.status_code, 200)
        self.assertIn("never been loaded", response.text)

        # 4. Unload fans out; the adapter then reloads on demand (intended
        #    `_resolve_lora_path` behavior) with identical outputs.
        response = self._unload("adapter_a")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["success"], response.text)

        reload_texts = {p: self._generate_on_all_ranks(p, "adapter_a") for p in PROMPTS}
        self.assertEqual(reload_texts, lora_texts)

        # 5. Two adapters served interleaved stay consistent per replica.
        response = self._load("adapter_b", ADAPTER_B)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["success"], response.text)

        for lora_name in ("adapter_a", "adapter_b"):
            self._generate_on_all_ranks(PROMPTS[0], lora_name)

        for lora_name in ("adapter_a", "adapter_b"):
            response = self._unload(lora_name)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(response.json()["success"], response.text)


if __name__ == "__main__":
    unittest.main()
