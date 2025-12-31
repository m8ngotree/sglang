import os
import unittest

# Disable JIT DeepGemm BEFORE importing sglang modules (must be at module level)
os.environ["SGLANG_ENABLE_JIT_DEEPGEMM"] = "0"

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

# Runs on both H200 and B200 via nightly-8-gpu-common suite
register_cuda_ci(est_time=12000, suite="nightly-8-gpu-common", nightly=True)

QWEN3_235B_MODEL_PATH = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"

# Accuracy baselines from HuggingFace model card
AIME25_BASELINE = 0.703
GPQA_BASELINE = 0.775


class TestQwen3235BFP8Unified(unittest.TestCase):
    """Unified test class for Qwen3-235B-FP8 performance and accuracy.

    Tests run on both H200 and B200 systems.
    Uses official cookbook configuration: TP=8, EP=2
    Evaluates:
    - Performance with low latency batch sizes [1, 2, 4, 8, 16, 64]
    - Accuracy on AIME25 and GPQA datasets
    """

    def test_qwen3_235b_fp8_aime25(self):
        """Run performance and AIME25 accuracy for Qwen3-235B-FP8."""
        # Use official cookbook configuration: TP=8, EP=2
        # Let backends auto-select (no explicit attention/moe backend)
        # Add chat template and reasoning support for AIME25
        base_args = [
            "--tp=8",
            "--ep=2",
            "--trust-remote-code",
            "--chat-template=qwen",
            "--reasoning-parser=qwen3",
        ]

        variants = [
            ModelLaunchSettings(
                QWEN3_235B_MODEL_PATH,
                tp_size=8,
                extra_args=base_args,
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="Qwen3-235B-FP8 AIME25",
            accuracy_params=AccuracyTestParams(
                dataset="aime25",
                baseline_accuracy=AIME25_BASELINE,
            ),
            performance_params=PerformanceTestParams(
                batch_sizes=[1, 2, 4, 8, 16, 64],
                profile_dir="performance_profiles_qwen3_235b_fp8",
            ),
        )

    def test_qwen3_235b_fp8_gpqa(self):
        """Run GPQA accuracy for Qwen3-235B-FP8 (performance already tested in AIME25)."""
        # Use official cookbook configuration: TP=8, EP=2
        # Let backends auto-select (no explicit attention/moe backend)
        base_args = [
            "--tp=8",
            "--ep=2",
            "--trust-remote-code",
        ]

        variants = [
            ModelLaunchSettings(
                QWEN3_235B_MODEL_PATH,
                tp_size=8,
                extra_args=base_args,
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="Qwen3-235B-FP8 GPQA",
            accuracy_params=AccuracyTestParams(
                dataset="gpqa",
                baseline_accuracy=GPQA_BASELINE,
            ),
            performance_params=None,  # Skip performance test, already tested in AIME25
        )


if __name__ == "__main__":
    unittest.main()
