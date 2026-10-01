Recorded vLLM startups used by `tests/test_ground_truth_runs.py`.

Add one by running, on a machine with an NVIDIA GPU and vLLM installed:

    python scripts/collect_ground_truth.py <model> [--tensor-parallel-size N] \
        [--gpu-memory-utilization 0.9] [--max-model-len 8192] [--enforce-eager]

Re-record on each vLLM minor release; file names include the vLLM version.
