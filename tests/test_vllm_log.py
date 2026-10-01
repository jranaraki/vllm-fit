from vllm_fit.vllm_log import parse_startup_log

LOG = """\
INFO 10-01 12:00:01 [gpu_model_runner.py:2100] Model loading took 14.9927 GiB memory and 5.312345 seconds
INFO 10-01 12:00:09 [gpu_worker.py:298] Available KV cache memory: 52.31 GiB
INFO 10-01 12:00:09 [kv_cache_utils.py:1087] GPU KV cache size: 190,464 tokens, Maximum concurrency for 32,768 tokens per request: 5.81x
INFO 10-01 12:00:21 [gpu_model_runner.py:2700] Graph capturing finished in 12 secs, took 0.53 GiB
"""


def test_parse_startup_log():
    assert parse_startup_log(LOG) == {
        "weights_gib": 14.9927,
        "available_kv_gib": 52.31,
        "kv_tokens": 190464,
        "max_concurrency": 5.81,
        "cudagraph_gib": 0.53,
    }


def test_parse_cpu_and_partial_log():
    log = "INFO CPU KV cache size: 65,536 tokens, Maximum concurrency for 8,192 tokens per request: 8.00x"
    assert parse_startup_log(log) == {"kv_tokens": 65536, "max_concurrency": 8.0}
    assert parse_startup_log("nothing here") == {}
