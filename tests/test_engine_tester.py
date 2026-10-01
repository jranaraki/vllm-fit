import signal

import vllm_fit.engine_tester as et
from vllm_fit.engine_tester import ProfilingAborted, _is_memory_failure


def _gpu_initial():
    return {"gpu_ids": [0], "gpu_memory_utilization": 0.9, "max_model_len": 4096,
            "tensor_parallel_size": 1, "max_num_seqs": 4}


def _cpu_initial(max_model_len=3000):
    return {"max_model_len": max_model_len, "max_num_seqs": 4, "enforce_eager": True,
            "kv_cache_space_gb": 8}


def test_memory_failures_are_recognised():
    assert _is_memory_failure("torch.OutOfMemoryError: CUDA out of memory.", 1)
    assert _is_memory_failure("ValueError: No available memory for the cache blocks.", 1)
    assert _is_memory_failure(
        "ValueError: To serve at least one request ... KV cache is needed, which is larger "
        "than the available KV cache memory (1.2 GiB).", 1)
    assert _is_memory_failure(
        "ValueError: Free memory on device (6.9/8.0 GiB) on startup is less than desired "
        "GPU memory utilization (0.9, 7.2 GiB).", 1)
    # Host OOM killer: SIGKILL, no traceback.
    assert _is_memory_failure("", -getattr(signal, "SIGKILL", 9))


def test_other_failures_are_not_memory():
    assert not _is_memory_failure("ModuleNotFoundError: No module named 'vllm'", 1)
    assert not _is_memory_failure("GatedRepoError: 401 Client Error", 1)
    assert not _is_memory_failure(
        "ValueError: User-specified max_model_len (262144) is greater than the derived "
        "max_model_len (131072).", 1)


def test_profile_stops_when_vllm_missing(monkeypatch):
    monkeypatch.setattr(et.importlib.util, "find_spec", lambda name: None)
    called = []
    monkeypatch.setattr(et, "_test_configuration", lambda *a, **k: called.append(1))
    res = et.profile_parameters("org/model", _gpu_initial())
    assert res["profiling_success"] is False
    assert "not installed" in res["error"]
    assert called == []


def test_profile_stops_on_non_memory_failure(monkeypatch):
    monkeypatch.setattr(et, "check_vllm_installed", lambda: None)
    calls = []

    def fake(*args, **kwargs):
        calls.append(args)
        raise ProfilingAborted("vLLM failed for a reason other than memory", "GatedRepoError")

    monkeypatch.setattr(et, "_test_configuration", fake)
    res = et.profile_parameters("org/model", _gpu_initial())
    assert res["profiling_success"] is False
    assert res["error_detail"] == "GatedRepoError"
    # Must not keep shrinking the configuration after a non-memory failure.
    assert len(calls) == 1
    assert res["max_model_len"] == 4096 and res["max_num_seqs"] == 4


def test_profile_passes_timeout_to_every_probe(monkeypatch):
    monkeypatch.setattr(et, "check_vllm_installed", lambda: None)
    timeouts = []

    def fake(*args, timeout=None, **kwargs):
        timeouts.append(timeout)
        return True, False

    monkeypatch.setattr(et, "_test_configuration", fake)
    res = et.profile_parameters("org/model", _gpu_initial(), timeout=1234)
    assert res["profiling_success"] is True
    assert timeouts and set(timeouts) == {1234}


def test_cpu_search_never_lowers_a_working_length(monkeypatch):
    monkeypatch.setattr(et, "check_vllm_installed", lambda: None)
    # Oracle: anything up to 3500 tokens fits.
    monkeypatch.setattr(
        et, "_test_configuration_cpu",
        lambda model, length, seqs, eager, **kw: (length <= 3500, False),
    )
    res = et.profile_parameters_cpu("org/model", _cpu_initial(3000))
    assert res["profiling_success"] is True
    assert res["max_model_len"] == 3500


def test_gpu_probe_pins_devices_in_pci_order(monkeypatch):
    seen = {}

    def fake_probe(env, llm_kwargs, timeout):
        seen.update(env)
        return True

    monkeypatch.setattr(et, "_run_probe", fake_probe)
    et._test_configuration("org/model", 0.9, 4096, 2, 4, False, [2, 3])
    assert seen["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert seen["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_cpu_probe_matches_served_command(monkeypatch):
    seen = {}
    monkeypatch.setattr(et, "_run_probe", lambda env, kw, timeout: seen.update(kw) or True)
    et._test_configuration_cpu("org/model-GGUF:Q8_0", 8192, 4, True,
                               config_repo_id="org/model")
    # Some vLLM CPU builds disable chunked prefill; the served command pins this too.
    assert seen["max_num_batched_tokens"] == 8192
    assert seen["hf_config_path"] == "org/model"
    assert seen["tokenizer"] == "org/model"


def test_gpu_probe_has_no_config_override_for_regular_repo(monkeypatch):
    seen = {}
    monkeypatch.setattr(et, "_run_probe", lambda env, kw, timeout: seen.update(kw) or True)
    et._test_configuration("org/model", 0.9, 4096, 1, 4, False, [0], config_repo_id="org/model")
    assert "hf_config_path" not in seen and "tokenizer" not in seen


def test_profile_threads_config_repo_to_every_probe(monkeypatch):
    monkeypatch.setattr(et, "check_vllm_installed", lambda: None)
    repos = []

    def fake(*args, config_repo_id=None, **kwargs):
        repos.append(config_repo_id)
        return True, False

    monkeypatch.setattr(et, "_test_configuration", fake)
    et.profile_parameters("org/model-GGUF:Q4_K_M", _gpu_initial(), config_repo_id="org/model")
    assert repos and set(repos) == {"org/model"}


def test_length_search_never_exceeds_model_limit(monkeypatch):
    # Qwen3-0.6B: vLLM rejects max_model_len > 40960 with a non-memory error.
    monkeypatch.setattr(et, "check_vllm_installed", lambda: None)
    tried = []

    def fake(model, util, length, tp, seqs, eager, gpu_ids, **kw):
        tried.append(length)
        if length > 40960:
            raise ProfilingAborted("vLLM failed for a reason other than memory")
        return True, False

    monkeypatch.setattr(et, "_test_configuration", fake)
    res = et.profile_parameters("Qwen/Qwen3-0.6B", {**_gpu_initial(), "max_model_len": 40960},
                                max_len_cap=40960)
    assert res["profiling_success"] is True
    assert max(tried) <= 40960 and res["max_model_len"] == 40960


def test_cpu_length_search_respects_model_limit(monkeypatch):
    monkeypatch.setattr(et, "check_vllm_installed", lambda: None)
    tried = []
    monkeypatch.setattr(et, "_test_configuration_cpu",
                        lambda model, length, seqs, eager, **kw: (tried.append(length) or True, False))
    res = et.profile_parameters_cpu("org/model", _cpu_initial(3000), max_len_cap=4096)
    assert max(tried) <= 4096 and res["max_model_len"] == 4096
