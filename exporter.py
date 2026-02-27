import os
import time
import json
import threading
import urllib.request
import traceback

from aiohttp import web


def _load_env_file(path: str):
    """Minimal .env loader (no external deps).

    - Lines: KEY=VALUE
    - Supports quoted values (single/double)
    - Ignores blank lines and comments starting with #
    - Does not override already-set environment variables
    """

    try:
        if not path or not os.path.exists(path):
            return

        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower().startswith("export "):
                    line = line[7:].strip()
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()
                if not k or k in os.environ:
                    continue

                if (len(v) >= 2) and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
                    v = v[1:-1]

                os.environ[k] = v
    except Exception:
        # Avoid breaking ComfyUI startup on .env issues.
        return


# Load .env from the same folder as this exporter, unless explicitly pointed elsewhere.
_DEFAULT_ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
_load_env_file(os.getenv("COMFY_PROM_ENV_FILE", _DEFAULT_ENV_PATH))

EXPORTER_VERSION = "0.7.0"

START_TIME = time.time()

_METRICS_PATH = os.getenv("COMFY_PROM_METRICS_PATH", "/metrics")
_API_BASE = os.getenv("COMFY_PROM_API_BASE", "http://127.0.0.1:8188")
_API_BASE_EXPLICIT = "COMFY_PROM_API_BASE" in os.environ
_API_KEY = os.getenv("COMFY_API_KEY")
_CACHE_TTL = float(os.getenv("COMFY_PROM_CACHE_TTL_SECONDS", "2.0"))
_SYSSTATS_TTL = float(os.getenv("COMFY_PROM_SYSSTATS_TTL_SECONDS", "10.0"))
_LOG_LEVEL = os.getenv("COMFY_PROM_LOG_LEVEL", "info").strip().lower()

_lock = threading.Lock()
_instrumented = False

_last_log_ts: dict[str, float] = {}
_last_request_base: str | None = None

# Pre-defined marker sets for failure classification (module-level constants for performance)
_CUDA_MARKERS = frozenset([
    "cuda error",
    "cublas",
    "cudnn",
    "device-side assert",
    "illegal memory access",
    "misaligned address",
    "driver shutting down",
    "nvrtc",
])

_MODEL_MARKERS = frozenset([
    "safetensors",
    "checkpoint",
    "ckpt",
    "failed to load",
    "can't load",
    "cannot load",
    "no such file or directory",
    "file not found",
])

_MODEL_CONTEXT_MARKERS = frozenset(["model", "ckpt", "checkpoint", "safetensors"])

_GRAPH_MARKERS = frozenset([
    "prompt is invalid",
    "invalid prompt",
    "validation",
    "required input",
    "missing input",
    "unknown node",
    "does not exist",
    "type mismatch",
])


def _log(level: str, msg: str, exc: BaseException | None = None):
    levels = {"debug": 10, "info": 20, "warn": 30, "warning": 30, "error": 40}
    want = levels.get(_LOG_LEVEL, 20)
    have = levels.get(level, 20)
    if have < want:
        return

    prefix = f"[comfyui_prometheus_exporter] {level.upper()}:"
    if exc is None:
        print(f"{prefix} {msg}")
    else:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        print(f"{prefix} {msg}\n{tb}")


def _log_rate_limited(key: str, level: str, msg: str, interval_s: float = 30.0, exc: BaseException | None = None):
    now = time.time()
    last = _last_log_ts.get(key, 0.0)
    if now - last >= interval_s:
        _last_log_ts[key] = now
        _log(level, msg, exc=exc)


def _get_api_base() -> str:
    if _API_BASE_EXPLICIT:
        return _API_BASE
    if _last_request_base:
        return _last_request_base
    return _API_BASE


_cache = {
    "ts": 0.0,
    "queue": None,
    "proc": None,
    "sys": None,
    "torch": None,
    "nvml_gpu": None,
    "system_stats": None,
    "system_stats_ts": 0.0,
}

_counters = {
    "comfyui_prompt_executions_total": 0,
    "comfyui_prompt_execution_failures_total": 0,
    "comfyui_prompt_failures_total": {},  # by reason

    # Failure classification totals (convenience counters)
    "comfyui_prompt_oom_failures_total": 0,
    "comfyui_prompt_cuda_failures_total": 0,
    "comfyui_prompt_model_load_failures_total": 0,
    "comfyui_prompt_graph_failures_total": 0,

    # Node-type timing breakdown
    "comfyui_node_duration_seconds_sum": {},  # by class_type
    "comfyui_node_duration_seconds_count": {},  # by class_type
    "comfyui_node_failures_total": {},  # by class_type

    # Model usage
    "comfyui_checkpoint_requests_total": {},  # by ckpt
    "comfyui_lora_requests_total": {},  # by lora
    "comfyui_controlnet_requests_total": {},  # by model

    "comfyui_ws_messages_total": {},  # by type
    "comfyui_ws_progress_events_total": 0,
    "comfyui_ws_execution_cached_nodes_total": 0,
    "comfyui_node_executions_total": 0,
    "comfyui_prompt_completions_total": 0,

    # Output volume
    "comfyui_images_generated_total": {},  # by type
    "comfyui_output_bytes_total": {},  # by type

    # Workload characteristics (sum/count)
    "comfyui_prompt_requested_width_sum": 0.0,
    "comfyui_prompt_requested_width_count": 0,
    "comfyui_prompt_requested_height_sum": 0.0,
    "comfyui_prompt_requested_height_count": 0,
    "comfyui_prompt_requested_pixels_sum": 0.0,
    "comfyui_prompt_requested_pixels_count": 0,
    "comfyui_prompt_batch_size_sum": 0.0,
    "comfyui_prompt_batch_size_count": 0,
    "comfyui_prompt_steps_sum": 0.0,
    "comfyui_prompt_steps_count": 0,
    "comfyui_prompt_cfg_sum": 0.0,
    "comfyui_prompt_cfg_count": 0,
    "comfyui_prompt_loras_sum": 0.0,
    "comfyui_prompt_loras_count": 0,
    "comfyui_prompt_controlnets_sum": 0.0,
    "comfyui_prompt_controlnets_count": 0,
    "comfyui_sampler_requests_total": {},  # by sampler
    "comfyui_scheduler_requests_total": {},  # by scheduler
    "comfyui_model_family_requests_total": {},  # by family

    # Latency breakdown (sum/count)
    "comfyui_prompt_queue_wait_seconds_sum": 0.0,
    "comfyui_prompt_queue_wait_seconds_count": 0,
    "comfyui_prompt_execution_seconds_sum": 0.0,
    "comfyui_prompt_execution_seconds_count": 0,
    "comfyui_prompt_end_to_end_seconds_sum": 0.0,
    "comfyui_prompt_end_to_end_seconds_count": 0,
}

_gauges = {
    "comfyui_queue_remaining": None,
    "comfyui_progress_value": None,
    "comfyui_progress_max": None,
    "comfyui_progress_ratio": None,

    # Concurrency & saturation
    "comfyui_prompt_in_flight": 0,
    "comfyui_node_in_flight": 0,
    "comfyui_active_prompts": 0,
    "comfyui_active_nodes": 0,
    "comfyui_queue_age_seconds": None,
    "comfyui_queue_throughput_prompts_per_min": None,

    # Output volume (per last completed prompt)
    "comfyui_last_prompt_images": None,
    "comfyui_last_prompt_output_bytes": None,

    # Workload (per last completed prompt)
    "comfyui_last_prompt_width": None,
    "comfyui_last_prompt_height": None,
    "comfyui_last_prompt_pixels": None,
    "comfyui_last_prompt_batch_size": None,
    "comfyui_last_prompt_steps": None,
    "comfyui_last_prompt_cfg": None,
    "comfyui_last_prompt_loras": None,
    "comfyui_last_prompt_controlnets": None,

    # Per-run peak GPU/VRAM stats (last completed prompt)
    "comfyui_prompt_cuda_max_memory_allocated_bytes": None,
    "comfyui_prompt_cuda_max_memory_reserved_bytes": None,
    "comfyui_prompt_nvml_vram_peak_by_gpu_bytes": {},

    # Last failure
    "comfyui_last_failure_reason": None,

    # Latency breakdown (last/max gauges)
    "comfyui_prompt_queue_wait_last_seconds": None,
    "comfyui_prompt_queue_wait_max_seconds": 0.0,
    "comfyui_prompt_end_to_end_last_seconds": None,
    "comfyui_prompt_end_to_end_max_seconds": 0.0,
}

_hist = {
    "comfyui_prompt_execution_last_seconds": None,
    "comfyui_prompt_execution_max_seconds": 0.0,
    "comfyui_prompt_execution_seconds_sum": 0.0,

    "comfyui_node_execution_last_seconds": None,
    "comfyui_node_execution_max_seconds": 0.0,
    "comfyui_node_execution_seconds_sum": 0.0,
}

_state = {
    "node_start_ts": None,
    "current_node": None,

    # prompt_id -> ts
    "enqueue_by_id": {},
    "exec_start_by_id": {},

    # output volume per prompt_id
    "images_by_id": {},  # prompt_id -> int
    "bytes_by_id": {},  # prompt_id -> int

    # workload per prompt_id
    "workload_by_id": {},  # prompt_id -> dict

    # prompt_id -> {node_id(str): class_type(str)}
    "node_class_by_prompt": {},

    # prompt_id -> {gpu_index(str): peak_used_bytes(int)} (NVML)
    "nvml_peak_by_prompt": {},

    # Throughput tracking (updated on scrape)
    "throughput_last_ts": None,
    "throughput_last_completions": None,
}


_nvml_ctx = {
    "init": False,
    "ok": False,
    "pynvml": None,
    "handles": None,
    "names": None,
}


def _nvml_init_once():
    with _lock:
        if _nvml_ctx["init"]:
            return
        _nvml_ctx["init"] = True
        try:
            import pynvml  # optional

            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            handles = []
            names = []
            for i in range(count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                handles.append(h)
                try:
                    name = pynvml.nvmlDeviceGetName(h)
                    # Handle both old (bytes) and new (str) pynvml versions
                    if isinstance(name, bytes):
                        name = name.decode("utf-8", "ignore")
                    names.append(name)
                except Exception:
                    names.append("unknown")

            _nvml_ctx["ok"] = True
            _nvml_ctx["pynvml"] = pynvml
            _nvml_ctx["handles"] = handles
            _nvml_ctx["names"] = names
        except Exception:
            _nvml_ctx["ok"] = False


def _nvml_used_by_gpu() -> list[dict] | None:
    _nvml_init_once()
    if not _nvml_ctx.get("ok"):
        return None

    try:
        pynvml = _nvml_ctx["pynvml"]
        handles = _nvml_ctx["handles"] or []
        names = _nvml_ctx["names"] or []
        out = []
        for i, h in enumerate(handles):
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            out.append({"index": i, "name": names[i] if i < len(names) else "unknown", "used": int(mem.used)})
        return out
    except Exception:
        return None


def _nvml_update_peak(prompt_id: str | None):
    if not isinstance(prompt_id, str) or not prompt_id:
        return

    gpus = _nvml_used_by_gpu()
    if not gpus:
        return

    with _lock:
        peaks = _state["nvml_peak_by_prompt"].get(prompt_id)
        if not isinstance(peaks, dict):
            peaks = {}
            _state["nvml_peak_by_prompt"][prompt_id] = peaks

        for g in gpus:
            idx = str(g.get("index"))
            used = g.get("used")
            if isinstance(idx, str) and isinstance(used, int):
                peaks[idx] = max(int(peaks.get(idx, 0)), int(used))


def _esc_label(v: str) -> str:
    return str(v).replace("\\", "\\\\").replace('"', '\\"')


def _prom_line(name: str, value, labels: dict | None = None) -> str:
    if labels:
        lbl = ",".join([f'{k}="{_esc_label(v)}"' for k, v in labels.items()])
        return f"{name}{{{lbl}}} {value}"
    return f"{name} {value}"


def _route_exists_aiohttp(ps, path: str) -> bool:
    try:
        router = getattr(getattr(ps, "app", None), "router", None)
        if router is None:
            return False

        routes_attr = getattr(router, "routes", None)
        routes_iter = routes_attr() if callable(routes_attr) else routes_attr
        if callable(routes_iter):
            routes_iter = routes_iter()
        if routes_iter is None:
            return False

        try:
            it = iter(routes_iter)
        except TypeError:
            return False

        for r in it:
            res = getattr(r, "resource", None)
            canonical = getattr(res, "canonical", None)
            if canonical == path:
                return True
        return False
    except Exception:
        return False


def _prune_prompt_maps(now: float):
    """Prune old entries from prompt-scoped state maps to prevent memory leaks."""
    enqueue = _state["enqueue_by_id"]
    execs = _state["exec_start_by_id"]
    imgs = _state["images_by_id"]
    bts = _state["bytes_by_id"]
    wl = _state["workload_by_id"]
    node_class = _state["node_class_by_prompt"]
    nvml_peak = _state["nvml_peak_by_prompt"]

    if len(enqueue) <= 2000 and len(execs) <= 2000:
        cutoff = now - 3600
    else:
        cutoff = now - 300

    for pid, ts in list(enqueue.items()):
        if ts < cutoff:
            enqueue.pop(pid, None)
            execs.pop(pid, None)
            imgs.pop(pid, None)
            bts.pop(pid, None)
            wl.pop(pid, None)
            node_class.pop(pid, None)
            nvml_peak.pop(pid, None)


def _record_enqueue(prompt_id: str, now: float):
    if not prompt_id:
        return
    with _lock:
        _state["enqueue_by_id"].setdefault(prompt_id, now)
        _prune_prompt_maps(now)


def _inc_by_label(counter_map: dict, label: str, inc: int = 1):
    counter_map[label] = counter_map.get(label, 0) + inc


def _coerce_int(v) -> int | None:
    try:
        if v is None:
            return None
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, str) and v.strip() != "":
            return int(float(v))
    except Exception:
        return None
    return None


def _coerce_float(v) -> float | None:
    try:
        if v is None:
            return None
        if isinstance(v, bool):
            return float(int(v))
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str) and v.strip() != "":
            return float(v)
    except Exception:
        return None
    return None


def _classify_model_family(ckpt_name: str | None) -> str | None:
    if not ckpt_name:
        return None
    s = ckpt_name.lower()
    if "flux" in s:
        return "flux"
    if "sdxl" in s or "xl" in s:
        return "sdxl"
    if "sd15" in s or "1.5" in s or "sd_15" in s:
        return "sd15"
    if "sd3" in s or "stable-diffusion-3" in s:
        return "sd3"
    return "other"


def _extract_workload_from_item(item) -> dict:
    # Queue items are commonly tuples like:
    # (number, prompt_id, prompt, extra_data, outputs_to_execute)
    prompt_obj = None
    extra_data = None
    try:
        if isinstance(item, (tuple, list)) and len(item) >= 3:
            prompt_obj = item[2]
        if isinstance(item, (tuple, list)) and len(item) >= 4:
            extra_data = item[3]
    except Exception:
        return {}

    graph = None
    if isinstance(prompt_obj, dict):
        # Some versions store the graph directly; others wrap under 'prompt'
        if isinstance(prompt_obj.get("prompt"), dict):
            graph = prompt_obj.get("prompt")
        else:
            graph = prompt_obj

    if not isinstance(graph, dict):
        return {}

    nodes = graph

    width = height = batch = None
    steps = None
    cfg = None
    sampler = None
    scheduler = None
    loras = 0
    controlnets = 0
    model_family = None

    # Prefer size from EmptyLatentImage if present
    preferred_size_nodes = {"EmptyLatentImage", "EmptyLatentImageAdvanced"}

    for node in nodes.values():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}

        if isinstance(class_type, str):
            if class_type in preferred_size_nodes or (width is None and "Latent" in class_type):
                w = _coerce_int(inputs.get("width"))
                h = _coerce_int(inputs.get("height"))
                b = _coerce_int(inputs.get("batch_size"))
                if w and h:
                    width = width or w
                    height = height or h
                if b is not None:
                    batch = batch or b

            if "KSampler" in class_type or class_type in {"SamplerCustom", "SamplerCustomAdvanced"}:
                st = _coerce_int(inputs.get("steps"))
                if st is not None:
                    steps = max(steps or 0, st)
                cf = _coerce_float(inputs.get("cfg"))
                if cf is not None:
                    cfg = max(cfg or 0.0, cf)
                sname = inputs.get("sampler_name") or inputs.get("sampler")
                if isinstance(sname, str) and sname:
                    sampler = sampler or sname
                sch = inputs.get("scheduler")
                if isinstance(sch, str) and sch:
                    scheduler = scheduler or sch

            if "LoraLoader" in class_type or class_type in {"LoraLoader", "LoraLoaderModelOnly"}:
                loras += 1

            if "ControlNet" in class_type:
                controlnets += 1

            if model_family is None and "CheckpointLoader" in class_type:
                ckpt = inputs.get("ckpt_name")
                if isinstance(ckpt, str) and ckpt:
                    model_family = _classify_model_family(ckpt)

    if batch is None:
        # Some workflows stash batch in extra_data
        if isinstance(extra_data, dict):
            batch = _coerce_int(extra_data.get("batch_size"))

    if width is not None and height is not None:
        pixels = int(width) * int(height) * int(batch or 1)
    else:
        pixels = None

    out = {}
    if width is not None:
        out["width"] = int(width)
    if height is not None:
        out["height"] = int(height)
    if batch is not None:
        out["batch"] = int(batch)
    if pixels is not None:
        out["pixels"] = int(pixels)
    if steps is not None:
        out["steps"] = int(steps)
    if cfg is not None:
        out["cfg"] = float(cfg)
    if sampler is not None:
        out["sampler"] = sampler
    if scheduler is not None:
        out["scheduler"] = scheduler
    out["loras"] = int(loras)
    out["controlnets"] = int(controlnets)
    if model_family is not None:
        out["model_family"] = model_family

    return out


def _http_get_json(path: str):
    base = _get_api_base().rstrip("/")
    url = f"{base}{path}"
    req = urllib.request.Request(url, method="GET")
    if _API_KEY:
        req.add_header("X-API-Key", _API_KEY)
    with urllib.request.urlopen(req, timeout=3) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _extract_queue_counts(info):
    """Best-effort normalization across ComfyUI versions."""
    try:
        if info is None:
            return None

        # Typical REST response: {queue_pending: [...], queue_running: [...]}
        if isinstance(info, dict):
            if "queue_pending" in info or "queue_running" in info:
                qp = info.get("queue_pending") or []
                qr = info.get("queue_running") or []
                if isinstance(qp, list) and isinstance(qr, list):
                    return {"pending": len(qp), "running": len(qr)}

            # Some builds may already provide counts.
            if "pending" in info or "running" in info:
                pending = info.get("pending")
                running = info.get("running")
                if pending is not None and running is not None:
                    return {"pending": int(pending), "running": int(running)}

            return None

        # Some internal methods may return a 2-tuple.
        if isinstance(info, (list, tuple)) and len(info) == 2:
            a, b = info[0], info[1]
            if isinstance(a, list) and isinstance(b, list):
                # Often (queue_running, queue_pending)
                return {"pending": len(b), "running": len(a)}
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                # Often (pending, running)
                return {"pending": int(a), "running": int(b)}

        return None
    except Exception:
        return None


def _get_queue_counts_internal():
    try:
        import server

        ps = getattr(server.PromptServer, "instance", None)
        if not ps:
            return None
        pq = getattr(ps, "prompt_queue", None)
        if not pq:
            return None

        # Prefer official methods if available (more stable across versions)
        for obj in (pq, ps):
            for method in ("get_current_queue", "get_queue_info"):
                fn = getattr(obj, method, None)
                if fn and callable(fn):
                    try:
                        counts = _extract_queue_counts(fn())
                        if counts is not None:
                            return counts
                    except Exception:
                        pass

        pending = running = None

        # Pending queue (many builds keep the actual pending list/deque here)
        for attr in ("queue_pending", "pending", "_queue_pending", "queue", "_queue"):
            if hasattr(pq, attr):
                try:
                    pending = len(getattr(pq, attr))
                    break
                except Exception:
                    pass

        # Running queue
        for attr in (
            "queue_running",
            "currently_running",
            "running",
            "_queue_running",
            "_currently_running",
        ):
            if hasattr(pq, attr):
                try:
                    running = len(getattr(pq, attr))
                    break
                except Exception:
                    pass

        if pending is None or running is None:
            return None
        return {"pending": int(pending), "running": int(running)}
    except Exception:
        return None


def _get_queue_counts_api():
    # ComfyUI typically exposes /api/queue; some builds/plugins may expose /queue.
    try:
        data = _http_get_json("/api/queue")
    except Exception:
        data = _http_get_json("/queue")

    counts = _extract_queue_counts(data)
    return counts


def _get_proc_metrics():
    out = {}

    try:
        import psutil  # optional

        p = psutil.Process()
        out["rss_bytes"] = int(p.memory_info().rss)
        cpu_times = p.cpu_times()
        out["cpu_seconds_total"] = float(cpu_times.user + cpu_times.system)
        out["num_threads"] = int(p.num_threads())
        return out
    except Exception:
        pass

    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF)
        rss_kb = ru.ru_maxrss  # typically KB on Linux
        out["rss_bytes"] = int(rss_kb * 1024)
        out["cpu_seconds_total"] = float(ru.ru_utime + ru.ru_stime)
    except Exception:
        pass

    return out


def _get_sys_metrics():
    out = {}
    try:
        import psutil  # optional

        vm = psutil.virtual_memory()
        out["mem_total_bytes"] = int(vm.total)
        out["mem_available_bytes"] = int(vm.available)
        out["mem_used_bytes"] = int(vm.used)
        out["cpu_count_logical"] = int(psutil.cpu_count(logical=True) or 0)
        out["loadavg_1"] = float(os.getloadavg()[0]) if hasattr(os, "getloadavg") else None
        return out
    except Exception:
        pass

    if hasattr(os, "getloadavg"):
        try:
            out["loadavg_1"] = float(os.getloadavg()[0])
        except Exception:
            pass

    return out


def _get_torch_metrics():
    out = {}
    try:
        import torch

        out["torch_version"] = getattr(torch, "__version__", "unknown")
        out["cuda_available"] = 1 if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            out["cuda_device_count"] = int(torch.cuda.device_count())
            idx = int(torch.cuda.current_device())
            out["cuda_current_device"] = idx
            out["cuda_device_name"] = torch.cuda.get_device_name(idx)
            try:
                out["cuda_mem_allocated_bytes"] = int(torch.cuda.memory_allocated(idx))
                out["cuda_mem_reserved_bytes"] = int(torch.cuda.memory_reserved(idx))
                out["cuda_max_mem_allocated_bytes"] = int(torch.cuda.max_memory_allocated(idx))
                out["cuda_max_mem_reserved_bytes"] = int(torch.cuda.max_memory_reserved(idx))
            except Exception:
                pass
        return out
    except Exception:
        return out


def _get_nvml_gpu_metrics():
    """Get GPU metrics using cached NVML handles."""
    _nvml_init_once()
    if not _nvml_ctx.get("ok"):
        return None

    try:
        pynvml = _nvml_ctx["pynvml"]
        handles = _nvml_ctx["handles"] or []
        names = _nvml_ctx["names"] or []
        gpus = []
        for i, h in enumerate(handles):
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            temp = None
            try:
                temp = int(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
            except Exception:
                pass
            gpus.append(
                {
                    "index": i,
                    "name": names[i] if i < len(names) else "unknown",
                    "mem_used": int(mem.used),
                    "mem_total": int(mem.total),
                    "gpu_util": int(util.gpu),
                    "mem_util": int(util.memory),
                    "temp_c": temp,
                }
            )
        return gpus
    except Exception:
        return None


def _get_system_stats_api_cached(now: float):
    with _lock:
        last = _cache.get("system_stats_ts", 0.0)
        if now - last < _SYSSTATS_TTL:
            return _cache.get("system_stats")

    try:
        data = _http_get_json("/api/system_stats")
    except Exception:
        data = None

    with _lock:
        _cache["system_stats"] = data
        _cache["system_stats_ts"] = now

    return data


def _collect_cached():
    now = time.time()
    with _lock:
        if now - _cache["ts"] < _CACHE_TTL:
            return dict(_cache)

    queue = _get_queue_counts_internal()
    if queue is None:
        try:
            queue = _get_queue_counts_api()
        except Exception as e:
            queue = None
            _log_rate_limited("queue_api_fail", "warn", "Queue API fallback failed; queue metrics will be missing.", exc=e)

    if queue is None:
        _log_rate_limited(
            "queue_missing",
            "debug",
            "Queue metrics are currently unavailable (internal + API fallback returned None).",
            interval_s=60.0,
        )

    proc = _get_proc_metrics()
    sysm = _get_sys_metrics()
    torchm = _get_torch_metrics()
    nvml = _get_nvml_gpu_metrics()
    sysstats = _get_system_stats_api_cached(now)

    with _lock:
        _cache.update(
            {
                "ts": now,
                "queue": queue,
                "proc": proc,
                "sys": sysm,
                "torch": torchm,
                "nvml_gpu": nvml,
                "system_stats": sysstats,
            }
        )
        return dict(_cache)


def _inc_ws_type(t: str):
    d = _counters["comfyui_ws_messages_total"]
    d[t] = d.get(t, 0) + 1


def _failure_text(data: dict) -> str:
    try:
        parts: list[str] = []

        def take(v):
            if isinstance(v, str) and v:
                parts.append(v)

        for k in ("error", "exception_message", "message", "traceback", "exception_type"):
            take(data.get(k))

        # Some builds nest error details
        detail = data.get("detail")
        if isinstance(detail, dict):
            for k in ("message", "error", "traceback", "exception_type"):
                take(detail.get(k))

        if parts:
            return "\n".join(parts).lower()

        return json.dumps(data, ensure_ascii=False, default=str).lower()
    except Exception:
        return str(data).lower()


def _classify_failure_reason(message_type: str, data: dict) -> str:
    if message_type == "execution_interrupted":
        return "interrupted"

    text = _failure_text(data)

    # Out-of-memory (common for CUDA + some CPU situations)
    if "out of memory" in text or "cuda out of memory" in text or "cudamalloc" in text:
        return "oom"

    # CUDA/runtime errors (but not explicitly OOM)
    if any(m in text for m in _CUDA_MARKERS):
        return "cuda"

    # Model / checkpoint load issues
    if any(m in text for m in _MODEL_MARKERS) and any(m in text for m in _MODEL_CONTEXT_MARKERS):
        return "model_load"

    # Graph/workflow validation issues
    if any(m in text for m in _GRAPH_MARKERS):
        return "graph"

    return "other"


def _extract_model_usage_from_item(item) -> dict:
    # Returns {"ckpts": set[str], "loras": set[str], "controlnets": set[str]}
    try:
        if not isinstance(item, (tuple, list)) or len(item) < 3:
            return {"ckpts": set(), "loras": set(), "controlnets": set()}

        prompt_obj = item[2]
        graph = None
        if isinstance(prompt_obj, dict):
            if isinstance(prompt_obj.get("prompt"), dict):
                graph = prompt_obj.get("prompt")
            else:
                graph = prompt_obj

        if not isinstance(graph, dict):
            return {"ckpts": set(), "loras": set(), "controlnets": set()}

        ckpts: set[str] = set()
        loras: set[str] = set()
        controlnets: set[str] = set()

        for node in graph.values():
            if not isinstance(node, dict):
                continue
            class_type = node.get("class_type")
            inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}

            if not isinstance(class_type, str):
                continue

            ct = class_type.lower()

            # Checkpoints
            if "checkpointloader" in ct:
                for k in ("ckpt_name", "checkpoint", "checkpoint_name", "model", "model_name"):
                    v = inputs.get(k)
                    if isinstance(v, str) and v.strip():
                        ckpts.add(v.strip())

            # LoRAs
            if "loraloader" in ct or "lora" in ct:
                for k, v in inputs.items():
                    if not isinstance(v, str) or not v.strip():
                        continue
                    kk = str(k).lower()
                    if kk in ("lora_name", "lora") or ("lora" in kk and kk.endswith("name")):
                        loras.add(v.strip())

            # ControlNets
            if "controlnet" in ct:
                for k, v in inputs.items():
                    if not isinstance(v, str) or not v.strip():
                        continue
                    kk = str(k).lower()
                    if kk in ("control_net_name", "controlnet_name", "controlnet", "control_net") or (
                        "control" in kk and "name" in kk
                    ):
                        controlnets.add(v.strip())

        return {"ckpts": ckpts, "loras": loras, "controlnets": controlnets}
    except Exception:
        return {"ckpts": set(), "loras": set(), "controlnets": set()}


def _extract_node_class_map_from_item(item) -> dict[str, str]:
    # Best-effort parse of the enqueued prompt graph to map node_id -> class_type.
    try:
        if not isinstance(item, (tuple, list)) or len(item) < 3:
            return {}

        prompt_obj = item[2]
        graph = None
        if isinstance(prompt_obj, dict):
            if isinstance(prompt_obj.get("prompt"), dict):
                graph = prompt_obj.get("prompt")
            else:
                graph = prompt_obj

        if not isinstance(graph, dict):
            return {}

        out: dict[str, str] = {}
        for node_id, node in graph.items():
            if not isinstance(node, dict):
                continue
            ct = node.get("class_type")
            if isinstance(ct, str) and ct:
                out[str(node_id)] = ct
        return out
    except Exception:
        return {}


def _node_class_type(prompt_id: str | None, node_id) -> str | None:
    if not prompt_id or node_id is None:
        return None
    try:
        pid = str(prompt_id)
        nid = str(node_id)
        mp = _state["node_class_by_prompt"].get(pid)
        if isinstance(mp, dict):
            ct = mp.get(nid)
            if isinstance(ct, str) and ct:
                return ct
        return None
    except Exception:
        return None


def _count_images_in_executed_payload(data):
    if not isinstance(data, dict):
        return []
    images = data.get("images", [])
    if isinstance(images, list):
        return images
    return []


def _image_meta_size_bytes(meta):
    if not isinstance(meta, dict):
        return "unknown", 0
    img_type = meta.get("type", "unknown")
    size_bytes = _coerce_int(meta.get("size")) or 0
    return img_type, size_bytes


def _handle_ws_event(message_type: str, data: dict):
    now = time.time()
    with _lock:
        _inc_ws_type(message_type)
        _prune_prompt_maps(now)

        if message_type == "status":
            exec_info = data.get("exec_info")
            if isinstance(exec_info, dict) and "queue_remaining" in exec_info:
                try:
                    _gauges["comfyui_queue_remaining"] = int(exec_info["queue_remaining"])
                except Exception:
                    pass

        elif message_type == "execution_start":
            prompt_id = data.get("prompt_id")
            if isinstance(prompt_id, str) and prompt_id:
                _state["exec_start_by_id"][prompt_id] = now

                # Reset per-prompt peak memory stats (Torch) and initialize NVML peaks.
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass

                _nvml_update_peak(prompt_id)

                enq = _state["enqueue_by_id"].get(prompt_id)
                if enq is not None:
                    qwait = now - enq
                    _counters["comfyui_prompt_queue_wait_seconds_sum"] += qwait
                    _counters["comfyui_prompt_queue_wait_seconds_count"] += 1
                    _gauges["comfyui_prompt_queue_wait_last_seconds"] = qwait
                    if qwait > _gauges["comfyui_prompt_queue_wait_max_seconds"]:
                        _gauges["comfyui_prompt_queue_wait_max_seconds"] = qwait

            _gauges["comfyui_prompt_in_flight"] = 1
            _gauges["comfyui_active_prompts"] = 1

        elif message_type in ("execution_error", "execution_interrupted"):
            _counters["comfyui_prompt_execution_failures_total"] += 1
            reason = _classify_failure_reason(message_type, data)
            _inc_by_label(_counters["comfyui_prompt_failures_total"], reason, 1)
            _gauges["comfyui_last_failure_reason"] = reason

            # Node-level failure attribution if we can detect a node_id
            failing_node = None
            for k in ("node_id", "node", "failed_node", "node_name"):
                if k in data:
                    failing_node = data.get(k)
                    break
            detail = data.get("detail")
            if failing_node is None and isinstance(detail, dict):
                for k in ("node_id", "node", "failed_node", "node_name"):
                    if k in detail:
                        failing_node = detail.get(k)
                        break

            prompt_id = data.get("prompt_id")
            ct = _node_class_type(prompt_id if isinstance(prompt_id, str) else None, failing_node)
            if isinstance(ct, str) and ct:
                _inc_by_label(_counters["comfyui_node_failures_total"], ct, 1)

            # Convenience totals for common failure buckets
            if reason == "oom":
                _counters["comfyui_prompt_oom_failures_total"] += 1
            elif reason == "cuda":
                _counters["comfyui_prompt_cuda_failures_total"] += 1
            elif reason == "model_load":
                _counters["comfyui_prompt_model_load_failures_total"] += 1
            elif reason == "graph":
                _counters["comfyui_prompt_graph_failures_total"] += 1

        elif message_type == "execution_cached":
            nodes = data.get("nodes")
            if isinstance(nodes, list):
                _counters["comfyui_ws_execution_cached_nodes_total"] += len(nodes)

        elif message_type == "progress":
            _counters["comfyui_ws_progress_events_total"] += 1
            try:
                v = float(data.get("value"))
                m = float(data.get("max"))
                _gauges["comfyui_progress_value"] = v
                _gauges["comfyui_progress_max"] = m
                _gauges["comfyui_progress_ratio"] = (v / m) if m > 0 else None
            except Exception:
                pass

            # Best-effort NVML peak sampling during execution.
            prompt_id = data.get("prompt_id")
            _nvml_update_peak(prompt_id if isinstance(prompt_id, str) else None)

        elif message_type == "executed":
            prompt_id = data.get("prompt_id")
            if isinstance(prompt_id, str) and prompt_id:
                metas = _count_images_in_executed_payload(data)
                if metas:
                    img_count = len(metas)
                    _state["images_by_id"][prompt_id] = _state["images_by_id"].get(prompt_id, 0) + img_count

                    bytes_added = 0
                    for meta in metas:
                        img_type, size_bytes = _image_meta_size_bytes(meta)
                        _inc_by_label(_counters["comfyui_images_generated_total"], img_type, 1)
                        if size_bytes:
                            bytes_added += size_bytes
                            _inc_by_label(_counters["comfyui_output_bytes_total"], img_type, int(size_bytes))

                    if bytes_added:
                        _state["bytes_by_id"][prompt_id] = _state["bytes_by_id"].get(prompt_id, 0) + bytes_added

        elif message_type == "executing":
            prompt_id = data.get("prompt_id")
            node = data.get("node")

            # Best-effort NVML peak sampling during execution.
            _nvml_update_peak(prompt_id if isinstance(prompt_id, str) else None)

            # close previous node timing on transition
            if _state["node_start_ts"] is not None:
                dt = now - _state["node_start_ts"]
                _hist["comfyui_node_execution_last_seconds"] = dt
                _hist["comfyui_node_execution_seconds_sum"] += dt
                if dt > _hist["comfyui_node_execution_max_seconds"]:
                    _hist["comfyui_node_execution_max_seconds"] = dt
                _counters["comfyui_node_executions_total"] += 1

                prev_node_id = _state.get("current_node")
                prev_prompt_id = prompt_id if isinstance(prompt_id, str) else None
                ct = _node_class_type(prev_prompt_id, prev_node_id)
                if ct:
                    _counters["comfyui_node_duration_seconds_sum"][ct] = _counters["comfyui_node_duration_seconds_sum"].get(
                        ct, 0.0
                    ) + float(dt)
                    _counters["comfyui_node_duration_seconds_count"][ct] = _counters[
                        "comfyui_node_duration_seconds_count"
                    ].get(ct, 0) + 1

            if node is None:
                # prompt completion
                if isinstance(prompt_id, str) and prompt_id:
                    # Record execution time
                    exec_start = _state["exec_start_by_id"].get(prompt_id)
                    if exec_start is not None:
                        exec_dt = now - exec_start
                        _hist["comfyui_prompt_execution_last_seconds"] = exec_dt
                        _hist["comfyui_prompt_execution_seconds_sum"] += exec_dt
                        if exec_dt > _hist["comfyui_prompt_execution_max_seconds"]:
                            _hist["comfyui_prompt_execution_max_seconds"] = exec_dt
                        _counters["comfyui_prompt_execution_seconds_sum"] += exec_dt
                        _counters["comfyui_prompt_execution_seconds_count"] += 1

                    # Record end-to-end time
                    enq = _state["enqueue_by_id"].get(prompt_id)
                    if enq is not None:
                        e2e = now - enq
                        _counters["comfyui_prompt_end_to_end_seconds_sum"] += e2e
                        _counters["comfyui_prompt_end_to_end_seconds_count"] += 1
                        _gauges["comfyui_prompt_end_to_end_last_seconds"] = e2e
                        if e2e > _gauges["comfyui_prompt_end_to_end_max_seconds"]:
                            _gauges["comfyui_prompt_end_to_end_max_seconds"] = e2e

                    # Record per-prompt peaks at completion
                    try:
                        import torch

                        if torch.cuda.is_available():
                            idx = int(torch.cuda.current_device())
                            _gauges["comfyui_prompt_cuda_max_memory_allocated_bytes"] = int(
                                torch.cuda.max_memory_allocated(idx)
                            )
                            _gauges["comfyui_prompt_cuda_max_memory_reserved_bytes"] = int(
                                torch.cuda.max_memory_reserved(idx)
                            )
                    except Exception:
                        pass

                    peaks = _state["nvml_peak_by_prompt"].get(prompt_id)
                    if isinstance(peaks, dict):
                        _gauges["comfyui_prompt_nvml_vram_peak_by_gpu_bytes"] = dict(peaks)

                    # Update last prompt gauges from workload
                    wl = _state["workload_by_id"].get(prompt_id)
                    if isinstance(wl, dict):
                        _gauges["comfyui_last_prompt_width"] = wl.get("width")
                        _gauges["comfyui_last_prompt_height"] = wl.get("height")
                        _gauges["comfyui_last_prompt_pixels"] = wl.get("pixels")
                        _gauges["comfyui_last_prompt_batch_size"] = wl.get("batch")
                        _gauges["comfyui_last_prompt_steps"] = wl.get("steps")
                        _gauges["comfyui_last_prompt_cfg"] = wl.get("cfg")
                        _gauges["comfyui_last_prompt_loras"] = wl.get("loras")
                        _gauges["comfyui_last_prompt_controlnets"] = wl.get("controlnets")

                    # Update last prompt output gauges
                    _gauges["comfyui_last_prompt_images"] = _state["images_by_id"].get(prompt_id)
                    _gauges["comfyui_last_prompt_output_bytes"] = _state["bytes_by_id"].get(prompt_id)

                    # cleanup prompt-scoped state
                    _state["enqueue_by_id"].pop(prompt_id, None)
                    _state["exec_start_by_id"].pop(prompt_id, None)
                    _state["images_by_id"].pop(prompt_id, None)
                    _state["bytes_by_id"].pop(prompt_id, None)
                    _state["workload_by_id"].pop(prompt_id, None)
                    _state["node_class_by_prompt"].pop(prompt_id, None)
                    _state["nvml_peak_by_prompt"].pop(prompt_id, None)

                _counters["comfyui_prompt_completions_total"] += 1
                _state["node_start_ts"] = None
                _state["current_node"] = None
                _gauges["comfyui_prompt_in_flight"] = 0
                _gauges["comfyui_node_in_flight"] = 0
                _gauges["comfyui_active_prompts"] = 0
                _gauges["comfyui_active_nodes"] = 0
            else:
                _state["current_node"] = node
                _state["node_start_ts"] = now
                _gauges["comfyui_node_in_flight"] = 1
                _gauges["comfyui_active_nodes"] = 1


def _mb_to_bytes(v):
    try:
        vv = float(v)
        return int(vv * 1024 * 1024)
    except Exception:
        return None


def _render_metrics():
    c = _collect_cached()
    now = time.time()

    lines = []

    lines.append("# HELP comfyui_exporter_info Exporter build info.")
    lines.append("# TYPE comfyui_exporter_info gauge")
    lines.append(_prom_line("comfyui_exporter_info", 1, {"version": EXPORTER_VERSION, "path": _METRICS_PATH}))

    lines.append("# HELP comfyui_uptime_seconds Uptime of the ComfyUI process in seconds.")
    lines.append("# TYPE comfyui_uptime_seconds gauge")
    lines.append(_prom_line("comfyui_uptime_seconds", int(now - START_TIME)))

    # Queue (pending/running)
    q = c.get("queue") or {}
    lines.append("# HELP comfyui_queue_pending Number of pending prompts.")
    lines.append("# TYPE comfyui_queue_pending gauge")
    if "pending" in q:
        lines.append(_prom_line("comfyui_queue_pending", int(q["pending"])))

    lines.append("# HELP comfyui_queue_running Number of running prompts.")
    lines.append("# TYPE comfyui_queue_running gauge")
    if "running" in q:
        lines.append(_prom_line("comfyui_queue_running", int(q["running"])))

    # WS-derived metrics
    with _lock:
        # Concurrency/saturation derived gauges
        try:
            enqueue = _state.get("enqueue_by_id") or {}
            execs = _state.get("exec_start_by_id") or {}
            queued_ts = [ts for pid, ts in enqueue.items() if pid not in execs]
            _gauges["comfyui_queue_age_seconds"] = (now - min(queued_ts)) if queued_ts else None

            last_ts = _state.get("throughput_last_ts")
            last_c = _state.get("throughput_last_completions")
            curr_c = int(_counters.get("comfyui_prompt_completions_total", 0))
            if isinstance(last_ts, (int, float)) and isinstance(last_c, int) and now > float(last_ts) + 0.001 and curr_c >= last_c:
                _gauges["comfyui_queue_throughput_prompts_per_min"] = ((curr_c - last_c) / (now - float(last_ts))) * 60.0
            _state["throughput_last_ts"] = now
            _state["throughput_last_completions"] = curr_c
        except Exception:
            pass

        ws_by_type = dict(_counters["comfyui_ws_messages_total"])
        gauges = dict(_gauges)
        hist = dict(_hist)
        images_by_type = dict(_counters["comfyui_images_generated_total"])
        bytes_by_type = dict(_counters["comfyui_output_bytes_total"])
        sampler_by = dict(_counters["comfyui_sampler_requests_total"])
        scheduler_by = dict(_counters["comfyui_scheduler_requests_total"])
        fam_by = dict(_counters["comfyui_model_family_requests_total"])
        failures_by_reason = dict(_counters["comfyui_prompt_failures_total"])

        ckpt_by = dict(_counters["comfyui_checkpoint_requests_total"])
        lora_by = dict(_counters["comfyui_lora_requests_total"])
        controlnet_by = dict(_counters["comfyui_controlnet_requests_total"])

        counters = {
            k: v
            for k, v in _counters.items()
            if k
            not in (
                "comfyui_ws_messages_total",
                "comfyui_images_generated_total",
                "comfyui_output_bytes_total",
                "comfyui_sampler_requests_total",
                "comfyui_scheduler_requests_total",
                "comfyui_model_family_requests_total",
                "comfyui_prompt_failures_total",
                "comfyui_checkpoint_requests_total",
                "comfyui_lora_requests_total",
                "comfyui_controlnet_requests_total",
            )
        }

    # Workload last-prompt gauges
    lines.append("# HELP comfyui_last_prompt_width Width (px) detected for the last completed prompt.")
    lines.append("# TYPE comfyui_last_prompt_width gauge")
    if gauges.get("comfyui_last_prompt_width") is not None:
        lines.append(_prom_line("comfyui_last_prompt_width", gauges["comfyui_last_prompt_width"]))

    lines.append("# HELP comfyui_last_prompt_height Height (px) detected for the last completed prompt.")
    lines.append("# TYPE comfyui_last_prompt_height gauge")
    if gauges.get("comfyui_last_prompt_height") is not None:
        lines.append(_prom_line("comfyui_last_prompt_height", gauges["comfyui_last_prompt_height"]))

    lines.append("# HELP comfyui_last_prompt_pixels Pixels (width*height*batch) detected for the last completed prompt.")
    lines.append("# TYPE comfyui_last_prompt_pixels gauge")
    if gauges.get("comfyui_last_prompt_pixels") is not None:
        lines.append(_prom_line("comfyui_last_prompt_pixels", gauges["comfyui_last_prompt_pixels"]))

    lines.append("# HELP comfyui_last_prompt_batch_size Batch size detected for the last completed prompt.")
    lines.append("# TYPE comfyui_last_prompt_batch_size gauge")
    if gauges.get("comfyui_last_prompt_batch_size") is not None:
        lines.append(_prom_line("comfyui_last_prompt_batch_size", gauges["comfyui_last_prompt_batch_size"]))

    lines.append("# HELP comfyui_last_prompt_steps Steps detected for the last completed prompt.")
    lines.append("# TYPE comfyui_last_prompt_steps gauge")
    if gauges.get("comfyui_last_prompt_steps") is not None:
        lines.append(_prom_line("comfyui_last_prompt_steps", gauges["comfyui_last_prompt_steps"]))

    lines.append("# HELP comfyui_last_prompt_cfg CFG detected for the last completed prompt.")
    lines.append("# TYPE comfyui_last_prompt_cfg gauge")
    if gauges.get("comfyui_last_prompt_cfg") is not None:
        lines.append(_prom_line("comfyui_last_prompt_cfg", gauges["comfyui_last_prompt_cfg"]))

    lines.append("# HELP comfyui_last_prompt_loras Number of LoRA loader nodes detected for the last completed prompt.")
    lines.append("# TYPE comfyui_last_prompt_loras gauge")
    if gauges.get("comfyui_last_prompt_loras") is not None:
        lines.append(_prom_line("comfyui_last_prompt_loras", gauges["comfyui_last_prompt_loras"]))

    lines.append("# HELP comfyui_last_prompt_controlnets Number of ControlNet-related nodes detected for the last completed prompt.")
    lines.append("# TYPE comfyui_last_prompt_controlnets gauge")
    if gauges.get("comfyui_last_prompt_controlnets") is not None:
        lines.append(_prom_line("comfyui_last_prompt_controlnets", gauges["comfyui_last_prompt_controlnets"]))

    # Workload counters for averages
    for base, help_text in [
        ("comfyui_prompt_requested_width", "Requested width (px) sum/count from queued prompts."),
        ("comfyui_prompt_requested_height", "Requested height (px) sum/count from queued prompts."),
        ("comfyui_prompt_requested_pixels", "Requested pixels (width*height*batch) sum/count from queued prompts."),
        ("comfyui_prompt_batch_size", "Requested batch size sum/count from queued prompts."),
        ("comfyui_prompt_steps", "Requested steps sum/count from queued prompts."),
        ("comfyui_prompt_cfg", "Requested CFG sum/count from queued prompts."),
        ("comfyui_prompt_loras", "LoRA loader count sum/count from queued prompts."),
        ("comfyui_prompt_controlnets", "ControlNet count sum/count from queued prompts."),
    ]:
        lines.append(f"# HELP {base}_sum {help_text}")
        lines.append(f"# TYPE {base}_sum counter")
        lines.append(_prom_line(f"{base}_sum", counters.get(f"{base}_sum", 0.0)))

        lines.append(f"# HELP {base}_count Count for {base}.")
        lines.append(f"# TYPE {base}_count counter")
        lines.append(_prom_line(f"{base}_count", counters.get(f"{base}_count", 0)))

    # Discrete workload counters
    lines.append("# HELP comfyui_sampler_requests_total Number of prompts observed by sampler_name.")
    lines.append("# TYPE comfyui_sampler_requests_total counter")
    for s, v in sampler_by.items():
        lines.append(_prom_line("comfyui_sampler_requests_total", v, {"sampler": s}))

    lines.append("# HELP comfyui_scheduler_requests_total Number of prompts observed by scheduler.")
    lines.append("# TYPE comfyui_scheduler_requests_total counter")
    for s, v in scheduler_by.items():
        lines.append(_prom_line("comfyui_scheduler_requests_total", v, {"scheduler": s}))

    lines.append("# HELP comfyui_model_family_requests_total Number of prompts observed by coarse model family.")
    lines.append("# TYPE comfyui_model_family_requests_total counter")
    for f, v in fam_by.items():
        lines.append(_prom_line("comfyui_model_family_requests_total", v, {"family": f}))

    # Model usage
    lines.append("# HELP comfyui_checkpoint_requests_total Number of prompts observed by checkpoint name.")
    lines.append("# TYPE comfyui_checkpoint_requests_total counter")
    for ckpt, v in ckpt_by.items():
        lines.append(_prom_line("comfyui_checkpoint_requests_total", v, {"ckpt": ckpt}))

    lines.append("# HELP comfyui_lora_requests_total Number of prompts observed by LoRA name.")
    lines.append("# TYPE comfyui_lora_requests_total counter")
    for lora, v in lora_by.items():
        lines.append(_prom_line("comfyui_lora_requests_total", v, {"lora": lora}))

    lines.append("# HELP comfyui_controlnet_requests_total Number of prompts observed by ControlNet model name.")
    lines.append("# TYPE comfyui_controlnet_requests_total counter")
    for m, v in controlnet_by.items():
        lines.append(_prom_line("comfyui_controlnet_requests_total", v, {"model": m}))

    # Failure classification
    lines.append("# HELP comfyui_prompt_failures_total Prompt failures observed by reason.")
    lines.append("# TYPE comfyui_prompt_failures_total counter")
    for r, v in failures_by_reason.items():
        lines.append(_prom_line("comfyui_prompt_failures_total", v, {"reason": r}))

    lines.append("# HELP comfyui_prompt_oom_failures_total Prompt failures classified as out-of-memory.")
    lines.append("# TYPE comfyui_prompt_oom_failures_total counter")
    lines.append(_prom_line("comfyui_prompt_oom_failures_total", counters.get("comfyui_prompt_oom_failures_total", 0)))

    lines.append("# HELP comfyui_prompt_cuda_failures_total Prompt failures classified as CUDA/runtime errors (non-OOM).")
    lines.append("# TYPE comfyui_prompt_cuda_failures_total counter")
    lines.append(_prom_line("comfyui_prompt_cuda_failures_total", counters.get("comfyui_prompt_cuda_failures_total", 0)))

    lines.append("# HELP comfyui_prompt_model_load_failures_total Prompt failures classified as model/checkpoint load errors.")
    lines.append("# TYPE comfyui_prompt_model_load_failures_total counter")
    lines.append(_prom_line("comfyui_prompt_model_load_failures_total", counters.get("comfyui_prompt_model_load_failures_total", 0)))

    lines.append("# HELP comfyui_prompt_graph_failures_total Prompt failures classified as graph/workflow validation errors.")
    lines.append("# TYPE comfyui_prompt_graph_failures_total counter")
    lines.append(_prom_line("comfyui_prompt_graph_failures_total", counters.get("comfyui_prompt_graph_failures_total", 0)))

    lines.append("# HELP comfyui_last_failure_reason Last observed failure reason (emitted as label).")
    lines.append("# TYPE comfyui_last_failure_reason gauge")
    if gauges.get("comfyui_last_failure_reason") is not None:
        lines.append(_prom_line("comfyui_last_failure_reason", 1, {"reason": gauges["comfyui_last_failure_reason"]}))

    # Output volume
    lines.append("# HELP comfyui_images_generated_total Total images observed in executed outputs, by type.")
    lines.append("# TYPE comfyui_images_generated_total counter")
    for t, v in images_by_type.items():
        lines.append(_prom_line("comfyui_images_generated_total", v, {"type": t}))

    lines.append("# HELP comfyui_output_bytes_total Total bytes observed for executed output images (best-effort file size), by type.")
    lines.append("# TYPE comfyui_output_bytes_total counter")
    for t, v in bytes_by_type.items():
        lines.append(_prom_line("comfyui_output_bytes_total", v, {"type": t}))

    lines.append("# HELP comfyui_last_prompt_images Images observed for the last completed prompt.")
    lines.append("# TYPE comfyui_last_prompt_images gauge")
    if gauges.get("comfyui_last_prompt_images") is not None:
        lines.append(_prom_line("comfyui_last_prompt_images", gauges["comfyui_last_prompt_images"]))

    lines.append("# HELP comfyui_last_prompt_output_bytes Bytes observed for the last completed prompt outputs (best-effort).")
    lines.append("# TYPE comfyui_last_prompt_output_bytes gauge")
    if gauges.get("comfyui_last_prompt_output_bytes") is not None:
        lines.append(_prom_line("comfyui_last_prompt_output_bytes", gauges["comfyui_last_prompt_output_bytes"]))

    # Latency breakdown: queue wait, execution, end-to-end
    lines.append("# HELP comfyui_prompt_queue_wait_seconds_sum Sum of queue wait time (enqueue -> execution_start), seconds.")
    lines.append("# TYPE comfyui_prompt_queue_wait_seconds_sum counter")
    lines.append(_prom_line("comfyui_prompt_queue_wait_seconds_sum", counters["comfyui_prompt_queue_wait_seconds_sum"]))

    lines.append("# HELP comfyui_prompt_queue_wait_seconds_count Count of queue wait observations.")
    lines.append("# TYPE comfyui_prompt_queue_wait_seconds_count counter")
    lines.append(_prom_line("comfyui_prompt_queue_wait_seconds_count", counters["comfyui_prompt_queue_wait_seconds_count"]))

    lines.append("# HELP comfyui_prompt_queue_wait_last_seconds Last observed queue wait time, seconds.")
    lines.append("# TYPE comfyui_prompt_queue_wait_last_seconds gauge")
    if gauges["comfyui_prompt_queue_wait_last_seconds"] is not None:
        lines.append(_prom_line("comfyui_prompt_queue_wait_last_seconds", gauges["comfyui_prompt_queue_wait_last_seconds"]))

    lines.append("# HELP comfyui_prompt_queue_wait_max_seconds Max observed queue wait time, seconds.")
    lines.append("# TYPE comfyui_prompt_queue_wait_max_seconds gauge")
    lines.append(_prom_line("comfyui_prompt_queue_wait_max_seconds", gauges["comfyui_prompt_queue_wait_max_seconds"]))

    lines.append("# HELP comfyui_prompt_execution_seconds_sum Sum of execution time (execution_start -> completion), seconds.")
    lines.append("# TYPE comfyui_prompt_execution_seconds_sum counter")
    lines.append(_prom_line("comfyui_prompt_execution_seconds_sum", counters["comfyui_prompt_execution_seconds_sum"]))

    lines.append("# HELP comfyui_prompt_execution_seconds_count Count of execution time observations.")
    lines.append("# TYPE comfyui_prompt_execution_seconds_count counter")
    lines.append(_prom_line("comfyui_prompt_execution_seconds_count", counters["comfyui_prompt_execution_seconds_count"]))

    lines.append("# HELP comfyui_prompt_end_to_end_seconds_sum Sum of end-to-end time (enqueue -> completion), seconds.")
    lines.append("# TYPE comfyui_prompt_end_to_end_seconds_sum counter")
    lines.append(_prom_line("comfyui_prompt_end_to_end_seconds_sum", counters["comfyui_prompt_end_to_end_seconds_sum"]))

    lines.append("# HELP comfyui_prompt_end_to_end_seconds_count Count of end-to-end time observations.")
    lines.append("# TYPE comfyui_prompt_end_to_end_seconds_count counter")
    lines.append(_prom_line("comfyui_prompt_end_to_end_seconds_count", counters["comfyui_prompt_end_to_end_seconds_count"]))

    lines.append("# HELP comfyui_prompt_end_to_end_last_seconds Last observed end-to-end time, seconds.")
    lines.append("# TYPE comfyui_prompt_end_to_end_last_seconds gauge")
    if gauges["comfyui_prompt_end_to_end_last_seconds"] is not None:
        lines.append(_prom_line("comfyui_prompt_end_to_end_last_seconds", gauges["comfyui_prompt_end_to_end_last_seconds"]))

    lines.append("# HELP comfyui_prompt_end_to_end_max_seconds Max observed end-to-end time, seconds.")
    lines.append("# TYPE comfyui_prompt_end_to_end_max_seconds gauge")
    lines.append(_prom_line("comfyui_prompt_end_to_end_max_seconds", gauges["comfyui_prompt_end_to_end_max_seconds"]))

    # Existing WS message & progress metrics
    lines.append("# HELP comfyui_ws_messages_total Total WS messages sent by ComfyUI, by message type.")
    lines.append("# TYPE comfyui_ws_messages_total counter")
    for t, v in ws_by_type.items():
        lines.append(_prom_line("comfyui_ws_messages_total", v, {"type": t}))

    lines.append("# HELP comfyui_ws_progress_events_total Total progress events observed.")
    lines.append("# TYPE comfyui_ws_progress_events_total counter")
    lines.append(_prom_line("comfyui_ws_progress_events_total", counters["comfyui_ws_progress_events_total"]))

    lines.append("# HELP comfyui_ws_execution_cached_nodes_total Total nodes reported as cached.")
    lines.append("# TYPE comfyui_ws_execution_cached_nodes_total counter")
    lines.append(_prom_line("comfyui_ws_execution_cached_nodes_total", counters["comfyui_ws_execution_cached_nodes_total"]))

    lines.append("# HELP comfyui_queue_remaining Queue remaining as reported in WS status.")
    lines.append("# TYPE comfyui_queue_remaining gauge")
    if gauges["comfyui_queue_remaining"] is not None:
        lines.append(_prom_line("comfyui_queue_remaining", gauges["comfyui_queue_remaining"]))

    lines.append("# HELP comfyui_prompt_in_flight Whether a prompt is currently executing (1/0).")
    lines.append("# TYPE comfyui_prompt_in_flight gauge")
    lines.append(_prom_line("comfyui_prompt_in_flight", gauges["comfyui_prompt_in_flight"]))

    lines.append("# HELP comfyui_node_in_flight Whether a node is currently executing (1/0).")
    lines.append("# TYPE comfyui_node_in_flight gauge")
    lines.append(_prom_line("comfyui_node_in_flight", gauges["comfyui_node_in_flight"]))

    lines.append("# HELP comfyui_progress_value Current progress value.")
    lines.append("# TYPE comfyui_progress_value gauge")
    if gauges["comfyui_progress_value"] is not None:
        lines.append(_prom_line("comfyui_progress_value", gauges["comfyui_progress_value"]))

    lines.append("# HELP comfyui_progress_max Current progress max.")
    lines.append("# TYPE comfyui_progress_max gauge")
    if gauges["comfyui_progress_max"] is not None:
        lines.append(_prom_line("comfyui_progress_max", gauges["comfyui_progress_max"]))

    lines.append("# HELP comfyui_progress_ratio Current progress ratio (0..1).")
    lines.append("# TYPE comfyui_progress_ratio gauge")
    if gauges["comfyui_progress_ratio"] is not None:
        lines.append(_prom_line("comfyui_progress_ratio", gauges["comfyui_progress_ratio"]))

    lines.append("# HELP comfyui_prompt_executions_total Prompt executions observed (best-effort).")
    lines.append("# TYPE comfyui_prompt_executions_total counter")
    lines.append(_prom_line("comfyui_prompt_executions_total", counters.get("comfyui_prompt_executions_total", 0)))

    lines.append("# HELP comfyui_prompt_execution_failures_total Prompt executions that ended in error/interruption (best-effort).")
    lines.append("# TYPE comfyui_prompt_execution_failures_total counter")
    lines.append(_prom_line("comfyui_prompt_execution_failures_total", counters.get("comfyui_prompt_execution_failures_total", 0)))

    lines.append("# HELP comfyui_prompt_completions_total Prompts observed as completed.")
    lines.append("# TYPE comfyui_prompt_completions_total counter")
    lines.append(_prom_line("comfyui_prompt_completions_total", counters["comfyui_prompt_completions_total"]))

    lines.append("# HELP comfyui_prompt_execution_last_seconds Execution duration of last completed prompt, seconds.")
    lines.append("# TYPE comfyui_prompt_execution_last_seconds gauge")
    if hist["comfyui_prompt_execution_last_seconds"] is not None:
        lines.append(_prom_line("comfyui_prompt_execution_last_seconds", hist["comfyui_prompt_execution_last_seconds"]))

    lines.append("# HELP comfyui_prompt_execution_max_seconds Max execution duration observed, seconds.")
    lines.append("# TYPE comfyui_prompt_execution_max_seconds gauge")
    lines.append(_prom_line("comfyui_prompt_execution_max_seconds", hist["comfyui_prompt_execution_max_seconds"]))

    lines.append("# HELP comfyui_prompt_execution_seconds_sum_legacy (Legacy) Sum of execution durations observed, seconds.")
    lines.append("# TYPE comfyui_prompt_execution_seconds_sum_legacy counter")
    lines.append(_prom_line("comfyui_prompt_execution_seconds_sum_legacy", hist["comfyui_prompt_execution_seconds_sum"]))

    lines.append("# HELP comfyui_node_executions_total Node executions observed (via executing transitions).")
    lines.append("# TYPE comfyui_node_executions_total counter")
    lines.append(_prom_line("comfyui_node_executions_total", counters["comfyui_node_executions_total"]))

    lines.append("# HELP comfyui_node_execution_last_seconds Duration of last observed node execution.")
    lines.append("# TYPE comfyui_node_execution_last_seconds gauge")
    if hist["comfyui_node_execution_last_seconds"] is not None:
        lines.append(_prom_line("comfyui_node_execution_last_seconds", hist["comfyui_node_execution_last_seconds"]))

    lines.append("# HELP comfyui_node_execution_max_seconds Max node duration observed.")
    lines.append("# TYPE comfyui_node_execution_max_seconds gauge")
    lines.append(_prom_line("comfyui_node_execution_max_seconds", hist["comfyui_node_execution_max_seconds"]))

    lines.append("# HELP comfyui_node_execution_seconds_sum Sum of observed node execution durations.")
    lines.append("# TYPE comfyui_node_execution_seconds_sum counter")
    lines.append(_prom_line("comfyui_node_execution_seconds_sum", hist["comfyui_node_execution_seconds_sum"]))

    # Node-type timing
    lines.append("# HELP comfyui_node_duration_seconds_sum Sum of observed node execution durations, by class_type.")
    lines.append("# TYPE comfyui_node_duration_seconds_sum counter")
    for ct, v in dict(_counters["comfyui_node_duration_seconds_sum"]).items():
        lines.append(_prom_line("comfyui_node_duration_seconds_sum", v, {"class_type": ct}))

    lines.append("# HELP comfyui_node_duration_seconds_count Count of observed node execution durations, by class_type.")
    lines.append("# TYPE comfyui_node_duration_seconds_count counter")
    for ct, v in dict(_counters["comfyui_node_duration_seconds_count"]).items():
        lines.append(_prom_line("comfyui_node_duration_seconds_count", v, {"class_type": ct}))

    lines.append("# HELP comfyui_node_failures_total Node failures observed, by class_type (best-effort).")
    lines.append("# TYPE comfyui_node_failures_total counter")
    for ct, v in dict(_counters["comfyui_node_failures_total"]).items():
        lines.append(_prom_line("comfyui_node_failures_total", v, {"class_type": ct}))

    # Per-run peak GPU/VRAM (last completed prompt)
    lines.append("# HELP comfyui_prompt_cuda_max_memory_allocated_bytes Torch max_memory_allocated captured per prompt (reset at execution_start, recorded at completion).")
    lines.append("# TYPE comfyui_prompt_cuda_max_memory_allocated_bytes gauge")
    if gauges.get("comfyui_prompt_cuda_max_memory_allocated_bytes") is not None:
        lines.append(_prom_line("comfyui_prompt_cuda_max_memory_allocated_bytes", gauges["comfyui_prompt_cuda_max_memory_allocated_bytes"]))

    lines.append("# HELP comfyui_prompt_cuda_max_memory_reserved_bytes Torch max_memory_reserved captured per prompt (reset at execution_start, recorded at completion).")
    lines.append("# TYPE comfyui_prompt_cuda_max_memory_reserved_bytes gauge")
    if gauges.get("comfyui_prompt_cuda_max_memory_reserved_bytes") is not None:
        lines.append(_prom_line("comfyui_prompt_cuda_max_memory_reserved_bytes", gauges["comfyui_prompt_cuda_max_memory_reserved_bytes"]))

    lines.append("# HELP comfyui_prompt_nvml_vram_peak_bytes NVML peak VRAM used captured per prompt (best-effort), by GPU index.")
    lines.append("# TYPE comfyui_prompt_nvml_vram_peak_bytes gauge")
    peaks = gauges.get("comfyui_prompt_nvml_vram_peak_by_gpu_bytes")
    if isinstance(peaks, dict):
        for gpu, used in peaks.items():
            if used is None:
                continue
            lines.append(_prom_line("comfyui_prompt_nvml_vram_peak_bytes", int(used), {"gpu": str(gpu)}))

    # Concurrency & saturation
    lines.append("# HELP comfyui_active_prompts Number of prompts currently executing (best-effort; 0/1 on single-worker ComfyUI).")
    lines.append("# TYPE comfyui_active_prompts gauge")
    lines.append(_prom_line("comfyui_active_prompts", int(gauges.get("comfyui_active_prompts") or 0)))

    lines.append("# HELP comfyui_active_nodes Whether a node is currently executing (best-effort; 0/1).")
    lines.append("# TYPE comfyui_active_nodes gauge")
    lines.append(_prom_line("comfyui_active_nodes", int(gauges.get("comfyui_active_nodes") or 0)))

    lines.append("# HELP comfyui_queue_age_seconds Age of the oldest queued (not yet executing) prompt in seconds.")
    lines.append("# TYPE comfyui_queue_age_seconds gauge")
    if gauges.get("comfyui_queue_age_seconds") is not None:
        lines.append(_prom_line("comfyui_queue_age_seconds", float(gauges["comfyui_queue_age_seconds"])))

    lines.append("# HELP comfyui_queue_throughput_prompts_per_min Prompt completions per minute (scrape-to-scrape estimate).")
    lines.append("# TYPE comfyui_queue_throughput_prompts_per_min gauge")
    if gauges.get("comfyui_queue_throughput_prompts_per_min") is not None:
        lines.append(_prom_line("comfyui_queue_throughput_prompts_per_min", float(gauges["comfyui_queue_throughput_prompts_per_min"])))

    # Process
    p = c.get("proc") or {}
    if "rss_bytes" in p:
        lines.append("# HELP process_resident_memory_bytes Resident memory size in bytes.")
        lines.append("# TYPE process_resident_memory_bytes gauge")
        lines.append(_prom_line("process_resident_memory_bytes", p["rss_bytes"]))
    if "cpu_seconds_total" in p:
        lines.append("# HELP process_cpu_seconds_total Total user+system CPU time in seconds.")
        lines.append("# TYPE process_cpu_seconds_total counter")
        lines.append(_prom_line("process_cpu_seconds_total", p["cpu_seconds_total"]))
    if "num_threads" in p:
        lines.append("# HELP process_threads Number of process threads.")
        lines.append("# TYPE process_threads gauge")
        lines.append(_prom_line("process_threads", p["num_threads"]))

    # System (psutil if present)
    s = c.get("sys") or {}
    if "mem_total_bytes" in s:
        lines.append("# HELP node_memory_total_bytes Total system memory in bytes.")
        lines.append("# TYPE node_memory_total_bytes gauge")
        lines.append(_prom_line("node_memory_total_bytes", s["mem_total_bytes"]))
    if "mem_available_bytes" in s:
        lines.append("# HELP node_memory_available_bytes Available system memory in bytes.")
        lines.append("# TYPE node_memory_available_bytes gauge")
        lines.append(_prom_line("node_memory_available_bytes", s["mem_available_bytes"]))
    if "mem_used_bytes" in s:
        lines.append("# HELP node_memory_used_bytes Used system memory in bytes.")
        lines.append("# TYPE node_memory_used_bytes gauge")
        lines.append(_prom_line("node_memory_used_bytes", s["mem_used_bytes"]))
    if "cpu_count_logical" in s:
        lines.append("# HELP node_cpu_count_logical Logical CPU count.")
        lines.append("# TYPE node_cpu_count_logical gauge")
        lines.append(_prom_line("node_cpu_count_logical", s["cpu_count_logical"]))
    if s.get("loadavg_1") is not None:
        lines.append("# HELP node_load1 1-minute load average.")
        lines.append("# TYPE node_load1 gauge")
        lines.append(_prom_line("node_load1", s["loadavg_1"]))

    # Torch/CUDA
    t = c.get("torch") or {}
    lines.append("# HELP comfyui_torch_info Torch version info.")
    lines.append("# TYPE comfyui_torch_info gauge")
    if t.get("torch_version"):
        lines.append(_prom_line("comfyui_torch_info", 1, {"version": t["torch_version"]}))

    lines.append("# HELP comfyui_cuda_available Whether CUDA is available (1/0).")
    lines.append("# TYPE comfyui_cuda_available gauge")
    if "cuda_available" in t:
        lines.append(_prom_line("comfyui_cuda_available", int(t["cuda_available"])))

    if t.get("cuda_available") == 1:
        lines.append("# HELP comfyui_cuda_device_count CUDA device count.")
        lines.append("# TYPE comfyui_cuda_device_count gauge")
        lines.append(_prom_line("comfyui_cuda_device_count", int(t.get("cuda_device_count", 0))))

        if t.get("cuda_device_name") is not None:
            lines.append("# HELP comfyui_cuda_device_info Current CUDA device info.")
            lines.append("# TYPE comfyui_cuda_device_info gauge")
            lines.append(_prom_line(
                "comfyui_cuda_device_info",
                1,
                {"index": str(t.get("cuda_current_device", 0)), "name": t["cuda_device_name"]},
            ))

        for k, help_text in [
            ("cuda_mem_allocated_bytes", "Torch allocated CUDA memory in bytes."),
            ("cuda_mem_reserved_bytes", "Torch reserved CUDA memory in bytes."),
            ("cuda_max_mem_allocated_bytes", "Torch max allocated CUDA memory in bytes."),
            ("cuda_max_mem_reserved_bytes", "Torch max reserved CUDA memory in bytes."),
        ]:
            if k in t:
                metric = "comfyui_" + k
                lines.append(f"# HELP {metric} {help_text}")
                lines.append(f"# TYPE {metric} gauge")
                lines.append(_prom_line(metric, t[k]))

    # NVML GPU (if available)
    gpus = c.get("nvml_gpu")
    if gpus:
        lines.append("# HELP comfyui_gpu_memory_used_bytes GPU memory used in bytes.")
        lines.append("# TYPE comfyui_gpu_memory_used_bytes gauge")
        lines.append("# HELP comfyui_gpu_memory_total_bytes GPU memory total in bytes.")
        lines.append("# TYPE comfyui_gpu_memory_total_bytes gauge")
        lines.append("# HELP comfyui_gpu_utilization_percent GPU utilization percent.")
        lines.append("# TYPE comfyui_gpu_utilization_percent gauge")
        lines.append("# HELP comfyui_gpu_temperature_c GPU temperature in Celsius.")
        lines.append("# TYPE comfyui_gpu_temperature_c gauge")
        for g in gpus:
            labels = {"gpu": str(g["index"]), "name": g["name"]}
            lines.append(_prom_line("comfyui_gpu_memory_used_bytes", g["mem_used"], labels))
            lines.append(_prom_line("comfyui_gpu_memory_total_bytes", g["mem_total"], labels))
            lines.append(_prom_line("comfyui_gpu_utilization_percent", g["gpu_util"], labels))
            if g.get("temp_c") is not None:
                lines.append(_prom_line("comfyui_gpu_temperature_c", g["temp_c"], labels))

    # /api/system_stats (RAM + VRAM + versions)
    sysstats = c.get("system_stats")
    if isinstance(sysstats, dict):
        system = sysstats.get("system") if isinstance(sysstats.get("system"), dict) else {}
        devices = sysstats.get("devices") if isinstance(sysstats.get("devices"), list) else []

        comfyui_version = str(system.get("comfyui_version", "unknown"))
        python_version = str(system.get("python_version", "unknown"))
        pytorch_version = str(system.get("pytorch_version", "unknown"))
        os_name = str(system.get("os", "unknown"))

        lines.append("# HELP comfyui_system_info System/version info from /api/system_stats.")
        lines.append("# TYPE comfyui_system_info gauge")
        lines.append(_prom_line(
            "comfyui_system_info",
            1,
            {
                "os": os_name,
                "python_version": python_version,
                "comfyui_version": comfyui_version,
                "pytorch_version": pytorch_version,
            },
        ))

        rt = _mb_to_bytes(system.get("ram_total"))
        rf = _mb_to_bytes(system.get("ram_free"))

        if rt is not None:
            lines.append("# HELP comfyui_system_ram_total_bytes Total RAM (bytes) from /api/system_stats.")
            lines.append("# TYPE comfyui_system_ram_total_bytes gauge")
            lines.append(_prom_line("comfyui_system_ram_total_bytes", rt))

        if rf is not None:
            lines.append("# HELP comfyui_system_ram_free_bytes Free RAM (bytes) from /api/system_stats.")
            lines.append("# TYPE comfyui_system_ram_free_bytes gauge")
            lines.append(_prom_line("comfyui_system_ram_free_bytes", rf))

        lines.append("# HELP comfyui_device_vram_total_bytes Device VRAM total (bytes) from /api/system_stats.")
        lines.append("# TYPE comfyui_device_vram_total_bytes gauge")
        lines.append("# HELP comfyui_device_vram_free_bytes Device VRAM free (bytes) from /api/system_stats.")
        lines.append("# TYPE comfyui_device_vram_free_bytes gauge")

        for d in devices:
            if not isinstance(d, dict):
                continue
            name = str(d.get("name", "unknown"))
            dtype = str(d.get("type", "unknown"))
            vt = _mb_to_bytes(d.get("vram_total"))
            vf = _mb_to_bytes(d.get("vram_free"))
            labels = {"type": dtype, "name": name}
            if vt is not None:
                lines.append(_prom_line("comfyui_device_vram_total_bytes", vt, labels))
            if vf is not None:
                lines.append(_prom_line("comfyui_device_vram_free_bytes", vf, labels))

    return "\n".join(lines) + "\n"


def setup_metrics_endpoint():
    try:
        import server

        ps = getattr(server.PromptServer, "instance", None)
        if ps is None:
            _log("warn", "PromptServer.instance not available; /metrics not registered.")
            return

        _log("info", f"Loading exporter v{EXPORTER_VERSION} from {__file__}")
        _log("info", f"Attempting to register metrics route at {_METRICS_PATH}")

        try:
            if _route_exists_aiohttp(ps, _METRICS_PATH):
                _log("info", f"Metrics route already exists at {_METRICS_PATH}")
                return
        except Exception as e:
            _log("warn", "Route existence check failed; attempting to register anyway.", exc=e)

        async def metrics(request):
            global _last_request_base
            try:
                # request.host includes port if non-default
                _last_request_base = f"{request.scheme}://{request.host}"
            except Exception:
                pass

            body = _render_metrics()
            return web.Response(
                text=body,
                headers={"Content-Type": "text/plain; version=0.0.4; charset=utf-8"},
            )

        try:
            ps.routes.get(_METRICS_PATH)(metrics)
            _log("info", f"Registered {_METRICS_PATH}")
        except Exception as e:
            _log("error", f"Failed to register metrics endpoint at {_METRICS_PATH}", exc=e)

    except Exception as e:
        _log("error", "Failed to set up metrics endpoint (unexpected).", exc=e)


def try_install_instrumentation():
    global _instrumented
    if _instrumented:
        return

    # Wrap PromptQueue.put to capture enqueue timestamp + workload
    try:
        import execution  # type: ignore

        PQ = getattr(execution, "PromptQueue", None)
        if PQ is not None:
            for fn_name in ("put", "put_nowait"):
                fn = getattr(PQ, fn_name, None)
                if fn and callable(fn) and not getattr(fn, "__comfy_prom_wrapped__", False):

                    def _make_pq_wrapper(orig):
                        def wrapped(self, item, *args, **kwargs):
                            try:
                                pid = None
                                if isinstance(item, (tuple, list)) and len(item) >= 2:
                                    pid = item[1]
                                if isinstance(pid, str) and pid:
                                    _record_enqueue(pid, time.time())

                                    wl = _extract_workload_from_item(item)
                                    if wl:
                                        with _lock:
                                            _state["workload_by_id"][pid] = wl

                                            # Update workload sum/count counters
                                            if "width" in wl:
                                                _counters["comfyui_prompt_requested_width_sum"] += wl["width"]
                                                _counters["comfyui_prompt_requested_width_count"] += 1
                                            if "height" in wl:
                                                _counters["comfyui_prompt_requested_height_sum"] += wl["height"]
                                                _counters["comfyui_prompt_requested_height_count"] += 1
                                            if "pixels" in wl:
                                                _counters["comfyui_prompt_requested_pixels_sum"] += wl["pixels"]
                                                _counters["comfyui_prompt_requested_pixels_count"] += 1
                                            if "batch" in wl:
                                                _counters["comfyui_prompt_batch_size_sum"] += wl["batch"]
                                                _counters["comfyui_prompt_batch_size_count"] += 1
                                            if "steps" in wl:
                                                _counters["comfyui_prompt_steps_sum"] += wl["steps"]
                                                _counters["comfyui_prompt_steps_count"] += 1
                                            if "cfg" in wl:
                                                _counters["comfyui_prompt_cfg_sum"] += wl["cfg"]
                                                _counters["comfyui_prompt_cfg_count"] += 1
                                            if "loras" in wl:
                                                _counters["comfyui_prompt_loras_sum"] += wl["loras"]
                                                _counters["comfyui_prompt_loras_count"] += 1
                                            if "controlnets" in wl:
                                                _counters["comfyui_prompt_controlnets_sum"] += wl["controlnets"]
                                                _counters["comfyui_prompt_controlnets_count"] += 1
                                            if "sampler" in wl:
                                                _inc_by_label(_counters["comfyui_sampler_requests_total"], wl["sampler"])
                                            if "scheduler" in wl:
                                                _inc_by_label(_counters["comfyui_scheduler_requests_total"], wl["scheduler"])
                                            if "model_family" in wl:
                                                _inc_by_label(_counters["comfyui_model_family_requests_total"], wl["model_family"])

                                    node_map = _extract_node_class_map_from_item(item)
                                    if node_map:
                                        with _lock:
                                            _state["node_class_by_prompt"][pid] = node_map

                                    usage = _extract_model_usage_from_item(item)
                                    ckpts = usage.get("ckpts")
                                    loras = usage.get("loras")
                                    cns = usage.get("controlnets")
                                    with _lock:
                                        if isinstance(ckpts, set):
                                            for ckpt in ckpts:
                                                _inc_by_label(_counters["comfyui_checkpoint_requests_total"], ckpt, 1)
                                        if isinstance(loras, set):
                                            for lora in loras:
                                                _inc_by_label(_counters["comfyui_lora_requests_total"], lora, 1)
                                        if isinstance(cns, set):
                                            for cn in cns:
                                                _inc_by_label(_counters["comfyui_controlnet_requests_total"], cn, 1)

                            except Exception as e:
                                _log_rate_limited("pq_put_parse_fail", "warn", "Failed parsing prompt workload on enqueue.", exc=e)
                            return orig(self, item, *args, **kwargs)

                        wrapped.__comfy_prom_wrapped__ = True
                        return wrapped

                    setattr(PQ, fn_name, _make_pq_wrapper(fn))

    except Exception:
        pass

    # A) Wrap PromptExecutor (coarse fallback)
    try:
        import execution  # type: ignore

        cls = getattr(execution, "PromptExecutor", None)
        if cls:
            for fn_name in ("execute", "execute_prompt", "run"):
                fn = getattr(cls, fn_name, None)
                if fn and callable(fn) and not getattr(fn, "__comfy_prom_wrapped__", False):

                    def _make_wrapper(orig):
                        def wrapped(self, *args, **kwargs):
                            t0 = time.time()
                            ok = False
                            try:
                                res = orig(self, *args, **kwargs)
                                ok = True
                                return res
                            finally:
                                dt = time.time() - t0
                                with _lock:
                                    _counters["comfyui_prompt_executions_total"] += 1
                                    if not ok:
                                        _counters["comfyui_prompt_execution_failures_total"] += 1

                        wrapped.__comfy_prom_wrapped__ = True
                        return wrapped

                    setattr(cls, fn_name, _make_wrapper(fn))
                    break
    except Exception:
        pass

    # B) Wrap PromptServer.send_sync to observe WS events
    try:
        import server

        PS = getattr(server, "PromptServer", None)
        if PS is None:
            _instrumented = True
            return

        orig = getattr(PS, "send_sync", None)
        if orig and callable(orig) and not getattr(orig, "__comfy_prom_wrapped__", False):

            def send_sync_wrapped(self, message_type, data, *args, **kwargs):
                try:
                    if isinstance(message_type, str) and isinstance(data, dict):
                        _handle_ws_event(message_type, data)
                except Exception as e:
                    _log_rate_limited("send_sync_handle_fail", "warn", "Failed handling WS event for metrics.", exc=e)
                return orig(self, message_type, data, *args, **kwargs)

            send_sync_wrapped.__comfy_prom_wrapped__ = True
            setattr(PS, "send_sync", send_sync_wrapped)

        _instrumented = True
        _log("info", "Instrumentation installed (PromptQueue + PromptExecutor + send_sync).")
    except Exception as e:
        _log("error", "Failed to install instrumentation.", exc=e)
        _instrumented = True