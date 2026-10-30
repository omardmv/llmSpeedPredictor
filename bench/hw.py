"""What the machine is and what it is doing right now (no sudo needed).

On Apple Silicon there is no separate VRAM: CPU and GPU share one pool of
"unified memory". The GPU numbers below come from the GPU driver via `ioreg`.
"""

import re
import subprocess
import threading
import time

from . import ollama

# Memory bandwidth from Apple's spec sheets (GB/s). There is no API for it.
BANDWIDTH_GBS = {
    "Apple M1": 68, "Apple M1 Pro": 200, "Apple M1 Max": 400, "Apple M1 Ultra": 800,
    "Apple M2": 100, "Apple M2 Pro": 200, "Apple M2 Max": 400, "Apple M2 Ultra": 800,
    "Apple M3": 100, "Apple M3 Pro": 150, "Apple M3 Max": 400,
    "Apple M4": 120, "Apple M4 Pro": 273, "Apple M4 Max": 546,
}

GB = 1024 ** 3


def _run(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def _sysctl(name):
    return _run("sysctl", "-n", name)


def machine():
    chip = _sysctl("machdep.cpu.brand_string")
    gpu_cores = re.search(r'"gpu-core-count" = (\d+)', _run("ioreg", "-r", "-d", "1", "-c", "IOAccelerator"))
    return {
        "chip": chip,
        "macos": _run("sw_vers", "-productVersion"),
        "cpu_performance_cores": int(_sysctl("hw.perflevel0.physicalcpu") or 0),
        "cpu_efficiency_cores": int(_sysctl("hw.perflevel1.physicalcpu") or 0),
        "gpu_cores": int(gpu_cores.group(1)) if gpu_cores else None,
        "unified_memory_bytes": int(_sysctl("hw.memsize")),
        "bandwidth_gbs": BANDWIDTH_GBS.get(chip),
        # 0 = macOS default: the GPU may use roughly 2/3-3/4 of unified memory.
        "gpu_wired_limit_mb": int(_sysctl("iogpu.wired_limit_mb") or 0),
    }


def gpu_now():
    """GPU busy % and how much memory the GPU is holding right now."""
    out = _run("ioreg", "-r", "-d", "1", "-c", "IOAccelerator")
    stats = {}
    for key, name in [("Device Utilization %", "gpu_util"),
                      ("Renderer Utilization %", "renderer_util"),
                      ("In use system memory", "gpu_mem_in_use"),
                      ("Alloc system memory", "gpu_mem_alloc")]:
        m = re.search(rf'"{re.escape(key)}"=(\d+)', out)
        stats[name] = int(m.group(1)) if m else 0
    return stats


def ram_now():
    """Unified memory split the way Activity Monitor shows it."""
    out = _run("vm_stat")
    page = int(re.search(r"page size of (\d+)", out).group(1))
    pages = {k.strip(): int(v.strip(" .")) for k, v in re.findall(r"^(.+?):\s+(\d+)\.?$", out, re.M)}
    used = lambda *keys: sum(pages.get(k, 0) for k in keys) * page
    free_pct = re.search(r"(\d+)%", _run("memory_pressure", "-Q"))
    return {
        "app_bytes": used("Anonymous pages") - used("Pages purgeable"),
        "wired_bytes": used("Pages wired down"),
        "compressed_bytes": used("Pages occupied by compressor"),
        "cached_files_bytes": used("File-backed pages"),
        "free_pct": int(free_pct.group(1)) if free_pct else None,
    }


class GpuSampler:
    """Polls the GPU in the background while a benchmark runs.

    with GpuSampler() as gpu:
        ...do work...
    gpu.samples -> [(seconds_since_start, gpu_util, gpu_mem_in_use), ...]
    """

    def __init__(self, interval=0.1):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()

    def __enter__(self):
        self.start = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def _loop(self):
        while not self._stop.is_set():
            s = gpu_now()
            self.samples.append((time.perf_counter() - self.start, s["gpu_util"], s["gpu_mem_in_use"]))
            self._stop.wait(self.interval)

    def summary(self):
        if not self.samples:
            return {"gpu_util_avg": 0, "gpu_util_max": 0, "gpu_mem_max": 0}
        utils = [u for _, u, _ in self.samples]
        return {"gpu_util_avg": sum(utils) / len(utils), "gpu_util_max": max(utils),
                "gpu_mem_max": max(m for _, _, m in self.samples)}


# ---------- printing ----------

def _gb(b):
    return f"{b / GB:5.2f} GB"


def _bar(pct, width=30):
    filled = round(width * pct / 100)
    return "█" * filled + "·" * (width - filled)


def print_machine():
    m = machine()
    total = m["unified_memory_bytes"]
    print("MACHINE")
    print(f"  chip              {m['chip']}  (macOS {m['macos']})")
    print(f"  CPU cores         {m['cpu_performance_cores']} performance + {m['cpu_efficiency_cores']} efficiency")
    print(f"  GPU cores         {m['gpu_cores']}")
    print(f"  unified memory    {_gb(total)}   (shared by CPU and GPU, there is no separate VRAM)")
    limit = (f"{m['gpu_wired_limit_mb'] / 1024:.1f} GB (set by you)" if m["gpu_wired_limit_mb"]
             else "macOS default (about 2/3-3/4 of unified memory)")
    print(f"  GPU memory limit  {limit}")
    bw = f"{m['bandwidth_gbs']} GB/s" if m["bandwidth_gbs"] else "unknown chip, add it to BANDWIDTH_GBS"
    print(f"  memory bandwidth  {bw}   (spec sheet: how many bytes per second can move between memory and chip)")


def print_now():
    g, r = gpu_now(), ram_now()
    print("RIGHT NOW")
    print(f"  GPU busy          {_bar(g['gpu_util'])} {g['gpu_util']:3d}%")
    print(f"  GPU memory        {_gb(g['gpu_mem_in_use'])} in use  ({_gb(g['gpu_mem_alloc'])} allocated)")
    print(f"  RAM: apps         {_gb(r['app_bytes'])}")
    print(f"  RAM: wired        {_gb(r['wired_bytes'])}   (locked in place; GPU buffers live here)")
    print(f"  RAM: compressed   {_gb(r['compressed_bytes'])}")
    print(f"  RAM: file cache   {_gb(r['cached_files_bytes'])}")
    print(f"  memory free       {r['free_pct']}%")
    try:
        models = ollama.loaded()
        print(f"OLLAMA {ollama.version()}")
        if not models:
            print("  no models loaded")
        for mod in models:
            print(f"  {mod['name']:16} {_gb(mod['size'])} total, {_gb(mod['size_vram'])} on GPU, "
                  f"context {mod.get('context_length', '?')}, unloads at {mod['expires_at'][11:19]}")
    except OSError:
        print("OLLAMA  not running (start it with `ollama serve` or the Ollama app)")


def watch(interval=1.0):
    """Live view. Leave it open in a second terminal while you send requests."""
    try:
        while True:
            print("\033[2J\033[H", end="")
            print_now()
            print("\n(ctrl-c to stop)", flush=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
