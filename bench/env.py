"""Is the machine quiet enough to trust a benchmark? And what was it like during the run?

The model shares the GPU, memory bandwidth and cooling with everything else on the Mac.
A browser, a screen recorder or a video call can change the numbers by 10-30%, so we
check before measuring and save the conditions next to every result.
"""

import os
import platform
import re
import subprocess
import time
from datetime import datetime

from . import heat, hw, ollama

# Limits for "quiet". The GPU ones matter most: decode competes for memory bandwidth.
GPU_BUSY_AVG_MAX = 5      # % the GPU may already be busy before we start
GPU_BUSY_PEAK_MAX = 30
CPU_MACHINE_MAX = 25      # % of all cores together
CPU_PROCESS_MAX = 20      # % of one core for any single other process

# Known heavy GPU users. A match is reported even when it is quiet right now.
GPU_HEAVY = ["OBS", "zoom.us", "Teams", "FaceTime", "Final Cut", "Premiere", "DaVinci", "Blender",
             "Steam", "Unity", "Screen Studio", "Loom", "ScreenFlow"]


def _run(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def power():
    batt = _run("pmset", "-g", "batt")
    settings = _run("pmset", "-g")
    m = re.search(r"^\s*(?:lowpowermode|powermode)\s+(\d)", settings, re.M)
    mode = {"0": "automatic", "1": "low power", "2": "high power"}.get(m.group(1) if m else "", "unknown")
    return {"on_ac_power": "AC Power" in batt, "power_mode": mode}


def thermal():
    """macOS only writes a level here once it has started slowing the chip down."""
    out = _run("pmset", "-g", "therm")
    warnings = [line.strip() for line in out.splitlines()
                if line.strip() and not line.strip().startswith("Note: No")]
    return {"throttling": bool(warnings), "detail": warnings}


def open_apps():
    """Names of running apps (from /Something/Name.app/Contents/MacOS/...), not background helpers."""
    names = set()
    for cmd in _run("ps", "-Ao", "comm=").splitlines():
        m = re.search(r"/([^/]+)\.app/Contents/MacOS/", cmd)
        if m:
            names.add(m.group(1))
    return names


def busy_processes():
    """Other processes using CPU right now, busiest first."""
    me = {os.getpid(), os.getppid()}
    rows = []
    for line in _run("ps", "-Ao", "pid=,pcpu=,comm=").splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, cpu, cmd = int(parts[0]), float(parts[1]), parts[2]
        if pid in me or "ollama" in cmd.lower() or "llama-server" in cmd:
            continue
        rows.append((cpu, os.path.basename(cmd.rstrip("/"))))
    rows.sort(reverse=True)
    return rows


def preflight(seconds=5):
    """Measure the machine while nothing of ours runs, and list what could spoil a benchmark."""
    samples = []
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        samples.append(hw.gpu_now()["gpu_util"])
        time.sleep(0.2)
    procs = busy_processes()
    cores = os.cpu_count() or 1
    cpu_total = sum(c for c, _ in procs) / cores
    apps = open_apps()
    heavy = sorted({h for h in GPU_HEAVY for a in apps if a.lower().startswith(h.lower())})

    result = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "gpu_busy_avg": sum(samples) / len(samples),
        "gpu_busy_peak": max(samples),
        "cpu_busy_machine": cpu_total,
        "top_processes": [{"name": n, "cpu": round(c, 1)} for c, n in procs[:5]],
        "gpu_heavy_apps": heavy,
        **power(),
        **{"throttling": thermal()["throttling"], "thermal_detail": thermal()["detail"]},
        "memory_free_pct": hw.ram_now()["free_pct"],
    }
    h = heat.now()
    if h:  # starting temperature: a run that starts hot is not comparable with one that starts cold
        result.update({"gpu_temp": h["gpu_temp"], "cpu_temp": h["cpu_temp"], "fan_rpm": h["fan_rpm"],
                       "swap_used_gb": h["swap_used"] / 1024 ** 3})

    problems = []
    if result["gpu_busy_avg"] > GPU_BUSY_AVG_MAX or result["gpu_busy_peak"] > GPU_BUSY_PEAK_MAX:
        problems.append(f"GPU already busy: avg {result['gpu_busy_avg']:.0f}%, peak {result['gpu_busy_peak']}% "
                        f"(want avg <= {GPU_BUSY_AVG_MAX}%)")
    if cpu_total > CPU_MACHINE_MAX:
        problems.append(f"CPU {cpu_total:.0f}% busy across all cores (want <= {CPU_MACHINE_MAX}%)")
    for c, n in procs:
        if c > CPU_PROCESS_MAX:
            problems.append(f"{n} is using {c:.0f}% CPU")
    if heavy:
        problems.append("GPU-heavy apps open: " + ", ".join(heavy))
    if not result["on_ac_power"]:
        problems.append("on battery: macOS may lower GPU speed to save power")
    if result["power_mode"] == "low power":
        problems.append("Low Power Mode is on")
    if result["throttling"]:
        problems.append("the chip is being slowed down for heat: " + "; ".join(result["thermal_detail"]))
    if result["memory_free_pct"] is not None and result["memory_free_pct"] < 20:
        problems.append(f"only {result['memory_free_pct']}% memory free: the system may swap")
    # Two servers holding models split memory, and if both work they split the bandwidth too.
    try:
        in_ollama = [m["name"] for m in ollama.loaded()]
    except OSError:
        in_ollama = []
    own_llama = [line for line in _run("ps", "-Ao", "command=").splitlines()
                 if "llama-server" in line and "Ollama.app" not in line and "--port" in line
                 and "-ngl 0" not in line]  # -ngl 0 = CPU only (the app's tokenizer), no GPU use
    result["ollama_loaded"] = in_ollama
    result["llama_servers"] = len(own_llama)
    if in_ollama and own_llama:
        problems.append(f"Ollama has {', '.join(in_ollama)} loaded while your llama-server runs too: "
                        "run `ollama stop <model>` so only one server holds the model")
    if len(own_llama) > 1:
        problems.append(f"{len(own_llama)} llama-servers are running: keep one")
    result["problems"] = problems
    result["quiet"] = not problems
    return result


def print_preflight(r):
    print(f"QUIET CHECK ({'ok' if r['quiet'] else 'NOT quiet'})")
    print(f"  GPU busy before start  avg {r['gpu_busy_avg']:.0f}%, peak {r['gpu_busy_peak']}%")
    print(f"  CPU busy               {r['cpu_busy_machine']:.0f}% of all cores")
    print(f"  power                  {'AC' if r['on_ac_power'] else 'battery'}, mode: {r['power_mode']}")
    print(f"  thermal                {'THROTTLING' if r['throttling'] else 'no throttling'}")
    if "gpu_temp" in r:
        print(f"  temperature            GPU {r['gpu_temp']:.0f}°C, CPU {r['cpu_temp']:.0f}°C, fans {r['fan_rpm']:.0f} rpm")
        print(f"  swap in use            {r['swap_used_gb']:.1f} GB (memory the system moved to disk)")
    print("  busiest other apps     " + ", ".join(f"{p['name']} {p['cpu']:.0f}%" for p in r["top_processes"][:3]))
    for p in r["problems"]:
        print(f"  ! {p}")


def environment(model):
    """Software and settings, so two runs can be compared later."""
    try:
        details = ollama.show(model)["details"]
        version, gguf = ollama.version(), os.path.basename(ollama.gguf_path(model))
    except Exception:  # Ollama may be stopped while llama-server is measured
        details, version, gguf = {}, None, None
    return {
        "machine": hw.machine(),
        "python": platform.python_version(),
        "ollama_version": version,
        "model": ollama.resolve(model),
        "quantization": details.get("quantization_level"),
        "gguf": gguf,
        # Ollama reads these when it starts; they change how many requests it serves at once.
        "ollama_env": {k: os.environ[k] for k in os.environ if k.startswith("OLLAMA_")},
    }


def after_run():
    return {"time": datetime.now().isoformat(timespec="seconds"), **power(),
            "throttling": thermal()["throttling"], "thermal_detail": thermal()["detail"]}
