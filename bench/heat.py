"""GPU temperature, clock speed and power while a benchmark runs (via macmon, no sudo).

Why: a hot chip lowers its clock to stay cool, and a lower clock means fewer tokens per
second. Without these numbers, "config B is slower" might just mean "the chip was hotter".
This is the Mac version of watching `nvidia-smi` clocks/temperature/power on a server.

    brew install macmon

    with HeatSampler() as heat:
        ...run requests...
    heat.summary(t_start, t_end)   # averages for one window of the run
"""

import json
import shutil
import subprocess
import threading
import time
from functools import lru_cache

MACMON = shutil.which("macmon") or "/opt/homebrew/bin/macmon"


def available():
    return shutil.which("macmon") is not None or shutil.os.path.exists(MACMON)


def _parse(line):
    d = json.loads(line)
    fans = d.get("fans") or []
    return {
        "gpu_temp": d["temp"]["gpu_temp_avg"],
        "cpu_temp": d["temp"]["cpu_temp_avg"],
        "gpu_mhz": d["gpu_freq_mhz"],
        "gpu_busy": d["gpu_active_ratio"] * 100,
        "gpu_power": d["gpu_power"],      # watts, GPU only
        "chip_power": d["all_power"],     # watts, CPU + GPU + neural engine
        "fan_rpm": max((f["rpm"] for f in fans), default=0),
        "swap_used": d.get("memory", {}).get("swap_usage", 0),
    }


@lru_cache(maxsize=1)
def max_gpu_mhz():
    """Highest GPU clock the chip can run at (the 'boost' clock)."""
    if not available():
        return None
    out = subprocess.run([MACMON, "pipe", "-s", "1", "--soc-info"], capture_output=True, text=True).stdout
    return max(json.loads(out.splitlines()[0])["soc"]["gpu_freqs"])


def now():
    """One reading, e.g. for the quiet check. None if macmon isn't installed."""
    if not available():
        return None
    out = subprocess.run([MACMON, "pipe", "-s", "1", "-i", "500"], capture_output=True, text=True).stdout
    return _parse(out.splitlines()[0]) if out.strip() else None


class HeatSampler:
    """Runs `macmon pipe` in the background and keeps every reading with its time.

    Times are time.perf_counter() values, the same clock the benchmark uses.
    """

    def __init__(self, interval_ms=500):
        self.interval_ms = interval_ms
        self.samples = []  # (perf_counter time, reading dict)
        self.proc = None

    def __enter__(self):
        if available():
            self.proc = subprocess.Popen([MACMON, "pipe", "-i", str(self.interval_ms)],
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            self._thread = threading.Thread(target=self._read, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()
            self.proc.wait()

    def _read(self):
        for line in self.proc.stdout:
            try:
                self.samples.append((time.perf_counter(), _parse(line)))
            except (ValueError, KeyError):
                pass

    def latest(self):
        return self.samples[-1][1] if self.samples else None

    def window(self, t_from, t_to):
        return [s for t, s in self.samples if t_from <= t <= t_to]

    def summary(self, t_from, t_to):
        """Averages over one window, plus the energy used in it (joules = watts x seconds)."""
        w = self.window(t_from, t_to)
        if not w:
            return {}
        avg = lambda k: sum(s[k] for s in w) / len(w)
        seconds = t_to - t_from
        top = max_gpu_mhz()
        # Clock only while the GPU is actually working: when idle (e.g. between levels) it
        # drops on purpose to save power, which is not the same as slowing down from heat.
        busy = [s for s in w if s["gpu_busy"] >= 50] or w
        mhz = sum(s["gpu_mhz"] for s in busy) / len(busy)
        return {
            "gpu_temp_avg": avg("gpu_temp"),
            "gpu_temp_max": max(s["gpu_temp"] for s in w),
            "gpu_mhz_avg": mhz,
            "gpu_mhz_min_busy": min(s["gpu_mhz"] for s in busy),
            "gpu_clock_pct": mhz / top * 100 if top else None,
            "gpu_power_w": avg("gpu_power"),
            "chip_power_w": avg("chip_power"),
            "chip_energy_j": avg("chip_power") * seconds,
            "fan_rpm_max": max(s["fan_rpm"] for s in w),
        }

    def series(self, t0):
        """All readings, with time relative to t0, for charts and the saved .json."""
        return [{"t": round(t - t0, 2), **{k: round(v, 2) for k, v in s.items() if k != "swap_used"}}
                for t, s in self.samples]
