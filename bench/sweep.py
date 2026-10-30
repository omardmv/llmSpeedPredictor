"""Many users at once: what happens to latency and throughput as concurrency grows?

Each "user" is a thread that sends requests back to back (a closed loop).
Every request has a different real prompt, so the KV cache can't reuse anything.
"""

import csv
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from . import backend, env, heat, hw, ollama
from .run import DOLLY

OUT = Path(__file__).parent.parent / "research" / "bench"

# Validated default chart palette (dataviz skill, light mode).
BLUE, ORANGE, GRAY, AQUA = "#2a78d6", "#eb6834", "#c3c2b7", "#1baf7a"
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def prompts(n):
    out = []
    with open(DOLLY) as f:
        for line in f:
            item = json.loads(line)
            if item["category"] in ("open_qa", "brainstorming", "creative_writing", "general_qa"):
                out.append(item["instruction"].strip() + "\nAnswer in detail.")
            if len(out) == n:
                break
    return out


def one_level(model, users, per_user, max_tokens, pool, t0, on_row=None, round_no=1):
    rows, lock = [], threading.Lock()

    def user(uid):
        for k in range(per_user):
            prompt = pool.pop()
            start = time.perf_counter() - t0
            r = backend.chat(model, [{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=0.7,
                            seed=uid * 1000 + k)
            t = r["t_tokens"]
            ttft = t[0] if t else r["wall_s"]
            n = r.get("eval_count", 0)
            row = {
                "round": round_no, "users": users, "user": uid, "start_s": start, "end_s": start + r["wall_s"],
                "ttft_s": ttft, "total_s": r["wall_s"],
                "prompt_tokens": r.get("prompt_eval_count", 0), "output_tokens": n,
                "server_prefill_s": r.get("prompt_eval_duration", 0) / 1e9,
                "server_decode_s": r.get("eval_duration", 0) / 1e9,
                # time between tokens as this user experiences it, after the first one arrived
                "tpot_ms": (r["wall_s"] - ttft) / max(n - 1, 1) * 1000,
            }
            with lock:
                rows.append(row)
                if on_row:
                    on_row(row)

    threads = [threading.Thread(target=user, args=(u,)) for u in range(users)]
    with hw.GpuSampler(interval=0.25) as gpu:
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    return rows, gpu.summary()


def emit(kind, **data):
    """One JSON line for the GUI app (python -m bench sweep --json)."""
    print(json.dumps({"type": kind, **{k: float(v) if isinstance(v, np.floating) else v for k, v in data.items()}}),
          flush=True)


def summarize(users, rows, gpu):
    ttft = np.array([r["ttft_s"] for r in rows])
    wall = max(r["end_s"] for r in rows) - min(r["start_s"] for r in rows)
    return {
        "users": users,
        "requests": len(rows),
        "ttft_p50_s": np.percentile(ttft, 50),
        "ttft_p95_s": np.percentile(ttft, 95),
        "tpot_p50_ms": np.median([r["tpot_ms"] for r in rows]),
        "per_user_tok_s": np.median([r["output_tokens"] / max(r["total_s"] - r["ttft_s"], 1e-9) for r in rows]),
        "total_tok_s": sum(r["output_tokens"] for r in rows) / wall,
        "gpu_util_avg": gpu["gpu_util_avg"],
    }


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK2)
    if ylabel != "request":
        ax.set_ylim(0, ax.get_ylim()[1] * 1.1)   # start at zero so small wobbles don't look like big changes


def chart(model, summary, rows, path):
    u = [s["users"] for s in summary]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor="white")

    ax = axes[0, 0]
    ax.plot(u, [s["ttft_p50_s"] for s in summary], "-o", color=BLUE, lw=2, ms=8, label="p50")
    ax.plot(u, [s["ttft_p95_s"] for s in summary], "-o", color=ORANGE, lw=2, ms=8, label="p95")
    style(ax, "Time to first token", "users at once", "seconds")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(u, [s["per_user_tok_s"] for s in summary], "-o", color=BLUE, lw=2, ms=8)
    style(ax, "Speed each user sees while their answer streams", "users at once", "tokens / second")

    ax = axes[1, 0]
    ax.plot(u, [s["total_tok_s"] for s in summary], "-o", color=BLUE, lw=2, ms=8)
    style(ax, "Total throughput of the server", "users at once", "tokens / second")

    # Gantt of the busiest level: grey = waiting for the first token, blue = streaming tokens.
    ax = axes[1, 1]
    top = max(u)
    level = sorted([r for r in rows if r["users"] == top], key=lambda r: r["start_s"])
    t_min = min(r["start_s"] for r in level)
    for i, r in enumerate(level):
        s = r["start_s"] - t_min
        ax.barh(i, r["ttft_s"], left=s, color=GRAY, height=0.8)
        ax.barh(i, r["total_s"] - r["ttft_s"], left=s + r["ttft_s"], color=BLUE, height=0.8)
    ax.invert_yaxis()
    style(ax, f"Every request at {top} users", "seconds", "request")
    ax.legend(handles=[Patch(color=GRAY, label="waiting for first token"), Patch(color=BLUE, label="streaming answer")],
              frameon=False, loc="lower left", bbox_to_anchor=(0, 1.04), ncol=2, borderaxespad=0)
    ax.set_title(f"Every request at {top} users", loc="left", color=INK, fontsize=11, pad=26)

    fig.suptitle(f"{backend.label(model)} under load", x=0.06, ha="left", fontsize=14, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


SPREAD_MAX = 10  # % difference between repeated runs above which a level is not trusted
SPREAD_KEYS = ["ttft_p95_s", "per_user_tok_s", "total_tok_s"]


def spread(per_round):
    """How much repeated runs of the same level disagree: (max - min) / mean, in %."""
    out = {}
    for key in SPREAD_KEYS:
        vals = np.array([s[key] for s in per_round])
        out[key] = float((vals.max() - vals.min()) / max(vals.mean(), 1e-9) * 100) if len(vals) > 1 else 0.0
    # Tiny times look wildly different in % (0.01 s vs 0.02 s = 100%) but don't matter,
    # so TTFT only counts as disagreeing if the rounds are also more than 0.1 s apart.
    ttft = [s["ttft_p95_s"] for s in per_round]
    ttft_ok = out["ttft_p95_s"] <= SPREAD_MAX or max(ttft) - min(ttft) < 0.1
    out["trusted"] = bool(ttft_ok and out["per_user_tok_s"] <= SPREAD_MAX and out["total_tok_s"] <= SPREAD_MAX)
    return out


def heat_chart(model, rows, series, path):
    """Decode speed, GPU clock and GPU temperature over the whole run, one panel each."""
    t = [p["t"] for p in series]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, facecolor="white")

    ax = axes[0]
    mid = [r["start_s"] + r["ttft_s"] + (r["total_s"] - r["ttft_s"]) / 2 for r in rows]
    speed = [r["output_tokens"] / max(r["total_s"] - r["ttft_s"], 1e-9) for r in rows]
    ax.scatter(mid, speed, s=14, color=BLUE, alpha=0.7)
    style(ax, "Decode speed of each request (tok/s, as the user sees it)", "", "tok/s")
    # Level boundaries, labelled once at the top.
    for (rnd, users), group in sorted({(r["round"], r["users"]): None for r in rows}.items()):
        start = min(r["start_s"] for r in rows if r["round"] == rnd and r["users"] == users)
        for a in axes:
            a.axvline(start, color="#d9d8d3", lw=0.8, zorder=0)
        ax.annotate(f"{users}u" + (f" r{rnd}" if rnd > 1 else ""), (start, 1.0), xycoords=("data", "axes fraction"),
                    fontsize=8, color=INK2, va="bottom")

    ax = axes[1]
    ax.plot(t, [p["gpu_mhz"] for p in series], color=ORANGE, lw=1.5)
    top = heat.max_gpu_mhz()
    if top:
        ax.axhline(top, color=INK2, lw=1, ls="--")
        ax.annotate(f"max {top} MHz", (t[0], top), fontsize=8, color=INK2, va="bottom")
    style(ax, "GPU clock (MHz): drops when the chip protects itself from heat", "", "MHz")

    ax = axes[2]
    ax.plot(t, [p["gpu_temp"] for p in series], color=AQUA, lw=1.5)
    style(ax, "GPU temperature (°C)", "seconds (negative = warm-up)", "°C")
    ax.set_ylim(min(p["gpu_temp"] for p in series) - 3, max(p["gpu_temp"] for p in series) + 3)

    for a in axes:
        if t[0] < 0:
            a.axvspan(t[0], 0, color="#efeeea", zorder=-1)
    fig.suptitle(f"{backend.label(model)}: heat during the run", x=0.06, ha="left", fontsize=14, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def combine(users, per_round):
    """Average the rounds of one level into one summary."""
    keys = [k for k in per_round[0] if k not in ("users", "requests", "round")
            and all(isinstance(r.get(k), (int, float)) for r in per_round)]
    return {"users": users, "requests": sum(s["requests"] for s in per_round),
            **{k: float(np.mean([s[k] for s in per_round])) for k in keys}}


def warm_up(model, hs, pool, say, as_json, min_s=60, max_s=360, stable_c=0.5, window_s=60):
    """Keep the GPU busy until its temperature stops rising, so every run starts equally hot.

    "Stable" = the GPU warmed up less than `stable_c` degrees over the last `window_s` seconds.
    A real server runs hot all day, so that is the state worth measuring.
    Without macmon we can't see the temperature and just run for 3 minutes.
    """
    stop = threading.Event()
    start = time.perf_counter()

    def load(uid):
        k = 0
        while not stop.is_set():
            backend.chat(model, [{"role": "user", "content": pool[(uid * 50 + k) % len(pool)]}],
                         max_tokens=400, temperature=0.7, seed=uid * 1000 + k)
            k += 1

    threads = [threading.Thread(target=load, args=(u,), daemon=True) for u in range(4)]
    for th in threads:
        th.start()
    reason = "fixed 180 s (install macmon to wait for a stable temperature)"
    while True:
        time.sleep(5)
        elapsed = time.perf_counter() - start
        now = hs.latest()
        if now:
            past = [s["gpu_temp"] for t, s in hs.samples if t >= time.perf_counter() - window_s]
            rise = now["gpu_temp"] - past[0] if past else 99
            say(f"  warm-up {elapsed:4.0f}s  GPU {now['gpu_temp']:5.1f}°C  {now['gpu_mhz']:5.0f} MHz  "
                f"(+{rise:.1f}°C in the last {window_s}s)", flush=True)
            if as_json:
                emit("warmup", seconds=elapsed, gpu_temp=now["gpu_temp"], gpu_mhz=now["gpu_mhz"], rise=rise)
            if elapsed >= min_s and rise < stable_c:
                reason = f"temperature stable ({now['gpu_temp']:.1f}°C, +{rise:.1f}°C in {window_s}s)"
                break
            if elapsed >= max_s:
                reason = f"stopped after {max_s}s, temperature still rising"
                break
        elif elapsed >= 180:
            break
    stop.set()
    for th in threads:
        th.join()  # let in-flight requests finish, so they don't overlap the first level
    return {"seconds": time.perf_counter() - start, "reason": reason}


def ask_to_continue(check):
    if not sys.stdin.isatty():
        return False
    return input("\nThe machine is not quiet; results may be off. Continue anyway? [y/N] ").strip().lower() == "y"


def main(model, levels, per_user, max_tokens, as_json=False, repeat=1, force=False, warmup=True):
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backend.prepare(model)
    name = backend.file_tag(model)
    # Every round gets the same prompts in the same order, so rounds are comparable.
    base_pool = prompts(sum(levels) * per_user + 10)
    say = (lambda *a, **k: None) if as_json else print

    check = env.preflight()
    if as_json:
        emit("preflight", **check)
    else:
        env.print_preflight(check)
        if not check["quiet"] and not force and not ask_to_continue(check):
            print("Stopped. Close what is listed above, or pass --force to run anyway.")
            return

    if not heat.available():
        say("\n(macmon not installed: no temperature, clock or power numbers. brew install macmon)")
    with heat.HeatSampler() as hs:
        say(f"\n{backend.label(model)}: first request...")
        backend.chat(model, [{"role": "user", "content": "hi"}], max_tokens=4)
        warm = None
        if warmup:
            say("WARM-UP: keeping the GPU busy until its temperature is stable")
            warm = warm_up(model, hs, base_pool, say, as_json)
            say(f"  done: {warm['reason']}")

        t0 = time.perf_counter()
        all_rows, by_level = [], {u: [] for u in levels}
        for round_no in range(1, repeat + 1):
            pool = list(reversed(base_pool))
            if repeat > 1:
                say(f"\nROUND {round_no}/{repeat}")
            say(f"\n{'users':>5} {'reqs':>5} {'TTFT p50':>9} {'TTFT p95':>9} {'ms/token':>9} "
                f"{'tok/s per user':>15} {'tok/s total':>12} {'GPU °C':>7} {'GPU MHz':>8} {'tok/J':>6}")
            for users in levels:
                if as_json:
                    emit("level_start", users=users, requests=users * per_user, round=round_no)
                rows, gpu = one_level(model, users, per_user, max_tokens, pool, t0, round_no=round_no,
                                      on_row=(lambda r: emit("request", **r)) if as_json else None)
                s = summarize(users, rows, gpu)
                s["round"] = round_no
                # Heat over exactly the time this level ran.
                h = hs.summary(t0 + min(r["start_s"] for r in rows), t0 + max(r["end_s"] for r in rows))
                s.update(h)
                if h.get("chip_energy_j"):
                    s["tokens_per_joule"] = sum(r["output_tokens"] for r in rows) / h["chip_energy_j"]
                all_rows += rows
                by_level[users].append(s)
                if as_json:
                    emit("level", **s)
                heat_cols = (f"{s['gpu_temp_avg']:7.1f} {s['gpu_mhz_avg']:8.0f} {s['tokens_per_joule']:6.1f}"
                             if "tokens_per_joule" in s else f"{'-':>7} {'-':>8} {'-':>6}")
                say(f"{users:5d} {s['requests']:5d} {s['ttft_p50_s']:8.2f}s {s['ttft_p95_s']:8.2f}s "
                    f"{s['tpot_p50_ms']:9.1f} {s['per_user_tok_s']:15.1f} {s['total_tok_s']:12.1f} {heat_cols}")

    summary = [combine(u, by_level[u]) for u in levels]
    spreads = {u: spread(by_level[u]) for u in levels}
    if repeat > 1:
        say(f"\nSPREAD between the {repeat} rounds (want <= {SPREAD_MAX}%)")
        say(f"{'users':>5} {'TTFT p95':>9} {'tok/s user':>11} {'tok/s total':>12}")
        for u in levels:
            sp = spreads[u]
            say(f"{u:5d} {sp['ttft_p95_s']:8.1f}% {sp['per_user_tok_s']:10.1f}% {sp['total_tok_s']:11.1f}%"
                + ("" if sp["trusted"] else "   <- rounds disagree: don't trust this level"))
            if as_json:
                emit("spread", users=u, **sp)

    csv_path = OUT / f"sweep_{name}_{stamp}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    png_path = OUT / f"sweep_{name}_{stamp}.png"
    # The Gantt panel shows round 1 only; the other panels average the rounds.
    chart(model, summary, [r for r in all_rows if r["round"] == 1], png_path)
    heat_series = hs.series(t0)
    heat_png = OUT / f"sweep_{name}_{stamp}_heat.png"
    if heat_series:
        heat_chart(model, all_rows, heat_series, heat_png)

    meta_path = OUT / f"sweep_{name}_{stamp}.json"
    meta = {
        "server": backend.describe(),
        "settings": {"model": ollama.resolve(model), "users": levels, "requests_per_user": per_user,
                     "max_tokens": max_tokens, "temperature": 0.7, "repeat": repeat, "forced": force},
        "environment": env.environment(model),
        "before": check,
        "after": env.after_run(),
        "levels": summary,
        "rounds": {str(u): by_level[u] for u in levels},
        "warmup": warm,
        "max_gpu_mhz": heat.max_gpu_mhz(),
        "heat": heat_series,
        "spread": {str(u): spreads[u] for u in levels},
        "files": {"csv": csv_path.name, "png": png_path.name, "heat_png": heat_png.name if heat_series else None},
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=float))
    if meta["after"]["throttling"] and not check["throttling"]:
        say("\n! the chip started throttling for heat during the run; later levels may be slower")
    if as_json:
        emit("done", csv=str(csv_path), png=str(png_path), meta=str(meta_path))
    for path in (csv_path, png_path, heat_png if heat_series else None, meta_path):
        if path:
            say(f"saved {path.relative_to(OUT.parent.parent)}")
