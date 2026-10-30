"""Many users at once: what happens to latency and throughput as concurrency grows?

Each "user" is a thread that sends requests back to back (a closed loop).
Every request has a different real prompt, so the KV cache can't reuse anything.
"""

import csv
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from . import hw, ollama
from .run import DOLLY

OUT = Path(__file__).parent.parent / "research" / "bench"

# Validated default chart palette (dataviz skill, light mode).
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#c3c2b7"
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


def one_level(model, users, per_user, max_tokens, pool, t0, on_row=None):
    rows, lock = [], threading.Lock()

    def user(uid):
        for k in range(per_user):
            prompt = pool.pop()
            start = time.perf_counter() - t0
            r = ollama.chat(model, [{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=0.7,
                            seed=uid * 1000 + k)
            t = r["t_tokens"]
            ttft = t[0] if t else r["wall_s"]
            n = r.get("eval_count", 0)
            row = {
                "users": users, "user": uid, "start_s": start, "end_s": start + r["wall_s"],
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

    fig.suptitle(f"{ollama.resolve(model)} under load", x=0.06, ha="left", fontsize=14, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(model, levels, per_user, max_tokens, as_json=False):
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = ollama.resolve(model).replace(":", "-")
    pool = prompts(sum(levels) * per_user + 10)
    pool.reverse()
    say = (lambda *a, **k: None) if as_json else print

    say(f"{ollama.resolve(model)}: warming up...")
    ollama.chat(model, [{"role": "user", "content": "hi"}], max_tokens=4)

    t0 = time.perf_counter()
    all_rows, summary = [], []
    say(f"\n{'users':>5} {'reqs':>5} {'TTFT p50':>9} {'TTFT p95':>9} {'ms/token':>9} "
        f"{'tok/s per user':>15} {'tok/s total':>12} {'GPU busy':>9}")
    for users in levels:
        if as_json:
            emit("level_start", users=users, requests=users * per_user)
        rows, gpu = one_level(model, users, per_user, max_tokens, pool, t0,
                              on_row=(lambda r: emit("request", **r)) if as_json else None)
        s = summarize(users, rows, gpu)
        all_rows += rows
        summary.append(s)
        if as_json:
            emit("level", **s)
        say(f"{users:5d} {s['requests']:5d} {s['ttft_p50_s']:8.2f}s {s['ttft_p95_s']:8.2f}s "
            f"{s['tpot_p50_ms']:9.1f} {s['per_user_tok_s']:15.1f} {s['total_tok_s']:12.1f} "
            f"{s['gpu_util_avg']:8.0f}%")

    csv_path = OUT / f"sweep_{name}_{stamp}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    png_path = OUT / f"sweep_{name}_{stamp}.png"
    chart(model, summary, all_rows, png_path)
    if as_json:
        emit("done", csv=str(csv_path), png=str(png_path))
    say(f"\nsaved {csv_path.relative_to(OUT.parent.parent)}")
    say(f"saved {png_path.relative_to(OUT.parent.parent)}")
