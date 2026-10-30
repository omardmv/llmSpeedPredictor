"""Time one request and split it into its phases.

    load     -> copy the weights from disk into (GPU) memory. Only on a cold start.
    prefill  -> read the whole prompt in one go, fill the KV cache. Ends at the first token.
    decode   -> write the answer, one token per step.
"""

import json
from pathlib import Path

import numpy as np

from . import hw, ollama

DOLLY = Path(__file__).parent.parent / "data" / "prompts" / "databricks-dolly-15k.jsonl"
SPARK = " ▁▂▃▄▅▆▇█"


def padding(n):
    """n different real paragraphs from the Dolly dataset, to make a prompt longer."""
    paras = []
    with open(DOLLY) as f:
        for line in f:
            ctx = json.loads(line).get("context", "").strip()
            if len(ctx) > 400:
                paras.append(ctx)
            if len(paras) == n:
                break
    return "\n\n".join(paras)


def measure(model, prompt, max_tokens, cold=False):
    if cold:
        ollama.unload(model)
    messages = [{"role": "user", "content": prompt}]
    with hw.GpuSampler() as gpu:
        r = ollama.chat(model, messages, max_tokens=max_tokens)
    t = np.array(r["t_tokens"])
    gaps = np.diff(t) if len(t) > 1 else np.array([0.0])
    s = lambda key: r.get(key, 0) / 1e9
    return {
        "load_s": s("load_duration"),
        "prefill_s": s("prompt_eval_duration"),
        "decode_s": s("eval_duration"),
        "server_total_s": s("total_duration"),
        "wall_s": r["wall_s"],
        "ttft_s": t[0] if len(t) else r["wall_s"],
        "prompt_tokens": r.get("prompt_eval_count", 0),
        "output_tokens": r.get("eval_count", 0),
        "stop": r.get("done_reason", ""),
        "gap_p50_ms": np.percentile(gaps, 50) * 1000,
        "gap_p95_ms": np.percentile(gaps, 95) * 1000,
        "gap_max_ms": gaps.max() * 1000,
        "gpu": gpu.summary(),
        "gpu_samples": gpu.samples,
        "t_tokens": r["t_tokens"],
        "text": r["text"],
    }


def timeline(m, width=60):
    parts = [("load", m["load_s"], "L"), ("prefill", m["prefill_s"], "P"), ("decode", m["decode_s"], "D")]
    other = max(m["wall_s"] - sum(v for _, v, _ in parts), 0)
    parts.append(("other", other, "·"))
    bar = "".join(ch * max(round(width * v / m["wall_s"]), 1 if v > 0.001 else 0) for _, v, ch in parts)
    return bar, other


def sparkline(samples, width=60):
    if not samples:
        return ""
    utils = np.array([u for _, u, _ in samples], dtype=float)
    idx = np.linspace(0, len(utils) - 1, min(width, len(utils))).astype(int)
    return "".join(SPARK[int(u / 100 * (len(SPARK) - 1))] for u in utils[idx])


def report(m):
    per = lambda n, secs: f"{n / secs:8.1f} tok/s" if secs > 0 and n else "        -    "
    bar, other = timeline(m)
    print(f"  timeline   {bar}   L=load P=prefill D=decode ·=other")
    print(f"  GPU busy   {sparkline(m['gpu_samples'])}   avg {m['gpu']['gpu_util_avg']:.0f}%, "
          f"max {m['gpu']['gpu_util_max']}%")
    print(f"  {'phase':10} {'seconds':>8} {'tokens':>7} {'speed':>14}")
    print(f"  {'load':10} {m['load_s']:8.3f}")
    # Ollama counts reused (cached) prompt tokens too, so a reused prompt shows an impossible speed.
    cached = m["prompt_tokens"] > 50 and m["prompt_tokens"] / max(m["prefill_s"], 1e-9) > 20_000
    print(f"  {'prefill':10} {m['prefill_s']:8.3f} {m['prompt_tokens']:7d} {per(m['prompt_tokens'], m['prefill_s'])}"
          + ("   <- too fast to be real: this prompt was found in the KV cache" if cached else ""))
    print(f"  {'decode':10} {m['decode_s']:8.3f} {m['output_tokens']:7d} {per(m['output_tokens'], m['decode_s'])}")
    print(f"  {'other':10} {other:8.3f}            (HTTP, template, sampling setup...)")
    print(f"  {'TOTAL':10} {m['wall_s']:8.3f}            stop reason: {m['stop']} "
          f"({'hit max_tokens' if m['stop'] == 'length' else 'model chose to stop'})")
    print(f"  time to first token  {m['ttft_s'] * 1000:7.0f} ms   (what the user waits before text appears)")
    print(f"  gap between tokens   p50 {m['gap_p50_ms']:.1f} ms, p95 {m['gap_p95_ms']:.1f} ms, "
          f"max {m['gap_max_ms']:.1f} ms   (measured by this client)")
    print(f"  GPU memory peak      {m['gpu']['gpu_mem_max'] / 1024 ** 3:.2f} GB")


def main(model, prompt, pad, max_tokens, cold, repeat, as_json=False):
    if pad:
        prompt = padding(pad) + "\n\n" + prompt
    if as_json:
        # One JSON line per request, for the GUI app.
        for i in range(repeat):
            m = measure(model, prompt, max_tokens, cold=cold and i == 0)
            m["prompt_chars"] = len(prompt)
            print(json.dumps(m, default=float), flush=True)
        return
    print(f"{ollama.resolve(model)}  |  prompt {len(prompt)} chars  |  max_tokens {max_tokens}  |  "
          f"{'cold start' if cold else 'model may already be loaded'}\n")
    rows = []
    for i in range(repeat):
        m = measure(model, prompt, max_tokens, cold=cold and i == 0)
        rows.append(m)
        print(f"REQUEST {i + 1}/{repeat}")
        report(m)
        print()
    if repeat == 1:
        answer = rows[0]["text"].strip().replace("\n", " ")
        print(f"  answer: {answer[:200]}{'...' if len(answer) > 200 else ''}")
    else:
        print(f"  {'#':>3} {'load':>7} {'prompt tok':>10} {'prefill':>8} {'TTFT':>7} {'decode tok/s':>13} {'total':>7}")
        for i, m in enumerate(rows, 1):
            print(f"  {i:3d} {m['load_s']:7.3f} {m['prompt_tokens']:10d} {m['prefill_s']:8.3f} {m['ttft_s']:7.3f} "
                  f"{m['output_tokens'] / max(m['decode_s'], 1e-9):13.1f} {m['wall_s']:7.3f}")
