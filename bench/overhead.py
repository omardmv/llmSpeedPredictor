"""Where does llama-server lose speed compared to the bare engine?

    python -m bench overhead quality --np 8

Starts N requests at exactly the same time, all forced to the same length (ignore_eos),
so the batch stays full and nothing new arrives. Then changes one thing at a time:
how the next token is picked (sampling), streaming, and sampling on the GPU.
The engine reference comes from llama-batched-bench with the same batch size.
"""

import json
import re
import subprocess
import threading
import time
import urllib.request

from . import ollama, slots

URL = slots.URL
TOKENS = 256


def _request(prompt, stream, temperature, samplers):
    payload = {"messages": [{"role": "user", "content": prompt}], "max_tokens": TOKENS, "ignore_eos": True,
               "temperature": temperature, "stream": stream, "cache_prompt": False}
    if samplers is not None:
        payload["samplers"] = samplers
    if stream:
        payload["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    timings = {}
    with urllib.request.urlopen(req, timeout=600) as resp:
        if stream:
            for raw in resp:
                line = raw.decode().strip()
                if line.startswith("data: {"):
                    chunk = json.loads(line[6:])
                    timings = chunk.get("timings") or timings
        else:
            timings = json.loads(resp.read()).get("timings", {})
    return timings


def _batch(n, prompts, **kw):
    """n requests released at the same instant. Returns server ms/step and total tok/s."""
    results, barrier = [None] * n, threading.Barrier(n)

    def go(i):
        barrier.wait()
        results[i] = _request(prompts[i], **kw)

    start = time.perf_counter()
    threads = [threading.Thread(target=go, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - start
    step_ms = sorted(r["predicted_ms"] / r["predicted_n"] for r in results)[n // 2]
    return step_ms, sum(r["predicted_n"] for r in results) / wall


def engine_step_ms(gguf, n):
    """Same batch size on the bare engine (no server, no HTTP, no sampling)."""
    out = subprocess.run(["llama-batched-bench", "-m", gguf, "-c", str(max(n, 1) * 1024), "-npp", "64",
                          "-ntg", str(TOKENS), "-npl", str(n)], capture_output=True, text=True).stdout
    row = [line for line in out.splitlines() if re.match(r"^\|\s+64\s+\|", line)][0].split("|")
    return float(row[7]) / TOKENS * 1000


def main(model, n):
    from .run import padding  # real Dolly paragraphs as prompts
    gguf = ollama.gguf_path(model)
    prompts = [f"Text {i}:\n" + padding(1 + i % 3) + "\n\nSummarize this." for i in range(n)]

    print(f"engine alone (llama-batched-bench), batch {n}: ", end="", flush=True)
    eng = engine_step_ms(gguf, n)
    print(f"{eng:.2f} ms/step  = {n / eng * 1000:.0f} tok/s")

    tests = [
        ("default sampling (temp 0.7, 9-step sampler chain), streaming", dict(stream=True, temperature=0.7, samplers=None)),
        ("greedy (temp 0: just take the top token), streaming", dict(stream=True, temperature=0.0, samplers=None)),
        ("only temperature sampler, streaming", dict(stream=True, temperature=0.7, samplers=["temperature"])),
        ("default sampling, no streaming", dict(stream=False, temperature=0.7, samplers=None)),
    ]
    for server_args, label in [([], "sampling on the CPU (default)"), (["--backend-sampling"], "sampling on the GPU (--backend-sampling)")]:
        print(f"\nllama-server -np {n}: {label}")
        proc = subprocess.Popen(["llama-server", "-m", gguf, "--port", "8080", "-np", str(n), "-c", str(n * 4096)]
                                + server_args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            slots._wait_ready(proc)
            _batch(n, prompts, stream=True, temperature=0.7, samplers=None)  # warm-up, not counted
            for name, kw in tests:
                runs = [_batch(n, prompts, **kw) for _ in range(2)]
                step = sum(r[0] for r in runs) / 2
                tps = sum(r[1] for r in runs) / 2
                print(f"  {name:62} {step:6.2f} ms/step (+{(step / eng - 1) * 100:3.0f}% vs engine)  {tps:5.0f} tok/s")
        finally:
            proc.terminate()
            proc.wait()
            time.sleep(2)
