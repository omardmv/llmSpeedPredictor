"""Collect real latency data from local Ollama models under realistic traffic.

What makes it close to the real world:
  - Real human-written prompts (databricks-dolly-15k), not templates.
  - Some requests carry earlier chat turns, like a real conversation, so prompts
    range from a short question to a long history.
  - Requests arrive at random (Poisson process) and overlap, like real users.
    Traffic switches between quiet, light, normal and busy periods.
  - Answer limits are any value from 16 to 1024, not just 3 choices.
  - Models unload by themselves after sitting idle (no forced cold starts).
  - Records what a real app knows at send time: requests already in flight,
    how busy the last minute was, and how long the model has been idle.

Usage:
    python collect.py                       # 2000 requests
    python collect.py --n 20 --out /tmp/smoke.csv
    python collect.py --split               # split into train.csv / test.csv
"""

import argparse
import csv
import json
import math
import random
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
OLLAMA = "http://localhost:11434"
PROMPTS_FILE = ROOT / "data" / "prompts" / "databricks-dolly-15k.jsonl"

MODELS = {"fast": "qwen2.5:0.5b", "quality": "qwen2.5:1.5b"}
MODEL_WEIGHTS = [0.55, 0.45]
MAX_TOKENS_RANGE = (16, 1024)      # picked on a log scale, so short limits are more common
TEMPERATURES = [0.0, 0.3, 0.7, 1.0]
KEEP_ALIVE = "20s"                 # idle models unload after this, which creates natural cold starts
CONTEXT_SIZE = 4096
MAX_PROMPT_CHARS = 12000           # keep the whole conversation inside the context
HISTORY_CHANCE = 0.3               # share of requests that continue an earlier conversation
HISTORY_TURNS = (1, 6)             # how many earlier question/answer pairs they carry

# Traffic periods: (name, average requests per second, how long it lasts in seconds)
TRAFFIC = {
    "quiet":  dict(rate=1 / 40, duration=(60, 150),  weight=0.20),
    "light":  dict(rate=1 / 12, duration=(120, 300), weight=0.25),
    "normal": dict(rate=0.5,    duration=(60, 240),  weight=0.35),
    "busy":   dict(rate=1.2,    duration=(30, 90),   weight=0.20),
}

COLUMNS = [
    "request_id", "send_time", "traffic", "model", "model_name", "prompt_category",
    "history_turns", "prompt_chars", "prompt_tokens", "max_tokens", "temperature",
    "in_flight_total", "in_flight_same_model", "requests_last_60s",
    "seconds_since_last_same_model", "model_was_loaded",
    "output_tokens", "stop_reason",
    "load_time_s", "prompt_eval_time_s", "time_to_first_token_s", "total_time_s", "error",
]


# ---------- Ollama ----------

def loaded_models():
    with urllib.request.urlopen(OLLAMA + "/api/ps", timeout=10) as resp:
        return {m["name"] for m in json.loads(resp.read()).get("models", [])}


def generate(model_name, messages, max_tokens, temperature):
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": True,
        "keep_alive": KEEP_ALIVE,
        "options": {"num_predict": max_tokens, "num_ctx": CONTEXT_SIZE, "temperature": temperature},
    }
    req = urllib.request.Request(OLLAMA + "/api/chat", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first_token = None
    with urllib.request.urlopen(req, timeout=900) as resp:
        for line in resp:
            chunk = json.loads(line)
            if first_token is None and chunk.get("message", {}).get("content"):
                first_token = time.perf_counter() - start
            if chunk.get("done"):
                chunk["wall_total"] = time.perf_counter() - start
                chunk["wall_first_token"] = first_token if first_token is not None else chunk["wall_total"]
                return chunk
    raise RuntimeError("stream ended without a final message")


# ---------- prompts ----------

def load_prompts():
    prompts = []
    with open(PROMPTS_FILE) as f:
        for line in f:
            item = json.loads(line)
            text = item["instruction"].strip()
            if item.get("context", "").strip():
                text += "\n\n" + item["context"].strip()
            if len(text) <= MAX_PROMPT_CHARS:
                prompts.append((item["category"], text, item["response"].strip()))
    return prompts


def make_messages(prompts):
    """One new question, sometimes preceded by earlier turns of a conversation."""
    category, question, _ = random.choice(prompts)
    messages = [{"role": "user", "content": question}]
    if random.random() < HISTORY_CHANCE:
        for _ in range(random.randint(*HISTORY_TURNS)):
            _, old_q, old_a = random.choice(prompts)
            turn = [{"role": "user", "content": old_q}, {"role": "assistant", "content": old_a}]
            if sum(len(m["content"]) for m in turn + messages) > MAX_PROMPT_CHARS:
                break
            messages = turn + messages
    return category, messages


# ---------- shared state between request threads ----------

class Tracker:
    """Keeps track of what is happening right now, as a real app could."""

    def __init__(self):
        self.lock = threading.Lock()
        self.in_flight = {name: 0 for name in MODELS}
        self.recent_sends = deque()
        self.last_send = {name: None for name in MODELS}

    def on_send(self, model):
        now = time.monotonic()
        with self.lock:
            while self.recent_sends and now - self.recent_sends[0] > 60:
                self.recent_sends.popleft()
            last = self.last_send[model]
            snapshot = {
                "in_flight_total": sum(self.in_flight.values()),
                "in_flight_same_model": self.in_flight[model],
                "requests_last_60s": len(self.recent_sends),
                "seconds_since_last_same_model": round(now - last, 2) if last else -1,
            }
            self.in_flight[model] += 1
            self.recent_sends.append(now)
            self.last_send[model] = now
        return snapshot

    def on_done(self, model):
        with self.lock:
            self.in_flight[model] -= 1


# ---------- one request (runs in its own thread) ----------

def run_request(request_id, traffic, model, category, messages, max_tokens, temperature,
                tracker, writer, write_lock, f):
    model_name = MODELS[model]

    try:
        was_loaded = model_name in loaded_models()
    except Exception:
        was_loaded = None
    snapshot = tracker.on_send(model)
    row = {
        "request_id": request_id,
        "send_time": datetime.now().isoformat(timespec="milliseconds"),
        "traffic": traffic,
        "model": model,
        "model_name": model_name,
        "prompt_category": category,
        "history_turns": (len(messages) - 1) // 2,
        "prompt_chars": sum(len(m["content"]) for m in messages),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "model_was_loaded": "" if was_loaded is None else int(was_loaded),
        **snapshot,
    }
    try:
        r = generate(model_name, messages, max_tokens, temperature)
        row.update({
            "prompt_tokens": r.get("prompt_eval_count", 0),
            "output_tokens": r.get("eval_count", 0),
            "stop_reason": r.get("done_reason", ""),
            "load_time_s": round(r.get("load_duration", 0) / 1e9, 4),
            "prompt_eval_time_s": round(r.get("prompt_eval_duration", 0) / 1e9, 4),
            "time_to_first_token_s": round(r["wall_first_token"], 4),
            "total_time_s": round(r["wall_total"], 4),
            "error": "",
        })
    except Exception as e:
        row["error"] = type(e).__name__
    finally:
        tracker.on_done(model)

    with write_lock:
        writer.writerow(row)
        f.flush()
    print(f"[{request_id}] {traffic:6} {model:7} in_flight={row['in_flight_total']} "
          f"loaded={row['model_was_loaded']} hist={row['history_turns']} "
          f"prompt={row.get('prompt_tokens', '?')} max={max_tokens} out={row.get('output_tokens', '?')} "
          f"total={row.get('total_time_s', 'ERR')}", flush=True)


# ---------- traffic generator ----------

def collect(n, out_path, seed):
    random.seed(seed)
    prompts = load_prompts()
    print(f"Loaded {len(prompts)} real prompts. Sending {n} requests into {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tracker = Tracker()
    write_lock = threading.Lock()
    threads = []

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()

        sent = 0
        while sent < n:
            name = random.choices(list(TRAFFIC), weights=[t["weight"] for t in TRAFFIC.values()])[0]
            period = TRAFFIC[name]
            period_end = time.monotonic() + random.uniform(*period["duration"])
            print(f"--- traffic: {name} ({period['rate']:.2f} req/s)", flush=True)

            while sent < n:
                gap = random.expovariate(period["rate"])  # Poisson process: random gaps between arrivals
                if time.monotonic() + gap > period_end:
                    time.sleep(max(0, period_end - time.monotonic()))
                    break
                time.sleep(gap)
                sent += 1
                model = random.choices(list(MODELS), weights=MODEL_WEIGHTS)[0]
                category, messages = make_messages(prompts)
                max_tokens = int(math.exp(random.uniform(*map(math.log, MAX_TOKENS_RANGE))))
                temperature = random.choice(TEMPERATURES)
                t = threading.Thread(target=run_request, daemon=True,
                                     args=(sent, name, model, category, messages, max_tokens, temperature,
                                           tracker, writer, write_lock, f))
                t.start()
                threads.append(t)

        for t in threads:
            t.join()
    print("Done.")


def split(src, test_fraction=0.2, seed=42):
    with open(src) as f:
        rows = [r for r in csv.DictReader(f) if not r["error"]]
    random.Random(seed).shuffle(rows)
    n_test = int(len(rows) * test_fraction)
    for name, part in {"test.csv": rows[:n_test], "train.csv": rows[n_test:]}.items():
        part.sort(key=lambda r: int(r["request_id"]))
        with open(src.parent / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(part)
        print(f"{name}: {len(part)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "real" / "requests.csv")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split", action="store_true")
    args = parser.parse_args()

    if args.split:
        split(args.out)
    else:
        collect(args.n, args.out, args.seed)
