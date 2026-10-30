"""Small llama-server (llama.cpp) client.

chat() returns the same dict as ollama.chat(), so run.py and sweep.py work unchanged.
llama-server serves ONE model, chosen when you start it:

    llama-server -m <gguf> --port 8080 -np 4 -c 16384

-np = parallel slots (users decoded together), -c = total context, split across slots.
"""

import json
import os
import time
import urllib.request

URL = "http://localhost:8080"


def _get(url, path):
    with urllib.request.urlopen(url + path, timeout=5) as resp:
        return json.loads(resp.read())


def server(url=URL):
    """How the running server was started: model file, slots, context per slot, build."""
    props = _get(url, "/props")
    slots = _get(url, "/slots")
    return {
        "url": url,
        "model_path": props.get("model_path", ""),
        "slots": props.get("total_slots", len(slots)),
        "ctx_per_slot": slots[0]["n_ctx"] if slots else None,
        "build": props.get("build_info", ""),
    }


def check_model(model, url=URL):
    """llama-server ignores the model name in requests, so make sure it runs the model we think."""
    from . import ollama
    running = os.path.basename(server(url)["model_path"])
    wanted = os.path.basename(ollama.gguf_path(model))
    if running != wanted:
        raise SystemExit(f"llama-server at {url} runs {running[:19]}…, but you asked for {ollama.resolve(model)} "
                         f"({wanted[:19]}…). Restart it with -m {ollama.gguf_path(model)}")


def chat(model, messages, max_tokens=256, temperature=0.0, num_ctx=None, keep_alive=None, seed=42, url=URL):
    """Send one streaming chat request and time every piece of the answer.

    model / num_ctx / keep_alive are accepted only so the call looks like ollama.chat():
    the model and context are fixed when llama-server starts (-m, -c).
    """
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})

    start = time.perf_counter()
    t_tokens, pieces = [], []
    finish_reason, timings = "", {}

    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue  # blank lines between events
            data = line[len("data: "):]
            if data == "[DONE]":
                break
            chunk = json.loads(data)

            choices = chunk.get("choices") or []  # the last (usage) line has an empty list
            if choices:
                # The very first line has content = None (it only announces the role): not a token.
                piece = choices[0].get("delta", {}).get("content")
                if piece:
                    t_tokens.append(time.perf_counter() - start)
                    pieces.append(piece)
                if choices[0].get("finish_reason"):
                    finish_reason = choices[0]["finish_reason"]

            if chunk.get("timings"):
                timings = chunk["timings"]

    ms_to_ns = 1_000_000  # Ollama reports nanoseconds, llama-server milliseconds
    return {
        "wall_s": time.perf_counter() - start,
        "t_tokens": t_tokens,
        "text": "".join(pieces),
        "done_reason": finish_reason,
        "load_duration": 0,  # the model was loaded when the server started
        # prompt_n = tokens actually computed, cache_n = reused from the cache.
        # Ollama counts both, so add them to keep the two backends comparable.
        "prompt_eval_count": timings.get("prompt_n", 0) + timings.get("cache_n", 0),
        "prompt_cached": timings.get("cache_n", 0),
        "prompt_eval_duration": timings.get("prompt_ms", 0) * ms_to_ns,
        "eval_count": timings.get("predicted_n", 0),
        "eval_duration": timings.get("predicted_ms", 0) * ms_to_ns,
    }
