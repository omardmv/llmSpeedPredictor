"""Small Ollama client that records *when* every streamed token arrived."""

import json
import re
import time
import urllib.request

OLLAMA = "http://localhost:11434"
MODELS = {"fast": "qwen2.5:0.5b", "quality": "qwen2.5:1.5b"}


def resolve(model):
    """Accept 'fast' / 'quality' or a real Ollama model name."""
    return MODELS.get(model, model)


def _post(path, payload, timeout=900):
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def _get(path):
    with urllib.request.urlopen(OLLAMA + path, timeout=10) as resp:
        return json.loads(resp.read())


def version():
    return _get("/api/version")["version"]


def loaded():
    """Models in memory right now: name, size (bytes), size_vram, context_length, expires_at."""
    return _get("/api/ps").get("models", [])


def show(model):
    with _post("/api/show", {"model": resolve(model)}, timeout=30) as resp:
        return json.loads(resp.read())


def gguf_path(model):
    """Path of the GGUF file on disk (the actual weights Ollama loads)."""
    match = re.search(r"^FROM (.+)$", show(model)["modelfile"], re.M)
    return match.group(1).strip()


def unload(model):
    """Ask Ollama to drop the model from memory, so the next request is a cold start."""
    with _post("/api/generate", {"model": resolve(model), "keep_alive": 0}) as resp:
        resp.read()
    for _ in range(50):
        if not any(m["name"] == resolve(model) for m in loaded()):
            return
        time.sleep(0.1)


def chat(model, messages, max_tokens=256, temperature=0.0, num_ctx=4096, keep_alive="5m", seed=42):
    """Send one streaming chat request.

    Returns Ollama's final stats plus what the client saw:
      t_tokens: seconds after sending at which each streamed piece arrived
      text:     the full answer
      wall_s:   seconds from sending until the last byte
    """
    payload = {
        "model": resolve(model),
        "messages": messages,
        "stream": True,
        "keep_alive": keep_alive,
        "options": {"num_predict": max_tokens, "num_ctx": num_ctx,
                    "temperature": temperature, "seed": seed},
    }
    start = time.perf_counter()
    t_tokens, pieces = [], []
    with _post("/api/chat", payload) as resp:
        for line in resp:
            chunk = json.loads(line)
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                t_tokens.append(time.perf_counter() - start)
                pieces.append(piece)
            if chunk.get("done"):
                chunk["wall_s"] = time.perf_counter() - start
                chunk["t_tokens"] = t_tokens
                chunk["text"] = "".join(pieces)
                return chunk
    raise RuntimeError("stream ended without a final message")
