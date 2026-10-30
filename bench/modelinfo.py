"""What a model is made of, and how much memory it really takes once loaded."""

import json
import os

from . import hw, ollama

MB = 1024 ** 2


def info(model):
    s = ollama.show(model)
    mi = s["model_info"]
    arch = mi["general.architecture"]
    get = lambda key: mi.get(f"{arch}.{key}")

    params = mi["general.parameter_count"]
    file_bytes = os.path.getsize(ollama.gguf_path(model))
    layers, hidden = get("block_count"), get("embedding_length")
    heads, kv_heads = get("attention.head_count"), get("attention.head_count_kv")
    head_dim = hidden // heads
    # Every token in the context keeps one Key and one Value vector per layer per KV head.
    # Ollama stores them as 16-bit floats (2 bytes) by default.
    kv_bytes_per_token = 2 * layers * kv_heads * head_dim * 2

    return {
        "name": ollama.resolve(model),
        "architecture": arch,
        "quantization": s["details"]["quantization_level"],
        "parameters": params,
        "file_bytes": file_bytes,
        "bits_per_parameter": file_bytes * 8 / params,
        "layers": layers,
        "hidden_size": hidden,
        "ffn_size": get("feed_forward_length"),
        "attention_heads": heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "max_context": get("context_length"),
        "kv_bytes_per_token": kv_bytes_per_token,
        "template": s.get("template", ""),
    }


def load_and_measure(model, num_ctx):
    """Load the model with this context size and ask Ollama how much memory it took."""
    ollama.unload(model)
    before = hw.gpu_now()["gpu_mem_in_use"]
    with ollama._post("/api/generate", {"model": ollama.resolve(model), "keep_alive": "2m",
                                        "options": {"num_ctx": num_ctx}}) as resp:
        resp.read()
    after = hw.gpu_now()["gpu_mem_in_use"]
    entry = next(m for m in ollama.loaded() if m["name"] == ollama.resolve(model))
    return {"ollama_total": entry["size"], "ollama_on_gpu": entry["size_vram"],
            "gpu_mem_delta": after - before, "context_length": entry.get("context_length", num_ctx)}


def print_info(model, num_ctx=4096, load=True):
    m = info(model)
    print(f"MODEL  {m['name']}  ({m['architecture']}, {m['quantization']})")
    print(f"  parameters          {m['parameters'] / 1e9:.3f} billion")
    print(f"  file on disk        {m['file_bytes'] / MB:7.1f} MB  -> {m['bits_per_parameter']:.2f} bits per parameter "
          f"(16 = full fp16, ~4.5-5 = Q4)")
    print(f"  layers              {m['layers']}   (every new token passes through all of them)")
    print(f"  hidden size         {m['hidden_size']}   (length of the vector that represents one token)")
    print(f"  feed-forward size   {m['ffn_size']}")
    print(f"  attention heads     {m['attention_heads']} query heads, {m['kv_heads']} key/value heads "
          f"of size {m['head_dim']}   (fewer KV heads = grouped-query attention, smaller KV cache)")
    print(f"  max context         {m['max_context']} tokens (trained); Ollama uses num_ctx instead")
    print(f"  KV cache per token  {m['kv_bytes_per_token'] / 1024:.1f} KB  "
          f"= 2 (K and V) x {m['layers']} layers x {m['kv_heads']} heads x {m['head_dim']} x 2 bytes")
    print(f"  KV cache @ {num_ctx:<6}  {m['kv_bytes_per_token'] * num_ctx / MB:7.1f} MB for ONE conversation slot")

    if load:
        L = load_and_measure(model, num_ctx)
        other = L["ollama_total"] - m["file_bytes"]
        print(f"LOADED with num_ctx={L['context_length']}")
        print(f"  Ollama reports      {L['ollama_total'] / MB:7.1f} MB total, {L['ollama_on_gpu'] / MB:7.1f} MB on GPU")
        print(f"    weights           {m['file_bytes'] / MB:7.1f} MB")
        print(f"    everything else   {other / MB:7.1f} MB   (KV cache for all slots + scratch buffers for the math)")
        print(f"  GPU memory grew by  {L['gpu_mem_delta'] / MB:7.1f} MB   (from the GPU driver; noisy, other apps use the GPU too)")
    return m


def dump(model, num_ctx=4096, load=True):
    """Everything above as one JSON object, for the GUI app."""
    out = {k: v for k, v in info(model).items() if k != "template"}
    out["num_ctx"] = num_ctx
    if load:
        out["loaded"] = load_and_measure(model, num_ctx)
    print(json.dumps(out))
