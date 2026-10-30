"""Which server the benchmarks talk to: Ollama (default) or llama-server.

    backend.use("llama", "http://localhost:8080")
    backend.chat(...)   # same arguments and same result dict for both
"""

from . import llama, ollama

NAME = "ollama"
URL = llama.URL


def use(name, url=None):
    global NAME, URL
    if name not in ("ollama", "llama"):
        raise SystemExit(f"unknown backend {name!r}: use ollama or llama")
    NAME, URL = name, url or URL


def chat(model, messages, **kwargs):
    if NAME == "llama":
        return llama.chat(model, messages, url=URL, **kwargs)
    return ollama.chat(model, messages, **kwargs)


def prepare(model):
    """Fail early if the server is down or runs a different model than asked for."""
    if NAME == "llama":
        try:
            llama.check_model(model, URL)
        except OSError:
            raise SystemExit(f"llama-server is not answering at {URL}. Start it first, e.g.\n"
                             f"  llama-server -m {ollama.gguf_path(model)} --port 8080 -np 4 -c 16384")


def describe():
    """Server setup, saved with every result so you know which config produced it."""
    if NAME == "llama":
        return {"backend": "llama-server", **llama.server(URL)}
    return {"backend": "ollama", "version": ollama.version()}


def label(model):
    """Short human name: 'qwen2.5:1.5b · llama-server 4 slots'."""
    if NAME == "llama":
        return f"{ollama.resolve(model)} · llama-server {llama.server(URL)['slots']} slots"
    return f"{ollama.resolve(model)} · Ollama"


def file_tag(model):
    """Part of the result file name: 'qwen2.5-1.5b_llama-np4-c4096' or 'qwen2.5-1.5b_ollama'."""
    name = ollama.resolve(model).replace(":", "-")
    if NAME == "llama":
        s = llama.server(URL)
        return f"{name}_llama-np{s['slots']}-c{s['ctx_per_slot']}"
    return f"{name}_ollama"
