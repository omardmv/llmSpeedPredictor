"""See how text is cut into tokens.

Ollama has no tokenize endpoint, so we start llama-server (llama.cpp) on the very
same GGUF file Ollama uses. Same file, same tokenizer, same token IDs.
"""

import json
import subprocess
import sys
import time
import urllib.request

from . import ollama

PORT = 8091
URL = f"http://localhost:{PORT}"

SAMPLES = [
    ("English", "Large language models split text into tokens and generate one token at a time."),
    ("Azerbaijani", "Böyük dil modelləri mətni tokenlərə bölür və hər dəfə bir token yaradır."),
    ("Uzbek", "Katta til modellari matnni tokenlarga boʻladi va har safar bitta token yaratadi."),
    ("Russian", "Большие языковые модели делят текст на токены и генерируют по одному токену за раз."),
    ("Python", "def mean(xs):\n    return sum(xs) / len(xs) if xs else 0.0\n"),
    ("JSON", '{"model": "qwen2.5:1.5b", "max_tokens": 256, "temperature": 0.7}'),
    ("Numbers", "3.14159 2653589 1000000 2026-10-30 192.168.0.1"),
    ("Emoji", "🚀🔥😂👍 ok"),
]

COLORS = ["\033[30;46m", "\033[30;43m", "\033[30;42m", "\033[30;45m", "\033[30;47m"]
RESET = "\033[0m"


class Tokenizer:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        path = ollama.gguf_path(self.model)
        # -ngl 0: keep it on the CPU, we only need the tokenizer, not the GPU.
        self.proc = subprocess.Popen(["llama-server", "-m", path, "--port", str(PORT), "-c", "512", "-ngl", "0"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                with urllib.request.urlopen(URL + "/health", timeout=1) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        return self
            except OSError:
                pass
            time.sleep(0.2)
        self.proc.kill()
        raise RuntimeError("llama-server did not start (is something else on port 8091?)")

    def __exit__(self, *exc):
        self.proc.terminate()
        self.proc.wait()

    def _post(self, path, payload):
        req = urllib.request.Request(URL + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def tokenize(self, text):
        """[(token_id, piece), ...]"""
        out = self._post("/tokenize", {"content": text, "with_pieces": True, "add_special": False,
                                       "parse_special": True})
        return [(t["id"], t["piece"]) for t in out["tokens"]]

    def chat_template(self, messages):
        return self._post("/apply-template", {"messages": messages})["prompt"]

    def vocab_size(self):
        with urllib.request.urlopen(URL + "/v1/models", timeout=5) as r:
            return json.loads(r.read())["data"][0]["meta"]["n_vocab"]


def show_pieces(toks):
    color = sys.stdout.isatty()
    parts = []
    for i, (_, piece) in enumerate(toks):
        shown = piece.replace(" ", "␣").replace("\n", "↵")
        parts.append(f"{COLORS[i % len(COLORS)]}{shown}{RESET}" if color else f"[{shown}]")
    print("  " + "".join(parts))
    print("  ids: " + " ".join(str(t) for t, _ in toks))


def main(model, text, compare=False, chat=False):
    with Tokenizer(model) as tok:
        print(f"TOKENIZER of {ollama.resolve(model)}: {tok.vocab_size():,} different tokens in the vocabulary\n")

        if text:
            toks = tok.tokenize(text)
            print(f"{len(text)} characters -> {len(toks)} tokens ({len(text) / max(len(toks), 1):.2f} chars per token)")
            show_pieces(toks)
            print()

        if compare or not (text or chat):
            print(f"{'':12} {'chars':>6} {'tokens':>7} {'chars/token':>12}")
            for name, sample in SAMPLES:
                toks = tok.tokenize(sample)
                ratio = len(sample) / len(toks)
                print(f"{name:12} {len(sample):6d} {len(toks):7d} {ratio:12.2f}  {'█' * round(ratio * 3)}")
            print()
            for name, sample in SAMPLES:
                print(name)
                show_pieces(tok.tokenize(sample))
            print()

        if chat:
            user_text = text or "Hi"
            messages = [{"role": "user", "content": user_text}]
            rendered = tok.chat_template(messages)
            print("What the model actually reads (llama.cpp's copy of the chat template):")
            print("  " + rendered.replace("\n", "\n  "))
            n_rendered = len(tok.tokenize(rendered))
            n_text = len(tok.tokenize(user_text))
            print(f"  your text: {n_text} tokens, after the template: {n_rendered} tokens")

            # Ollama uses its own template, so ask Ollama how many tokens it really processed.
            r = ollama.chat(model, messages, max_tokens=1)
            print(f"  Ollama says it processed {r.get('prompt_eval_count')} prompt tokens for the same message")
