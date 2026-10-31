"""Look inside your local LLM server.

    python -m bench hw                     # what the machine is, what it is doing now
    python -m bench hw --watch             # live GPU / memory view (keep it open in a 2nd terminal)
    python -m bench model quality          # architecture, memory, KV cache size
    python -m bench tokens "some text"     # how text is cut into tokens
    python -m bench tokens --compare       # tokens per character across languages and code
    python -m bench run quality            # one request, split into load / prefill / decode
    python -m bench run quality --cold     # same, starting from an unloaded model
    python -m bench check                  # is the machine quiet enough to benchmark?
    python -m bench sweep quality          # 1, 2, 4, 8, 16 users at once (runs the check first)
    python -m bench sweep quality --repeat 2   # twice, and report how much the runs differ
    python -m bench sweep quality --backend llama   # measure llama-server (start it yourself first)
    python -m bench slots quality --np 1,2,4,8,16    # start llama-server with each slot count and sweep it

`fast` = qwen2.5:0.5b and `quality` = qwen2.5:1.5b; any Ollama model name also works.
"""

import argparse


def main():
    p = argparse.ArgumentParser(prog="python -m bench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("hw", help="machine + live GPU / memory")
    s.add_argument("--watch", action="store_true")

    s = sub.add_parser("model", help="model architecture and memory")
    s.add_argument("model", nargs="?", default="quality")
    s.add_argument("--ctx", type=int, default=4096, help="context size to load with")
    s.add_argument("--no-load", action="store_true", help="only read the file, don't load it")
    s.add_argument("--json", action="store_true", help="machine-readable output (used by the GUI app)")

    s = sub.add_parser("tokens", help="see how text becomes tokens")
    s.add_argument("text", nargs="?")
    s.add_argument("--model", default="quality")
    s.add_argument("--compare", action="store_true", help="compare languages / code / numbers")
    s.add_argument("--chat", action="store_true", help="show what the chat template adds around your text")

    s = sub.add_parser("run", help="time one request, phase by phase")
    s.add_argument("model", nargs="?", default="quality")
    s.add_argument("--prompt", default="Explain how a car engine works.")
    s.add_argument("--pad", type=int, default=0, help="prepend N paragraphs to make the prompt longer")
    s.add_argument("--max-tokens", type=int, default=256)
    s.add_argument("--cold", action="store_true", help="unload the model first")
    s.add_argument("--repeat", type=int, default=1, help="send the same request N times in a row")
    s.add_argument("--backend", default="ollama", choices=["ollama", "llama"], help="which server to measure")
    s.add_argument("--url", default="http://localhost:8080", help="llama-server address (with --backend llama)")
    s.add_argument("--json", action="store_true", help="machine-readable output (used by the GUI app)")

    s = sub.add_parser("sweep", help="many users at once")
    s.add_argument("model", nargs="?", default="quality")
    s.add_argument("--users", default="1,2,4,8,16", help="concurrency levels")
    s.add_argument("--requests", type=int, default=3, help="requests per user at each level")
    s.add_argument("--max-tokens", type=int, default=400)
    s.add_argument("--repeat", type=int, default=1, help="run the whole sweep N times and report how much the runs differ")
    s.add_argument("--force", action="store_true", help="run even if the quiet check fails")
    s.add_argument("--no-warmup", action="store_true", help="skip heating the GPU to a stable temperature first")
    s.add_argument("--backend", default="ollama", choices=["ollama", "llama"], help="which server to measure")
    s.add_argument("--url", default="http://localhost:8080", help="llama-server address (with --backend llama)")
    s.add_argument("--json", action="store_true", help="machine-readable output (used by the GUI app)")

    s = sub.add_parser("slots", help="sweep several llama-server slot counts, one after another")
    s.add_argument("model", nargs="?", default="quality")
    s.add_argument("--np", default="1,2,4,8,16", help="slot counts to test")
    s.add_argument("--users", default="1,2,4,8,16", help="concurrency levels in each sweep")
    s.add_argument("--requests", type=int, default=3, help="requests per user at each level")
    s.add_argument("--max-tokens", type=int, default=400)
    s.add_argument("--repeat", type=int, default=2)

    s = sub.add_parser("overhead", help="where llama-server loses speed vs the bare engine")
    s.add_argument("model", nargs="?", default="quality")
    s.add_argument("--np", type=int, default=8, help="slots = requests decoded together")

    s = sub.add_parser("check", help="is the machine quiet enough to benchmark?")
    s.add_argument("--seconds", type=float, default=5)
    s.add_argument("--json", action="store_true", help="machine-readable output (used by the GUI app)")

    a = p.parse_args()
    if getattr(a, "backend", None):
        from . import backend
        backend.use(a.backend, a.url)

    if a.cmd == "hw":
        from . import hw
        if a.watch:
            hw.watch()
        else:
            hw.print_machine()
            print()
            hw.print_now()
    elif a.cmd == "model":
        from . import modelinfo
        if a.json:
            modelinfo.dump(a.model, num_ctx=a.ctx, load=not a.no_load)
        else:
            modelinfo.print_info(a.model, num_ctx=a.ctx, load=not a.no_load)
    elif a.cmd == "tokens":
        from . import tokens
        tokens.main(a.model, a.text, compare=a.compare, chat=a.chat)
    elif a.cmd == "run":
        from . import run
        run.main(a.model, a.prompt, a.pad, a.max_tokens, a.cold, a.repeat, as_json=a.json)
    elif a.cmd == "sweep":
        from . import sweep
        sweep.main(a.model, [int(u) for u in a.users.split(",")], a.requests, a.max_tokens, as_json=a.json,
                   repeat=a.repeat, force=a.force, warmup=not a.no_warmup)
    elif a.cmd == "slots":
        from . import slots
        slots.main(a.model, [int(n) for n in a.np.split(",")], [int(u) for u in a.users.split(",")],
                   a.requests, a.max_tokens, a.repeat)
    elif a.cmd == "overhead":
        from . import overhead
        overhead.main(a.model, a.np)
    elif a.cmd == "check":
        import json
        from . import env
        result = env.preflight(seconds=a.seconds)
        if a.json:
            print(json.dumps(result))
        else:
            env.print_preflight(result)


if __name__ == "__main__":
    main()
