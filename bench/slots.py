"""Sweep several llama-server slot counts one after another, unattended.

    python -m bench slots quality --np 1,2,4,8,16

For each slot count: start llama-server with -np N and -c N x 4096 (same context per user
every time), wait until it is ready, run the normal sweep (with warm-up), stop the server.
Configs never run at the same time: they would share the GPU and spoil each other's numbers.
"""

import json
import subprocess
import time
import urllib.request
from pathlib import Path

from . import backend, llama, ollama, sweep

PORT = 8080
URL = f"http://localhost:{PORT}"
LOGS = Path(__file__).parent.parent / "research" / "bench" / "logs"


def _port_busy():
    try:
        urllib.request.urlopen(URL + "/health", timeout=1)
        return True
    except OSError:
        return False


def _wait_ready(proc, timeout=180):
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"llama-server exited early (code {proc.returncode}); see its log in {LOGS}")
        try:
            with urllib.request.urlopen(URL + "/health", timeout=1) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return
        except OSError:
            pass
        time.sleep(1)
    raise SystemExit("llama-server did not become ready in time")


def main(model, slot_counts, levels, per_user, max_tokens, repeat, ctx_per_slot=4096):
    if _port_busy():
        raise SystemExit(f"Something already answers on port {PORT}. Stop that llama-server first (Ctrl+C in its terminal).")
    if ollama.loaded():
        raise SystemExit("Ollama has a model loaded: run `ollama stop <model>` so llama-server has the GPU to itself.")
    LOGS.mkdir(parents=True, exist_ok=True)
    gguf = ollama.gguf_path(model)
    backend.use("llama", URL)

    for n in slot_counts:
        ctx = n * ctx_per_slot
        print(f"\n{'=' * 70}\n{n} SLOT{'S' if n > 1 else ''}: llama-server -np {n} -c {ctx}\n{'=' * 70}", flush=True)
        log = open(LOGS / f"llama-server-np{n}-c{ctx}.log", "w")
        proc = subprocess.Popen(["llama-server", "-m", gguf, "--port", str(PORT), "-np", str(n), "-c", str(ctx)],
                                stdout=log, stderr=subprocess.STDOUT)
        try:
            _wait_ready(proc)
            s = llama.server(URL)
            print(f"ready: {s['slots']} slots x {s['ctx_per_slot']} tokens context", flush=True)
            # force=True: we checked the machine once at the start; each sweep still records its own check.
            sweep.main(model, levels, per_user, max_tokens, repeat=repeat, force=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
            log.close()
        time.sleep(3)  # let the port close before the next server starts
    print("\nAll slot counts done. Compare them in the app: History.")
