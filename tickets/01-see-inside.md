# Ticket 01: See inside the server

## Background
Users complain the chat is slow when many people use it at once. Before anyone
changes anything, ops wants to know what the machine and the models are actually
doing. You can't tune what you haven't measured.

You have a measuring kit in `bench/`. Run everything from the project root with the
venv active (`source .venv/bin/activate`).

| Command | What it shows |
|---|---|
| `python -m bench hw` | the chip, unified memory, bandwidth, GPU right now, loaded models |
| `python -m bench hw --watch` | live view; keep it open in a second terminal during the others |
| `python -m bench model quality` | layers, heads, bits per weight, KV cache size, real memory once loaded |
| `python -m bench tokens --compare` | how English, Azerbaijani, Russian, code, JSON and numbers get cut up |
| `python -m bench tokens --chat "your text"` | what the chat template adds around your message |
| `python -m bench run quality --cold --repeat 3` | one request split into load / prefill / decode, then the same request warm |
| `python -m bench run quality --pad 8 --max-tokens 64 --repeat 2` | a long prompt, then the same long prompt again |
| `python -m bench sweep quality` | 1, 2, 4, 8, 16 users at once; saves a CSV and a chart in `research/bench/` |

Prefer windows? `app/run.sh` builds and opens **Inference Lab**, a native Mac app with the
same five views (Live, Model, Tokens, One request, Load test). It calls the same `bench`
code, so the numbers match the terminal.

Use `fast` instead of `quality` for the small model. Close heavy apps while you
measure: the GPU and memory are shared with everything else on the laptop.

## Deliverables
1. A write-up in `research/bench/findings.md` answering every question below, with the
   numbers you measured (not the ones in this ticket).
2. For every answer that has a formula, write the formula and plug in the numbers, the
   same way you did the gradient in `notes.md`.

## Rules
Same as the predictor ticket: code mechanics can come from anywhere, the thinking is yours.
Predict before you measure. Write the prediction down first, then run it and compare.

## Questions

### Memory
1. The 1.5B model has 1.54 billion parameters. How many bytes would it take at 16 bits
   per parameter? At the bits-per-parameter `bench model` reports? Why is the 0.5B model
   at more bits per parameter than the 1.5B one, even though both are `Q4_K_M`?
   (Hint: look at the vocabulary size and the hidden size.)
2. Check the KV-cache-per-token formula by hand. Why does it use `kv_heads` and not
   `attention_heads`? How much KV cache would 16 users with a 4,096-token context need?
   Does that fit on this laptop?
3. `bench model` shows "everything else" next to the weights. Split it into KV cache and
   other buffers. How many conversation slots did Ollama reserve? Reload with
   `--ctx 8192` and `--ctx 16384`. Does the number grow the way your formula says?

### Tokens
4. Why does Azerbaijani cost so many more tokens than English for the same sentence?
   What does that mean for the speed and cost an Azerbaijani user gets?
5. Why are numbers one token per digit? What could that mean for a model doing arithmetic?
6. How many tokens does the chat template add to every message? When does that overhead matter?

### One request
7. In `run --cold`, which phase does the cold start add? Where do the weights come from
   when "load" is that short? (What would happen right after a reboot?)
8. Prefill reads hundreds or thousands of tokens per second, decode only ~150. Both use the
   same weights. Explain the difference with "compute-bound" and "memory-bound" from your notes.
9. Roofline: during decode, every new token needs all the weights read from memory once.
   Using the bandwidth from `bench hw` and the weight size from `bench model`, what is the
   *fastest* decode could possibly be? How close does the real number get? Do the same for `fast`.
10. Compare prefill tok/s for a 37-token prompt and a ~1,900-token prompt. Why is the long
    one faster per token?
11. The second `--pad 8` request has the same prompt and shows an "impossible" prefill
    speed. What happened? Which kind of real traffic gets this for free, and which doesn't?

### Many users
12. Run `sweep`. Describe the chart in your own words: what does TTFT do, what does
    total throughput do, what does the Gantt chart show?
13. Is the server batching requests or queuing them? Prove it from the numbers.
14. Quit the Ollama app, then start it by hand with
    `OLLAMA_NUM_PARALLEL=4 ollama serve` and run `sweep` again. What changed for TTFT,
    per-user speed, and total throughput? Use your answer to question 9 to explain why
    total throughput can go *up* when the GPU is already "95% busy".
15. Write a one-line SLO the product team could use, e.g. "p95 TTFT under X s for up to N
    users." Which setting gives the most users under that SLO?
