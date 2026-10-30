# Ticket 02: Serve 4× more users on the same Mac

## Background
The product team wants to open the chat to more people. Finance says no new hardware:
it has to run on the one M4 Pro we have.

Ticket 01 showed the problem. With 8 users at once, the `quality` model still gives about
150 tokens/s *in total*, and the wait for the first token grows by roughly 0.75 s with
every extra user. Requests wait in line one after another; nothing is shared.

Production servers (vLLM, TGI, SGLang, llama.cpp's `llama-server`) handle this with
**continuous batching**: every decode step runs the next token for *all* active users
together. Ollama can do it too, but on this machine it starts its internal `llama-server`
with one slot.

Your job: find a server setup that serves **at least 4× more users** than today while
keeping the same latency promise, and write it up so ops can deploy it.

## The promise (SLO)
For the `quality` model, `max_tokens` 400, real Dolly prompts:

- **p95 time to first token ≤ 2.0 s**
- **median speed each user sees ≥ 20 tokens/s** (a little faster than people read)

"Users served" = the largest concurrency level in your sweep where **both** hold.

## What you have
- `bench/` and the Inference Lab app from ticket 01 (they only talk to Ollama for now)
- `llama-server` (llama.cpp) installed with Homebrew
- The GGUF files Ollama downloaded: `python -c "from bench import ollama; print(ollama.gguf_path('quality'))"`
- Your sweep from ticket 01 in `research/bench/` as the baseline

## Tasks

### A. Baseline (no code)
1. Run `python -m bench sweep quality --users 1,2,4,8,16` against Ollama, with the
   machine quiet. Record how many users meet the SLO. This is the number to beat ×4.

### B. Teach `bench` to talk to llama-server
2. Add a `--backend ollama|llama` option to `run` and `sweep` (default `ollama`), plus
   `--url` for where llama-server listens. Keep the output identical for both backends
   (same columns, same JSON lines), so charts, CSVs and the app work unchanged.
3. Start the server yourself in a second terminal, e.g.
   `llama-server -m <gguf> --port 8080 -np 4 -c 16384 --metrics`

   Facts you need (checked on this machine, llama.cpp build 11669):
   - Endpoint: `POST /v1/chat/completions` with `"stream": true` and
     `"stream_options": {"include_usage": true}`. Each line looks like `data: {...}`;
     the stream ends with `data: [DONE]`.
   - Text pieces arrive in `choices[0].delta.content`, one chunk per token (check this).
   - The last chunk before `[DONE]` has `usage` and `timings`: `prompt_n`, `prompt_ms`,
     `predicted_n`, `predicted_ms`, `cache_n`. Those map onto Ollama's
     `prompt_eval_count/duration` and `eval_count/duration`. Watch the units:
     Ollama sends nanoseconds, llama-server milliseconds.
   - `-np N` = number of parallel slots (users served at the same time).
   - `-c` is the **total** context, split evenly across slots: `-c 8192 -np 4` gives
     each slot 2,048 tokens (the startup log shows `n_ctx_slot`).
   - `GET /slots` shows what every slot is doing; `GET /metrics` (with `--metrics`) gives counters.
   - Continuous batching is on by default (`-cb`); `-nocb` turns it off.
4. Prove the new backend measures the same thing as the old one: with `-np 1`, the
   single-user numbers from `bench run` must be within ~10% of Ollama's. If they're not,
   find out why before going on.

### C. The batching experiment
5. **Predict first.** Before any run, write down for `-np 1, 2, 4, 8, 16`: total
   throughput, per-user speed, p95 TTFT. Use your roofline answer from ticket 01: if decode
   only has to read the weights once per step, what does adding a second user cost?
6. Sweep every `-np` value with users = 1, 2, 4, 8, 16, 32. Same `max_tokens`, same prompts.
7. For each run, pick `-c` on purpose. Each slot needs room for prompt + answer; your
   longest Dolly prompt + 400 tokens is the minimum. Show the math with your
   KV-per-token formula and check it against the memory the Live view shows.

### D. Quantization (the second lever)
8. `ollama pull qwen2.5:1.5b-instruct-q8_0` (about 1.6 GB) gives an 8-bit version of the
   same model (if the tag has changed, look it up at ollama.com/library/qwen2.5/tags). Repeat your best `-np` setting with Q8 and compare with Q4: decode speed,
   total throughput, memory.
9. Speed alone isn't enough. Make a small quality check: 30 questions with short, known
   answers (Dolly `closed_qa`, or write your own facts and arithmetic). Run them at
   temperature 0 on Q4 and Q8 and score both. Was the speed worth what it cost?

### E. Decision memo
10. Write `research/bench/02-decision.md`, at most one page:
    - a table of every setup you tried (backend, `-np`, `-c`, quant, users served, total tok/s,
      p95 TTFT, per-user tok/s, memory)
    - the setup you recommend, and the exact command to start it
    - users served before → after, and the ratio
    - what got worse (there is always something) and when you'd choose differently
    - one chart: total throughput vs. p95 TTFT, one line per `-np`

## Acceptance criteria
- `bench run` and `bench sweep` work with both backends; Ollama results unchanged.
- The `-np 1` check (task 4) passes, or the memo explains the difference.
- The recommended setup serves **≥ 4× the baseline users** within the SLO, measured, not
  predicted. Stretch goal: 6×.
- Every prediction from task 5 sits next to the measured number, with one sentence on each
  one you got wrong.

## Rules
- Same as before: code mechanics can come from anywhere; the predictions, the math and the
  decision are yours.
- Change one thing at a time. If two settings changed between runs, the run doesn't count.
- Close heavy apps and keep the laptop plugged in. Run each important setting twice; if the
  two runs differ by more than 10%, find out why before trusting either.
- Never run two servers on the GPU at once: quit Ollama (or `ollama stop` every model)
  while you test llama-server.

## Questions to answer in the memo
1. Why does total throughput go up with more slots, and why does it stop going up? What is
   the GPU's limit at that point: memory bandwidth or compute? Show it with numbers.
2. Why does each user's speed drop as slots increase, even though the server gets faster overall?
3. What happens to TTFT when users > slots? When slots > users?
4. Turn off continuous batching (`-nocb`) at your best `-np`. What changes, and why?
5. A new request arrives while 7 others are decoding. Its prefill has to share the GPU with
   their decode steps. What does that do to *their* token gaps? (Look at p95/max gap, not
   the median.) This is the problem "chunked prefill" and "disaggregated serving" solve.
6. With your chosen `-c` and `-np`, what happens to a user whose conversation is longer
   than one slot? How would you choose `-c` for real traffic like `data/real/`?
7. How many tokens per day can this one Mac serve at your setting, at 50% average load?

## Not in this ticket
Speculative decoding, multiple models at once, prefix-cache tuning. They come next, and
each one needs the batching numbers from this ticket as its baseline.
