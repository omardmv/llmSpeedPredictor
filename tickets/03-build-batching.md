# Ticket 03: Build an LLM engine with batching, from scratch (NumPy)

## Background
In ticket 02 you measured batching in llama-server: 8 slots served 8 users at ~520 tok/s,
1 slot served 1 user at ~150 tok/s. You also measured *why* it works: decode is memory-bound,
so 8 users can share each read of the weights. But it was a flag (`-np 8`) on someone else's engine.

Now you build the engine. Same rule as the predictor ticket: **Python + NumPy only for the
model.** No PyTorch, no transformers, no llama.cpp. You write the forward pass, the KV cache,
sampling, and batching. When batching speeds things up, you'll know exactly which line of your
code made that happen.

## The model: Qwen2.5-0.5B-Instruct
The same model as `fast` (qwen2.5:0.5b), but the original 16-bit weights from Hugging Face
instead of Ollama's 4-bit file (unpacking that 4-bit format is its own project).

Download it yourself, into a folder git ignores (~1 GB):

```bash
mkdir -p models/qwen2.5-0.5b-instruct && cd models/qwen2.5-0.5b-instruct
curl -L -O https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors
curl -L -O https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/config.json
curl -L -O https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json
```

Facts (checked against the real files):

| | |
|---|---|
| layers | 24 |
| hidden size | 896 |
| query heads / key-value heads | 14 / 2 (head size 64; every 7 query heads share one K/V head) |
| feed-forward size | 4864, activation SiLU (SwiGLU) |
| vocabulary | 151,936 |
| RoPE base (theta) | 1,000,000 |
| RMSNorm epsilon | 1e-6 |
| weights | `model.safetensors`, 290 tensors, all **bfloat16**, 988 MB |
| output layer | **none in the file**: the embedding matrix is reused (`tie_word_embeddings`), so scores = hidden @ embed.T |

Tensor names for layer 0 (shapes are `[out, in]`, so a layer computes `x @ W.T + b`):

```
model.embed_tokens.weight                     [151936, 896]
model.layers.0.input_layernorm.weight         [896]
model.layers.0.self_attn.q_proj.weight/.bias  [896, 896] / [896]
model.layers.0.self_attn.k_proj.weight/.bias  [128, 896] / [128]
model.layers.0.self_attn.v_proj.weight/.bias  [128, 896] / [128]
model.layers.0.self_attn.o_proj.weight        [896, 896]     (no bias)
model.layers.0.post_attention_layernorm.weight [896]
model.layers.0.mlp.gate_proj.weight           [4864, 896]
model.layers.0.mlp.up_proj.weight             [4864, 896]
model.layers.0.mlp.down_proj.weight           [896, 4864]
model.norm.weight                             [896]
```

## What runs where
NumPy runs on the **CPU**, not the GPU. That's fine: CPU decode is memory-bound too, so the
batching lesson is the same. You'll compare your engine **with itself** (1 user vs 8),
not with llama-server's absolute speed. A later step can port it to MLX (NumPy-like, runs on
the GPU).

Tokenizer: tokenization isn't the lesson, so you may use the `tokenizers` library with
`tokenizer.json` (`pip install tokenizers`). Writing the byte-level BPE yourself is a stretch goal.

## Parts

### A. Read the weights yourself
1. A `.safetensors` file is: 8 bytes (little-endian integer N), then N bytes of JSON
   (name → dtype, shape, byte offsets), then raw data. Parse it with `struct` + `json` +
   `np.frombuffer`.
2. NumPy has no bfloat16. bfloat16 is the top 16 bits of a float32, so read the data as `uint16`,
   shift it into the top half of a `uint32`, and view that as `float32`.
3. Check: total parameters (≈ 494 million) and memory in float32. Predict both first.

### B. One forward pass, no cache
4. Write the pieces as functions: RMSNorm, RoPE, attention (with grouped KV heads and a causal
   mask), SwiGLU feed-forward, the final norm and the tied output.
5. RoPE detail that trips everyone: this model uses the "rotate half" layout. For each head,
   the first 32 numbers pair with the last 32 (not neighbour pairs).
6. Run the whole prompt through all 24 layers and take the top-scoring next token.
   **Correctness check:** for 10 different prompts, compare your greedy next tokens with
   llama-server running the 0.5B model. Its file is 4-bit, so small differences are expected;
   the top token should match almost always.

### C. KV cache and decode, one user
7. Keep each layer's K and V in arrays that grow by one row per token. Prefill = the whole
   prompt at once; decode = one token per step, reusing the cache.
8. Time it: prefill tok/s, decode tok/s, ms per step. **Predict first:** each decode step reads
   all the weights once. How many bytes is that in float32, and what speed does that give at
   your CPU's memory bandwidth?
9. Check your KV cache size against your ticket 01 formula (12 KB per token in 16-bit;
   what is it in your float32 arrays?).

### D. Static batching
10. Decode B users in one step: stack their current tokens into a `(B, 896)` matrix, so every
    weight multiply becomes `matrix @ W.T` instead of `vector @ W.T`. Attention stays per user
    (each has its own cache and length).
11. Measure ms per step and total tok/s for B = 1, 2, 4, 8, 16, 32. Where does adding a user
    stop being nearly free on your CPU? Explain it with bytes read vs math done.
12. Count the capacity wasted when one user's answer ends early but the batch keeps running.

### E. Continuous batching
13. A scheduler loop: every step, remove finished users (free their cache), admit waiting ones
    (prefill them), build one batch of everyone's next token, run the step, sample, stream.
14. Decide how a newcomer's prefill fits in (same step as everyone's decode, its own step, or
    in chunks) and measure what each choice does to the other users' token gaps.

### F. Optional
15. Serve it over HTTP like llama-server, so `bench sweep --backend llama --url ...` and the
    History screen can measure it (Claude may write that HTTP shell; the engine stays yours).
16. Port the engine to MLX and run it on the GPU. How close do you get to llama.cpp?

## Acceptance criteria
- Weights loaded by your own parser; forward pass uses only NumPy.
- Greedy next token matches llama-server on at least 9 of 10 test prompts.
- KV-cache decode produces the same tokens as the no-cache forward pass.
- Batching: total tok/s at B = 8 is **at least 3×** B = 1, measured.
- Continuous batching keeps at least 90% of slots busy under a steady stream of requests.
- Write-up `research/bench/03-engine.md`: every prediction next to its measurement, and the
  answers below.

## Rules
- The engine is yours: weight loading, every layer, KV cache, sampling, batching, scheduler.
  Claude explains math, NumPy syntax and concepts when asked, and may write measuring tools
  and the optional HTTP shell.
- Predict before you measure, in writing.
- Same conditions as before: plugged in, OBS closed, nothing heavy running.

## Questions for the write-up
1. In one decode step, how many bytes do you read and how many multiply-adds do you do?
   What does that ratio say about memory-bound vs compute-bound, for B = 1 and B = 32?
2. Why does the KV cache make decode fast, and what does it cost?
3. Why can the weight multiplies be batched across users, but attention can't share the same way?
4. How many query heads share each K/V head here, and how much smaller does that make the cache?
5. Static vs continuous: how much capacity did continuous batching recover, and why?
6. What did your engine teach you about the 1→2 jump you saw in llama.cpp?
