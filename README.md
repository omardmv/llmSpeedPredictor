# Ticket: Request Latency Predictor

## Background
You are the first ML engineer at a startup that runs its own small LLM server.
It serves two models to many users at once:

- `fast`: `qwen2.5:0.5b`, quick, lower quality
- `quality`: `qwen2.5:1.5b`, slower, better answers

Users complain they never know how long a response will take: sometimes a blink,
sometimes 15 seconds. The product manager wants:

1. **An ETA**: before a request is sent, show "≈ 2 seconds".
2. **Routing**: each user sets a wait limit (e.g. 1.5 s). If `quality` is predicted to
   exceed it, send the request to `fast` instead.

The ops team recorded 2,000 real requests from the server. The data is in `data/real/`.

## How the data was recorded
`collect.py` sent real traffic to the real models on one machine (Apple M4 Pro):

- **Real human prompts** from the databricks-dolly-15k dataset (questions, summaries,
  brainstorming, classification, …). About 30% continue an earlier conversation, so they
  carry 1–6 previous question/answer pairs.
- **Random arrivals**: requests arrive at random times and can overlap. Traffic switches
  between quiet, light, normal and busy periods, so sometimes a request has the server to
  itself and sometimes it waits behind 10 others.
- **Natural cold starts**: a model unloads from memory after 20 s without requests. The
  next request has to load it again.
- **Answer limits** anywhere from 16 to 1,024 tokens, and 4 different temperatures.

Run it again with `python collect.py` (about 1.5–2 h). Don't do that while you are
training. It needs the machine to itself.

## The data

| File | Rows | Use |
|---|---|---|
| `data/real/train.csv` | 1,600 | Training. Split off your own validation set from this. |
| `data/real/test.csv` | 400 | **Do not open until you are done.** Run your final model on it exactly once. |
| `data/real/requests.csv` | 2,000 | Everything, before the split. Don't use it for training. |

### Columns your predictor MAY use (known before the request is sent)

| Column | Meaning |
|---|---|
| `model` | `fast` or `quality` |
| `prompt_category` | Kind of request: `open_qa`, `brainstorming`, `summarization`, … |
| `history_turns` | Earlier question/answer pairs sent along with the new question (0–6) |
| `prompt_chars` | Total length of everything sent, in characters |
| `prompt_tokens` | Same, in tokens (an app can count these before sending) |
| `max_tokens` | Answer length limit for this request (16–1024) |
| `temperature` | Randomness setting: 0, 0.3, 0.7 or 1.0 |
| `in_flight_total` | Requests already running on the server when this one was sent |
| `in_flight_same_model` | Same, counting only the same model |
| `requests_last_60s` | Requests sent in the previous minute (how busy it is) |
| `seconds_since_last_same_model` | Seconds since this model last got a request (`-1` = first ever) |
| `model_was_loaded` | `1` if the model was already in memory, `0` if it had to be loaded |

### Columns your predictor may NOT use

| Column | Why not |
|---|---|
| `total_time_s` | **The target**: seconds from sending until the answer finished |
| `output_tokens` | Only known after the answer finishes |
| `stop_reason` | `stop` = the model finished, `length` = it hit `max_tokens`. Only known after |
| `time_to_first_token_s` | Seconds until the first word appeared. Only known after |
| `load_time_s` | Time spent loading the model into memory. Only known after |
| `prompt_eval_time_s` | Time spent reading the prompt. Only known after |
| `traffic` | Label from the traffic generator. A real server doesn't know it; use `requests_last_60s` instead |
| `request_id`, `send_time`, `model_name` | IDs and timestamps, not useful inputs |
| `error` | Always empty in this data |

The "may NOT use" columns are still very useful: study them to understand *why* a
request took as long as it did.

## Deliverables
1. **Predictor built from scratch**: Python + NumPy only. No scikit-learn, no PyTorch,
   no autograd. You write the forward pass, loss, gradients and updates yourself.
2. **Training report** with:
   - loss curve over epochs
   - training vs validation loss on one chart
   - comparison of at least 3 learning rates
   - comparison of batch, stochastic and mini-batch training
3. **Command-line tool**: takes the allowed inputs plus a wait limit, prints the predicted
   time and which model it routes to.
4. **One-page write-up**: what you found, what didn't work, and the tokens per second of
   each model (work this out from the data).

## Acceptance criteria
- **Baseline** = "always predict the average `total_time_s` of the training data".
- Your mean absolute error on `test.csv` must be **at least 35% lower** than the baseline's.
  Stretch goal: 45%.
- `test.csv` is used exactly once, at the end.
- Your write-up answers every question below.

Why only 35%? This is real data. The biggest part of the time is how long the answer turns
out to be, and nobody knows that before the request runs. Some error can't be removed.
Part of the job is finding out how much.

## Rules
- Code mechanics (pandas, NumPy, plotting syntax) can come from anywhere, including AI.
- The thinking is yours: what to check, what the charts mean, which inputs to use, and the
  model math (gradients and updates).
- When stuck: write down what you tried and what happened before asking anyone.

## Questions you must answer in the write-up
1. Why does training blow up if you feed raw `prompt_tokens` or `max_tokens` into the neuron?
2. Plot `total_time_s` against `output_tokens` for each model, using only requests that
   were alone on the server and already loaded. Is it a straight line? What do the slope
   and starting point mean physically?
3. Why can't you use `output_tokens` as an input, and what do you use instead? How often
   does the answer use its whole `max_tokens` limit?
4. How much slower is a cold start? Which column predicts it?
5. How does a long conversation history change the time? Which part of the time grows,
   the first token or the rest?
6. What happens to a request's time when other requests are in flight? Is the effect a
   straight line?
7. Does one neuron beat the baseline? Does a hidden layer help, or does it overfit?
8. Which requests does your model get most wrong, and why?
