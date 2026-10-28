# Ticket: Request Latency Predictor

## Background
You are the first ML engineer at a startup whose writing assistant runs AI models
locally on the user's computer. It ships two models:

- `fast`: quick, lower quality
- `quality`: slower, better answers

Users complain they never know how long "Generate" will take: sometimes 1 second,
sometimes 20. The product manager wants:

1. **An ETA**: before a request runs, show "≈ 4 seconds".
2. **Routing**: each user sets a wait limit (e.g. 5 s). If `quality` is predicted to
   exceed it, send the request to `fast` instead.

The ops team logged 3,000 real requests for you. The data is in `data/`.

## The data

| File | Rows | Use |
|---|---|---|
| `data/train.csv` | 2,400 | Training. Split off your own validation set from this. |
| `data/test.csv` | 600 | **Do not open until you are done.** Run your final model on it exactly once. |

| Column | Meaning | Known before the request runs? |
|---|---|---|
| `request_id` | Order the request arrived in | yes (but meaningless as an input) |
| `model` | `fast` or `quality` | yes |
| `prompt_tokens` | Length of the user's prompt, in tokens | yes |
| `max_tokens` | Answer length the user picked: 50, 150 or 400 | yes |
| `seconds_since_last_request` | How long the app sat idle before this request | yes |
| `output_tokens` | How many tokens the model actually generated | **no**: only known after |
| `time_to_first_token_s` | Seconds until the first word appeared | **no**: only known after |
| `total_time_s` | Seconds until the answer finished. **This is what you predict.** | (target) |

Your predictor may use only the columns marked **yes**. The "no" columns are still
useful: study them to understand *why* requests take as long as they do.

Note: the logs are real-world messy. Not every row is "normal".

## Deliverables
1. **Predictor built from scratch.** Python + NumPy only (`pip3 install numpy`).
   No scikit-learn, no PyTorch, no autograd. You write the forward pass, loss,
   gradients and updates yourself.
2. **Training report** with:
   - loss curve over epochs
   - training vs validation loss on one chart
   - comparison of at least 3 learning rates
   - comparison of batch, stochastic and mini-batch training
   (charts: `pip3 install matplotlib`, or print tables if you prefer)
3. **Command-line tool**: takes model, prompt length, max_tokens, idle seconds and a
   wait limit. It prints the predicted time and which model it routes to.
4. **One-page write-up**: what you found, what didn't work, and the tokens per second
   of each model (work this out from the data).

## Acceptance criteria
- **Baseline** = "always predict the average `total_time_s` of the training data".
  Your mean absolute error on `test.csv` must be **at least 60% lower** than the
  baseline's error on `test.csv`.
- `test.csv` is used exactly once, at the end.
- Your write-up answers every question below.

## Rules
- No AI coding help: no Copilot, no ChatGPT code, no Claude code.
- Allowed: Python docs, NumPy docs, your own notes.
- When stuck: write down what you tried and what happened before asking anyone.

## Questions you must answer in the write-up
1. Why does training blow up if you feed raw `prompt_tokens` into the neuron?
2. Plot `total_time_s` against `output_tokens` for each model. Is it a straight line?
   What do the slope and the starting point mean physically?
3. Why can't you use `output_tokens` as an input, and what do you use instead?
4. Some requests have a very slow first token. Which column explains it, and why
   would a real computer behave like that?
5. A few rows are far slower than anything similar. What do you do with them, and
   how does that choice change your loss?
6. Does one neuron already beat the baseline? Does a hidden layer help, or overfit?
