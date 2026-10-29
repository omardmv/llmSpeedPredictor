import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

w = 0.2
b= 0.1 
lr= 0.1
batch_size = 32
epochs = 500

w_start, b_start = w, b          # remember the starting guess, to draw it later
loss_history, w_history, b_history = [], [], []

df = pd.read_csv("data/real/train.csv")

clean = df[(df["model"] == "fast") & (df["model_was_loaded"] == 1) & (df["in_flight_total"] == 0)]

## pred = w(seconds/token) * x(tokens Qwen wrote) + b(starting cost which predictor learns)
## error = pred - y(seconds Qwen actually took) 

tokens = clean["output_tokens"].to_numpy() / 1000    # 201 tokens → 0.201
real_times = clean["total_time_s"].to_numpy()

for epoch in range(epochs):
    order = np.random.permutation(len(tokens))

    for start in range(0, len(tokens), batch_size): 
        batch = order[start:start + batch_size] 
        x_batch = tokens[batch]
        y_batch = real_times[batch]


        pred = w * x_batch + b
        error = pred - y_batch

        grad_w =  np.mean(error * x_batch)
        grad_b =  np.mean(error)

        w -= lr * grad_w
        b -= lr * grad_b

    loss = np.mean(0.5 * (w * tokens + b - real_times) ** 2)
    loss_history.append(loss)
    w_history.append(w / 1000)   # store in seconds per token
    b_history.append(b)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss={loss:.4f}, w={w / 1000:.6f} sec/token, b={b:.6f}")

print(f"Final model: time = {w / 1000:.6f} * tokens + {b:.6f}")
print(f"Speed: {1000 / w:.0f} tokens per second")



fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1. Loss: is it learning?
axes[0, 0].plot(loss_history)
axes[0, 0].set_title("Loss (lower = better)")
axes[0, 0].set_xlabel("epoch")

# 2. w: heading to the right answer?
axes[0, 1].plot(w_history, label="w (learned)")
axes[0, 1].axhline(0.0044, color="green", linestyle="--", label="correct answer ≈ 0.0044")
axes[0, 1].set_title("w: seconds per token")
axes[0, 1].set_xlabel("epoch")
axes[0, 1].legend()

# 3. b: heading to the right answer?
axes[1, 0].plot(b_history, label="b (learned)")
axes[1, 0].axhline(0.088, color="green", linestyle="--", label="correct answer ≈ 0.088")
axes[1, 0].set_title("b: starting cost (seconds)")
axes[1, 0].set_xlabel("epoch")
axes[1, 0].legend()

# 4. The data with the line before and after training
real_tokens = tokens * 1000                      # back to normal tokens for the picture
line_x = np.linspace(0, real_tokens.max(), 100)  # 100 evenly spaced points for drawing lines
axes[1, 1].scatter(real_tokens, real_times, s=6, alpha=0.4, label="real requests")
axes[1, 1].plot(line_x, w_start / 1000 * line_x + b_start, "r--", label="before training")
axes[1, 1].plot(line_x, w / 1000 * line_x + b, "g-", linewidth=2, label="after training")
axes[1, 1].set_title("Your predictor vs the data")
axes[1, 1].set_xlabel("output tokens")
axes[1, 1].set_ylabel("seconds")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("research/charts/training.png")
plt.show()
