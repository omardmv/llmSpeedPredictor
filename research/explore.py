from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent.parent
CHARTS = Path(__file__).parent / "charts"

MODELS = ["fast", "quality"]
MAX_TOKEN_BINS = [0, 32, 64, 128, 256, 512, 1024]
PROMPT_BINS = [0, 50, 100, 250, 500, 1000, 4000]

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


# ---------- helpers ----------

def load_data():
    df = pd.read_csv(ROOT / "data" / "real" / "train.csv")
    df["max_tokens_bin"] = pd.cut(df["max_tokens"], bins=MAX_TOKEN_BINS)
    df["prompt_bin"] = pd.cut(df["prompt_tokens"], bins=PROMPT_BINS)
    return df


def clean(df):
    """Requests with no outside effects: model already loaded, nothing else running."""
    return df[(df["model_was_loaded"] == 1) & (df["in_flight_total"] == 0)]


def section(title):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def save_and_show(name):
    CHARTS.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(CHARTS / f"{name}.png")
    plt.show()


# ---------- tables ----------

def overview(df):
    section("Overview")
    print("rows:", len(df), "| columns:", len(df.columns))
    print(df.head())
    print(df.describe())


def time_by_max_tokens(df):
    section("Total time by model x max_tokens range")
    print(df.groupby(["model", "max_tokens_bin"], observed=True)["total_time_s"].agg(["mean", "median", "count"]).round(3))


def max_tokens_usage(df):
    section("How often does the answer hit its max_tokens limit?")
    hit = df["stop_reason"] == "length"
    print((hit.groupby([df["model"], df["max_tokens_bin"]], observed=True).mean() * 100).round(1).rename("% hit limit"))


def cold_vs_warm(df):
    section("Cold start (model_was_loaded = 0) vs already loaded")
    cols = ["time_to_first_token_s", "total_time_s", "load_time_s"]
    print(df.groupby(["model", "model_was_loaded"])[cols].median().round(3))
    print(df.groupby(["model", "model_was_loaded"]).size().rename("count"))


def history_effect(df):
    section("Effect of conversation history (clean requests only)")
    c = clean(df)
    cols = ["prompt_tokens", "time_to_first_token_s", "total_time_s"]
    print(c.groupby("history_turns")[cols].median().round(3))


def concurrency_effect(df):
    section("Effect of other requests in flight")
    in_flight = df["in_flight_total"].clip(upper=6).rename("in_flight (6 = 6+)")
    print(df.groupby(in_flight)["total_time_s"].agg(["mean", "median", "count"]).round(3))


# ---------- charts ----------

def chart_output_vs_time(df):
    """Clean requests only, so you see the pure tokens -> time relationship."""
    c = clean(df)
    plt.figure(figsize=(8, 5))
    for model in MODELS:
        m = c[c["model"] == model]
        plt.scatter(m["output_tokens"], m["total_time_s"], s=6, alpha=0.5, label=model)
    plt.xlabel("output tokens")
    plt.ylabel("total time (s)")
    plt.title("Output length vs total time (loaded, alone on server)")
    plt.legend()
    save_and_show("output_vs_time")


def chart_max_tokens_vs_time(df):
    """What the predictor can actually see: the limit, not the real answer length."""
    plt.figure(figsize=(8, 5))
    for model in MODELS:
        m = df[df["model"] == model]
        plt.scatter(m["max_tokens"], m["total_time_s"], s=6, alpha=0.5, label=model)
    plt.xscale("log")
    plt.xlabel("max_tokens (log scale)")
    plt.ylabel("total time (s)")
    plt.title("Answer limit vs total time (all requests)")
    plt.legend()
    save_and_show("max_tokens_vs_time")


def chart_prompt_vs_first_token(df):
    c = clean(df)
    plt.figure(figsize=(8, 5))
    for model in MODELS:
        m = c[c["model"] == model]
        plt.scatter(m["prompt_tokens"], m["time_to_first_token_s"], s=6, alpha=0.5, label=model)
    plt.xscale("log")
    plt.xlabel("prompt tokens (log scale)")
    plt.ylabel("time to first token (s)")
    plt.title("Prompt length vs time to first token (loaded, alone on server)")
    plt.legend()
    save_and_show("prompt_vs_first_token")


def chart_in_flight_vs_time(df):
    groups = [df[df["in_flight_total"].clip(upper=6) == k]["total_time_s"] for k in range(7)]
    plt.figure(figsize=(8, 5))
    plt.boxplot(groups, tick_labels=[str(k) for k in range(6)] + ["6+"], showfliers=False)
    plt.xlabel("requests already in flight")
    plt.ylabel("total time (s)")
    plt.title("Server load vs total time")
    save_and_show("in_flight_vs_time")


# ---------- run ----------

if __name__ == "__main__":
    df = load_data()

    overview(df)
    time_by_max_tokens(df)
    max_tokens_usage(df)
    cold_vs_warm(df)
    history_effect(df)
    concurrency_effect(df)

    chart_output_vs_time(df)
    chart_max_tokens_vs_time(df)
    chart_prompt_vs_first_token(df)
    chart_in_flight_vs_time(df)
