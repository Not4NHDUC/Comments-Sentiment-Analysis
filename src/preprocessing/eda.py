import os
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8")

_LABEL_COLORS = {"Positive": "green", "Neutral": "gray", "Negative": "red"}
_LABEL_ORDER = ["Positive", "Neutral", "Negative"]


def plot_label_distribution(df: pd.DataFrame, save_path: str) -> None:
    counts = df["label"].value_counts().reindex(_LABEL_ORDER, fill_value=0)
    colors = [_LABEL_COLORS[l] for l in counts.index]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white")
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.01,
            str(int(bar.get_height())),
            ha="center", va="bottom", fontsize=11,
        )
    ax.set_title("Label Distribution", fontsize=14)
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.set_ylim(0, counts.max() * 1.12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_review_length(df: pd.DataFrame, save_path: str) -> None:
    lengths = df["text_clean"].dropna().apply(lambda t: len(str(t).split()))
    mean_len = lengths.mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lengths, bins=40, color="steelblue", edgecolor="white")
    ax.axvline(mean_len, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean: {mean_len:.1f}")
    ax.set_title("Review Length Distribution (words)", fontsize=14)
    ax.set_xlabel("Word count")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_top_words(df: pd.DataFrame, save_path: str, top_n: int = 20) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, label in zip(axes, _LABEL_ORDER):
        subset = df[df["label"] == label]["text_clean"].dropna()
        words = [w for text in subset for w in str(text).split()]
        top = Counter(words).most_common(top_n)
        if not top:
            ax.set_title(label)
            continue
        words_list, counts = zip(*reversed(top))
        ax.barh(words_list, counts, color=_LABEL_COLORS[label])
        ax.set_title(f"Top {top_n} words — {label}", fontsize=12)
        ax.set_xlabel("Frequency")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def run_eda(df: pd.DataFrame, out_dir: str = "reports/eda") -> None:
    os.makedirs(out_dir, exist_ok=True)

    plot_label_distribution(df, os.path.join(out_dir, "label_distribution.png"))
    plot_review_length(df, os.path.join(out_dir, "review_length.png"))
    plot_top_words(df, os.path.join(out_dir, "top_words.png"))

    print(f"[eda] Saved plots to {out_dir}")
