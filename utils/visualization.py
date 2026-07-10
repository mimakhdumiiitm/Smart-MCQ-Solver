# visualization.py
# All plotting functions for the EDA pipeline

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

from config.config import (
    OPTION_COLS, PLOT_STYLE, COLORS, FIGURE_DPI,
    SAVE_PLOTS, OUTPUT_DIR, ANSWER_COL
)


# ------------------------------------------------------------------
# SETUP
# ------------------------------------------------------------------
def setup_plot_style() -> None:
    """Apply the global matplotlib style from config."""
    plt.style.use(PLOT_STYLE)
    sns.set_palette(COLORS)


def save_figure(fig: plt.Figure, filename: str) -> None:
    if SAVE_PLOTS:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, filename)
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"Figure saved to {path}")


# ------------------------------------------------------------------
# ANSWER DISTRIBUTION
# ------------------------------------------------------------------

def plot_answer_distribution(df: pd.DataFrame,
                              answer_col = ANSWER_COL,
                              title: str = "Answer Label Distribution") -> None:
    setup_plot_style()
    counts = df[answer_col].value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    # Bar
    bars = axes[0].bar(
        counts.index, counts.values,
        color=COLORS[:len(counts)], edgecolor="black", linewidth=0.8
    )
    axes[0].set_title("Count per Label")
    axes[0].set_xlabel("Answer Label")
    axes[0].set_ylabel("Count")
    for bar, val in zip(bars, counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            str(val), ha="center", fontweight="bold"
        )

    # Pie
    axes[1].pie(
        counts.values, labels=counts.index,
        autopct="%1.1f%%", colors=COLORS[:len(counts)],
        startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2)
    )
    axes[1].set_title("Proportion per Label")

    plt.tight_layout()
    save_figure(fig, "answer_distribution.png")
    plt.show()


# ------------------------------------------------------------------
# TEXT LENGTH DISTRIBUTIONS
# ------------------------------------------------------------------

def plot_text_length_distributions(df: pd.DataFrame,
                                   cols: list = None) -> None:

    setup_plot_style()
    if cols is None:
        cols = ["prompt"] + OPTION_COLS

    n    = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    fig.suptitle("Word Count Distributions", fontsize=14, fontweight="bold")

    for ax, col, color in zip(axes, cols, COLORS * 3):
        if col in df.columns:
            lengths = df[col].apply(lambda x: len(str(x).split()))
            ax.hist(lengths, bins=20, color=color, edgecolor="black", alpha=0.85)
            ax.set_title(f"Column: {col}")
            ax.set_xlabel("Word Count")
            ax.set_ylabel("Frequency")
            ax.axvline(
                lengths.mean(), color="red",
                linestyle="--", label=f"Mean={lengths.mean():.1f}"
            )
            ax.legend(fontsize=8)

    plt.tight_layout()
    save_figure(fig, "text_length_distributions.png")
    plt.show()


# ------------------------------------------------------------------
# WORD FREQUENCY
# ------------------------------------------------------------------

def plot_top_words(word_freq: Counter,
                   title: str = "Top Words",
                   n: int     = 20,
                   color: str = None,
                   filename: str = "top_words.png") -> None:
    setup_plot_style()
    if color is None:
        color = COLORS[0]

    top    = word_freq.most_common(n)
    words, counts = zip(*top)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(list(words)[::-1], list(counts)[::-1], color=color, edgecolor="black")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Frequency")
    ax.axvline(
        np.mean(list(counts)), color="red",
        linestyle="--", label="Mean"
    )
    ax.legend()
    plt.tight_layout()
    save_figure(fig, filename)
    plt.show()


def plot_wordcloud(tokens: list,
                   title: str    = "Word Cloud",
                   colormap: str = "Blues",
                   filename: str = "wordcloud.png") -> None:

    setup_plot_style()
    text = " ".join(tokens)
    wc   = WordCloud(
        width=800, height=400,
        background_color="white",
        colormap=colormap,
        max_words=100
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, filename)
    plt.show()


# ------------------------------------------------------------------
# SIMILARITY PLOTS
# ------------------------------------------------------------------

def plot_similarity_distributions(correct_sims: list,
                                  incorrect_sims: list,
                                  method_name: str = "TF-IDF",
                                  filename: str    = "similarity_dist.png") -> None:

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(
        incorrect_sims, bins=20, alpha=0.6, color=COLORS[3],
        label=f"Incorrect (mean={np.mean(incorrect_sims):.3f})",
        edgecolor="black"
    )
    ax.hist(
        correct_sims, bins=20, alpha=0.6, color=COLORS[0],
        label=f"Correct (mean={np.mean(correct_sims):.3f})",
        edgecolor="black"
    )
    ax.set_title(
        f"{method_name} Cosine Similarity: Correct vs Incorrect",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.legend()

    plt.tight_layout()
    save_figure(fig, filename)
    plt.show()


def plot_inter_option_heatmap(sim_matrix: np.ndarray,
                              option_cols: list = None,
                              title: str = "Inter-Option Cosine Similarity",
                              filename: str = "inter_option_heatmap.png") -> None:

    setup_plot_style()
    if option_cols is None:
        option_cols = OPTION_COLS

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        sim_matrix,
        xticklabels=option_cols, yticklabels=option_cols,
        annot=True, fmt=".3f", cmap="Blues",
        vmin=0, vmax=1, ax=ax,
        linewidths=0.5, linecolor="white"
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, filename)
    plt.show()


def plot_mean_similarity_per_option(df: pd.DataFrame,
                                    sim_prefix: str   = "tfidf_sim",
                                    option_cols: list = None,
                                    title: str        = "Mean Similarity per Option",
                                    filename: str     = "mean_sim_per_option.png") -> None:

    setup_plot_style()
    if option_cols is None:
        option_cols = OPTION_COLS

    sim_cols  = [f"{sim_prefix}_{c}" for c in option_cols if f"{sim_prefix}_{c}" in df.columns]
    labels    = [c.replace(f"{sim_prefix}_", "") for c in sim_cols]
    means     = [df[c].mean() for c in sim_cols]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, means, color=COLORS[:len(labels)], edgecolor="black")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Option")
    ax.set_ylabel("Mean Cosine Similarity")
    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.3f}", ha="center", fontsize=10
        )
    plt.tight_layout()
    save_figure(fig, filename)
    plt.show()


# ------------------------------------------------------------------
# MAP@K PLOTS
# ------------------------------------------------------------------

def plot_map_comparison(map_scores: dict,
                        k: int = 3,
                        filename: str = "map_comparison.png") -> None:

    setup_plot_style()
    strategies = list(map_scores.keys())
    scores     = list(map_scores.values())

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(
        strategies, scores,
        color=COLORS[:len(strategies)], edgecolor="black"
    )
    ax.set_title(f"MAP@{k} Comparison Across Strategies",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel(f"MAP@{k} Score")
    ax.set_xlim(0, 1.05)
    for bar, val in zip(bars, scores):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontweight="bold"
        )
    plt.tight_layout()
    save_figure(fig, filename)
    plt.show()


def plot_rank_distribution(rank_dist: dict,
                           k: int = 3,
                           filename: str = "rank_distribution.png") -> None:
    setup_plot_style()
    labels = [f"Rank {r}" for r in range(1, k + 1)] + ["Not Found"]
    values = [rank_dist.get(r, 0) for r in range(1, k + 1)]
    values.append(rank_dist.get("not_found", 0))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=COLORS[:len(labels)], edgecolor="black")
    ax.set_title("Rank Distribution of Correct Answer",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Rank Position")
    ax.set_ylabel("Count")
    total = sum(values)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val}\n({val/total*100:.1f}%)",
            ha="center", fontsize=9
        )
    plt.tight_layout()
    save_figure(fig, filename)
    plt.show()


# ------------------------------------------------------------------
# WORD2VEC PCA PLOT
# ------------------------------------------------------------------
def plot_w2v_pca(reduced: np.ndarray,
                 words: list,
                 n_label: int  = 50,
                 title: str    = "Word2Vec PCA Projection",
                 filename: str = "w2v_pca.png") -> None:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.4, s=25, c=COLORS[0])
    for i, word in enumerate(words[:n_label]):
        ax.annotate(
            word, (reduced[i, 0], reduced[i, 1]),
            fontsize=7, alpha=0.85,
            xytext=(3, 3), textcoords="offset points"
        )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    plt.tight_layout()
    save_figure(fig, filename)
    plt.show()
print("done")
