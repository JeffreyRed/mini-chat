"""
visualize.py — Training curves, loss mask visualisation, attention heatmaps.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from typing import List

PALETTE = {
    "bg":        "#0d1117",
    "grid":      "#21262d",
    "text":      "#e6edf3",
    "accent":    "#58a6ff",
    "highlight": "#f78166",
    "muted":     "#8b949e",
    "green":     "#3fb950",
    "purple":    "#bc8cff",
    "yellow":    "#e3b341",
    "orange":    "#d29922",
}


# ── Loss mask visualisation ───────────────────────────────────────────────────

def plot_loss_mask(
    tokens    : List[int],
    mask      : List[int],
    idx2word  : dict,
    title     : str = "Loss mask for one training example",
    save_path : str = None,
) -> None:
    """
    Plots the loss mask as a colour-coded token strip.

    Blue  = prompt token (no gradient)
    Orange = AI response token (gradient flows)

    This makes the masking mechanism visually obvious.
    """
    words  = [idx2word.get(t, "?") for t in tokens]
    n      = len(words)

    fig, ax = plt.subplots(figsize=(max(10, n * 0.7), 2.5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.axis("off")

    for i, (word, m) in enumerate(zip(words, mask)):
        color  = PALETTE["orange"] if m == 1 else PALETTE["accent"]
        alpha  = 0.85 if m == 1 else 0.4
        rect   = mpatches.FancyBboxPatch(
            (i, 0.2), 0.85, 0.6,
            boxstyle="round,pad=0.05",
            facecolor=color, alpha=alpha,
            edgecolor=PALETTE["bg"], linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(
            i + 0.42, 0.52, word,
            ha="center", va="center",
            fontsize=8, color=PALETTE["text"],
            fontfamily="monospace",
        )
        ax.text(
            i + 0.42, 0.16, str(m),
            ha="center", va="center",
            fontsize=7, color=PALETTE["muted"],
        )

    ax.set_xlim(-0.1, n)
    ax.set_ylim(0, 1)

    prompt_patch   = mpatches.Patch(color=PALETTE["accent"], alpha=0.4,
                                     label="Prompt  (mask=0, no gradient)")
    response_patch = mpatches.Patch(color=PALETTE["orange"],
                                     label="AI response  (mask=1, gradient flows)")
    ax.legend(handles=[prompt_patch, response_patch],
              loc="upper right", facecolor=PALETTE["grid"],
              labelcolor=PALETTE["text"], fontsize=8)

    ax.set_title(title, color=PALETTE["text"], fontsize=10, pad=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        print(f"Loss mask plot saved → {save_path}")
    plt.show()


# ── Train vs val perplexity ───────────────────────────────────────────────────

def plot_training(
    train_history : list,
    val_history   : list,
    save_path     : str = None,
) -> None:
    """Plots train and validation perplexity + loss with shaded overfitting gap."""
    train_ppl  = [h[1] for h in train_history]
    val_ppl    = [h[1] for h in val_history]
    train_loss = [h[0] for h in train_history]
    val_loss   = [h[0] for h in val_history]
    epochs     = list(range(1, len(train_ppl) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(PALETTE["bg"])

    for ax, (tr, vl, ylabel) in zip(axes, [
        (train_ppl, val_ppl, "Perplexity  (lower = better)"),
        (train_loss, val_loss, "Cross-Entropy Loss"),
    ]):
        ax.set_facecolor(PALETTE["bg"])
        ax.grid(color=PALETTE["grid"], linewidth=0.5)
        ax.plot(epochs, tr, color=PALETTE["accent"],  linewidth=2, label="Train")
        ax.plot(epochs, vl, color=PALETTE["highlight"], linewidth=2, label="Val")
        ax.fill_between(
            epochs, tr, vl,
            where=[v > t for t, v in zip(tr, vl)],
            alpha=0.15, color=PALETTE["highlight"], label="Overfitting gap",
        )
        ax.set_xlabel("Epoch", color=PALETTE["muted"])
        ax.set_ylabel(ylabel, color=PALETTE["muted"])
        ax.tick_params(colors=PALETTE["muted"])
        for sp in ax.spines.values(): sp.set_edgecolor(PALETTE["grid"])
        ax.legend(facecolor=PALETTE["grid"], labelcolor=PALETTE["text"], fontsize=9)

    axes[0].set_title("Train vs Val Perplexity  (AI tokens only)",
                      color=PALETTE["text"], fontsize=11, pad=8, loc="left")
    axes[1].set_title("Train vs Val Loss  (masked)",
                      color=PALETTE["text"], fontsize=11, pad=8, loc="left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        print(f"Training curves saved → {save_path}")
    plt.show()


# ── Attention heatmap ─────────────────────────────────────────────────────────

def plot_attention(
    all_weights    : list,
    words          : List[str],
    sentence_label : str = "",
    save_path      : str = None,
) -> None:
    """Plots all layers × all heads for one conversation example."""
    n_layers = len(all_weights)
    n_heads  = all_weights[0].shape[0]

    fig, axes = plt.subplots(n_layers, n_heads,
                             figsize=(4 * n_heads, 4 * n_layers))
    fig.patch.set_facecolor(PALETTE["bg"])

    if n_layers == 1: axes = [axes] if n_heads > 1 else [[axes]]
    elif n_heads == 1: axes = [[ax] for ax in axes]

    for li, weights in enumerate(all_weights):
        for hi in range(n_heads):
            ax = axes[li][hi]
            w  = weights[hi].numpy()
            ax.set_facecolor(PALETTE["bg"])
            ax.imshow(w, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(words)))
            ax.set_xticklabels(words, rotation=45, ha="right",
                               fontsize=6, color=PALETTE["text"],
                               fontfamily="monospace")
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words, fontsize=6, color=PALETTE["text"],
                               fontfamily="monospace")
            for sp in ax.spines.values(): sp.set_edgecolor(PALETTE["grid"])
            ax.set_title(f"L{li+1} H{hi}", color=PALETTE["text"],
                         fontsize=8, pad=4)

    fig.suptitle(f"Attention  ·  \"{sentence_label}\"",
                 color=PALETTE["text"], fontsize=10, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        print(f"Attention plot saved → {save_path}")
    plt.show()
