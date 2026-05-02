"""
train.py — Training loop with per-token loss masking.

The key new piece: masked cross-entropy loss.

Standard cross-entropy (mini-gpt):
    loss = mean( -log P(target_t) for all t )

Masked cross-entropy (mini-chat):
    loss = sum( mask_t * -log P(target_t) for all t )
           / sum(mask_t)

Where mask_t = 1 only on [AI] response tokens.
This focuses all gradient signal on what the model produces,
not on reproducing the human prompt.
"""

import math
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Tuple

from src.model   import MiniChat
from src.dataset import ChatDataset, make_collate


def masked_cross_entropy(
    logits  : torch.Tensor,   # (B, T, V)
    targets : torch.Tensor,   # (B, T)
    mask    : torch.Tensor,   # (B, T) float 0/1
    pad_idx : int = 0,
) -> torch.Tensor:
    """
    Cross-entropy loss computed only on positions where mask == 1.

    Args:
        logits:  raw model output  (batch, seq_len, vocab_size)
        targets: target token indices  (batch, seq_len)
        mask:    1.0 on [AI] tokens, 0.0 elsewhere  (batch, seq_len)
        pad_idx: padding index — always excluded

    Returns:
        Scalar mean loss over all unmasked positions.
    """
    B, T, V = logits.shape

    # Per-token cross-entropy — no reduction yet
    loss_per_token = nn.functional.cross_entropy(
        logits.view(B * T, V),
        targets.view(B * T),
        ignore_index=pad_idx,
        reduction="none",
    ).view(B, T)                    # (B, T)

    # Zero out prompt positions
    masked_loss = loss_per_token * mask

    # Normalise by number of active (AI) tokens
    n_active = mask.sum().clamp(min=1)
    return masked_loss.sum() / n_active


def cosine_lr(step, warmup_steps, total_steps, peak_lr, min_lr=1e-5):
    if step < warmup_steps:
        return peak_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(
    model      : MiniChat,
    dataset    : ChatDataset,
    pad_idx    : int,
    batch_size : int = 8,
) -> Tuple[float, float]:
    """Evaluates masked loss and perplexity on the val set."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size,
                        collate_fn=make_collate(pad_idx))
    total, n = 0.0, 0
    for src, tgt, msk in loader:
        logits, _ = model(src)
        loss = masked_cross_entropy(logits, tgt, msk, pad_idx)
        total += loss.item()
        n += 1
    mean = total / max(n, 1)
    return mean, math.exp(min(mean, 30))


def train(
    model          : MiniChat,
    train_dataset  : ChatDataset,
    val_dataset    : ChatDataset,
    pad_idx        : int,
    epochs         : int   = 300,
    lr             : float = 3e-3,
    min_lr         : float = 1e-5,
    batch_size     : int   = 8,
    warmup_steps   : int   = 100,
    verbose        : bool  = True,
    snapshots      : list  = None,
    snapshot_every : int   = 10,
) -> Tuple[list, list, MiniChat]:
    """
    Trains MiniChat using masked cross-entropy.

    Returns:
        train_history : list of (loss, perplexity) per epoch
        val_history   : list of (loss, perplexity) per epoch
        best_model    : deep copy at best val perplexity
    """
    loader    = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, collate_fn=make_collate(pad_idx),
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=lr,
        betas=(0.9, 0.95), weight_decay=0.1,
    )

    total_steps  = epochs * len(loader)
    step         = 0
    best_val_ppl = float("inf")
    best_model   = None
    train_hist   = []
    val_hist     = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for src, tgt, msk in loader:
            step += 1
            for pg in optimizer.param_groups:
                pg["lr"] = cosine_lr(step, warmup_steps, total_steps, lr, min_lr)

            optimizer.zero_grad()
            logits, _ = model(src)
            loss = masked_cross_entropy(logits, tgt, msk, pad_idx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        t_loss = total_loss / len(loader)
        t_ppl  = math.exp(min(t_loss, 30))
        train_hist.append((t_loss, t_ppl))

        v_loss, v_ppl = evaluate(model, val_dataset, pad_idx, batch_size)
        val_hist.append((v_loss, v_ppl))

        if v_ppl < best_val_ppl:
            best_val_ppl = v_ppl
            best_model   = copy.deepcopy(model)

        if snapshots is not None and (epoch % snapshot_every == 0 or epoch == 1):
            model.eval()
            with torch.no_grad():
                src0, _, _ = next(iter(loader))
                _, all_w   = model(src0[:1])
                snapshots.append((epoch, [w.detach().clone() for w in all_w]))

        if verbose and (epoch % 10 == 0 or epoch == 1):
            gap     = v_ppl - t_ppl
            warning = "  ⚠ overfit" if gap > 5 and epoch > 30 else ""
            print(
                f"Epoch [{epoch:>3}/{epochs}]  "
                f"train_ppl={t_ppl:>7.2f}  "
                f"val_ppl={v_ppl:>7.2f}"
                f"{warning}"
            )

    return train_hist, val_hist, best_model
