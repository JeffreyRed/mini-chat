"""
model.py — MiniChat: GPT-style causal language model for instruction following.

Architecture is identical to MiniGPT (learned PE, pre-norm, GELU, weight tying).
The only model-level change is that the forward pass returns logits aligned
with the full sequence — the loss masking happens in the training loop,
not inside the model.

This keeps the model architecture clean and general: the same model can be
used for plain language modelling (mini-gpt) or instruction following (mini-chat)
just by changing the loss computation.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from src.attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, emb_dim: int, ff_dim: int = None, dropout: float = 0.1) -> None:
        super().__init__()
        ff_dim = ff_dim or 4 * emb_dim
        self.net = nn.Sequential(
            nn.Linear(emb_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, n_heads, ff_dim=None, dropout=0.1):
        super().__init__()
        self.attn  = MultiHeadAttention(emb_dim, n_heads, dropout)
        self.ff    = FeedForward(emb_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, w = self.attn(self.norm1(x), mask)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x, w


class MiniChat(nn.Module):
    """
    MiniChat — instruction-following language model.

    Identical architecture to MiniGPT. Can be initialised from scratch
    or warm-started from a MiniGPT checkpoint via load_pretrained().

    Args:
        vocab_size (int):   vocabulary size (including chat special tokens)
        emb_dim    (int):   embedding dimension
        n_heads    (int):   attention heads
        n_layers   (int):   stacked transformer blocks
        max_len    (int):   max sequence length
        ff_dim     (int):   feedforward inner dim
        dropout    (float): dropout probability
        pad_idx    (int):   padding token index
    """

    def __init__(
        self,
        vocab_size : int,
        emb_dim    : int,
        n_heads    : int,
        n_layers   : int   = 4,
        max_len    : int   = 128,
        ff_dim     : int   = None,
        dropout    : float = 0.1,
        pad_idx    : int   = 0,
    ) -> None:
        super().__init__()
        self.pad_idx  = pad_idx
        self.n_layers = n_layers
        self.emb_dim  = emb_dim
        self.max_len  = max_len

        self.tok_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        self.drop    = nn.Dropout(dropout)
        self.blocks  = nn.ModuleList([
            TransformerBlock(emb_dim, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def _causal_mask(self, T, device):
        return torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
        ).unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List]:
        B, T = x.shape
        assert T <= self.max_len
        pos  = torch.arange(T, device=x.device).unsqueeze(0)
        h    = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        mask = self._causal_mask(T, x.device)
        all_w = []
        for block in self.blocks:
            h, w = block(h, mask)
            all_w.append(w)
        return self.head(self.norm(h)), all_w

    @torch.no_grad()
    def generate(
        self,
        prompt_ids     : torch.Tensor,
        max_new_tokens : int   = 40,
        temperature    : float = 0.8,
        top_k          : int   = 40,
        top_p          : float = 0.9,
        greedy         : bool  = False,
        eos_idx        : int   = None,
    ) -> torch.Tensor:
        self.eval()
        ids = prompt_ids.clone()
        for _ in range(max_new_tokens):
            ctx         = ids[:, -self.max_len:]
            logits, _   = self(ctx)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)
            if greedy:
                next_id = next_logits.argmax(dim=-1, keepdim=True)
            else:
                if top_k > 0:
                    vals, _ = torch.topk(next_logits, top_k)
                    next_logits[next_logits < vals[:, -1:]] = float("-inf")
                probs = torch.softmax(next_logits, dim=-1)
                if top_p > 0.0:
                    sp, si  = torch.sort(probs, descending=True)
                    cum     = torch.cumsum(sp, dim=-1)
                    remove  = cum - sp > top_p
                    sp[remove] = 0.0
                    sp      = sp / sp.sum(dim=-1, keepdim=True)
                    next_id = si.gather(-1, torch.multinomial(sp, 1))
                else:
                    next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id], dim=1)
            if eos_idx is not None and next_id.item() == eos_idx:
                break
        return ids

    def __repr__(self):
        params = sum(p.numel() for p in self.parameters())
        v, d   = self.tok_emb.weight.shape
        return (
            f"MiniChat(\n"
            f"  vocab={v}, emb_dim={d}, n_layers={self.n_layers},\n"
            f"  {self.blocks[0].attn},\n"
            f"  ff_dim={self.blocks[0].ff.net[0].out_features},\n"
            f"  parameters={params:,}\n"
            f")"
        )
