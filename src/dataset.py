"""
dataset.py — Chat dataset with per-token loss masking.

The critical difference from mini-gpt:
  mini-gpt      → loss computed on every token
  mini-chat     → loss computed ONLY on [AI] response tokens

Without masking the model wastes capacity learning to predict the human
prompt, which is given at inference time and never needs to be generated.
Masking focuses all training signal on what the model actually produces.

Implementation:
  The loss mask is a 0/1 tensor aligned with the target sequence.
  We multiply the per-token cross-entropy by this mask before averaging,
  so masked positions contribute zero to the gradient.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple


class ChatDataset(Dataset):
    """
    Wraps (tokens, loss_mask) pairs as shifted input/target sequences.

    For sequence  [t0, t1, t2, t3, t4]:
        input:     [t0, t1, t2, t3]
        target:    [t1, t2, t3, t4]
        mask:      [m1, m2, m3, m4]   (mask shifted to align with target)

    Args:
        encoded_pairs: list of (tokens, loss_mask) from ChatTokenizer
        min_len: skip sequences shorter than this
    """

    def __init__(
        self,
        encoded_pairs : List[Tuple[List[int], List[int]]],
        min_len       : int = 4,
    ) -> None:
        self.examples: List[Tuple] = []
        for tokens, mask in encoded_pairs:
            if len(tokens) < min_len + 1:
                continue
            # shift: input = tokens[:-1], target = tokens[1:], mask = mask[1:]
            self.examples.append((tokens[:-1], tokens[1:], mask[1:]))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        src, tgt, msk = self.examples[idx]
        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long),
            torch.tensor(msk, dtype=torch.float),
        )

    def __repr__(self) -> str:
        return f"ChatDataset(examples={len(self.examples)})"


def collate_fn(
    batch   : list,
    pad_idx : int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pads a batch to the longest sequence.

    Returns:
        src:  (batch, max_len)
        tgt:  (batch, max_len)
        mask: (batch, max_len)  float, 0/1
    """
    srcs, tgts, masks = zip(*batch)
    max_len = max(s.size(0) for s in srcs)

    src_pad  = torch.full((len(srcs), max_len), pad_idx, dtype=torch.long)
    tgt_pad  = torch.full((len(tgts), max_len), pad_idx, dtype=torch.long)
    mask_pad = torch.zeros(len(masks), max_len)

    for i, (s, t, m) in enumerate(zip(srcs, tgts, masks)):
        src_pad[i,  :s.size(0)] = s
        tgt_pad[i,  :t.size(0)] = t
        mask_pad[i, :m.size(0)] = m

    return src_pad, tgt_pad, mask_pad


def make_collate(pad_idx: int):
    def _fn(batch):
        return collate_fn(batch, pad_idx)
    return _fn
