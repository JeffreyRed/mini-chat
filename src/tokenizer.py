"""
tokenizer.py — Chat-aware tokenizer with role tokens.

New vs mini-gpt:
  - Two new special tokens: [HUMAN] and [AI]
    These mark who is speaking in the conversation.
    The model learns to produce [AI]-style responses when prompted with [HUMAN].

  - parse_conversations() splits the raw file into structured turns.

  - encode_conversation() encodes a full conversation and returns a
    loss_mask that is 1 only on [AI] response tokens and 0 on everything else.
    This is the key new piece: gradients only flow on the AI turns.

Special tokens (indices 0-5):
    <PAD>   0
    <BOS>   1
    <EOS>   2
    <UNK>   3
    [HUMAN] 4
    [AI]    5
"""

import random
from collections import Counter
from typing import List, Dict, Tuple


class ChatTokenizer:

    PAD_TOKEN   = "<PAD>"
    BOS_TOKEN   = "<BOS>"
    EOS_TOKEN   = "<EOS>"
    UNK_TOKEN   = "<UNK>"
    HUMAN_TOKEN = "[HUMAN]"
    AI_TOKEN    = "[AI]"
    SPECIAL     = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN,
                   HUMAN_TOKEN, AI_TOKEN]

    def __init__(
        self,
        path      : str,
        val_split : float = 0.15,
        min_freq  : int   = 1,
        seed      : int   = 42,
    ) -> None:
        random.seed(seed)
        self.raw_pairs = self._parse_file(path)

        # Shuffle and split
        random.shuffle(self.raw_pairs)
        n_val = max(1, int(len(self.raw_pairs) * val_split))
        self.val_pairs   = self.raw_pairs[:n_val]
        self.train_pairs = self.raw_pairs[n_val:]

        self._build_vocab(self.raw_pairs, min_freq)

    # ------------------------------------------------------------------

    def _parse_file(self, path: str) -> List[Dict]:
        """
        Parses the conversation file into a list of dicts:
            {"human": [...words...], "ai": [...words...]}
        """
        pairs  = []
        human  = None
        ai     = None

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("[HUMAN]:"):
                    if human is not None and ai is not None:
                        pairs.append({"human": human, "ai": ai})
                    human = line[len("[HUMAN]:"):].strip().split()
                    ai    = None
                elif line.startswith("[AI]:"):
                    ai = line[len("[AI]:"):].strip().split()

        if human is not None and ai is not None:
            pairs.append({"human": human, "ai": ai})

        return pairs

    def _build_vocab(self, pairs: List[Dict], min_freq: int) -> None:
        all_words = [w for p in pairs for turn in (p["human"], p["ai"]) for w in turn]
        counts    = Counter(all_words)
        words     = sorted(w for w, c in counts.items() if c >= min_freq)
        all_tokens = self.SPECIAL + words

        self.word2idx : Dict[str, int] = {t: i for i, t in enumerate(all_tokens)}
        self.idx2word : Dict[int, str] = {i: t for t, i in self.word2idx.items()}
        self.vocab_size : int  = len(self.word2idx)

        self.pad_idx   = self.word2idx[self.PAD_TOKEN]
        self.bos_idx   = self.word2idx[self.BOS_TOKEN]
        self.eos_idx   = self.word2idx[self.EOS_TOKEN]
        self.unk_idx   = self.word2idx[self.UNK_TOKEN]
        self.human_idx = self.word2idx[self.HUMAN_TOKEN]
        self.ai_idx    = self.word2idx[self.AI_TOKEN]

    # ------------------------------------------------------------------

    def _encode_words(self, words: List[str]) -> List[int]:
        return [self.word2idx.get(w, self.unk_idx) for w in words]

    def encode_conversation(
        self,
        pair: Dict,
    ) -> Tuple[List[int], List[int]]:
        """
        Encodes one (human, ai) pair into a flat token sequence with a loss mask.

        Format:
            <BOS> [HUMAN] w0 w1 ... [AI] r0 r1 ... <EOS>

        Loss mask:
            0 on <BOS>, [HUMAN], human words, [AI]
            1 on AI response words and <EOS>

        Returns:
            tokens    : full integer sequence
            loss_mask : 0/1 list, same length as tokens
        """
        human_ids = self._encode_words(pair["human"])
        ai_ids    = self._encode_words(pair["ai"])

        tokens = (
            [self.bos_idx, self.human_idx]
            + human_ids
            + [self.ai_idx]
            + ai_ids
            + [self.eos_idx]
        )

        # Mask: 0 for prompt portion, 1 for response + EOS
        prompt_len = 2 + len(human_ids) + 1   # BOS + [HUMAN] + words + [AI]
        loss_mask  = [0] * prompt_len + [1] * (len(ai_ids) + 1)

        return tokens, loss_mask

    def encode_all_split(
        self,
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Returns (train_encoded, val_encoded).
        Each element is a list of (tokens, loss_mask) tuples.
        """
        train = [self.encode_conversation(p) for p in self.train_pairs]
        val   = [self.encode_conversation(p) for p in self.val_pairs]
        return train, val

    def decode(self, indices: List[int], strip_special: bool = True) -> str:
        words = [self.idx2word.get(i, self.UNK_TOKEN) for i in indices]
        if strip_special:
            words = [w for w in words if w not in self.SPECIAL]
        return " ".join(words)

    def decode_response(self, indices: List[int]) -> str:
        """
        Extracts and decodes only the AI response from a full sequence.
        Returns everything after [AI] up to (but not including) <EOS>.
        """
        try:
            ai_pos  = indices.index(self.ai_idx)
            rest    = indices[ai_pos + 1:]
            if self.eos_idx in rest:
                rest = rest[:rest.index(self.eos_idx)]
            return self.decode(rest, strip_special=False)
        except ValueError:
            return self.decode(indices)

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ChatTokenizer(\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  train_pairs={len(self.train_pairs)},\n"
            f"  val_pairs={len(self.val_pairs)},\n"
            f"  special={self.SPECIAL}\n"
            f")"
        )
