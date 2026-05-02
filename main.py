"""
main.py — End-to-end pipeline for mini-chat.

Usage:
    python main.py

Pipeline:
    1.  Load science Q&A corpus, build ChatTokenizer with train/val split
    2.  Show loss masking so the mechanism is fully visible
    3.  Build MiniChat model
    4.  Train with masked cross-entropy + cosine LR
    5.  Plot train vs val perplexity (overfitting on AI tokens only)
    6.  Demo fixed responses from best checkpoint
    7.  Interactive chat loop
    8.  Attention heatmap for one conversation turn
    9.  Save best checkpoint
"""

import torch
from pathlib import Path

from src.tokenizer  import ChatTokenizer
from src.dataset    import ChatDataset
from src.model      import MiniChat
from src.train      import train
from src.utils      import demo_responses, interactive_chat, show_loss_masking
from src.visualize  import plot_loss_mask, plot_training, plot_attention

# ── Config ────────────────────────────────────────────────────────────────────
CORPUS_PATH      = "data/science_qa.txt"
VAL_SPLIT        = 0.15
EMB_DIM          = 64
N_HEADS          = 4
N_LAYERS         = 2       # reduced from 4 — dataset too small for 4 layers
FF_DIM           = 128
MAX_LEN          = 128
EPOCHS           = 150     # early stopping will trigger well before this
LR               = 5e-4   # lower LR for fine-tuning (base model already trained)
MIN_LR           = 1e-6
BATCH_SIZE       = 8
WARMUP_STEPS     = 50
PATIENCE         = 20      # early stopping: halt if val_ppl doesn't improve for 20 epochs
DROPOUT          = 0.3     # higher dropout to slow memorisation (was 0.1)
OUTPUTS_DIR      = Path("outputs")
INSPECT_Q        = "What is imagination"

# Path to mini-gpt checkpoint — set to None to train from scratch
# Using pretrained weights means the model already knows language structure
# and only needs to learn the conversation format.
MINIGPT_CKPT     = "../mini-gpt/outputs/minigpt_best.pt"
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # ── 1. Tokeniser ──────────────────────────────────────────────────────────
    print("\n── Tokeniser ───────────────────────────────────────")
    tok = ChatTokenizer(CORPUS_PATH, val_split=VAL_SPLIT)
    print(tok)
    print(f"\n  Special tokens:")
    print(f"    PAD={tok.pad_idx}  BOS={tok.bos_idx}  EOS={tok.eos_idx}")
    print(f"    [HUMAN]={tok.human_idx}  [AI]={tok.ai_idx}\n")
    print(f"  Train pairs : {len(tok.train_pairs)}")
    print(f"  Val pairs   : {len(tok.val_pairs)}\n")

    # ── 2. Show loss masking ──────────────────────────────────────────────────
    print("── Loss masking ────────────────────────────────────")
    print("  The model trains on the FULL sequence but gradients flow")
    print("  ONLY on [AI] response tokens (mask=1).\n")
    show_loss_masking(tok, n_examples=2)

    # Visualise mask for first training pair
    first_pair = tok.train_pairs[0]
    tokens, mask = tok.encode_conversation(first_pair)
    plot_loss_mask(
        tokens[:20], mask[:20], tok.idx2word,
        title=f"Loss mask  ·  Q: {' '.join(first_pair['human'][:6])} ...",
        save_path=str(OUTPUTS_DIR / "loss_mask.png"),
    )

    # ── 3. Dataset ────────────────────────────────────────────────────────────
    train_enc, val_enc = tok.encode_all_split()
    train_ds = ChatDataset(train_enc)
    val_ds   = ChatDataset(val_enc)
    print(f"  {train_ds}  (train)")
    print(f"  {val_ds}    (val)\n")

    # ── 4. Model ──────────────────────────────────────────────────────────────
    print("── Model ───────────────────────────────────────────")
    from pathlib import Path as _Path
    ckpt_exists = MINIGPT_CKPT and _Path(MINIGPT_CKPT).exists()

    if ckpt_exists:
        print(f"  Loading pretrained weights from {MINIGPT_CKPT}")
        print("  (base model already knows language — only learning conversation format)\n")
        model = MiniChat.from_pretrained(
            MINIGPT_CKPT,
            new_vocab_size = tok.vocab_size,
            emb_dim        = EMB_DIM,
            n_heads        = N_HEADS,
            n_layers       = N_LAYERS,
            max_len        = MAX_LEN,
            ff_dim         = FF_DIM,
            dropout        = DROPOUT,
            pad_idx        = tok.pad_idx,
        )
    else:
        if MINIGPT_CKPT:
            print(f"  ⚠  mini-gpt checkpoint not found at: {MINIGPT_CKPT}")
            print("     Training from scratch. For better results run mini-gpt first.\n")
        else:
            print("  Training from scratch (MINIGPT_CKPT = None)\n")
        model = MiniChat(
            vocab_size = tok.vocab_size,
            emb_dim    = EMB_DIM,
            n_heads    = N_HEADS,
            n_layers   = N_LAYERS,
            max_len    = MAX_LEN,
            ff_dim     = FF_DIM,
            dropout    = DROPOUT,
            pad_idx    = tok.pad_idx,
        )
    print(model, "\n")

    # ── 5. Train ──────────────────────────────────────────────────────────────
    print("── Training ────────────────────────────────────────")
    print("  Note: perplexity is computed ONLY on [AI] response tokens.")
    print("  This is more honest than full-sequence perplexity.\n")

    snapshots: list = []
    train_hist, val_hist, best_model = train(
        model, train_ds, val_ds,
        pad_idx        = tok.pad_idx,
        epochs         = EPOCHS,
        lr             = LR,
        min_lr         = MIN_LR,
        batch_size     = BATCH_SIZE,
        warmup_steps   = WARMUP_STEPS,
        patience       = PATIENCE,
        snapshots      = snapshots,
        snapshot_every = max(1, EPOCHS // 30),
    )

    best_val_ppl = min(h[1] for h in val_hist)
    best_epoch   = val_hist.index(min(val_hist, key=lambda x: x[1])) + 1
    print(f"\n  Best val_ppl = {best_val_ppl:.2f} at epoch {best_epoch}")

    # ── 6. Training curves ────────────────────────────────────────────────────
    plot_training(
        train_hist, val_hist,
        save_path=str(OUTPUTS_DIR / "training_curves.png"),
    )

    # ── 7. Demo responses ─────────────────────────────────────────────────────
    demo_responses(best_model, tok)

    # ── 8. Interactive chat ───────────────────────────────────────────────────
    interactive_chat(best_model, tok)

    # ── 9. Attention heatmap ──────────────────────────────────────────────────
    inspect_words = (
        [tok.idx2word[tok.bos_idx], tok.idx2word[tok.human_idx]]
        + INSPECT_Q.split()
        + [tok.idx2word[tok.ai_idx]]
    )
    inspect_ids = (
        [tok.bos_idx, tok.human_idx]
        + [tok.word2idx.get(w, tok.unk_idx) for w in INSPECT_Q.split()]
        + [tok.ai_idx]
    )
    best_model.eval()
    with torch.no_grad():
        x = torch.tensor([inspect_ids])
        _, all_w = best_model(x)
    all_w = [w.squeeze(0) for w in all_w]

    plot_attention(
        all_w, inspect_words,
        sentence_label = INSPECT_Q,
        save_path      = str(OUTPUTS_DIR / "attention.png"),
    )

    # ── 10. Save ──────────────────────────────────────────────────────────────
    ckpt = OUTPUTS_DIR / "minichat_best.pt"
    torch.save({
        "model_state": best_model.state_dict(),
        "config": dict(
            vocab_size=tok.vocab_size, emb_dim=EMB_DIM,
            n_heads=N_HEADS, n_layers=N_LAYERS,
            max_len=MAX_LEN, ff_dim=FF_DIM,
        ),
        "best_val_ppl": best_val_ppl,
        "best_epoch":   best_epoch,
    }, ckpt)
    print(f"\nBest model saved → {ckpt}")


if __name__ == "__main__":
    main()