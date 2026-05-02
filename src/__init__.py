"""mini-chat — source package."""
from src.tokenizer  import ChatTokenizer
from src.dataset    import ChatDataset, collate_fn, make_collate
from src.attention  import MultiHeadAttention
from src.model      import MiniChat
from src.train      import train, evaluate, masked_cross_entropy, cosine_lr
from src.utils      import (
    chat_response, interactive_chat,
    demo_responses, show_loss_masking,
)
from src.visualize  import plot_loss_mask, plot_training, plot_attention

__all__ = [
    "ChatTokenizer",
    "ChatDataset", "collate_fn", "make_collate",
    "MultiHeadAttention",
    "MiniChat",
    "train", "evaluate", "masked_cross_entropy", "cosine_lr",
    "chat_response", "interactive_chat", "demo_responses", "show_loss_masking",
    "plot_loss_mask", "plot_training", "plot_attention",
]
