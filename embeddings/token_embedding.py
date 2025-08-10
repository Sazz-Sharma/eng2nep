from torch import nn
import torch
from math import sqrt

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size:int, embedding_dim:int, scale: bool = True, dropout: float = 0.0, padding_idx: int = None):
    
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = padding_idx)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.scale = scale
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()

        x = self.embedding(input_ids)
        if self.scale:
            x = x* sqrt(self.embedding_dim)

        return self.dropout(x)
    
    def extra_repr(self) -> str:
        return f"vocab_size={self.vocab_size}, embedding_dim={self.embedding_dim}, scale={self.scale}, dropout={self.dropout.p if hasattr(self.dropout, 'p') else 0.0}, padding_idx={self.embedding.padding_idx}"
    


