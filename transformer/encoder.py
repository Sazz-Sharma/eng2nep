

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from transformer.multihead_attention import MultiHeadAttention
import torch
import torch.nn.functional as F
from torch import nn
from transformer.layer_norm import LayerNorm

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float = 0.2):
        super().__init__()
        self.attn = MultiHeadAttention(d_model = d_model, heads = heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        residual = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x)))
        output = residual + self.dropout(self.ffn(self.norm2(residual)))
        return output


