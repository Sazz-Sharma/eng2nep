from torch import nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float = 0.2, bias: bool = True):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.bias = bias 
        self.head_dim = d_model // heads
        assert self.head_dim * heads == d_model, "d_model must be divisible by heads"

        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        batch_size = Q.size(0)
        seq_len = Q.size(1)

        Q = self.W_q(Q).reshape(batch_size, seq_len, self.heads, self.head_dim).transpose(1,2)
        K = self.W_k(K).reshape(batch_size, seq_len, self.heads, self.head_dim).transpose(1,2)
        V = self.W_v(V).reshape(batch_size, seq_len, self.heads, self.head_dim).transpose(1,2)

        attention_score = Q @ K.transpose(3,2)
        attention_score = attention_score / (self.head_dim ** 0.5)

        attention_weights = F.softmax(attention_score, dim = -1)
        attention_weights = self.dropout(attention_weights)

        multi_head_output = attention_weights @ V
        concatenated_single_head = multi_head_output.transpose(1,2).reshape(batch_size, seq_len, self.d_model)

        output = self.W_o(concatenated_single_head)
        output = self.dropout(output)
        return output
    

    





