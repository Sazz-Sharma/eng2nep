import torch
import math
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int): 
        """
            d_model: embedding dimension
        """
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        device = x.device

        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        denominator = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float, device=device)
            * (-math.log(10000.0) / self.d_model)
        )

        pe = torch.zeros(seq_len, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(position * denominator)
        pe[:, 1::2] = torch.cos(position * denominator)
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        return x + pe

    




# class PositionalEncoding(nn.Module):
    
#     def __init__(self, d_model: int):
#         super().__init__()
#         self.d_model = d_model
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         seq_len = x.size(1)
#         device = x.device 
#         position = torch.aranage(seq_len, dtype = torch.float, device = device).unsqueeze(1)
#         denominator = torch.exp(
#             torch.arange(0, self.d_model, 2, dtype=torch.float, device=device)
#             * (-math.log(10000.0) / self.d_model)
#         )

        
