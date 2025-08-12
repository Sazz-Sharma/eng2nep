from torch import nn
import torch
import torch.nn.functional as F

class LayerNorm(nn.Module):
    
    def __init__(self, d_model: int, epsilon: float = 1e-6):
        
        super().__init__()
        self.d_model = d_model
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized_x = (x - mean) / torch.sqrt(variance + self.epsilon)
        normalized_x = torch.where(variance <= 1e-12, torch.zeros_like(normalized_x), normalized_x)
        return self.weight * normalized_x + self.bias

