import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformer.encoder import EncoderBlock

def make_padding_mask(lengths, T, device):
    ar = torch.arange(T, device=device).unsqueeze(0)
    return (ar < lengths.unsqueeze(1)).unsqueeze(1).unsqueeze(1)  # (B,1,1,T)

def zero_module_(m):
    for p in m.parameters():
        if p.grad is not None:
            p.grad.zero_()
        with torch.no_grad():
            p.zero_() if p.ndim > 0 else p.fill_(0)

def test_shapes_and_forward(device="cpu"):
    torch.manual_seed(0)
    B, T, D, H, DFF = 2, 12, 64, 8, 256
    x = torch.randn(B, T, D, device=device, requires_grad=True)
    enc = EncoderBlock(d_model=D, heads=H, d_ff=DFF, dropout=0.0).to(device).train()
    mask = make_padding_mask(torch.tensor([T, T-3], device=device), T, device)
    y = enc(x, mask)
    assert y.shape == x.shape
    y.mean().backward()
    for p in enc.parameters():
        assert p.grad is not None
    assert x.grad is not None
    print("[OK] shape and gradients on", device)

def test_identity_when_zero_weights(device="cpu"):
    torch.manual_seed(0)
    B, T, D, H, DFF = 1, 5, 32, 4, 64
    x = torch.randn(B, T, D, device=device)
    enc = EncoderBlock(d_model=D, heads=H, d_ff=DFF, dropout=0.0).to(device).eval()

    # Zero all weights so residuals should pass input unchanged
    zero_module_(enc.attn.W_q); zero_module_(enc.attn.W_k); zero_module_(enc.attn.W_v); zero_module_(enc.attn.W_o)
    for layer in enc.ffn:
        if isinstance(layer, torch.nn.Linear):
            zero_module_(layer)

    y = enc(x, mask=None)
    assert torch.allclose(y, x, atol=1e-6)
    print("[OK] residual identity with zero weights on", device)

def test_cuda_if_available():
    if torch.cuda.is_available():
        test_shapes_and_forward("cuda")
        test_identity_when_zero_weights("cuda")
    else:
        print("[SKIP] CUDA not available")

if __name__ == "__main__":
    test_shapes_and_forward("cpu")
    test_identity_when_zero_weights("cpu")
    test_cuda_if_available()
    print("All EncoderBlock tests passed.")