import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from transformer.multihead_attention import MultiHeadAttention


def manual_forward(mha: MultiHeadAttention, Q, K, V, mask=None):
    # Replicates the module logic (with dropout disabled in tests)
    B, T, _ = Q.shape
    H, Hd = mha.heads, mha.head_dim
    q = mha.W_q(Q).reshape(B, T, H, Hd).transpose(1, 2)
    k = mha.W_k(K).reshape(B, T, H, Hd).transpose(1, 2)
    v = mha.W_v(V).reshape(B, T, H, Hd).transpose(1, 2)
    scores = q @ k.transpose(3, 2)
    scores = scores / (Hd ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = F.softmax(scores, dim=-1)
    out = attn @ v
    out = out.transpose(1, 2).reshape(B, T, H * Hd)
    out = mha.W_o(out)
    return out, attn


def make_padding_mask(lengths, T):
    # lengths: (B,) valid lengths; returns (B,1,1,T) with 1 for valid, 0 for pad
    B = lengths.size(0)
    ar = torch.arange(T, device=lengths.device).unsqueeze(0).expand(B, T)
    mask = (ar < lengths.unsqueeze(1)).unsqueeze(1).unsqueeze(1)  # (B,1,1,T)
    return mask


def test_shapes_and_forward_no_mask(device="cpu"):
    torch.manual_seed(0)
    B, T, D, H = 2, 8, 64, 8
    mha = MultiHeadAttention(d_model=D, heads=H, dropout=0.0).to(device).eval()
    Q = torch.randn(B, T, D, device=device)
    K = torch.randn(B, T, D, device=device)
    V = torch.randn(B, T, D, device=device)

    out = mha(Q, K, V)
    assert out.shape == (B, T, D)
    out_manual, _ = manual_forward(mha, Q, K, V, None)
    torch.testing.assert_close(out, out_manual, atol=1e-6, rtol=1e-6)
    print("[OK] shapes and forward (no mask)", device)


def test_padding_mask_effect(device="cpu"):
    torch.manual_seed(0)
    B, T, D, H = 2, 6, 32, 4
    mha = MultiHeadAttention(d_model=D, heads=H, dropout=0.0).to(device).eval()
    Q = torch.randn(B, T, D, device=device)
    K = torch.randn(B, T, D, device=device)
    V = torch.zeros(B, T, D, device=device)
    # Make last two key/value positions very distinctive; mask should zero them out
    V[:, -2:, :] = 1000.0

    lengths = torch.tensor([T - 2, T - 2], device=device)
    mask = make_padding_mask(lengths, T)  # (B,1,1,T)

    out_masked = mha(Q, K, V, mask=mask)
    # If mask works, no contribution from last two positions; values were huge, so output should remain finite and small
    assert torch.isfinite(out_masked).all()
    # Compare with no mask -> no-mask output should be much larger in magnitude due to huge V at last two keys
    out_nomask = mha(Q, K, V, mask=None)
    assert (out_nomask.abs().mean() > out_masked.abs().mean()).item()
    print("[OK] padding mask reduces contribution from masked keys", device)


def test_causal_mask(device="cpu"):
    torch.manual_seed(0)
    B, T, D, H = 1, 5, 32, 4
    mha = MultiHeadAttention(d_model=D, heads=H, dropout=0.0).to(device).eval()
    X = torch.randn(B, T, D, device=device)
    # Causal mask: allow attending to self and past only
    causal = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
    out_causal = mha(X, X, X, mask=causal)
    out_manual, attn = manual_forward(mha, X, X, X, causal)
    torch.testing.assert_close(out_causal, out_manual, atol=1e-6, rtol=1e-6)
    # Ensure no attention to future: attn[..., i, j] == 0 for j>i
    assert torch.all(attn[..., torch.triu(torch.ones(T, T, device=device), diagonal=1) == 1] == 0)
    print("[OK] causal mask blocks future attention", device)


def test_gradients(device="cpu"):
    torch.manual_seed(0)
    B, T, D, H = 3, 7, 48, 6
    mha = MultiHeadAttention(d_model=D, heads=H, dropout=0.0).to(device).train()
    Q = torch.randn(B, T, D, device=device, requires_grad=True)
    K = torch.randn(B, T, D, device=device, requires_grad=True)
    V = torch.randn(B, T, D, device=device, requires_grad=True)
    out = mha(Q, K, V)
    loss = out.pow(2).mean()
    loss.backward()
    # Gradients should exist for inputs and parameters
    assert Q.grad is not None and K.grad is not None and V.grad is not None
    for p in mha.parameters():
        assert p.grad is not None
    print("[OK] gradients flow", device)


def test_cuda_if_available():
    if torch.cuda.is_available():
        device = "cuda"
        test_shapes_and_forward_no_mask(device)
        test_padding_mask_effect(device)
        test_causal_mask(device)
        test_gradients(device)
    else:
        print("[SKIP] CUDA not available")


if __name__ == "__main__":
    # CPU tests
    test_shapes_and_forward_no_mask("cpu")
    test_padding_mask_effect("cpu")
    test_causal_mask("cpu")
    test_gradients("cpu")
    # CUDA (optional)
    test_cuda_if_available()
    print("All MultiHeadAttention tests passed.")