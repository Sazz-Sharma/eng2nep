from positional_encoding import PositionalEncoding
from token_embedding import TokenEmbedding
import torch
from math import sqrt


torch.manual_seed(0)
d_model = 64
bsz, seq_len = 2, 16

pe = PositionalEncoding(d_model=d_model)

x = torch.randn(bsz, seq_len, d_model, requires_grad=True)
y = pe(x)

print(f"in:  {x.shape}, dtype={x.dtype}, device={x.device}")
print(f"out: {y.shape}, mean={y.mean().item():.4f}, std={y.std().item():.4f}")

# Gradient check
y.sum().backward()
assert x.grad is not None and x.grad.shape == x.shape
print("grad OK:", x.grad.mean().item())

# Determinism check (no dropout/state)
y2 = pe(x.detach())
assert torch.allclose(y2, pe(x.detach())), "Output should be deterministic"
print("determinism OK")



def test_token_embedding_shapes_and_dtype():
    vocab_size, d_model = 100, 64
    pad_id = 0
    emb = TokenEmbedding(vocab_size, d_model, padding_idx=pad_id, dropout=0.0, scale=False)
    ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]], dtype=torch.long)
    out = emb(ids)
    assert out.shape == (2, 5, d_model)
    assert out.dtype == torch.float32

def test_token_embedding_padding_zero():
    vocab_size, d_model = 50, 32
    pad_id = 0
    emb = TokenEmbedding(vocab_size, d_model, padding_idx=pad_id, dropout=0.0, scale=False)
    ids = torch.tensor([[0, 0, 1, 2]])
    out = emb(ids)
    # padding rows should be all zeros
    assert torch.allclose(out[0, 0], torch.zeros_like(out[0, 0]))
    assert torch.allclose(out[0, 1], torch.zeros_like(out[0, 1]))

def test_token_embedding_scaling_factor():
    torch.manual_seed(0)
    vocab_size, d_model = 200, 64
    ids = torch.randint(1, vocab_size, (2, 10), dtype=torch.long)

    emb_no_scale = TokenEmbedding(vocab_size, d_model, padding_idx=None, dropout=0.0, scale=False)
    emb_scale = TokenEmbedding(vocab_size, d_model, padding_idx=None, dropout=0.0, scale=True)
    # Make weights identical
    emb_scale.load_state_dict(emb_no_scale.state_dict())

    out_no = emb_no_scale(ids)
    out_sc = emb_scale(ids)
    assert torch.allclose(out_sc, out_no * sqrt(d_model), atol=1e-6)

if __name__ == "__main__":
    # existing PE checks
    torch.manual_seed(0)
    d_model = 64
    bsz, seq_len = 2, 16

    pe = PositionalEncoding(d_model=d_model)
    x = torch.randn(bsz, seq_len, d_model, requires_grad=True)
    y = pe(x)

    print(f"in:  {x.shape}, dtype={x.dtype}, device={x.device}")
    print(f"out: {y.shape}, mean={y.mean().item():.4f}, std={y.std().item():.4f}")

    y.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape
    print("grad OK:", x.grad.mean().item())

    y2 = pe(x.detach())
    assert torch.allclose(y2, pe(x.detach())), "Output should be deterministic"
    print("determinism OK")

    # run TokenEmbedding tests
    test_token_embedding_shapes_and_dtype()
    test_token_embedding_padding_zero()
    test_token_embedding_scaling_factor()
    print("TokenEmbedding tests OK")