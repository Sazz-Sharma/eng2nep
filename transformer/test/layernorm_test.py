import sys
from pathlib import Path

# Make project root importable
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch import nn
from transformer.layer_norm import LayerNorm


def compare_with_torch_ln(device="cpu", dtype=torch.float32):
    torch.manual_seed(0)
    B, T, D = 4, 7, 32
    eps = 1e-6

    x = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)

    ln = LayerNorm(d_model=D, epsilon=eps).to(device=device, dtype=dtype)
    ref = nn.LayerNorm(normalized_shape=D, eps=eps, elementwise_affine=True).to(device=device, dtype=dtype)

    # Copy params so both modules are identical
    with torch.no_grad():
        ref.weight.copy_(ln.weight)
        ref.bias.copy_(ln.bias)

    y = ln(x)
    y_ref = ref(x_ref)

    # Value check
    torch.testing.assert_close(y, y_ref, atol=1e-6, rtol=1e-6)

    # Gradient check
    loss = y.pow(2).mean()
    loss_ref = y_ref.pow(2).mean()
    loss.backward()
    loss_ref.backward()

    torch.testing.assert_close(x.grad, x_ref.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(ln.weight.grad, ref.weight.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(ln.bias.grad, ref.bias.grad, atol=1e-6, rtol=1e-6)

    print(f"[OK] match torch.nn.LayerNorm on {device}, dtype={dtype}")


def test_stability_constant_input(device="cpu", dtype=torch.float32):
    # For constant input, normalized output should be zeros (then shifted by bias)
    B, T, D = 2, 5, 16
    eps = 1e-6
    x = torch.full((B, T, D), 3.14, device=device, dtype=dtype)

    ln = LayerNorm(d_model=D, epsilon=eps).to(device=device, dtype=dtype)
    with torch.no_grad():
        ln.weight.fill_(1.0)
        ln.bias.fill_(0.0)

    y = ln(x)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6, rtol=1e-5), "Output should be zeros for constant input"
    print(f"[OK] stability on constant input ({device}, {dtype})")


def test_shapes(device="cpu", dtype=torch.float32):
    B, T, D = 3, 9, 24
    x = torch.randn(B, T, D, device=device, dtype=dtype)
    ln = LayerNorm(d_model=D).to(device=device, dtype=dtype)
    y = ln(x)
    assert y.shape == x.shape
    print(f"[OK] shape preserved ({device}, {dtype})")


def run_cpu():
    compare_with_torch_ln("cpu", torch.float32)
    test_stability_constant_input("cpu", torch.float32)
    test_shapes("cpu", torch.float32)


def run_cuda_if_available():
    if torch.cuda.is_available():
        compare_with_torch_ln("cuda", torch.float32)
        test_stability_constant_input("cuda", torch.float32)
        test_shapes("cuda", torch.float32)
    else:
        print("[SKIP] CUDA not available")


if __name__ == "__main__":
    run_cpu()
    run_cuda_if_available()
    print("All LayerNorm tests passed.")