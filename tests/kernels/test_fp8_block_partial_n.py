"""Block-fp8 linears whose N stops inside a 128-row block (GLM-5.3 kv_a is 576)."""

from __future__ import annotations

import pytest
import torch

from freetoken.layers.quantization.linear.base import LinearConfig
from freetoken.layers.quantization.linear.fp8_block import Fp8BlockLinearMethod
from freetoken.layers.quantization.scheme import fp8_block_scheme

FP8 = torch.float8_e4m3fn


def _method(k: int, n: int, output_sizes: tuple[int, ...] | None = None) -> Fp8BlockLinearMethod:
    cfg = LinearConfig(k, n, output_sizes or (n,), fp8_block_scheme("float"))
    return Fp8BlockLinearMethod(cfg)


def test_partial_output_block_allocates_a_ceil_scale():
    layer = type("L", (), {})()
    _method(6144, 576).create_weights(layer)
    assert layer.weight.shape == (576, 6144)
    assert layer.weight.dtype == FP8
    assert layer.weight_scale_inv.shape == (5, 48)
    assert layer.weight_scale_inv.dtype == torch.bfloat16


def test_aligned_output_keeps_an_exact_scale():
    layer = type("L", (), {})()
    _method(256, 256).create_weights(layer)
    assert layer.weight.shape == (256, 256)
    assert layer.weight_scale_inv.shape == (2, 2)


def test_fused_partial_output_is_rejected():
    layer = type("L", (), {})()
    with pytest.raises(ValueError, match="divisible by 128"):
        _method(256, 200, (128, 72)).create_weights(layer)


def test_partial_k_is_rejected():
    layer = type("L", (), {})()
    with pytest.raises(ValueError, match="divisible by 128"):
        _method(100, 128).create_weights(layer)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_partial_n_prefill_matches_a_zero_padded_gemm():
    from freetoken.kernel.triton.fp8_block_linear import block_fp8_linear

    torch.manual_seed(0)
    n, k = 576, 256
    weight = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.uint8).view(FP8)
    scale = torch.randn(5, k // 128, device="cuda", dtype=torch.bfloat16).abs() + 0.1
    x = torch.randn(4, k, device="cuda", dtype=torch.bfloat16)
    padded = torch.cat((weight, torch.zeros(64, k, dtype=FP8, device="cuda")), dim=0)
    got = block_fp8_linear(x, weight, scale)
    ref = block_fp8_linear(x, padded, scale)[..., :n]
    assert got.shape == (4, n)
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_partial_n_decode_matches_dequant():
    from freetoken.kernel.triton.fp8_block_linear import block_fp8_linear, dequant_block_fp8

    torch.manual_seed(1)
    n, k = 576, 256
    weight = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.uint8).view(FP8)
    scale = torch.randn(5, k // 128, device="cuda", dtype=torch.bfloat16).abs() + 0.1
    x = torch.randn(1, k, device="cuda", dtype=torch.bfloat16)
    got = block_fp8_linear(x, weight, scale)
    ref = x.float() @ dequant_block_fp8(weight, scale).float().T
    assert got.shape == (1, n)
    torch.testing.assert_close(got.float(), ref, rtol=1e-2, atol=1e-2)
