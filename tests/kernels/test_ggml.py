"""Packed GGUF ops against gguf-py's CPU dequantization and PyTorch math."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from freetoken.kernel import ggml
from freetoken.third_party.gguf import GGMLQuantizationType as QType
from freetoken.third_party.gguf import quants

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
DTYPES = [torch.float32, torch.bfloat16, torch.float16]


def _q4_weights(shape, seed=690):
    values = np.random.default_rng(seed).normal(0, 0.1, shape).astype(np.float32)
    packed = quants.quantize(values, QType.Q4_0)
    reference = torch.from_numpy(quants.dequantize(packed, QType.Q4_0))
    # ggml's last MMQ tile may read an extra partial row.
    storage = torch.zeros(*packed.shape[:-2], packed.shape[-2] + 1, packed.shape[-1], dtype=torch.uint8, device="cuda")
    storage[..., :-1, :].copy_(torch.from_numpy(packed))
    return storage[..., :-1, :], reference


def _activation(shape, dtype, seed=19):
    rng = np.random.default_rng(seed)
    half = rng.integers(-127, 128, (*shape[:-1], shape[-1] // 32, 16))
    half[..., 0] = 127
    # Exactly representable q8_1 inputs with zero block sums isolate weight math.
    values = np.concatenate([half, -half], axis=-1).reshape(shape).astype(np.float32) / 64
    return torch.from_numpy(values).to("cuda", dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_q4_dequantize_and_rows(dtype):
    w, ref = _q4_weights((64, 256))
    actual = ggml.dequantize(w, int(QType.Q4_0), 256, dtype)
    torch.testing.assert_close(actual.cpu(), ref.to(dtype), atol=0, rtol=0)
    ids = torch.tensor([63, 0, 17, 17], dtype=torch.int32, device="cuda")
    rows = ggml.get_rows(w, int(QType.Q4_0), 256, ids, dtype)
    torch.testing.assert_close(rows.cpu(), ref[ids.cpu().long()].to(dtype), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch", [1, 17])
def test_q4_matrix_matches_cpu(dtype, batch):
    w, ref = _q4_weights((64, 256))
    x = _activation((batch, 256), dtype)
    actual = ggml.mul_mat(w, int(QType.Q4_0), x)
    expected = F.linear(x.cpu().float(), ref).to(dtype)
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-4, rtol=1e-3)


def _route_reference(x, ref, ids):
    return torch.einsum("tk, tjok -> tjo", x.cpu().float(), ref[ids.cpu().long()]).to(x.dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch", [1, 17])
def test_q4_moe_routes_match_cpu(dtype, batch):
    w, ref = _q4_weights((4, 64, 256))
    x = _activation((batch, 256), dtype)
    ids = torch.tensor([[i % 4, (i + 1) % 4] for i in range(batch)], dtype=torch.int32, device="cuda")
    actual = ggml.mul_mat_id(w, int(QType.Q4_0), x, ids)
    torch.testing.assert_close(actual.cpu(), _route_reference(x, ref, ids), atol=2e-4, rtol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
def test_q4_moe_vector_kernel_writes_repeated_ids(dtype):
    """The small-batch kernel evaluates every route, including a repeated expert id."""
    w, ref = _q4_weights((4, 64, 256))
    x = _activation((1, 256), dtype)
    ids = torch.tensor([[0, 0, 3]], dtype=torch.int32, device="cuda")
    actual = ggml.mul_mat_id(w, int(QType.Q4_0), x, ids)
    torch.testing.assert_close(actual.cpu(), _route_reference(x, ref, ids), atol=2e-4, rtol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("topk", [3, 8])
def test_q4_moe_matrix_kernel_collapses_repeated_ids(dtype, topk):
    """MMQ keeps one column when a token repeats an expert.

    Expert parallel points every unowned route at one pad row. Those extra columns stay zero,
    and a higher expert's column still belongs to that expert.
    """
    batch = 17
    w, ref = _q4_weights((4, 64, 256))
    x = _activation((batch, 256), dtype)
    ids = torch.zeros(batch, topk, dtype=torch.int32, device="cuda")
    ids[:, :-1] = 0
    ids[:, -1] = 3
    actual = ggml.mul_mat_id(w, int(QType.Q4_0), x, ids)
    assert torch.isfinite(actual).all()
    ref0 = _route_reference(x, ref, torch.zeros_like(ids[:, :1]))[:, 0]
    ref3 = _route_reference(x, ref, torch.full_like(ids[:, :1], 3))[:, 0]
    torch.testing.assert_close(actual[:, 0].cpu(), ref0, atol=2e-4, rtol=1e-3)
    torch.testing.assert_close(actual[:, -1].cpu(), ref3, atol=2e-4, rtol=1e-3)
    assert torch.count_nonzero(actual[:, 1:-1]) == 0

    # The pad row is the highest id, and every copy of it has weight 0.
    pad = torch.zeros_like(ids)
    pad[:, 0] = torch.arange(batch, device="cuda") % 3
    pad[:, 1:] = 3
    routed = ggml.mul_mat_id(w, int(QType.Q4_0), x, pad)
    weights = torch.zeros(batch, topk, dtype=torch.float32, device="cuda")
    weights[:, 0] = 1
    summed = ggml.moe_weighted_sum(routed, weights)
    owned = _route_reference(x, ref, pad[:, :1])[:, 0]
    assert torch.isfinite(summed).all()
    torch.testing.assert_close(summed.cpu(), owned, atol=2e-4, rtol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
def test_q4_shared_q8_matches_internal_quantization(dtype):
    w, _ = _q4_weights((64, 256))
    x = _activation((1, 256), dtype)
    expected = ggml.mul_mat(w, int(QType.Q4_0), x)
    actual = ggml.mul_mat(w, int(QType.Q4_0), x, xq=ggml.quantize_q8_1(x))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_q4_nondefault_stream():
    w, ref = _q4_weights((64, 256))
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = _activation((1, 256), torch.float32)
        actual = ggml.mul_mat(w, int(QType.Q4_0), x)
        copied = actual.clone()
    stream.synchronize()
    torch.testing.assert_close(copied.cpu(), F.linear(x.cpu(), ref), atol=2e-4, rtol=1e-3)
