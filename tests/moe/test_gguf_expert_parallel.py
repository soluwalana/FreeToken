"""GGUF expert banks under expert parallel: one process, no NCCL.

The file still holds every expert. Each rank copies ``id % size == rank`` and leaves the last local row zero.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from freetoken.distributed import set_ep_info, set_tp_info, try_get_tp_info
from freetoken.layers.quantization import QuantKind
from freetoken.layers.quantization.ggml_types import row_bytes
from freetoken.moe.expert_parallel import clear_ep_comm
from freetoken.third_party import gguf

Q8_0 = int(gguf.GGMLQuantizationType.Q8_0)
NVFP4 = int(gguf.GGMLQuantizationType.NVFP4)
_K = 32


class _Info:
    def __init__(self, name: str, ggml_type: int, ne: tuple[int, ...], payload: np.ndarray):
        self.name = name
        self.path = "/dev/null"
        self.offset = 0
        self.nbytes = int(payload.nbytes)
        self.ggml_type = ggml_type
        self.ne = ne
        self.root = "synthetic.gguf"
        self._payload = np.ascontiguousarray(payload)
        assert self._payload.dtype == np.uint8

    def data(self) -> np.ndarray:
        return self._payload


def _packed(name: str, ggml_type: int, experts: int, rows: int, fill) -> _Info:
    width = row_bytes(ggml_type, _K if ggml_type == Q8_0 else 64)
    raw = np.zeros((experts, rows, width), dtype=np.uint8)
    for expert in range(experts):
        raw[expert] = fill(expert)
    return _Info(name, ggml_type, (64 if ggml_type == NVFP4 else _K, rows, experts), raw.reshape(-1))


def _scales(name: str, values: np.ndarray) -> _Info:
    raw = np.ascontiguousarray(values, dtype=np.float32)
    return _Info(name, 0, (raw.shape[0],), raw.view(np.uint8).reshape(-1))


def _index(experts: int, *, layers: tuple[int, ...] = (3, 4), merged: bool = False, scaled: bool = False):
    index = {}
    ggml_type = NVFP4 if scaled else Q8_0
    for blk in layers:
        prefix = f"blk.{blk}"
        if merged:
            index[f"{prefix}.ffn_gate_up_exps.weight"] = _packed(
                f"{prefix}.ffn_gate_up_exps.weight", ggml_type, experts, 4, lambda e: e + 1
            )
        else:
            index[f"{prefix}.ffn_gate_exps.weight"] = _packed(
                f"{prefix}.ffn_gate_exps.weight", ggml_type, experts, 2, lambda e: e + 1
            )
            index[f"{prefix}.ffn_up_exps.weight"] = _packed(
                f"{prefix}.ffn_up_exps.weight", ggml_type, experts, 2, lambda e: e + 51
            )
        index[f"{prefix}.ffn_down_exps.weight"] = _packed(
            f"{prefix}.ffn_down_exps.weight", ggml_type, experts, 3, lambda e: e + 101
        )
        if scaled:
            for part, base in (("gate", 1), ("up", 11), ("down", 21)):
                vals = np.arange(experts, dtype=np.float32) + base
                index[f"{prefix}.ffn_{part}_exps.scale"] = _scales(f"{prefix}.ffn_{part}_exps.scale", vals)
    return index


def _install(monkeypatch, index) -> None:
    monkeypatch.setattr("freetoken.models.gguf.experts.gguf_tensor_index", lambda _path: index)
    monkeypatch.setattr("freetoken.models.gguf.tensors.gguf_tensor_index", lambda _path: index)


@pytest.fixture(scope="module", autouse=True)
def _tp():
    if try_get_tp_info() is None:
        set_tp_info(0, 1)


@pytest.fixture(autouse=True)
def _reset_ep():
    yield
    set_ep_info(0, 1)
    clear_ep_comm()


def _load(device, *, experts=4, merged=False, scaled=False, first_layer=3):
    from freetoken.models.gguf.experts import load_gguf_expert_banks

    return load_gguf_expert_banks(
        "synthetic.gguf", 2, first_layer=first_layer, device=device
    )


def test_size_one_keeps_every_expert(monkeypatch):
    _install(monkeypatch, _index(4))
    banks = _load(torch.device("cpu"))
    gate_up, down = banks.sources["gate_up"][0], banks.sources["down"][0]
    assert gate_up.shape[0] == 4 and down.shape[0] == 4
    assert "global" not in banks.sources
    assert torch.all(gate_up[3, :2] == 4)
    assert torch.all(gate_up[3, 2:] == 54)
    assert torch.all(down[3] == 104)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("device", [None, torch.device("cpu")])
def test_rank_keeps_its_stripe_and_a_zero_pad(monkeypatch, rank, device):
    if device is None:
        monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")
    _install(monkeypatch, _index(4))
    set_ep_info(rank, 2)
    banks = _load(device)
    gate_up = banks.sources["gate_up"][1]
    down = banks.sources["down"][1]
    assert gate_up.shape[0] == 3 and down.shape[0] == 3
    for local, expert in enumerate(range(rank, 4, 2)):
        assert torch.all(gate_up[local, :2] == expert + 1)
        assert torch.all(gate_up[local, 2:] == expert + 51)
        assert torch.all(down[local] == expert + 101)
    assert torch.all(gate_up[2] == 0) and torch.all(down[2] == 0)
    assert banks.sources["gate_up"][0].shape == gate_up.shape


def test_merged_gate_up_uses_the_same_stripe(monkeypatch):
    _install(monkeypatch, _index(4, merged=True))
    set_ep_info(1, 2)
    banks = _load(torch.device("cpu"))
    gate_up = banks.sources["gate_up"][0]
    assert gate_up.shape == (3, 4, row_bytes(Q8_0, _K))
    assert torch.all(gate_up[0] == 2) and torch.all(gate_up[1] == 4)
    assert torch.all(gate_up[2] == 0)


def test_nvfp4_scales_follow_the_stripe_and_the_pad_stays_one(monkeypatch):
    _install(monkeypatch, _index(4, scaled=True))
    set_ep_info(0, 2)
    banks = _load(torch.device("cpu"))
    assert banks.quant_format == "gguf_scaled"
    scales = banks.sources["global"][0]
    assert scales.shape == (3, 4)
    assert torch.equal(scales[0], torch.tensor([1.0, 11.0, 21.0, 1.0]))
    assert torch.equal(scales[1], torch.tensor([3.0, 13.0, 23.0, 1.0]))
    assert torch.equal(scales[2], torch.ones(4))


def test_ep_size_must_divide_the_expert_count(monkeypatch):
    _install(monkeypatch, _index(3))
    set_ep_info(0, 2)
    with pytest.raises(ValueError, match="must divide"):
        _load(torch.device("cpu"))


def test_load_expert_banks_accepts_gguf_under_expert_parallel(monkeypatch):
    from freetoken.moe.expert_banks import load_expert_banks

    _install(monkeypatch, _index(4))
    set_ep_info(1, 2)
    banks = load_expert_banks(
        "synthetic.gguf",
        SimpleNamespace(num_moe_layers=2, first_k_dense_replace=3),
        method=SimpleNamespace(kind=QuantKind.GGUF),
        device=torch.device("cpu"),
        parallel=False,
        resident=True,
    )
    assert banks.sources["down"][0].shape[0] == 3
    assert torch.all(banks.sources["down"][0][0] == 102)
    assert torch.all(banks.sources["down"][0][2] == 0)
