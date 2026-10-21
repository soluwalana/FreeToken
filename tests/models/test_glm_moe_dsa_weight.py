"""GLM dense projections: per-tensor FP8 stays FP8, block-fp8 stays block-fp8. No checkpoint."""

from __future__ import annotations

import freetoken.layers.quantization.configs as quant_configs
import torch
from freetoken.layers.quantization.configs.modelopt import ModelOptConfig
from freetoken.models.glm_moe_dsa.weight import _proj


class _Reader:
    def __init__(self, tensors):
        self.tensors = tensors

    def has(self, name: str) -> bool:
        return name in self.tensors

    def get(self, name: str) -> torch.Tensor:
        return self.tensors[name]


def _install(config):
    previous = quant_configs._QUANT_CONFIG
    quant_configs.set_quant_config(config)
    return previous


def _restore(previous) -> None:
    quant_configs._QUANT_CONFIG = previous


def test_fp8_tensor_yields_weight_scale_and_input_scale():
    weight = torch.ones(4, 8, dtype=torch.float8_e4m3fn)
    src = "model.layers.0.mlp.gate_proj"
    reader = _Reader({
        f"{src}.weight": weight,
        f"{src}.weight_scale": torch.tensor(2.0),
        f"{src}.input_scale": torch.tensor(0.5),
    })
    config = ModelOptConfig({
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {"model.layers.0.mlp": {"quant_algo": "FP8"}},
    })
    previous = _install(config)
    try:
        got = dict(_proj(reader, src, src))
    finally:
        _restore(previous)
    assert torch.equal(got[f"{src}.weight"].view(torch.uint8), weight.view(torch.uint8))
    assert got[f"{src}.weight_scale"].shape == (4,)
    assert torch.equal(got[f"{src}.weight_scale"], torch.full((4,), 2.0))
    assert got[f"{src}.input_scale"].shape == ()
    assert got[f"{src}.input_scale"].item() == 0.5
    assert f"{src}.weight_scale_inv" not in got


def test_block_fp8_still_yields_weight_scale_inv():
    src = "model.layers.0.mlp.gate_proj"
    weight = torch.ones(4, 8, dtype=torch.float8_e4m3fn)
    scale = torch.ones(1, 1, dtype=torch.float32)
    reader = _Reader({f"{src}.weight": weight, f"{src}.weight_scale_inv": scale})
    config = ModelOptConfig({"quant_method": "modelopt", "quant_algo": "FP8_PB_WO"})
    previous = _install(config)
    try:
        got = dict(_proj(reader, src, src))
    finally:
        _restore(previous)
    assert torch.equal(got[f"{src}.weight"].view(torch.uint8), weight.view(torch.uint8))
    assert torch.equal(got[f"{src}.weight_scale_inv"], scale)
    assert f"{src}.weight_scale" not in got


def test_bf16_projection_stays_bf16():
    src = "model.layers.3.mlp.shared_experts.gate_proj"
    weight = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    reader = _Reader({f"{src}.weight": weight})
    config = ModelOptConfig({
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {"model.layers.0.mlp": {"quant_algo": "FP8"}},
    })
    previous = _install(config)
    try:
        got = dict(_proj(reader, src, src))
    finally:
        _restore(previous)
    assert list(got) == [f"{src}.weight"]
    assert got[f"{src}.weight"].dtype == torch.bfloat16
    assert torch.equal(got[f"{src}.weight"], weight)
