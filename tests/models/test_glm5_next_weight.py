"""GLM-5.3-Flash dense projections. NVFP4 stays packed; an ignored shared expert stays bf16."""

from __future__ import annotations

import freetoken.layers.quantization.configs as quant_configs
import torch
from freetoken.layers.quantization import NameMap
from freetoken.layers.quantization.configs.modelopt import ModelOptConfig
from freetoken.models.glm5_next.weight import _proj

_ROOTS = NameMap(roots=(("model", "model.language_model"),))


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


def _flash_config() -> ModelOptConfig:
    """The NVIDIA Flash export: every Linear is NVFP4 except the ignore list."""
    return ModelOptConfig(
        {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "config_groups": {
                "group_0": {
                    "input_activations": {"num_bits": 4, "type": "float", "group_size": 16},
                    "weights": {"num_bits": 4, "type": "float", "group_size": 16},
                    "targets": ["Linear"],
                }
            },
            "ignore": ["model.language_model.layers.3.mlp.shared_experts*"],
        },
        name_map=_ROOTS,
    )


def test_nvfp4_dense_mlp_yields_packed_buffers():
    src = "model.language_model.layers.0.mlp.gate_proj"
    dst = "model.layers.0.mlp.gate_proj"
    weight = torch.arange(4 * 16, dtype=torch.uint8).reshape(4, 16)
    scale = torch.empty(4, 2, dtype=torch.float8_e4m3fn)
    scale.view(torch.uint8).fill_(7)
    reader = _Reader({
        f"{src}.weight": weight,
        f"{src}.weight_scale": scale,
        f"{src}.weight_scale_2": torch.tensor(2.5),
        f"{src}.input_scale": torch.tensor(0.25),
    })
    previous = _install(_flash_config())
    try:
        got = dict(_proj(reader, src, dst))
    finally:
        _restore(previous)
    assert torch.equal(got[f"{dst}.weight"], weight)
    assert torch.equal(got[f"{dst}.weight_scale"].view(torch.uint8), scale.view(torch.uint8))
    assert got[f"{dst}.weight_global"].shape == (4,)
    assert torch.equal(got[f"{dst}.weight_global"], torch.full((4,), 2.5))
    assert got[f"{dst}.input_scale"].shape == ()
    assert got[f"{dst}.input_scale"].item() == 0.25
    assert f"{dst}.weight_scale_2" not in got
    assert f"{dst}.weight_scale_inv" not in got


def test_ignored_shared_expert_stays_bf16():
    src = "model.language_model.layers.3.mlp.shared_experts.gate_proj"
    dst = "model.layers.3.mlp.shared_experts.gate_proj"
    weight = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    reader = _Reader({f"{src}.weight": weight})
    previous = _install(_flash_config())
    try:
        got = dict(_proj(reader, src, dst))
    finally:
        _restore(previous)
    assert list(got) == [f"{dst}.weight"]
    assert got[f"{dst}.weight"].dtype == torch.bfloat16
    assert torch.equal(got[f"{dst}.weight"], weight)
