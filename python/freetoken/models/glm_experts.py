"""Checkpoint names shared by GLM-5.3 (``glm_moe_dsa``) and GLM-5.3-Flash (``glm5_next``).

The two families store the same per-expert projections. Flash nests them under
``model.language_model.layers``; GLM-5.3 stores them under ``model.layers``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from freetoken.layers.quantization import QuantKind

_NVFP4_ALGOS = ("NVFP4", "W4A16_NVFP4")


@dataclass(frozen=True)
class DiskExpertKeys:
    """How ``--moe-disk`` finds one family's routed-expert tensors in a safetensors index."""

    pattern: re.Pattern[str]
    suffixes: dict[str, str]
    pieces: int
    names: str
    where: str
    input_pattern: re.Pattern[str] | None = None


def projection_disk_keys(prefix: str, kind: QuantKind) -> DiskExpertKeys | None:
    """Keys for ``{prefix}.{L}.mlp.experts.{E}.{gate,up,down}_proj``. None for other kinds."""
    stem = rf"^{re.escape(prefix)}\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    proj = r"(?P<proj>gate|up|down)_proj\."
    where = f"{prefix}.{{L}}.mlp.experts.{{E}}.{{gate,up,down}}_proj."
    if kind is QuantKind.NVFP4:
        return DiskExpertKeys(
            pattern=re.compile(stem + proj + r"(?P<kind>weight_scale_2|weight_scale|weight)$"),
            suffixes={"weight": "", "weight_scale": "_scale", "weight_scale_2": "_global"},
            pieces=9,
            names="{weight,weight_scale,weight_scale_2}",
            where=where,
            input_pattern=re.compile(stem + proj + r"input_scale$"),
        )
    if kind is QuantKind.FP8_BLOCK:
        return DiskExpertKeys(
            pattern=re.compile(stem + proj + r"(?P<kind>weight|weight_scale_inv)$"),
            suffixes={"weight": "", "weight_scale_inv": "_scale"},
            pieces=6,
            names="{weight,weight_scale_inv}",
            where=where,
        )
    return None


def routed_expert_quant(hf_config: Any, detected: str) -> str:
    """Expert format for a ModelOpt ``MIXED_PRECISION`` checkpoint.

    The top-level algo is ``mixed_precision``. The routed-expert algo is the
    ``quantized_layers`` entry whose key ends in ``.mlp.experts``. Any other
    detected format is left as it is, including a plain ``fp8`` checkpoint.
    """
    if detected != "mixed_precision":
        return detected
    quant = getattr(hf_config, "quantization_config", None)
    if quant is None:
        return detected
    get = quant.get if isinstance(quant, dict) else (lambda k, d=None: getattr(quant, k, d))
    layers = get("quantized_layers") or {}
    if not isinstance(layers, dict):
        return detected
    for key, spec in layers.items():
        if not str(key).endswith(".mlp.experts"):
            continue
        layer_algo = spec.get("quant_algo") if isinstance(spec, dict) else spec
        if str(layer_algo or "").upper() in _NVFP4_ALGOS:
            return "nvfp4"
    return detected
